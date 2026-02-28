#!/usr/bin/env python3
import argparse
import csv
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


DEFAULT_TRAIN_DIR = "/home/guangyu/patrick/CAP5516/Assignment1/datasets/chest-xray-pneumonia/chest_xray/train"
DEFAULT_VAL_DIR = "/home/guangyu/patrick/CAP5516/Assignment1/datasets/chest-xray-pneumonia/chest_xray/val"


def parse_args():
    parser = argparse.ArgumentParser(description="Train ResNet-18 for chest X-ray classification.")
    parser.add_argument("--train-dir", type=str, default=DEFAULT_TRAIN_DIR)
    parser.add_argument("--val-dir", type=str, default=DEFAULT_VAL_DIR)
    parser.add_argument("--mode", type=str, choices=["scratch", "finetune"], default="finetune")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3, help="Base learning rate before linear scaling.")
    parser.add_argument("--base-batch-size", type=int, default=64, help="Reference batch size for linear LR scaling.")
    parser.add_argument("--disable-linear-scaling-lr", action="store_true")
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--disable-no-bias-decay", action="store_true")
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--warmup-epochs", type=int, default=3)
    parser.add_argument("--disable-warmup", action="store_true")
    parser.add_argument("--disable-cosine-lr-decay", action="store_true")
    parser.add_argument("--enable-xavier-init", action="store_true")
    parser.add_argument("--random-erasing-prob", type=float, default=0.25)
    parser.add_argument("--random-erasing-scale-min", type=float, default=0.02)
    parser.add_argument("--random-erasing-scale-max", type=float, default=0.2)
    parser.add_argument("--cutout-size", type=int, default=32)
    parser.add_argument("--cutout-prob", type=float, default=0.5)
    parser.add_argument("--num-workers", type=int, default=min(8, os.cpu_count() or 4))
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--val-interval", type=int, default=5)
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class Cutout:
    def __init__(self, size: int, p: float = 0.5):
        self.size = size
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return img
        _, h, w = img.shape
        cutout_size = min(self.size, h, w)
        y = random.randint(0, h - 1)
        x = random.randint(0, w - 1)
        y1 = max(0, y - cutout_size // 2)
        y2 = min(h, y + cutout_size // 2)
        x1 = max(0, x - cutout_size // 2)
        x2 = min(w, x + cutout_size // 2)
        img[:, y1:y2, x1:x2] = 0.0
        return img


class LabelSmoothingCE(nn.Module):
    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = logits.size(1)
        log_probs = torch.log_softmax(logits, dim=1)
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / max(1, num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return torch.mean(torch.sum(-true_dist * log_probs, dim=1))


def build_model(mode: str, num_classes: int = 2):
    if mode == "finetune":
        try:
            model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except (AttributeError, TypeError):
            model = models.resnet18(pretrained=True)
    else:
        try:
            model = models.resnet18(weights=None)
        except TypeError:
            model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def init_xavier(model: nn.Module):
    for module in model.modules():
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)):
            if module.weight is not None:
                nn.init.ones_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)


def get_param_groups(model: nn.Module, weight_decay: float, no_bias_decay: bool):
    if not no_bias_decay:
        return model.parameters()

    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name.endswith(".bias") or "bn" in name.lower() or "downsample.1" in name:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def get_loaders(
    train_dir,
    val_dir,
    batch_size,
    num_workers,
    img_size,
    random_erasing_prob,
    random_erasing_scale_min,
    random_erasing_scale_max,
    cutout_size,
    cutout_prob,
):
    train_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            Cutout(size=cutout_size, p=cutout_prob),
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=(random_erasing_scale_min, random_erasing_scale_max),
                ratio=(0.3, 3.3),
                value=0.0,
            ),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    eval_tfms = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    train_ds = datasets.ImageFolder(train_dir, transform=train_tfms)
    val_ds = datasets.ImageFolder(val_dir, transform=eval_tfms)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_ds, val_ds, train_loader, val_loader


def run_eval(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            loss_sum += loss.item() * x.size(0)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    return loss_sum / max(1, total), correct / max(1, total)


def main():
    args = parse_args()
    set_seed(args.seed)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"resnet18_{args.mode}_{stamp}"
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    log_path = run_dir / "log.log"
    logger = logging.getLogger("train_resnet18")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    csv_path = run_dir / "metrics.csv"
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.DictWriter(
        csv_file,
        fieldnames=[
            "epoch",
            "train_loss",
            "train_acc",
            "lr",
            "epoch_seconds",
            "val_loss",
            "val_acc",
        ],
    )
    csv_writer.writeheader()
    csv_file.flush()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    use_linear_scaling_lr = not args.disable_linear_scaling_lr
    no_bias_decay = not args.disable_no_bias_decay
    use_warmup = (not args.disable_warmup) and (args.warmup_epochs > 0)
    use_cosine_lr_decay = not args.disable_cosine_lr_decay
    effective_lr = args.lr * (args.batch_size / float(args.base_batch_size)) if use_linear_scaling_lr else args.lr

    train_ds, val_ds, train_loader, val_loader = get_loaders(
        args.train_dir,
        args.val_dir,
        args.batch_size,
        args.num_workers,
        args.img_size,
        args.random_erasing_prob,
        args.random_erasing_scale_min,
        args.random_erasing_scale_max,
        args.cutout_size,
        args.cutout_prob,
    )
    class_names = train_ds.classes
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes, found {len(class_names)}: {class_names}")

    model = build_model(args.mode, num_classes=2).to(device)
    if args.enable_xavier_init:
        init_xavier(model)
        logger.info("Xavier init enabled.")

    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    except TypeError:
        criterion = LabelSmoothingCE(smoothing=args.label_smoothing)

    optimizer = optim.AdamW(
        get_param_groups(model, args.weight_decay, no_bias_decay=no_bias_decay),
        lr=effective_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if use_cosine_lr_decay:
        if use_warmup and args.warmup_epochs < args.epochs:
            warmup_scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=args.warmup_epochs,
            )
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, args.epochs - args.warmup_epochs),
            )
            scheduler = optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[args.warmup_epochs],
            )
        elif use_warmup and args.warmup_epochs >= args.epochs:
            scheduler = optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=args.epochs,
            )
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif use_warmup:
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=min(args.warmup_epochs, args.epochs),
        )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    writer = SummaryWriter(log_dir=str(run_dir / "tb"))
    writer.add_text("config/mode", args.mode)
    writer.add_text("config/classes", ", ".join(class_names))
    writer.add_text("config/train_dir", str(Path(args.train_dir).resolve()))
    writer.add_text("config/val_dir", str(Path(args.val_dir).resolve()))
    writer.add_text("config/effective_lr", f"{effective_lr:.8e}")
    writer.add_text("config/linear_scaling_lr", str(use_linear_scaling_lr))
    writer.add_text("config/no_bias_decay", str(no_bias_decay))
    writer.add_text("config/label_smoothing", str(args.label_smoothing))
    writer.add_text("config/warmup_enabled", str(use_warmup))
    writer.add_text("config/warmup_epochs", str(args.warmup_epochs))
    writer.add_text("config/cosine_lr_decay", str(use_cosine_lr_decay))
    writer.add_text("config/random_erasing_prob", str(args.random_erasing_prob))
    writer.add_text("config/cutout_size", str(args.cutout_size))
    writer.add_text("config/cutout_prob", str(args.cutout_prob))

    logger.info(
        "Tricks | xavier_init=%s warmup=%s no_bias_decay=%s label_smoothing=%.3f random_erasing_prob=%.2f cutout_prob=%.2f linear_scaling_lr=%s cosine_lr_decay=%s",
        args.enable_xavier_init,
        use_warmup,
        no_bias_decay,
        args.label_smoothing,
        args.random_erasing_prob,
        args.cutout_prob,
        use_linear_scaling_lr,
        use_cosine_lr_decay,
    )
    logger.info("Base LR=%.8e Effective LR=%.8e", args.lr, effective_lr)

    hparams = {
        "mode": args.mode,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "effective_lr": effective_lr,
        "base_batch_size": args.base_batch_size,
        "linear_scaling_lr": use_linear_scaling_lr,
        "weight_decay": args.weight_decay,
        "no_bias_decay": no_bias_decay,
        "label_smoothing": args.label_smoothing,
        "warmup_enabled": use_warmup,
        "warmup_epochs": args.warmup_epochs,
        "cosine_lr_decay": use_cosine_lr_decay,
        "enable_xavier_init": args.enable_xavier_init,
        "random_erasing_prob": args.random_erasing_prob,
        "random_erasing_scale_min": args.random_erasing_scale_min,
        "random_erasing_scale_max": args.random_erasing_scale_max,
        "cutout_size": args.cutout_size,
        "cutout_prob": args.cutout_prob,
        "num_workers": args.num_workers,
        "img_size": args.img_size,
        "val_interval": args.val_interval,
        "seed": args.seed,
        "train_dir": str(Path(args.train_dir).resolve()),
        "val_dir": str(Path(args.val_dir).resolve()),
    }
    hparams_yaml_path = run_dir / "hparams.yaml"
    OmegaConf.save(config=OmegaConf.create(hparams), f=str(hparams_yaml_path))
    logger.info("Saved hyperparameters YAML: %s", hparams_yaml_path)

    best_val_acc = -1.0
    global_step = 0
    last_train_loss = 0.0
    last_train_acc = 0.0
    last_val_loss = 0.0
    last_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pred = logits.argmax(dim=1)
            running_loss += loss.item() * x.size(0)
            running_correct += (pred == y).sum().item()
            running_total += y.size(0)

            writer.add_scalar("train/iter_loss", loss.item(), global_step)
            global_step += 1

        if scheduler is not None:
            scheduler.step()

        train_loss = running_loss / max(1, running_total)
        train_acc = running_correct / max(1, running_total)
        epoch_time = time.time() - epoch_start
        last_train_loss = train_loss
        last_train_acc = train_acc

        writer.add_scalar("train/epoch_loss", train_loss, epoch)
        writer.add_scalar("train/epoch_acc", train_acc, epoch)
        writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], epoch)
        writer.add_scalar("train/epoch_seconds", epoch_time, epoch)

        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            "Epoch [%03d/%d] train_loss=%.4f train_acc=%.4f lr=%.2e",
            epoch,
            args.epochs,
            train_loss,
            train_acc,
            current_lr,
        )

        should_validate = (epoch % args.val_interval == 0) or (epoch == args.epochs)
        val_loss = ""
        val_acc = ""
        if should_validate:
            val_loss, val_acc = run_eval(model, val_loader, criterion, device)
            last_val_loss = val_loss
            last_val_acc = val_acc
            writer.add_scalar("val/loss", val_loss, epoch)
            writer.add_scalar("val/acc", val_acc, epoch)
            logger.info("  -> val_loss=%.4f val_acc=%.4f", val_loss, val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_path = ckpt_dir / "best.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "best_val_acc": best_val_acc,
                        "class_names": class_names,
                        "mode": args.mode,
                        "img_size": args.img_size,
                    },
                    best_path,
                )
                logger.info("  -> Saved best checkpoint: %s", best_path)

        csv_writer.writerow(
            {
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_acc": f"{train_acc:.6f}",
                "lr": f"{current_lr:.8e}",
                "epoch_seconds": f"{epoch_time:.4f}",
                "val_loss": f"{val_loss:.6f}" if should_validate else "",
                "val_acc": f"{val_acc:.6f}" if should_validate else "",
            }
        )
        csv_file.flush()

    last_path = ckpt_dir / "last.pt"
    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_acc": best_val_acc,
            "class_names": class_names,
            "mode": args.mode,
            "img_size": args.img_size,
        },
        last_path,
    )
    logger.info("Saved last checkpoint: %s", last_path)

    hparam_metrics = {
        "hparam/last_train_loss": float(last_train_loss),
        "hparam/last_train_acc": float(last_train_acc),
        "hparam/last_val_loss": float(last_val_loss),
        "hparam/last_val_acc": float(last_val_acc),
        "hparam/best_val_acc": float(best_val_acc),
    }
    writer.add_hparams(hparams, hparam_metrics)
    logger.info("Logged hyperparameters to TensorBoard with add_hparams().")

    logger.info("TensorBoard log dir: %s", run_dir / "tb")
    logger.info("Log file: %s", log_path)
    logger.info("CSV metrics file: %s", csv_path)
    writer.close()
    csv_file.close()


if __name__ == "__main__":
    main()
