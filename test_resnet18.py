#!/usr/bin/env python3
import argparse
from datetime import datetime
from pathlib import Path
import shutil

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


DEFAULT_TEST_DIR = "/home/guangyu/patrick/CAP5516/Assignment1/datasets/chest-xray-pneumonia/chest_xray/test"


class PathImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        image, target = super().__getitem__(index)
        path, _ = self.samples[index]
        return image, target, path


def parse_args():
    parser = argparse.ArgumentParser(description="Test ResNet-18 on chest X-ray test set.")
    parser.add_argument("--test-dir", type=str, default=DEFAULT_TEST_DIR)
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to best.pt or last.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--output-dir", type=str, default="test_results")
    return parser.parse_args()


def build_model(num_classes=2):
    try:
        model = models.resnet18(weights=None)
    except TypeError:
        model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    class_names = ckpt.get("class_names", ["NORMAL", "PNEUMONIA"])
    if len(class_names) != 2:
        raise ValueError(f"Expected 2 classes in checkpoint, got {len(class_names)}: {class_names}")

    model = build_model(num_classes=2).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    tfms = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    test_ds = PathImageFolder(args.test_dir, transform=tfms)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    per_class_correct = torch.zeros(2, dtype=torch.long)
    per_class_total = torch.zeros(2, dtype=torch.long)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / f"test_{stamp}"
    log_dir = run_dir / "tb"
    failure_dir = run_dir / "failed_pneumonia"
    log_dir.mkdir(parents=True, exist_ok=True)
    failure_dir.mkdir(parents=True, exist_ok=True)

    pneumonia_idx = class_names.index("PNEUMONIA") if "PNEUMONIA" in class_names else 1
    saved_failed_pneumonia = []

    with torch.no_grad():
        for x, y, paths in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            loss = criterion(logits, y)

            pred = logits.argmax(dim=1)
            total_loss += loss.item() * x.size(0)
            total_correct += (pred == y).sum().item()
            total_count += y.size(0)

            for cls_idx in range(2):
                cls_mask = y == cls_idx
                per_class_total[cls_idx] += cls_mask.sum().cpu()
                per_class_correct[cls_idx] += ((pred == y) & cls_mask).sum().cpu()

            for sample_idx, src_path in enumerate(paths):
                if len(saved_failed_pneumonia) >= 5:
                    break
                true_label = y[sample_idx].item()
                pred_label = pred[sample_idx].item()
                if true_label == pneumonia_idx and pred_label != true_label:
                    src = Path(src_path)
                    dst = failure_dir / (
                        f"failed_pneumonia_{len(saved_failed_pneumonia)+1}_"
                        f"pred_{class_names[pred_label]}_{src.name}"
                    )
                    shutil.copy2(src, dst)
                    saved_failed_pneumonia.append(str(dst))

    test_loss = total_loss / max(1, total_count)
    test_acc = total_correct / max(1, total_count)
    per_class_acc = per_class_correct.float() / torch.clamp(per_class_total.float(), min=1.0)

    writer = SummaryWriter(log_dir=str(log_dir))
    writer.add_text("test/checkpoint", str(Path(args.checkpoint).resolve()))
    writer.add_text("test/test_dir", str(Path(args.test_dir).resolve()))
    writer.add_text("test/failed_pneumonia_dir", str(failure_dir.resolve()))
    writer.add_scalar("test/loss", test_loss, 0)
    writer.add_scalar("test/acc", test_acc, 0)
    for i, name in enumerate(class_names):
        writer.add_scalar(f"test_per_class_acc/{name}", per_class_acc[i].item(), 0)
    writer.close()

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")
    for i, name in enumerate(class_names):
        print(f"{name} accuracy: {per_class_acc[i].item():.4f} ({per_class_correct[i].item()}/{per_class_total[i].item()})")
    print(f"TensorBoard log dir: {log_dir}")
    if saved_failed_pneumonia:
        print("Saved failed PNEUMONIA examples:")
        for path in saved_failed_pneumonia:
            print(path)
    else:
        print("No failed PNEUMONIA examples were found to save.")


if __name__ == "__main__":
    main()
