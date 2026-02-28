#!/usr/bin/env python3
"""
Download the Chest X-Ray Pneumonia dataset from Kaggle into datasets/.

Dataset URL from assignment:
https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia
"""

from pathlib import Path
import shutil
import sys

import kagglehub

DATASET_HANDLE = "paultimothymooney/chest-xray-pneumonia"
OUTPUT_DIR = Path("datasets")
TARGET_DIR = OUTPUT_DIR / "chest-xray-pneumonia"


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    try:
        print(f"Downloading with kagglehub: {DATASET_HANDLE}")
        print("Note: Kaggle credentials are required (kaggle.json or env vars).")
        cached_dataset_path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    except Exception as exc:
        print(f"Download failed: {exc}")
        print("Make sure your Kaggle API credentials are configured.")
        print("See: https://www.kaggle.com/docs/api")
        return 1

    # Mirror kagglehub cache content into datasets/chest-xray-pneumonia
    shutil.copytree(cached_dataset_path, TARGET_DIR, dirs_exist_ok=True)

    print(f"Cached path: {cached_dataset_path}")
    print(f"Done. Dataset available under: {TARGET_DIR.resolve()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
