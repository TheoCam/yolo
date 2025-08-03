#!/usr/bin/env python3
"""Orchestrate YOLOv8 training using images stored across multiple S3 buckets.

Running ``python mastermind.py`` will:
  1. Download all PNG images and their YOLO labels encoded in S3 object
     metadata from a predefined list of buckets.
  2. Split the gathered data into train/val/test subsets.
  3. Launch a YOLOv8 training run on the prepared dataset.

The script assumes AWS credentials with ``s3:GetObject`` and
``s3:ListBucket`` permissions are configured in the environment.
"""

from __future__ import annotations

import os
import subprocess
from ultralytics import YOLO

# Buckets storing the training images.
BUCKETS = [
    "acc-amiens",
    "acc-bobigny",
    "acc-bordeaux",
    "acc-dijon",
    "acc-lille",
    "acc-marseille",
    "acc-montpellier",
    "acc-saclay",
    "acc-sorbonne",
    "acc-udp",
    "acc-upec",
    "acc-uvsq",
    "actu-bobigny",
    "actu-sorbonne",
    "actu-upc",
    "fiches-amiens",
    "fiches-bobigny",
    "fiches-bordeaux",
    "fiches-dijon",
    "fiches-lille",
    "fiches-marseille",
    "fiches-montpellier",
    "fiches-saclay",
    "fiches-sorbonne",
    "fiches-udp",
    "fiches-upec",
    "fiches-uvsq",
]

# Ensure the AWS client knows which region to hit if none is configured.
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")


def fetch_from_s3() -> None:
    """Download images and labels for all buckets using ``fetch_s3_dataset.py``."""
    cmd = ["python", "fetch_s3_dataset.py", *BUCKETS]
    subprocess.run(cmd, check=True)


def split_dataset() -> None:
    """Split downloaded files into train/val/test directories."""
    cmd = [
        "python",
        "split_dataset.py",
        "-i",
        "images",
        "-l",
        "labels",
        "-o",
        "dataset",
        "-r",
        "0.7",
        "-v",
        "0.2",
    ]
    subprocess.run(cmd, check=True)


def train_yolo() -> None:
    """Train a YOLOv8 model on the prepared dataset."""
    model = YOLO("yolo11n.pt")
    model.train(
        data="data.yaml",
        epochs=200,
        imgsz=640,
        batch=16,
        patience=15,
        project="models",
        name="exp_ex_corr",
        tensorboard=True,
    )


def main() -> None:
    fetch_from_s3()
    split_dataset()
    train_yolo()


if __name__ == "__main__":
    main()
