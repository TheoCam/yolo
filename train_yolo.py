#!/usr/bin/env python3
"""Train a YOLOv8 model given a data config."""

from __future__ import annotations

import argparse
from ultralytics import YOLO
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Path to data.yaml")
    parser.add_argument("--model", default="yolo11n.pt", help="YOLOv8 model weights")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    parser.add_argument("--batch", type=int, default=16, help="Batch size")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    parser.add_argument("--project", default="models", help="Ultralytics project directory")
    parser.add_argument("--name", default="exp_ex_corr", help="Training run name")
    parser.add_argument(
        "--device",
        default=None,
        help="Device to train on, e.g. '0' or 'cpu'. Auto-detect by default",
    )
    args = parser.parse_args()
    if args.device not in (None, "cpu") and not torch.cuda.is_available():
        print("[!] CUDA requested but not available, falling back to CPU")
        args.device = "cpu"
    print(f"[+] Loading model {args.model}")
    model = YOLO(args.model)
    print("[+] Starting training...")
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        project=args.project,
        name=args.name,
        device=args.device,
    )
    print("[+] Training finished")


if __name__ == "__main__":
    main()
