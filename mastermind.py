#!/usr/bin/env python3
"""Orchestrate YOLOv8 training workflow.

Usage: python mastermind.py <workspace_dir>
"""

from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

def run(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python mastermind.py <workspace_dir>", file=sys.stderr)
        sys.exit(1)

    workspace_dir = Path(sys.argv[1]).resolve()
    workspace_dir.mkdir(parents=True, exist_ok=True)

    # Ensure AWS region is set for boto3
    os.environ.setdefault("AWS_DEFAULT_REGION", "eu-north-1")

    images_dir = workspace_dir / "images"
    labels_dir = workspace_dir / "labels"
    metadata_dir = workspace_dir / "metadata"
    dataset_dir = workspace_dir / "dataset"

    script_dir = Path(__file__).resolve().parent

    fetch_script = script_dir / "fetch_s3_dataset.py"
    run([
        sys.executable,
        str(fetch_script),
        "fiches-udp",
        "fiches-sorbonne",
        "--images-dir",
        str(images_dir),
        "--labels-dir",
        str(labels_dir),
        "--metadata-dir",
        str(metadata_dir),
    ])

    split_script = script_dir / "split_dataset.py"
    run([
        sys.executable,
        str(split_script),
        "-i",
        str(images_dir),
        "-l",
        str(labels_dir),
        "-o",
        str(dataset_dir),
        "-r",
        "0.7",
        "-v",
        "0.2",
    ])

    data_yaml_path = workspace_dir / "data.yaml"
    yaml_content = "\n".join(
        [
            f"train: {dataset_dir / 'images' / 'train'}",
            f"val: {dataset_dir / 'images' / 'val'}",
            f"test: {dataset_dir / 'images' / 'test'}",
            "",
            "# Number of classes",
            "nc: 5",
            "",
            "# Class names",
            "names: ['schematic', 'table', 'qcm', 'preamble', 'question_year']",
            "",
        ]
    )
    data_yaml_path.write_text(yaml_content)

    train_script = script_dir / "train_yolo.py"
    model_path = script_dir / "yolo11n.pt"
    run([
        sys.executable,
        str(train_script),
        "--data",
        str(data_yaml_path),
        "--model",
        str(model_path),
    ])

if __name__ == "__main__":
    main()
