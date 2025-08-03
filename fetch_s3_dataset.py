#!/usr/bin/env python3
"""Download images and annotations from S3 metadata.

Each PNG object stored in the provided S3 buckets is expected to carry
YOLO-style bounding boxes inside custom metadata headers named
``x-amz-meta-<class>`` where ``<class>`` is one of the supported class
names. The header value should be a base64 encoded string containing one
or more lines in the form ``<class_id> x_center y_center width height``
(normalized coordinates) or just ``x_center y_center width height``.

The script downloads the PNG file to an ``images/`` directory and writes
its corresponding label file to ``labels/``.

Example usage:
    python fetch_s3_dataset.py my-bucket-1 my-bucket-2 --prefix data/
"""

from __future__ import annotations

import argparse
import base64
import os
from typing import Iterable

import boto3

CLASSES = ["schematic", "table", "qcm", "preamble", "question_year"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASSES)}


def _write_label(lines: Iterable[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _process_object(s3, bucket: str, key: str, img_dir: str, lbl_dir: str) -> None:
    obj = s3.get_object(Bucket=bucket, Key=key)
    body = obj["Body"].read()
    os.makedirs(img_dir, exist_ok=True)
    img_name = os.path.basename(key)
    img_path = os.path.join(img_dir, img_name)
    with open(img_path, "wb") as f:
        f.write(body)

    label_lines = []
    metadata = obj.get("Metadata", {})
    for cls_name, cls_id in CLASS_TO_ID.items():
        if cls_name in metadata:
            try:
                decoded = base64.b64decode(metadata[cls_name]).decode().strip().splitlines()
            except Exception:
                continue
            for line in decoded:
                parts = line.strip().split()
                if not parts:
                    continue
                # Ignore provided class id if present; use cls_id instead
                if len(parts) == 5 and parts[0].isdigit():
                    parts = parts[1:]
                if len(parts) == 4:
                    label_lines.append(f"{cls_id} {' '.join(parts)}")
    lbl_name = os.path.splitext(img_name)[0] + ".txt"
    lbl_path = os.path.join(lbl_dir, lbl_name)
    _write_label(label_lines, lbl_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("buckets", nargs="+", help="S3 buckets to scan")
    parser.add_argument(
        "--prefix", default="", help="Optional prefix inside each bucket")
    parser.add_argument(
        "--images-dir", default="images", help="Directory for downloaded images")
    parser.add_argument(
        "--labels-dir", default="labels", help="Directory for generated labels")
    args = parser.parse_args()

    s3 = boto3.client("s3")
    for bucket in args.buckets:
        paginator = s3.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=args.prefix):
            for item in page.get("Contents", []):
                key = item["Key"]
                if key.lower().endswith(".png"):
                    _process_object(s3, bucket, key, args.images_dir, args.labels_dir)


if __name__ == "__main__":
    main()
