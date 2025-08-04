#!/usr/bin/env python3
import os
import argparse
import random
import shutil
from pathlib import Path

def split_dataset_three_way(
    images_dir: Path,
    labels_dir: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.2,
    seed: int = 42
):
    """
    Splits images+labels into train/val/test directories.
    - train_ratio + val_ratio <= 1.0
    - test_ratio is implied as (1 - train_ratio - val_ratio)
    """
    print(f"[+] Splitting dataset: images={images_dir}, labels={labels_dir}, output={output_dir}")
    # 1) List all images
    IM_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    all_images = [
        f for f in sorted(images_dir.iterdir())
        if f.is_file() and f.suffix.lower() in IM_EXTS
    ]
    if not all_images:
        print(f"[!] No images found in {images_dir}")
        return

    # 2) Pair each image with its .txt
    pairs = []
    missing = []
    for img_path in all_images:
        lbl_path = labels_dir / (img_path.stem + ".txt")
        if lbl_path.exists():
            pairs.append((img_path, lbl_path))
        else:
            missing.append(img_path.name)
    if missing:
        print(f"[!] Skipping {len(missing)} images without labels. Examples: {missing[:3]}")
    if not pairs:
        print("[!] No matching image/label pairs found.")
        return

    # 3) Shuffle and split
    random.seed(seed)
    random.shuffle(pairs)

    n_total = len(pairs)
    n_train = int(n_total * train_ratio)
    n_val   = int(n_total * val_ratio)
    n_test  = n_total - n_train - n_val

    train_pairs = pairs[:n_train]
    val_pairs   = pairs[n_train : n_train + n_val]
    test_pairs  = pairs[n_train + n_val : ]

    print(f"[+] Total pairs: {n_total}")
    print(f"    → Train: {len(train_pairs)}")
    print(f"    → Val:   {len(val_pairs)}")
    print(f"    → Test:  {len(test_pairs)}")

    # 4) Create folders
    for sub in ["train", "val", "test"]:
        (output_dir / "images" / sub).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / sub).mkdir(parents=True, exist_ok=True)

    # 5) Copy files
    def copy_subset(subset, img_dest, lbl_dest):
        for img_file, lbl_file in subset:
            shutil.copy2(img_file, img_dest / img_file.name)
            shutil.copy2(lbl_file, lbl_dest / lbl_file.name)

    copy_subset(train_pairs, output_dir / "images" / "train", output_dir / "labels" / "train")
    copy_subset(val_pairs,   output_dir / "images" / "val",   output_dir / "labels" / "val")
    copy_subset(test_pairs,  output_dir / "images" / "test",  output_dir / "labels" / "test")

    print(f"[+] Done copying to {output_dir}/images/{{train,val,test}} and labels/{{train,val,test}}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split images+labels into train/val/test for YOLO."
    )
    parser.add_argument("--images_dir", "-i", type=Path, required=True,
                        help="Folder containing all image files.")
    parser.add_argument("--labels_dir", "-l", type=Path, required=True,
                        help="Folder containing all .txt label files (same basenames).")
    parser.add_argument("--output_dir", "-o", type=Path, required=True,
                        help="Root folder for images/{train,val,test} & labels/{train,val,test}.")
    parser.add_argument("--train_ratio", "-r", type=float, default=0.7,
                        help="Fraction for training (default=0.7)")
    parser.add_argument("--val_ratio", "-v", type=float, default=0.2,
                        help="Fraction for validation (default=0.2)")
    parser.add_argument("--seed", "-s", type=int, default=42,
                        help="Random seed (default=42)")
    args = parser.parse_args()

    if args.train_ratio + args.val_ratio >= 1.0:
        parser.error("train_ratio + val_ratio must be < 1.0 (test gets the remainder).")

    split_dataset_three_way(
        images_dir=args.images_dir,
        labels_dir=args.labels_dir,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed
    )
