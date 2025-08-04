# YOLO Pipeline

Utilities and scripts for training a YOLOv8 model on custom datasets.

## Installation

```bash
npm install
```

Running `npm install` triggers a `postinstall` script that installs the Python package and its dependencies via `pip`. It also exposes the following command line tools:

- `fetch-s3-dataset` – download images and annotations from S3.
- `split-dataset` – split images and labels into train/val/test sets.
- `train-yolo` – train a YOLOv8 model using `ultralytics`.

## Usage

See `--help` for each command for detailed options.

