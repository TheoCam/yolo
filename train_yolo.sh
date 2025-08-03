#!/usr/bin/env bash
# Train YOLOv8 model for document element detection.
# Usage: ./train_yolo.sh [epochs] [batch_size] [imgsz] [model] [data_cfg] [project_dir] [exp_name]

EPOCHS=${1:-200}
BATCH=${2:-16}
IMGSZ=${3:-640}
MODEL=${4:-XXX.pt}
DATA_CFG=${5:-data.yaml}
PROJECT_DIR=${6:-models}
EXP_NAME=${7:-exp_ex_corr}
PATIENCE=${8:-15}

# If S3 buckets are provided, fetch images and labels then create dataset split
if [ -n "$S3_BUCKETS" ]; then
    if [ -n "$S3_PREFIX" ]; then
        python fetch_s3_dataset.py $S3_BUCKETS --prefix "$S3_PREFIX"
    else
        python fetch_s3_dataset.py $S3_BUCKETS
    fi
    python split_dataset.py \
        -i images \
        -l labels \
        -o dataset \
        -r 0.7 \
        -v 0.2
fi

ultralytics train detect \
    model=$MODEL \
    data=$DATA_CFG \
    epochs=$EPOCHS \
    imgsz=$IMGSZ \
    batch=$BATCH \
    patience=$PATIENCE \
    project=$PROJECT_DIR \
    name=$EXP_NAME \
    tensorboard=True
