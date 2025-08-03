#!/usr/bin/env bash
# Train YOLOv8 model for exercise/correction detection.
# Usage: ./train_yolo.sh [epochs] [batch_size] [imgsz] [model] [data_cfg] [project_dir] [exp_name]
EPOCHS=${1:-200}
BATCH=${2:-16}
IMGSZ=${3:-640}
MODEL=${4:-yolov8n.pt}
DATA_CFG=${5:-data.yaml}
PROJECT_DIR=${6:-models}
EXP_NAME=${7:-exp_ex_corr}
PATIENCE=${8:-15} 

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