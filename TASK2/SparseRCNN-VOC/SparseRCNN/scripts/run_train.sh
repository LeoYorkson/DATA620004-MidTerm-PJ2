#!/bin/bash

CONFIG=SparseRCNN/configs/sparse_rcnn_voc.py
WORK_DIR=SparseRCNN/results
# CUDA_VISIBLE_DEVICES=4 python mmdetection/tools/train.py $CONFIG --work-dir $WORK_DIR
CUDA_VISIBLE_DEVICES=3,4,5 bash mmdetection/tools/dist_train.sh $CONFIG 3 --work-dir $WORK_DIR

echo "训练完成！模型、日志等已保存在 $WORK_DIR 下。"
