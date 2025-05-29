# Sparse R-CNN (MMDetection implementation)

## Preparation
1. Set work dir to SparseRCNN-VOC
2. Download and decompress the dataset
3. Git clone MMDetection and then substitute the provided "sparse-rcnn_r50_fpn_1x_voc.py" for "mmdetection\configs\sparse_rcnn\sparse-rcnn_r50_fpn_1x_coco.py"
## Training
```text
bash SparseRCNN/scripts/run_train.sh
```
You may modify the parameters in the file according to your environment.

The training results mainly consist of: training logs, checkpoints (.pth), a summarization of confuguration by MMDetection (config.py)

## Validation
```text
bash SparseRCNN/scripts/run_eval.sh
```
This will evaluate all checkpoints on VOC2007 test set.

## Visualization
Move the training log and the validation log to "vis" folder and run:
```text
python SparseRCNN/results/vis/vis.py
python SparseRCNN/results/vis/vis_eval.py
```
The plots in the experiment report were created by the two scripts.