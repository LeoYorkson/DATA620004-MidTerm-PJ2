# Mask R-CNN


## 环境配置：
* Python3.10
* Pytorch2.2.2
* 详细环境配置见`requirements.txt`

## 文件结构：
```
  ├── backbone: 特征提取网络
  ├── network_files: Mask R-CNN网络
  ├── train_utils: 训练验证相关模块（包括coco验证相关）
  ├── my_dataset_coco.py: 自定义dataset用于读取COCO2017数据集
  ├── my_dataset_voc.py: 自定义dataset用于读取Pascal VOC数据集
  ├── train.py: 单GPU/CPU训练脚本
  ├── train_multi_GPU.py: 针对使用多GPU的用户使用
  ├── predict.py: 简易的预测脚本，使用训练好的权重进行预测
  ├── validation.py: 利用训练好的权重验证/测试数据的COCO指标，并生成record_mAP.txt文件
  └── transforms.py: 数据预处理（随机水平翻转图像以及bboxes、将PIL图像转为Tensor）
```

## 预训练权重下载地址（下载后放入当前文件夹中）：
* Resnet50预训练权重 https://download.pytorch.org/models/resnet50-0676ba61.pth (注意，下载预训练权重后要重命名，
比如在train.py中读取的是`resnet50.pth`文件，不是`resnet50-0676ba61.pth`)
* Mask R-CNN(Resnet50+FPN)预训练权重 https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth (注意，
载预训练权重后要重命名，比如在train.py中读取的是`maskrcnn_resnet50_fpn_coco.pth`文件，不是`maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth`)
 
 
## 数据集，本例程使用的有COCO2017数据集和Pascal VOC2012数据集

### Pascal VOC2012数据集

* 解压后得到的文件夹结构如下：
```
VOCdevkit
    └── VOC2012
         ├── Annotations               所有的图像标注信息(XML文件)
         ├── ImageSets
         │   ├── Action                人的行为动作图像信息
         │   ├── Layout                人的各个部位图像信息
         │   │
         │   ├── Main                  目标检测分类图像信息
         │   │     ├── train.txt       训练集(5717)
         │   │     ├── val.txt         验证集(5823)
         │   │     └── trainval.txt    训练集+验证集(11540)
         │   │
         │   └── Segmentation          目标分割图像信息
         │         ├── train.txt       训练集(1464)
         │         ├── val.txt         验证集(1449)
         │         └── trainval.txt    训练集+验证集(2913)
         │
         ├── JPEGImages                所有图像文件
         ├── SegmentationClass         语义分割png图（基于类别）
         └── SegmentationObject        实例分割png图（基于目标）
```

## 训练方法
* 准备好数据集
* 下载好对应预训练模型权重
* 设置好`--num-classes=20`(适配VOC数据集)和`--data-path`
* 若要使用单GPU训练直接使用train.py训练脚本
* 若要使用多GPU训练，使用`torchrun --nproc_per_node=8 train_multi_GPU.py`指令,`nproc_per_node`参数为使用GPU数量

## 注意事项
1. 在使用训练脚本时，注意要将`--data-path`设置为自己存放数据集的**根目录**：
```
# 要使用Pascal VOC数据集，启用自定义数据集读取VOCInstances并数据集解压到成/data/VOCdevkit目录下
python train.py --data-path /data/VOCdevkit
```
2. 训练过程中保存的`det_results.txt`(目标检测任务)以及`seg_results.txt`(实例分割任务)是每个epoch在验证集上的COCO指标，前12个值是COCO指标，后面两个值是训练平均损失以及学习率
3. 在使用预测脚本时，将`weights_path`设置为生成的权重路径。

## 可视化
在pred.py文件中修改`img_path`和`weights_path`，以及输出路径，直接运行即可，得到的是既有第一阶段又有最终检测结果的对比图。如果只想得到最终检测图，可以将`comparison` 设置为`False`即可。
```
python predict.py
```

