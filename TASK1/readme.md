
## 数据处理
1. 数据集下载：在官网下载 Caltech-101 数据集，并将`101_ObjectCategories`置于当前项目目录。
2. 数据集划分: 运行`data_split.py`，在脚本新创建的`data`目录下可找到训练集与测试集。

## 模型训练
1. 参数设定：在main.py中修改超参数`batch_size`, `lr`, `lr_fc`, `epochs`等。
2. 模型训练：运行脚本
```bash
python main.py
```

## 结果查询
1. Loss与Accuracy曲线：日志保存在`tensor_board`文件夹中。其中的文件名为预设参数组合。使用TensorBoard可查看相应曲线。命令行如下：
```bash
python -m tensorboard.main --logdir=tensor_board
```
2. 最优模型：在checkpoint`文件夹中，可找到训练的最优模型。

