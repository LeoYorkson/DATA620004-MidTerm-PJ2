import torch.nn as nn
from torchvision import models


class MyModel:
    def __init__(self, pretrained=True):
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None  # pretrained or scratch 权重
        self.model = models.resnet18(weights=weights)  # resnet18 模型
        self.model.fc = nn.Linear(self.model.fc.in_features, 102)  # 全连接层，输出102类

    def parameters_tuning(self, lr, lr_fc):
        param_groups = [{'params': getattr(self.model, layer).parameters(), 'lr': lr}
                        for layer in ['conv1', 'bn1', 'layer1', 'layer2', 'layer3', 'layer4']]
        param_groups.append({'params': self.model.fc.parameters(), 'lr': lr_fc})
        return param_groups
