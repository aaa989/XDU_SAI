import torch.nn as nn
import torchvision.models as models

# 基于ResNet18的自定义分类模型
class MyModel(nn.Module):
    def __init__(self, num_classes=7, pretrained=True):
        super().__init__()
        self.model = models.resnet18(pretrained=pretrained)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)