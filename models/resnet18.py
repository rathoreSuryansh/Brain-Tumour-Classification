"""Member 2 - ResNet18 (convolutional, residual connections).

torchvision ResNet18 initialised with ImageNet weights; the final fully-connected
layer is replaced with a 4-way classifier and the whole network is fine-tuned.
"""
import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18

from common.config import PRETRAINED


def get_resnet18(num_classes=4, pretrained=PRETRAINED):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
