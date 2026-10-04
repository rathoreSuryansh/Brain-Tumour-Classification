"""Member 1 - MLP (classical baseline).

Flattens the 3x224x224 image and classifies it with three fully-connected
layers. No convolution, no attention: a lower bound for how much architectural
inductive bias contributes on this task.
"""
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, num_classes=4, img_size=224):
        super().__init__()
        self.flatten = nn.Flatten()
        self.net = nn.Sequential(
            nn.Linear(3*img_size*img_size, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.net(self.flatten(x))
