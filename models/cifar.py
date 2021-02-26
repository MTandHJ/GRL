


import torch
import torch.nn as nn
from .components import RevGrad





class Gf(nn.Module):
    """
    for features extraction
    """
    def __init__(self, dim_feature=256):
        super(Gf, self).__init__()
        self.dim_feature = dim_feature

        self.conv = nn.Sequential(  # 3 x 32 x 32
            nn.Conv2d(3, 64, 3),  # 64 x 30 x 30
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3),  # 64 x 28 x 28
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # 64 x 14 x 14
            nn.Conv2d(64, 128, 3),  # 128 x 12 x 12
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3),  # 128 x 10 x 10
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)  # 128 x 5 x 5
        )
        
        self.apply(init_paras)

    def forward(self, inputs):
        features = self.conv(inputs).flatten(start_dim=1)
        return features


class Gc(nn.Module):
    """
    for classification upon given features
    """
    def __init__(self, num_classes=10):
        super(Gc, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(128 * 5 * 5, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes)
        )

        self.apply(init_paras)

    def forward(self, features):
        return self.dense(features)


class Gd(nn.Module):
    """
    0 | 1
    """
    def __init__(self):
        super(Gd, self).__init__()
        self.dense = nn.Sequential(
            nn.Linear(128 * 5 * 5, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

        self.revgrad = RevGrad.apply # GRL layer
        self.apply(init_paras)

    def forward(self, features):
        return self.dense(self.revgrad(features)).squeeze()
        
@torch.no_grad()
def init_paras(cls):
    for m in cls.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.)
            nn.init.constant_(m.bias, 0.)

