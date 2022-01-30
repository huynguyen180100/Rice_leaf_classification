import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, shortcut=None):
        super(ResBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.right = shortcut

    def forward(self, x):
        out = self.left(x)
        residual = x if self.right is None else self.right(x)
        out += residual
        return F.relu(out)

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.pre = nn.Sequential(
            nn.Conv2d(3, 64, 7, 2, 3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.layers1 = self._make_layers(64, 128, 3)
        self.layers2 = self._make_layers(128, 256, 4, stride=2)
        self.layers3 = self._make_layers( 256, 512, 6, stride=2)
        self.layer4 = self._make_layers( 512, 512, 3, stride=2)

        self.fc = nn.Linear(512, num_classes)

    def _make_layers(self, in_channels, out_channels, block_num, stride=1):
        shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,1,stride, bias=False),
                nn.BatchNorm2d(out_channels))
        
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride, shortcut))
        
        for i in range(1, block_num):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = F.adaptive_avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)