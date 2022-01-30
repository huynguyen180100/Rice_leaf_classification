import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class TransitionLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(TransitionLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)
        return self.pool(out)

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        return torch.cat([x, out], 1)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(BottleneckBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels*4)
        self.conv2 = nn.Conv2d(out_channels*4, out_channels, kernel_size=1, stride=1, bias=False)

    def forward(self, x):
        out = self.conv1(self.relu(self.bn1(x)))
        out = self.conv2(self.relu(self.bn2(out)))
        return torch.cat([x, out], 1)

class DenseBlock(nn.Module):
    def __init__(self, nb_layers, in_channels, growth_rate, block):
        super(DenseBlock, self).__init__()
        self.layer = self._make_layers(block, in_channels, growth_rate, nb_layers)
    def _make_layers(self, block, in_channels, growth_rate, nb_layers):
        layers = []
        for i in range(nb_layers):
            layers.append(block(in_channels + i * growth_rate, growth_rate))
        return nn.Sequential(*layers)

class DenseNet121(nn.Module):
    def __init__(self, depth, num_classes, growth_rate=32,
                reduction=0.5, bottleneck=True):
        super(DenseNet121, self).__init__()
        in_channels = 2 * growth_rate
        n = (depth - 4) / 3
        if bottleneck == True:
            n = n / 2
            block = BottleneckBlock
        else:
            block = BasicBlock
        n = int(n)
        # 1st conv before any dense block
        self.conv1 = nn.Conv2d(3, in_channels, kernel_size=7, stride=2,
                               padding=3, bias=False)
        # 1st block
        self.block1 = DenseBlock(n, in_channels, growth_rate, block)
        in_channels = int(in_channels + n*growth_rate)
        self.trans1 = TransitionLayer(in_channels, int(math.floor(in_channels * reduction)))
        in_channels = int(math.floor(in_channels * reduction))
        # 2nd block
        self.block2 = DenseBlock(n, in_channels, growth_rate, block)
        in_channels = int(in_channels + n*growth_rate)
        self.trans2 = TransitionLayer(in_channels, int(math.floor(in_channels*reduction)))
        in_channels = int(math.floor(in_channels*reduction))
        # 3rd block
        self.block3 = DenseBlock(n, in_channels, growth_rate, block)
        in_channels = int(in_channels + n*growth_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(in_channels, num_classes)
        self.in_channels = in_channels

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.block1(out))
        out = self.trans2(self.block2(out))
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.in_channels)
        return self.fc(out)