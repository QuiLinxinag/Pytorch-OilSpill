# resunet_parts.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DeformableConvBlock(nn.Module):
    """可變形卷積塊"""

    def __init__(self, in_channels, out_channels):
        super(DeformableConvBlock, self).__init__()
        self.offset_conv = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        offset = self.offset_conv(x)
        out = self.deform_conv(x, offset)
        out = self.bn(out)
        out += self.shortcut(x)
        return self.relu(out)

class PyramidPoolingModule(nn.Module):
    """混合金字塔池化模塊"""

    def __init__(self, in_channels, pool_sizes):
        super(PyramidPoolingModule, self).__init__()
        self.stages = nn.ModuleList([nn.AdaptiveAvgPool2d(output_size=s) for s in pool_sizes])
        self.conv = nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            out = stage(x)
            out = F.interpolate(out, size=(h, w), mode='bilinear', align_corners=True)
            pyramids.append(out)
        out = torch.cat(pyramids, dim=1)
        out = self.conv(out)
        out = self.bn(out)
        return self.relu(out)

class Down(nn.Module):
    """下採樣與可變形卷積"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv = DeformableConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        return x

class Up(nn.Module):
    """上採樣與可變形卷積"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DeformableConvBlock(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # 調整尺寸
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Module):
    """輸出卷積層"""

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.ppm = PyramidPoolingModule(in_channels, pool_sizes=[1, 2, 3, 6])
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.ppm(x)
        return self.conv(x)