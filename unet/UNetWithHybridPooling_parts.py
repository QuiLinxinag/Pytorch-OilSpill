import torch
import torch.nn as nn
import torch.nn.functional as F

# DoubleConv 模組：執行兩層連續卷積，包含批次標準化與 ReLU 激活
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

# Down 模組：執行最大池化並跟隨 DoubleConv 操作
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

# 新增的機制：混合式深度池化（Hybrid Pooling）用於更好的特徵提取
class HybridPooling(nn.Module):
    def __init__(self, kernel_size=2, mode='max_avg'):
        super(HybridPooling, self).__init__()
        self.mode = mode
        self.max_pool = nn.MaxPool2d(kernel_size)
        self.avg_pool = nn.AvgPool2d(kernel_size)

    def forward(self, x):
        if self.mode == 'max_avg':
            return (self.max_pool(x) + self.avg_pool(x)) / 2
        elif self.mode == 'max':
            return self.max_pool(x)
        elif self.mode == 'avg':
            return self.avg_pool(x)
        else:
            raise ValueError("模式必須是 'max_avg', 'max' 或 'avg'")

# Up 模組：執行上採樣並跟隨 DoubleConv 操作
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# OutConv 模組：執行最後的卷積以減少輸出通道數
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
