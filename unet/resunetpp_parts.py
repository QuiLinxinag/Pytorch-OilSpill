import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block"""
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = F.adaptive_avg_pool2d(x, 1).view(b, c)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y

class ResidualBlock(nn.Module):
    """Residual Block with SE"""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, bias=False) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.bn_shortcut(self.shortcut(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        return self.relu(out)

class Down(nn.Module):
    """Downscaling with maxpool then Residual Block"""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_res = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlock(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_res(x)

class Up(nn.Module):
    """Upscaling then Residual Block with multi-scale fusion"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2)
        self.conv = ResidualBlock(in_channels, out_channels)

    def forward(self, x1, x2, x_skip=None):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        tensors = [x2, x1]
        if x_skip is not None:
            # 對 x_skip 做 padding 使其與 x1 尺寸一致
            diffY_skip = x1.size()[2] - x_skip.size()[2]
            diffX_skip = x1.size()[3] - x_skip.size()[3]
            x_skip = F.pad(x_skip, [diffX_skip // 2, diffX_skip - diffX_skip // 2,
                                    diffY_skip // 2, diffY_skip - diffY_skip // 2])
            tensors.append(x_skip)
        x = torch.cat(tensors, dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x):
        return self.conv(x)

class DeepSupervisionHead(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(DeepSupervisionHead, self).__init__()
        self.conv = nn.Conv2d(in_channels, n_classes, 1)

    def forward(self, x):
        return self.conv(x)