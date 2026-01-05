import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d

class DoubleConv(nn.Module):
    """(deformable convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.offsets1 = nn.Conv2d(in_channels, 18, kernel_size=3, padding=1, bias=False)
        self.offsets2 = nn.Conv2d(mid_channels, 18, kernel_size=3, padding=1, bias=False)
        self.double_conv = nn.Sequential(
            DeformConv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            DeformConv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        offset1 = self.offsets1(x)
        x = self.double_conv[0](x, offset1)
        x = self.double_conv[1](x)
        x = self.double_conv[2](x)
        offset2 = self.offsets2(x)
        x = self.double_conv[3](x, offset2)
        x = self.double_conv[4](x)
        x = self.double_conv[5](x)
        return x

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class HybridPyramidPooling(nn.Module):
    """Hybrid Pyramid Pooling Module"""

    def __init__(self, in_channels, pool_sizes=[1, 2, 3, 6]):
        super().__init__()
        self.stages = nn.ModuleList([self._make_stage(in_channels, size) for size in pool_sizes])
        self.bottleneck = nn.Conv2d(in_channels * (len(pool_sizes) + 1), in_channels, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def _make_stage(self, in_channels, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, x):
        h, w = x.size(2), x.size(3)
        pyramids = [x]
        for stage in self.stages:
            pyramids.append(F.interpolate(stage(x), size=(h, w), mode='bilinear', align_corners=True))
        output = self.bottleneck(torch.cat(pyramids, dim=1))
        return self.relu(output)