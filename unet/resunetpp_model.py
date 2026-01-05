import torch
import torch.nn as nn
from .resunetpp_parts import ResidualBlock, Down, Up, OutConv,DeepSupervisionHead


class ResUNetPP(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True, deep_supervision=True):
        super(ResUNetPP, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.deep_supervision = deep_supervision

        self.inc = ResidualBlock(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

        # 多尺度融合
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512 + 512 // factor, 256 // factor, bilinear)
        self.up3 = Up(256 + 256 // factor, 128 // factor, bilinear)
        self.up4 = Up(128 + 128 // factor, 64, bilinear)
        self.outc = OutConv(64, n_classes)

        if deep_supervision:
            self.ds1 = DeepSupervisionHead(512 // factor, n_classes)
            self.ds2 = DeepSupervisionHead(256 // factor, n_classes)
            self.ds3 = DeepSupervisionHead(128 // factor, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # 多尺度融合
        u1 = self.up1(x5, x4)
        u2 = self.up2(u1, x3, x2)
        u3 = self.up3(u2, x2, x1)
        u4 = self.up4(u3, x1)

        logits = self.outc(u4)

        if self.deep_supervision:
            ds1_out = self.ds1(u1)
            ds2_out = self.ds2(u2)
            ds3_out = self.ds3(u3)
            return logits, ds1_out, ds2_out, ds3_out
        else:
            return logits