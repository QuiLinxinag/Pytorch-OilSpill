import torch
import torch.nn as nn
from .attention_unet_parts_V1 import DoubleConv, Down, Up, OutConv, AttentionGate

class AttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(AttentionUNet, self).__init__()
        self.bilinear = bilinear
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)

        self.attention1 = AttentionGate(F_g=1024, F_l=512)
        self.attention2 = AttentionGate(F_g=512, F_l=256)
        self.attention3 = AttentionGate(F_g=256, F_l=128)
        self.attention4 = AttentionGate(F_g=128, F_l=64)

        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.attention1(x5, x4)
        x = self.up1(x5, x4)
        
        x4 = self.attention2(x, x3)
        x = self.up2(x4, x3)
        
        x3 = self.attention3(x, x2)
        x = self.up3(x3, x2)
        
        x2 = self.attention4(x, x1)
        x = self.up4(x2, x1)
        
        logits = self.outc(x)
        return logits

    def use_checkpointing(self):
        self.inc = torch.utils.checkpoint(self.inc)
        self.down1 = torch.utils.checkpoint(self.down1)
        self.down2 = torch.utils.checkpoint(self.down2)
        self.down3 = torch.utils.checkpoint(self.down3)
        self.down4 = torch.utils.checkpoint(self.down4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)