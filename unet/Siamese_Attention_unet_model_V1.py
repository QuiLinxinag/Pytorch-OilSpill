import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from .Siamese_Attention_unet_parts_V1 import DoubleConv, Down, Up, OutConv, MultiScaleAttentionGate

class SiameseAttentionUNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(SiameseAttentionUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder for both inputs
        self.inc1 = DoubleConv(n_channels, 64)
        self.inc2 = DoubleConv(n_channels, 64)
        
        self.down1_1 = Down(64, 128)
        self.down1_2 = Down(64, 128)
        
        self.down2_1 = Down(128, 256)
        self.down2_2 = Down(128, 256)

        self.down3_1 = Down(256, 512)
        self.down3_2 = Down(256, 512)

        self.down4_1 = Down(512, 1024)
        self.down4_2 = Down(512, 1024)

        # Attention gates for concatenation
        self.attention1 = MultiScaleAttentionGate(F_g=2048, F_l=1024)  # F_g 設為 2048
        self.attention2 = MultiScaleAttentionGate(F_g=512, F_l=512)
        self.attention3 = MultiScaleAttentionGate(F_g=256, F_l=256)
        self.attention4 = MultiScaleAttentionGate(F_g=128, F_l=128)

        # Upsampling layers
        self.up1 = Up(1024, 1024)  # 修改通道數為 1024
        self.up2 = Up(512, 512)
        self.up3 = Up(256, 256)
        self.up4 = Up(128, 128)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x1, x2):
        # Encoder stream for input 1
        x1_1 = self.inc1(x1)
        x1_2 = self.down1_1(x1_1)
        x1_3 = self.down2_1(x1_2)
        x1_4 = self.down3_1(x1_3)
        x1_5 = self.down4_1(x1_4)

        # Encoder stream for input 2
        x2_1 = self.inc2(x2)
        x2_2 = self.down1_2(x2_1)
        x2_3 = self.down2_2(x2_2)
        x2_4 = self.down3_2(x2_3)
        x2_5 = self.down4_2(x2_4)

        # Concatenate and use attention gates
        x1_5 = torch.cat([x1_5, x2_5], dim=1)
        x = self.attention1(x1_5, x1_4)

        x1_4 = torch.cat([x1_4, x2_4], dim=1)  # 通道數為 512
        x = self.up1(x, x1_4)

        x1_3 = torch.cat([x1_3, x2_3], dim=1)  # 通道數為 256
        x = self.attention2(x, x1_3)
        x = self.up2(x, x1_3)

        x1_2 = torch.cat([x1_2, x2_2], dim=1)  # 通道數為 128
        x = self.attention3(x, x1_2)
        x = self.up3(x, x1_2)

        x1_1 = torch.cat([x1_1, x2_1], dim=1)  # 通道數為 64
        x = self.attention4(x, x1_1)
        x = self.up4(x, x1_1)

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
