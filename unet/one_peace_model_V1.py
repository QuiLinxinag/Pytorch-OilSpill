# 匯入必要的函式庫
import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from .one_peace_parts_V1 import ConvBlock, UpsampleBlock

# 定義 ONE-PEACE 模型的主類別
class OnePeaceModel(nn.Module):
    def __init__(self, n_channels, n_classes, use_checkpoint: bool = False):
        super(OnePeaceModel, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.use_checkpoint = use_checkpoint
        # 其他模型結構定義

        # 初始卷積層
        self.initial_conv = nn.Conv2d(n_channels, 64, kernel_size=3, padding=1)
        
    # 下採樣層 (每個 block 後會做 pooling)
        self.down1 = ConvBlock(64, 128)
        self.down2 = ConvBlock(128, 256)
        self.down3 = ConvBlock(256, 512)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    # 瓶頸層
        self.bottleneck = ConvBlock(512, 1024)
        
        # 上採樣層
        self.up1 = UpsampleBlock(1024, 512)
        self.up2 = UpsampleBlock(512, 256)
        self.up3 = UpsampleBlock(256, 128)
        
    # 最終卷積層: 輸出通道數應該為類別數 (n_classes)
        self.final_conv = nn.Conv2d(128, n_classes, kernel_size=1)
    
    def forward(self, x):
        # 初始卷積層的輸出
        # 記住輸入的空間尺寸，最後保證輸出與之對齊
        in_h, in_w = x.size(-2), x.size(-1)
        x1 = self.initial_conv(x)
        
        # 下採樣層的輸出
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # 下採樣 (每個 down block 後 pooling)
        x2p = self.pool(x2)
        x3 = self.down2(x2p)
        x3p = self.pool(x3)
        x4 = self.down3(x3p)
        x4p = self.pool(x4)

        # 瓶頸層的輸出 (對 pooled feature 使用 checkpoint 時需傳入 use_reentrant)
        if self.use_checkpoint:
            x_bottleneck = cp.checkpoint(self.bottleneck, x4p, use_reentrant=False)
        else:
            x_bottleneck = self.bottleneck(x4p)
        
        # 上採樣層的輸出
        x_up1 = self.up1(x_bottleneck)
        x_up2 = self.up2(x_up1)
        x_up3 = self.up3(x_up2)
        
        # 最終輸出層
        output = self.final_conv(x_up3)
        # 若輸出空間尺寸與輸入不同，插值回輸入大小以確保 label/輸出對齊
        if output.dim() == 4 and (output.size(-2) != in_h or output.size(-1) != in_w):
            output = nn.functional.interpolate(output, size=(in_h, in_w), mode='bilinear', align_corners=False)
        return output
