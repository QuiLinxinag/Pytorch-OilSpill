import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int=None):
        super(AttentionGate, self).__init__()
        if F_int is None:
            F_int = F_l // 2

        # 根據輸入的 F_g 和 F_l 動態調整通道數
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 進行通道數檢查
        if g.size(1) != self.W_g[0].in_channels:
            raise ValueError(f"輸入 g 的通道數 ({g.size(1)}) 與 W_g 預期的通道數 ({self.W_g[0].in_channels}) 不匹配")

        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.size() != x1.size():
            diffY = x1.size()[2] - g1.size()[2]
            diffX = x1.size()[3] - g1.size()[3]
            g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


class MultiScaleAttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int=None):
        super(MultiScaleAttentionGate, self).__init__()
        if F_int is None:
            F_int = F_l // 2

        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True)

        self.psi = nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True)
        self.sigmoid = nn.Sigmoid()

        # 多尺度卷積
        self.scale1 = nn.Conv2d(F_l, F_l // 2, kernel_size=1)
        self.scale2 = nn.Conv2d(F_l // 2, F_l // 4, kernel_size=3, padding=1)
        self.scale3 = nn.Conv2d(F_l // 4, F_l // 8, kernel_size=5, padding=2)
        
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 計算 Attention 機制
        g1 = self.W_g(g)
        x1 = self.W_x(x)

        if g1.size() != x1.size():
            diffY = x1.size()[2] - g1.size()[2]
            diffX = x1.size()[3] - g1.size()[3]
            g1 = F.pad(g1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        attention_map = self.sigmoid(psi)

        # 多尺度卷積處理
        scale1_out = self.scale1(x)
        scale2_out = self.scale2(scale1_out)
        scale3_out = self.scale3(scale2_out)

        # 結合 Attention Map 與多尺度結果
        multi_scale_out = scale1_out + scale2_out + scale3_out
        return x * attention_map * multi_scale_out

# 雙層卷積模組
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels

        # 檢查 in_channels 是否可以被 2 整除，若不行則設定 groups=1
        groups = 2 if in_channels % 2 == 0 else 1
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5)
        )

    def forward(self, x):
        return self.double_conv(x)


# 下採樣模組
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        
        if x.shape[1] != 512:
            x = nn.Conv2d(1024, 512, kernel_size=1)(x)
        return self.maxpool_conv(x)

# 上採樣模組
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # 如果尺寸不同，使用填充對齊
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# 輸出層
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
