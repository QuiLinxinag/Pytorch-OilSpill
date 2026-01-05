import torch
import torch.nn as nn
import torch.nn.functional as F

def channel_shuffle(x, groups):
    batch_size, channels, height, width = x.size()
    assert channels % groups == 0, "通道數必須能被分組數整除"
    group_channels = channels // groups

    x = x.view(batch_size, groups, group_channels, height, width)
    x = x.permute(0, 2, 1, 3, 4).contiguous()
    x = x.view(batch_size, channels, height, width)
    return x

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        se = F.adaptive_avg_pool2d(x, 1)
        se = self.fc1(se)
        se = F.relu(se)
        se = self.fc2(se)
        se = self.sigmoid(se)
        return x * se

class ResidualBlockWithShuffle(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2, dropout_rate=0.2):
        super(ResidualBlockWithShuffle, self).__init__()
        self.groups = groups
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout2d(p=dropout_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False) if in_channels != out_channels else nn.Identity()
        self.bn_shortcut = nn.BatchNorm2d(out_channels) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        identity = self.bn_shortcut(self.shortcut(x))
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = channel_shuffle(out, self.groups)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out += identity
        return self.relu(out)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels, groups=2):
        super(Down, self).__init__()
        self.maxpool_res = nn.Sequential(
            nn.MaxPool2d(2),
            ResidualBlockWithShuffle(in_channels, out_channels, groups)
        )

    def forward(self, x):
        return self.maxpool_res(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True, groups=2):
        super(Up, self).__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = ResidualBlockWithShuffle(in_channels, out_channels, groups)

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