from torchinfo import summary
import torch
from unet.ResUNetWithShuffle_model import ResUNetWithShuffle
from unet.unet_model_V1 import UNet  # 引入你的模型
from unet.MultiScale_Attention_unet_model_V1 import MultiScaleAttentionUNet
from unet import UNet, AttentionUNet, EnhancedAttentionUNet, UNetWithHybridPooling,ResUNet, ImprovedResUNet, SiameseAttentionUNet, MultiScaleAttentionUNet, SEResUNet,ResUNetPlusPlus,InnovativeResUNet,HybridUNet
from unet.resunet_model import ResUNet
from unet import OnePeaceModel, ResUNetWithShuffle, AttentionUNetShuffle
# 創建模型
model = ImprovedResUNet(n_channels=3, n_classes=5, bilinear=False)  # 根據你的模型參數進行調整

# 選擇使用的裝置 (CPU 或 GPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 使用 torchinfo.summary 列出模型結構
summary(model, input_size=(1, 3, 1250, 650))  # 根據你模型的輸入大小進行調整
