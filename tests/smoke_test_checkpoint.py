import os
import sys
# 將專案根目錄加入 sys.path，確保可以找到 local package `unet`
PROJ_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJ_ROOT not in sys.path:
    sys.path.insert(0, PROJ_ROOT)

import torch
from unet import OnePeaceModel

def run():
    device = torch.device('cpu')
    model = OnePeaceModel(n_channels=3, n_classes=5, use_checkpoint=True)
    model.to(device)
    model.eval()
    x = torch.randn(1, 3, 128, 128, device=device)
    with torch.no_grad():
        out = model(x)
    print('Input shape:', x.shape)
    print('Output shape:', out.shape)

if __name__ == '__main__':
    run()
