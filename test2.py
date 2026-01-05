import torch
print(torch.cuda.is_available())  # 若為 True，表示 CUDA 可用
print(torch.cuda.device_count()) # 顯示可用 GPU 的數量
print(torch.cuda.get_device_name(0))  # 顯示 GPU 名稱（若可用）
