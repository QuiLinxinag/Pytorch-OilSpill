import torch
import sys

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    sys.exit(1)

checkpoint_path = sys.argv[1]
state_dict = torch.load(checkpoint_path, map_location='cpu')

print(f"\n=== Checkpoint: {checkpoint_path} ===")
print(f"Keys count: {len(state_dict)}")

# 抽樣顯示前 20 個 key
print("\nFirst 20 keys:")
for i, key in enumerate(list(state_dict.keys())[:20]):
    shape = state_dict[key].shape if hasattr(state_dict[key], 'shape') else 'N/A'
    print(f"  {i+1}. {key}: {shape}")

# 檢查架構特徵
keys_str = '|'.join(state_dict.keys())
print("\n=== Architecture clues ===")
print(f"Has 'se.' (SE-Net): {'se.' in keys_str}")
print(f"Has 'attention': {'attention' in keys_str.lower()}")
print(f"Has 'double_conv': {'double_conv' in keys_str}")
print(f"Has 'shuffle': {'shuffle' in keys_str.lower()}")
print(f"Has 'res' blocks: {'res' in keys_str.lower()}")
