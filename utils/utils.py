import matplotlib.pyplot as plt
import numpy as np

def plot_img_and_mask(img, mask):
    # 確認遮罩中的最大類別數
    classes = mask.max() + 1
    
    # 創建圖表，設定子圖數量為 classes + 1
    fig, ax = plt.subplots(1, classes + 1, figsize=(15, 5))
    
    # 顯示輸入影像
    ax[0].set_title('輸入影像')
    ax[0].imshow(img)
    ax[0].axis('off')  # 隱藏軸線
    
    # 顯示各類別的遮罩
    for i in range(classes):
        ax[i + 1].set_title(f'遮罩 (類別 {i + 1})')
        ax[i + 1].imshow(mask == i, cmap='gray')  # 使用灰階顯示
        ax[i + 1].axis('off')  # 隱藏軸線
    
    # 儲存圖表或顯示
    plt.tight_layout()
    plt.show()

# 測試用假資料
img = np.random.rand(128, 128, 3)  # 假設 128x128 的 RGB 圖像
mask = np.random.randint(0, 5, (128, 128))  # 假設 5 個類別的遮罩
plot_img_and_mask(img, mask)
