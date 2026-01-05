import argparse
import logging
import os
import torch
import torch.nn.functional as F
from pathlib import Path
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from torch.optim.lr_scheduler import ReduceLROnPlateau
import wandb
from evaluate import evaluate
from unet import UNet, AttentionUNet,MultiScaleAttentionUNet,SiameseAttentionUNet  # 修改導入 AttentionUNet
from utils.data_loading_test import BasicDataset, CarvanaDataset
from utils.dice_score import dice_coeff, dice_loss, accuracy, multiclass_dice_coeff
import time
from torch.nn import Dropout2d

# 設定資料夾路徑
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

def train_model(
        model,
        device,
        epochs: int = 1,
        batch_size: int = 1,
        learning_rate: float = 1e-4,
        val_percent: float = 0.1,
        save_checkpoint: bool = True,
        img_scale: float = 1,
        amp: bool = False,
        weight_decay: float = 1e-8,
        momentum: float = 0.999,
        gradient_clipping: float = 1.0,
        use_augmentation: bool = True  # 新增參數用於控制數據增強
):
    # 1. 創建數據集，嘗試使用 CarvanaDataset，如果失敗則使用 BasicDataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)  # 傳入數據增強參數
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)

    # 2. 拆分為訓練集和驗證集
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. 創建數據加載器
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)  # pin_memory=True 加速 GPU 訓練
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # 4. 初始化 WandB 實驗
    experiment = wandb.init(project='Attention_UNet', resume='allow', anonymous='must')
    experiment.config.update(dict(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        val_percent=val_percent,
        save_checkpoint=save_checkpoint,
        img_scale=img_scale,
        amp=amp,
        weight_decay=weight_decay,
        momentum=momentum,
        gradient_clipping=gradient_clipping,
        use_augmentation=use_augmentation
    ))

    # 5. 設置優化器、損失函數、學習率調度器和 AMP 的損失縮放
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # 調整 patience 以加速學習率下降
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

    # 6. 訓練開始日誌
    logging.info(f'''開始訓練:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
        Data Augmentation: {use_augmentation}
    ''')

    # 7. 訓練迴圈
    for epoch in range(1, epochs + 1):
        model.train()  # 設定模型為訓練模式
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_dice = {i: 0 for i in range(model.n_classes)}
        class_count = {i: 0 for i in range(model.n_classes)}
        epoch_start_time = time.time()  # 記錄 epoch 開始時間

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images_1, images_2, true_masks = batch['image_1'], batch['image_2'], batch['mask']

                images_1 = images_1.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                images_2 = images_2.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    # 假設你有兩個輸入影像，images_1 和 images_2
                    masks_pred = model(images_1, images_2)


                    if model.n_classes == 1:  # 二分類情況
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        accuracy_score = accuracy(F.sigmoid(masks_pred.squeeze(1)), true_masks)
                        dice_score = dice_coeff(F.sigmoid(masks_pred).float(), true_masks.float(), reduce_batch_first=True)
                    else:  # 多分類情況
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        accuracy_score = accuracy(F.softmax(masks_pred, dim=1), true_masks)
                        true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
                        dice_score = multiclass_dice_coeff(F.softmax(masks_pred, dim=1).float(), true_masks_one_hot, reduce_batch_first=True)

                        # 計算每個類別的 Dice coefficient 和 class-wise dice_score
                        for i in range(model.n_classes):
                            class_mask_pred = F.softmax(masks_pred, dim=1)[:, i, :, :]
                            class_mask_true = true_masks_one_hot[:, i, :, :]
                            dice_for_class = dice_coeff(class_mask_pred, class_mask_true, reduce_batch_first=True)
                            epoch_dice[i] += dice_for_class.item()
                            class_count[i] += 1

                # 梯度更新步驟
                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images_1.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_accuracy += accuracy_score.item()

                # 記錄 WandB
                experiment.log({
                    'train loss': loss.item(),
                    'train accuracy': accuracy_score.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item(), 'accuracy (batch)': accuracy_score.item()})

        # 8. 更新學習率調度器
        scheduler.step(epoch_loss)

        logging.info(f'Epoch {epoch} 訓練完成')

        # 9. 計算並記錄每個類別的平均 Dice coefficient 和總平均
        for i in range(model.n_classes):
            if class_count[i] > 0:
                avg_dice = epoch_dice[i] / class_count[i]
                logging.info(f'類別 {i} 的平均 Dice coefficient: {avg_dice}')
                experiment.log({f'class_{i}_dice': avg_dice, 'epoch': epoch})

        # 計算總平均 Dice coefficient
        avg_dice_total = sum(epoch_dice.values()) / sum(class_count.values())
        logging.info(f'總平均 Dice coefficient: {avg_dice_total}')
        experiment.log({'average_dice_total': avg_dice_total, 'epoch': epoch})

        # 10. 保存模型檔案
        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

        # 11. 記錄每個 epoch 的訓練時間
        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time
        logging.info(f'Epoch {epoch} 訓練時間: {epoch_training_time:.2f} 秒')

def get_args():
    parser = argparse.ArgumentParser(description='Train the Attention UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batchsize', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', dest='scale', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--val', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', help='Use mixed precision')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 加載模型並設置損失函數
    model = SiameseAttentionUNet(n_channels=3, n_classes=5, bilinear=False)
    model.to(device=device)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_model(model=model, epochs=args.epochs, batch_size=args.batchsize, learning_rate=args.lr, device=device, img_scale=args.scale, amp=args.amp)
    except torch.cuda.OutOfMemoryError:
        logging.error('訓練過程中遇到記憶體不足錯誤，請減少批次大小或圖像尺寸。')
