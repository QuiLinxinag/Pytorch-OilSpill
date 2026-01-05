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
from unet import UNet, AttentionUNet, EnhancedAttentionUNet, UNetWithHybridPooling,ResUNet, ImprovedResUNet, SiameseAttentionUNet, MultiScaleAttentionUNet, SEResUNet,ResUNetPlusPlus,InnovativeResUNet,HybridUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_scoreV2_1 import (
    dice_coeff,
    dice_loss,
    accuracy,
    multiclass_dice_coeff,
    iou_score,
    dice_per_class,
    iou_per_class,
    update_confusion_matrix,
    confusion_metrics,
    pixel_accuracy,
    EMA,
)
import time
from unet import OnePeaceModel, ResUNetWithShuffle, AttentionUNetShuffle
import logging
from pathlib import Path

# 設定資料夾路徑
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

# 初始化最佳指標變量
best_dice_total = 0
best_epoch = 0
best_metrics = {}


# 定義類別名稱對應字典
class_names = {
    0: 'SeaSurface',  # 黑色 [0, 0, 0]
    1: 'Land',        # 綠色 [0, 153, 0]
    2: 'OilSpill',    # 青色 [0, 255, 255]
    3: 'Lookalike',   # 紅色 [255, 0, 0]
    4: 'Ship'         # 棕色 [153, 76, 0]
}

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
        use_augmentation: bool = True  # 控制數據增強
):
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)
    except (AssertionError, RuntimeError, IndexError):
        dataset = BasicDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)

    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

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

    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    grad_scaler = torch.amp.GradScaler('cuda', enabled=amp)
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()
    global_step = 0

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

    # 在 train_model 函數內加入 IoU 計算
    # 初始化 EMA trackers
    ema_macro = EMA(decay=0.9)
    ema_pixel = EMA(decay=0.9)

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        # 累積 soft probability 與 target one-hot 的總和以便 epoch 結束計算 per-class dice / IoU
        per_class_inter = torch.zeros(model.n_classes, device=device)
        per_class_pred = torch.zeros(model.n_classes, device=device)
        per_class_true = torch.zeros(model.n_classes, device=device)
        per_class_union_inter = torch.zeros(model.n_classes, device=device)
        per_class_union_total = torch.zeros(model.n_classes, device=device)
        epoch_start_time = time.time()
        # 混淆矩陣 (C,C)
        confusion = torch.zeros((model.n_classes, model.n_classes), device=device, dtype=torch.long)

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    probs = None
                    if model.n_classes == 1:
                        probs = torch.sigmoid(masks_pred)
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(probs.squeeze(1), true_masks.float(), multiclass=False)
                        accuracy_score = accuracy(probs.squeeze(1), true_masks)
                    else:
                        probs = F.softmax(masks_pred, dim=1).float()
                        one_hot = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(probs, one_hot, multiclass=True)
                        accuracy_score = accuracy(probs, true_masks, n_classes=model.n_classes)

                        # 累積 dice 所需量 (使用 soft probabilities)
                        dims = (0, 2, 3)
                        inter = (probs * one_hot).sum(dim=dims)
                        pred_sum = probs.sum(dim=dims)
                        true_sum = one_hot.sum(dim=dims)
                        per_class_inter += inter.detach()
                        per_class_pred += pred_sum.detach()
                        per_class_true += true_sum.detach()

                        # IoU 使用 hard prediction
                        hard = probs.argmax(dim=1)
                        hard_oh = F.one_hot(hard, model.n_classes).permute(0, 3, 1, 2)
                        inter_u = (hard_oh & one_hot.bool()).sum(dim=dims).float()
                        union_u = (hard_oh | one_hot.bool()).sum(dim=dims).float()
                        per_class_union_inter += inter_u.detach()
                        per_class_union_total += union_u.detach()
                        # 混淆矩陣更新
                        update_confusion_matrix(confusion, hard, true_masks, model.n_classes)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                # 取得即時 macro accuracy (batch) 與 pixel accuracy
                with torch.no_grad():
                    if model.n_classes > 1:
                        batch_macro_acc = accuracy(probs, true_masks, n_classes=model.n_classes)
                        batch_pixel_acc = pixel_accuracy(update_confusion_matrix(torch.zeros_like(confusion), probs, true_masks, model.n_classes))
                    else:
                        # 二分類：pixel accuracy 等同 macro here
                        preds_bin = (probs.squeeze(1) > 0.5).long()
                        batch_pixel_acc = (preds_bin == true_masks).float().mean().item()
                        batch_macro_acc = batch_pixel_acc
                    ema_macro_val = ema_macro.update(batch_macro_acc)
                    ema_pixel_val = ema_pixel.update(batch_pixel_acc)

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_accuracy += batch_macro_acc

                experiment.log({
                    'train loss': loss.item(),
                    'train macro_acc_batch': batch_macro_acc,
                    'train pixel_acc_batch': batch_pixel_acc,
                    'ema_macro_acc': ema_macro_val,
                    'ema_pixel_acc': ema_pixel_val,
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{
                    'loss': f'{loss.item():.4f}',
                    'macro_acc(EMA)': f'{ema_macro_val:.3f}',
                    'pixel_acc(EMA)': f'{ema_pixel_val:.3f}'
                })

        scheduler.step(epoch_loss)

        logging.info(f'Epoch {epoch} 訓練完成')

        # Epoch 結束計算 per-class dice / iou (僅多分類情況)
        if model.n_classes > 1:
            epsilon = 1e-6
            dice_c = (2 * per_class_inter + epsilon) / (per_class_pred + per_class_true + epsilon)
            iou_c = (per_class_union_inter + epsilon) / (per_class_union_total + epsilon)
            valid = per_class_true > 0
            if valid.any():
                avg_dice_total = dice_c[valid].mean().item()
                avg_iou_total = iou_c[valid].mean().item()
            else:
                avg_dice_total = 1.0
                avg_iou_total = 1.0
            # 混淆矩陣指標
            cm_metrics = confusion_metrics(confusion)
            macro_precision = cm_metrics['macro_precision'].item()
            macro_recall = cm_metrics['macro_recall'].item()
            macro_f1 = cm_metrics['macro_f1'].item()
            macro_iou_conf = cm_metrics['macro_iou'].item()
            pix_acc_epoch = pixel_accuracy(confusion)
            experiment.log({
                'epoch_macro_precision': macro_precision,
                'epoch_macro_recall': macro_recall,
                'epoch_macro_f1': macro_f1,
                'epoch_macro_iou_conf': macro_iou_conf,
                'epoch_pixel_accuracy': pix_acc_epoch,
                'epoch': epoch
            })
            logging.info(f'PixelAcc:{pix_acc_epoch:.4f} MacroAcc:{(epoch_accuracy/(global_step or 1)):.4f} MacroF1:{macro_f1:.4f}')
            for i in range(model.n_classes):
                logging.info(f'類別 {i} Dice: {dice_c[i].item():.4f} | IoU: {iou_c[i].item():.4f} | 有效:{bool(valid[i].item())}')
                experiment.log({f'class_{i}_dice': dice_c[i].item(), f'class_{i}_iou': iou_c[i].item(), 'epoch': epoch})
            logging.info(f'總平均 Dice (有效類別): {avg_dice_total:.4f}')
            logging.info(f'總平均 IoU (有效類別): {avg_iou_total:.4f}')
            experiment.log({'average_dice_total': avg_dice_total, 'average_iou_total': avg_iou_total, 'epoch': epoch})
        else:
            avg_dice_total = dice_coeff(probs.squeeze(1), true_masks.float()) if 'probs' in locals() else 0
            avg_iou_total = iou_score(probs.squeeze(1), true_masks) if 'probs' in locals() else 0

        # 紀錄最佳 Epoch 和指標
        global best_dice_total, best_epoch, best_metrics
        if avg_dice_total > best_dice_total:
            best_dice_total = avg_dice_total
            best_epoch = epoch
            if model.n_classes > 1:
                best_metrics = {
                    'epoch': epoch,
                    'average_dice_total': avg_dice_total,
                    'average_iou_total': avg_iou_total,
                }
            else:
                best_metrics = {'epoch': epoch, 'dice': avg_dice_total, 'iou': avg_iou_total}
            logging.info(f'新最佳模型在 Epoch {epoch}，總平均 Dice coefficient: {avg_dice_total}')
            logging.info(f'新最佳模型的總平均 IoU: {avg_iou_total}')  # 新增 IoU 記錄
            experiment.log({'best_epoch': best_epoch, 'best_average_dice_total': best_dice_total, 'best_average_iou_total': avg_iou_total})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time
        logging.info(f'Epoch {epoch} 訓練時間: {epoch_training_time:.2f} 秒')

def get_args():
    parser = argparse.ArgumentParser(description='Train the Attention UNet on images and target masks')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batchsize', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', dest='scale', type=float, default=1.0, help='Downscaling factor of the images')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use automatic mixed precision')
    parser.add_argument('--weights-decay', '-w', dest='weight_decay', type=float, default=1e-8, help='Weight decay for optimizer')
    parser.add_argument('--momentum', '-m', metavar='Momentum', type=float, default=0.999, help='Momentum for optimizer')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型，確保 n_classes 被傳入
    model = ResUNetWithShuffle(n_channels=3, n_classes=5)
    model.to(device=device)

    train_model(
        model=model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batchsize,
        learning_rate=args.lr,
        val_percent=args.val / 100,
        save_checkpoint=True,
        img_scale=args.scale,
        amp=args.amp
    )

    # 訓練結束後紀錄最佳指標
    logging.info(f'最佳模型在 Epoch {best_epoch}，總平均 Dice coefficient: {best_dice_total}')
    logging.info(f'最佳模型的指標: {best_metrics}')

