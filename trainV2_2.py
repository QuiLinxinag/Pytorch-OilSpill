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
from unet import UNet, AttentionUNet, EnhancedAttentionUNet, UNetWithHybridPooling,ResUNet
from utils.data_loading import BasicDataset, CarvanaDataset
from utils.dice_scoreV2_1 import dice_coeff, dice_loss, accuracy, multiclass_dice_coeff
import time
from torch.nn import Dropout2d

# 設定資料夾路徑
dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')

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
        use_augmentation: bool = True
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
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
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

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        epoch_dice = {i: 0 for i in range(model.n_classes)}
        class_count = {i: 0 for i in range(model.n_classes)}
        epoch_start_time = time.time()

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                images, true_masks = batch['image'], batch['mask']

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)

                    if model.n_classes == 1:
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                        loss += dice_loss(F.sigmoid(masks_pred.squeeze(1)), true_masks.float(), multiclass=False)
                        accuracy_score = accuracy(F.sigmoid(masks_pred.squeeze(1)), true_masks)
                        dice_score = dice_coeff(F.sigmoid(masks_pred).float(), true_masks.float(), reduce_batch_first=True)
                    else:
                        loss = criterion(masks_pred, true_masks)
                        loss += dice_loss(
                            F.softmax(masks_pred, dim=1).float(),
                            F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float(),
                            multiclass=True
                        )
                        accuracy_score = accuracy(F.softmax(masks_pred, dim=1), true_masks)
                        true_masks_one_hot = F.one_hot(true_masks, num_classes=model.n_classes).permute(0, 3, 1, 2).float()
                        dice_score = multiclass_dice_coeff(F.softmax(masks_pred, dim=1).float(), true_masks_one_hot, reduce_batch_first=True)

                        for i in range(model.n_classes):
                            class_mask_pred = F.softmax(masks_pred, dim=1)[:, i, :, :]
                            class_mask_true = true_masks_one_hot[:, i, :, :]
                            dice_for_class = dice_coeff(class_mask_pred, class_mask_true, reduce_batch_first=True)
                            epoch_dice[i] += dice_for_class.item()
                            class_count[i] += 1

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)

                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                epoch_accuracy += accuracy_score.item() if isinstance(accuracy_score, torch.Tensor) else accuracy_score

                experiment.log({
                    'train loss': loss.item(),
                    'train accuracy': accuracy_score.item() if isinstance(accuracy_score, torch.Tensor) else accuracy_score,
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{
                    'loss (batch)': loss.item() if isinstance(loss, torch.Tensor) else loss,
                    'accuracy (batch)': accuracy_score.item() if isinstance(accuracy_score, torch.Tensor) else accuracy_score
                })

        scheduler.step(epoch_loss)

        # Validation Phase
        val_score, val_accuracy, val_dice = evaluate(model, val_loader, device)
        logging.info(f'Epoch {epoch} 驗證得分: Dice coefficient {val_score}, 驗證準確率: {val_accuracy}')
        experiment.log({
            'validation dice': val_score,
            'validation accuracy': val_accuracy,
            'epoch': epoch
        })

        for i in range(model.n_classes):
            if class_count[i] > 0:
                avg_dice = epoch_dice[i] / class_count[i]
                logging.info(f'類別 {i} 的平均 Dice coefficient: {avg_dice}')
                experiment.log({f'class_{i}_dice': avg_dice, 'epoch': epoch})

        avg_dice_total = sum(epoch_dice.values()) / sum(class_count.values())
        logging.info(f'總平均 Dice coefficient: {avg_dice_total}')
        experiment.log({'average_dice_total': avg_dice_total, 'epoch': epoch})

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))
            logging.info(f'Checkpoint {epoch} saved!')

        epoch_end_time = time.time()
        epoch_training_time = epoch_end_time - epoch_start_time
        logging.info(f'Epoch {epoch} 訓練時間: {epoch_training_time:.2f} 秒')

def get_args():
    parser = argparse.ArgumentParser(description='Train the Attention UNet on images and target masks')
    parser.add_argument('--model', '-m', metavar='MODEL', type=str, default='ResUNet', help='Model: unet, attention_unet, enhanced_attention_unet ,ResUNet')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=2, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--load', '-f', metavar='FILE', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--scale', '-s', metavar='S', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('--val', '-v', metavar='V', type=float, default=10.0, help='Validation percentage (0-100)')
    parser.add_argument('--amp', action='store_true', default=True, help='Use mixed precision')
    parser.add_argument('--use-augmentation', action='store_true', default=True, help='Use data augmentation')

    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model_dict = {
        'unet': UNet,
        'attention_unet': AttentionUNet,
        'enhanced_attention_unet': EnhancedAttentionUNet,
        'hybrid_unet': UNetWithHybridPooling,
        'ResUNet' : ResUNet
    }

    if args.model not in model_dict:
        raise ValueError(f"未知的模型名稱 '{args.model}'，請選擇 'unet'、'attention_unet'、'enhanced_attention_unet' 或 'hybrid_unet'。")
    
    model = model_dict[args.model](n_channels=3, n_classes=5, bilinear=True)
    model.to(device=device)

    if args.load:
        model.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'模型權重從 {args.load} 載入完畢！')

    try:
        train_model(
            model=model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            val_percent=args.val / 100,
            img_scale=args.scale,
            amp=args.amp,
            use_augmentation=args.use_augmentation
        )
    except torch.cuda.CudaError as e:
        logging.error(f'CUDA 錯誤：{e}')
    except Exception as e:
        logging.error(f'訓練過程中發生錯誤：{e}')