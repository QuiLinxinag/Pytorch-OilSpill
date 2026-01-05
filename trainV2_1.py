# -*- coding: utf-8 -*-  # 指定 UTF-8 編碼以正確顯示中文註解
# 在此檔中加入 Weights & Biases（wandb）收斂圖（Train/Validation Loss 與 Accuracy 等）  # 描述本檔目標
import argparse  # 匯入 argparse 以解析命令列參數
import logging  # 匯入 logging 以紀錄訓練過程重要訊息
import os  # 匯入 os 以存取作業系統功能（如 CPU 核心數）
import time  # 匯入 time 以計時每個 epoch 訓練時間
from pathlib import Path  # 匯入 Path 以處理路徑

import torch  # 匯入 PyTorch 核心
import torch.nn as nn  # 匯入神經網路模組
import torch.nn.functional as F  # 匯入函式式 API（如 softmax、one_hot）
from torch import optim  # 匯入優化器
from torch.optim.lr_scheduler import ReduceLROnPlateau  # 匯入學習率排程器
from torch.utils.data import DataLoader, random_split  # 匯入資料載入器與資料集切分
from tqdm import tqdm  # 匯入 tqdm 以建立訓練進度條
import wandb  # 匯入 wandb 用於實驗追蹤與可視化
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont

# 匯入外部自訂模組（依題目給定）
from evaluate import evaluate  # 匯入驗證流程函式（本回覆附上 evaluate.py 範例）
from unet import ResUNetWithShuffle,UNet, AttentionUNet, EnhancedAttentionUNet, UNetWithHybridPooling,ResUNet, ImprovedResUNet  # 匯入模型（此處示範使用 ResUNetWithShuffle）
from unet import SiameseAttentionUNet, MultiScaleAttentionUNet, SEResUNet,ResUNetPlusPlus,InnovativeResUNet,HybridUNet
from unet import OnePeaceModel
from utils.data_loading import BasicDataset, CarvanaDataset  # 匯入資料集定義
from utils.dice_scoreV2_1 import (  # 匯入評估指標工具
    dice_loss,  # Dice Loss 計算函式
    accuracy,  # Macro Accuracy 計算函式（支援多分類）
    update_confusion_matrix,  # 混淆矩陣更新函式
    confusion_metrics,  # 由混淆矩陣推得 Precision/Recall/F1/IoU
    pixel_accuracy,  # Pixel Accuracy 計算函式
    EMA,  # 指數移動平均工具
)

# 設定資料夾路徑（與題目一致）
dir_img = Path('./data/imgs/')  # 影像資料夾
dir_mask = Path('./data/masks/')  # 標註資料夾
dir_checkpoint = Path('./checkpoints/')  # 檢查點資料夾

# 初始化最佳指標變數（用於追蹤最佳模型）
best_dice_total = 0  # 全域最佳 Dice（僅示意，實務可改用最佳 Val Loss）
best_epoch = 0  # 全域最佳所在 Epoch
best_metrics = {}  # 全域最佳對應的其他指標

# 類別名稱（可用於日誌或圖表標示）
class_names = {
    0: 'SeaSurface',  # 海面
    1: 'Land',        # 陸地
    2: 'OilSpill',    # 油汙
    3: 'Lookalike',   # 相似物
    4: 'Ship'         # 船舶
}

def train_model(
    model: nn.Module,  # 傳入待訓練之模型
    device: torch.device,  # 指定運算裝置（CPU/GPU/MPS）
    epochs: int = 1,  # 訓練輪數
    batch_size: int = 1,  # 每批次資料量
    accumulation_steps: int = 1,  # 梯度累積步數（用於模擬較大 batch）
    learning_rate: float = 1e-4,  # 學習率
    val_percent: float = 0.1,  # 驗證集比例
    save_checkpoint: bool = True,  # 是否儲存權重
    img_scale: float = 1.0,  # 影像縮放倍率
    amp: bool = False,  # 是否使用混合精度
    weight_decay: float = 1e-8,  # 權重衰減
    momentum: float = 0.999,  # 動量（若使用 SGD 時會用到；此處以 Adam 為主）
    gradient_clipping: float = 1.0,  # 梯度裁切上限
    use_augmentation: bool = True,  # 是否啟用資料增強
    load: str = None,  # 可選：從此 checkpoint 路徑繼續訓練
) -> None:  # 無回傳，透過副作用（儲存權重、wandb 紀錄）達成目的
    # 嘗試載入 CarvanaDataset，若失敗則退回 BasicDataset
    try:
        dataset = CarvanaDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)  # 建立 CarvanaDataset
    except (AssertionError, RuntimeError, IndexError):  # 捕捉常見資料例外
        dataset = BasicDataset(dir_img, dir_mask, img_scale, augment=use_augmentation)  # 改用 BasicDataset

    # 切分訓練與驗證資料
    n_val = int(len(dataset) * val_percent)  # 計算驗證集大小
    n_train = len(dataset) - n_val  # 訓練集大小為總數減驗證數
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))  # 使用固定種子以重現切分

    # 若使用 CUDA 且使用者傳入的 batch_size > 1，為了避免 OOM 失敗，我們在此提供保護性 fallback
    if device.type == 'cuda' and batch_size > 1:
        logging.warning(f"Device is CUDA but batch_size={batch_size} may cause OOM. Falling back to batch_size=1.")
        batch_size = 1

    # 建立 DataLoader（依硬體 CPU 核心數調整 num_workers）
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)  # 設定載入器參數
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)  # 建立訓練資料載入器
    val_loader = DataLoader(val_set, shuffle=False, drop_last=False, **loader_args)  # 建立驗證資料載入器（不丟棄最後不足批次）

    # 啟動 wandb 專案（可自訂專案與執行名稱）
    run = wandb.init(project='Attention_UNet', resume='allow', anonymous='must')  # 初始化 wandb 執行，允許續傳
    run.config.update(dict(  # 紀錄本次實驗的超參數設定
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
        use_augmentation=use_augmentation,
        model_name=type(model).__name__,
    ))

    # 監看模型參數（避免過度記錄梯度以減少 overhead）
    wandb.watch(model, log='parameters', log_freq=100)

    # 設定優化器與學習率排程器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # 使用 Adam 優化器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)  # 以驗證損失做自動降學習率

    # AMP 相關工具：使用正確的 GradScaler 介面
    grad_scaler = torch.amp.GradScaler(enabled=amp)

    # 損失函式（多分類使用 CrossEntropyLoss，二分類使用 BCEWithLogitsLoss）
    criterion = nn.CrossEntropyLoss() if model.n_classes > 1 else nn.BCEWithLogitsLoss()  # 動態選擇損失函式

    # 全域步數計數器
    global_step = 0  # 用於 wandb 繪圖的步數軸

    # 處理從 checkpoint 繼續訓練（支援舊格式 state_dict 與新格式 dict）
    start_epoch = 1
    if load:
        ckpt_path = Path(load)
        if ckpt_path.exists():
            try:
                ckpt = torch.load(str(ckpt_path), map_location=device)
                # 若為完整 dict（包含 model_state_dict）
                if isinstance(ckpt, dict) and ('model_state_dict' in ckpt or 'epoch' in ckpt):
                    model_state = ckpt.get('model_state_dict', ckpt)
                    model.load_state_dict(model_state)
                    if 'optimizer_state_dict' in ckpt:
                        try:
                            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                        except Exception:
                            logging.warning('Could not load optimizer state_dict from checkpoint.')
                    if 'scheduler_state_dict' in ckpt:
                        try:
                            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                        except Exception:
                            logging.warning('Could not load scheduler state_dict from checkpoint.')
                    start_epoch = int(ckpt.get('epoch', 0)) + 1
                    # restore best metrics if available
                    if 'best_metrics' in ckpt:
                        try:
                            best = ckpt.get('best_metrics', {})
                            # try to restore best_dice_total from saved metrics
                            if 'best_dice_total' in best:
                                nonlocal_best = best.get('best_dice_total')
                                if nonlocal_best is not None:
                                    globals()['best_dice_total'] = nonlocal_best
                        except Exception:
                            pass
                else:
                    # assume ckpt is raw state_dict
                    model.load_state_dict(ckpt)
                    start_epoch = 1
                logging.info(f"Loaded checkpoint '{ckpt_path}' (starting epoch {start_epoch})")
            except Exception as e:
                logging.warning(f'Failed to load checkpoint {ckpt_path}: {e}')
        else:
            logging.warning(f'Checkpoint path {ckpt_path} does not exist. Starting from scratch.')

    # 紀錄器（用於自製收斂圖：每個 epoch 的 Train/Val Loss、Accuracy）
    history_epochs = []  # 儲存 epoch 編號
    history_train_loss = []  # 儲存每個 epoch 的訓練損失
    history_val_loss = []  # 儲存每個 epoch 的驗證損失
    history_train_macro_acc = []  # 儲存每個 epoch 的訓練 Macro Accuracy（以 batch EMA 表徵）
    history_val_macro_acc = []  # 儲存每個 epoch 的驗證 Macro Accuracy
    history_val_dice = []  # 儲存每個 epoch 的驗證平均 Dice

    # 設定紀錄器（以 INFO 級別輸出）
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

    # 建立 EMA（指數移動平均）用於平滑批次指標
    ema_macro = EMA(decay=0.9)  # 建立 Macro Accuracy 的 EMA
    ema_pixel = EMA(decay=0.9)  # 建立 Pixel Accuracy 的 EMA

    # 進入多輪訓練
    # 若從 checkpoint 繼續，從 start_epoch 開始並訓練 epochs 輪
    end_epoch = start_epoch + epochs - 1
    for epoch in range(start_epoch, end_epoch + 1):
        model.train()  # 切換到訓練模式
        epoch_loss_sum = 0.0  # 累計訓練損失總和
        epoch_macro_acc_sum = 0.0  # 累計 Macro Accuracy（僅作參考）
        num_train_batches = 0  # 計算實際訓練批次數
        epoch_start_time = time.time()  # 記錄 epoch 開始時間

        # 混淆矩陣與類別統計（僅多分類時使用）
        if model.n_classes > 1:  # 確認為多分類情境
            confusion = torch.zeros((model.n_classes, model.n_classes), device=device, dtype=torch.long)  # 建立混淆矩陣

        # 在進入訓練迴圈前確保梯度已清零（對梯度累積很重要）
        optimizer.zero_grad(set_to_none=True)

        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:  # 建立進度條，單位為張
            for batch in train_loader:  # 逐批訓練
                images, true_masks = batch['image'], batch['mask']  # 取出影像與標註

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)  # 將影像移動到裝置並啟用通道最後的記憶體格式
                true_masks = true_masks.to(device=device, dtype=torch.long)  # 將標註移動到裝置並轉為長整數以配合 CE Loss

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):  # 自動混合精度上下文
                    logits = model(images)  # 前向傳播取得 logits
                    # --- Safety: 若 model 輸出與標註尺寸不一致，將 logits 插值到標註大小 ---
                    # true_masks shape: [N, H, W]
                    if logits.dim() == 4:
                        _, _, h_logits, w_logits = logits.shape
                        try:
                            h_mask, w_mask = true_masks.shape[1], true_masks.shape[2]
                        except Exception:
                            h_mask = true_masks.size(-2)
                            w_mask = true_masks.size(-1)
                        if (h_logits != h_mask) or (w_logits != w_mask):
                            # 使用雙線性插值保留類別 logits 的平滑性；對二分類/多分類皆適用
                            logits = F.interpolate(logits, size=(h_mask, w_mask), mode='bilinear', align_corners=False)
                    # --- end safety resize ---
                    if model.n_classes == 1:  # 二分類情境
                        probs = torch.sigmoid(logits)  # 經 sigmoid 取得機率
                        loss = criterion(logits.squeeze(1), true_masks.float())  # 計算 BCE 損失
                        loss += dice_loss(probs.squeeze(1), true_masks.float(), multiclass=False)  # 加入 Dice Loss 強化邊界學習
                        macro_acc_batch = accuracy(probs.squeeze(1), true_masks)  # 以自訂函式計算 batch Macro Accuracy
                        pixel_acc_batch = (probs.squeeze(1) > 0.5).eq(true_masks).float().mean().item()  # 計算 Pixel Accuracy
                    else:  # 多分類情境
                        probs = F.softmax(logits, dim=1).float()  # 經 softmax 取得類別機率
                        one_hot = F.one_hot(true_masks, model.n_classes).permute(0, 3, 1, 2).float()  # 轉換標註為 one-hot
                        loss = criterion(logits, true_masks)  # 計算 CE 損失
                        loss += dice_loss(probs, one_hot, multiclass=True)  # 加入多分類 Dice Loss
                        macro_acc_batch = accuracy(probs, true_masks, n_classes=model.n_classes)  # 計算 Macro Accuracy
                        hard_pred = probs.argmax(dim=1)  # 取得硬分類結果
                        update_confusion_matrix(confusion, hard_pred, true_masks, model.n_classes)  # 更新混淆矩陣
                        pixel_acc_batch = pixel_accuracy(confusion)  # 由混淆矩陣計算 Pixel Accuracy（趨近整體）

                # 梯度累積邏輯：將 loss 平均到 accumulation_steps
                if accumulation_steps < 1:
                    accumulation_steps = 1
                loss_to_backprop = loss / accumulation_steps

                grad_scaler.scale(loss_to_backprop).backward()  # 以縮放後的損失進行反向傳播（累積）

                # 當達到累積步數時才執行優化器更新
                if (global_step + 1) % accumulation_steps == 0:
                    grad_scaler.unscale_(optimizer)  # 解除縮放以利梯度裁切
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clipping)  # 進行梯度裁切避免梯度爆炸
                    grad_scaler.step(optimizer)  # 執行優化器更新
                    grad_scaler.update()  # 更新縮放器內部狀態
                    optimizer.zero_grad(set_to_none=True)

                    # 釋放 CUDA 暫存快取以減少碎片化造成的 OOM（若為 CUDA 裝置）
                    if device.type == 'cuda':
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass

                # 使用 EMA 平滑批次指標，並同步至 wandb
                ema_macro_val = ema_macro.update(macro_acc_batch)  # 更新 Macro Accuracy 的 EMA
                ema_pixel_val = ema_pixel.update(pixel_acc_batch)  # 更新 Pixel Accuracy 的 EMA

                # 累計訓練期望輸出
                epoch_loss_sum += loss.item()  # 將當前批次損失相加
                epoch_macro_acc_sum += macro_acc_batch  # 將當前批次 Macro Accuracy 相加
                num_train_batches += 1  # 批次計數加一
                global_step += 1  # 全域步數加一

                # 將批次級指標記錄到 wandb（自動繪製 batch-level 線圖）
                wandb.log({
                    'train/batch_loss': float(loss.item()),  # 記錄訓練批次損失
                    'train/batch_macro_acc': float(macro_acc_batch),  # 記錄訓練批次 Macro Accuracy
                    'train/ema_macro_acc': float(ema_macro_val),  # 記錄 EMA 後的 Macro Accuracy
                    'train/ema_pixel_acc': float(ema_pixel_val),  # 記錄 EMA 後的 Pixel Accuracy
                    'trainer/step': int(global_step),  # 記錄當前步數
                    'trainer/epoch': int(epoch),  # 記錄當前 epoch
                }, step=global_step)  # 使用步數作為 X 軸

                # 更新進度條提示
                pbar.update(images.shape[0])  # 以影像張數更新進度條
                pbar.set_postfix(**{
                    'loss': f'{loss.item():.4f}',  # 顯示當前批次損失
                    'macro_acc(EMA)': f'{ema_macro_val:.3f}',  # 顯示 EMA 後 Macro Accuracy
                    'pixel_acc(EMA)': f'{ema_pixel_val:.3f}',  # 顯示 EMA 後 Pixel Accuracy
                })

        # 計算當前 epoch 的平均訓練損失與 Macro Accuracy
        train_loss_epoch = epoch_loss_sum / max(1, num_train_batches)  # 避免除以零
        train_macro_acc_epoch = epoch_macro_acc_sum / max(1, num_train_batches)  # 計算平均 Macro Accuracy

        # 執行驗證流程以取得驗證損失與 Macro Accuracy（於 evaluate.py 定義）
        # evaluate 的定義為 evaluate(net, dataloader, device, amp, criterion, ignore_background=True)
        val_loss_epoch, val_macro_acc_epoch, val_mean_dice, val_per_class_dice, val_mDice = evaluate(
            model,  # net
            val_loader,  # dataloader
            device,  # device
            amp,  # amp
            criterion,  # criterion
            True  # ignore_background
        )

        # 以驗證損失驅動學習率排程器（收斂較差則自動降學習率）
        scheduler.step(val_loss_epoch)  # 將驗證損失提供給 ReduceLROnPlateau 以調整學習率

        # 本輪訓練時間
        epoch_time_sec = time.time() - epoch_start_time  # 計算 epoch 花費秒數

        # 將 epoch 級別的指標同步至 wandb（自動繪製收斂曲線）
        wandb.log({
            'train/epoch_loss': float(train_loss_epoch),  # 紀錄訓練平均損失
            'train/epoch_macro_acc': float(train_macro_acc_epoch),  # 紀錄訓練平均 Macro Accuracy
            'val/epoch_loss': float(val_loss_epoch),  # 紀錄驗證平均損失
            'val/epoch_macro_acc': float(val_macro_acc_epoch),  # 紀錄驗證平均 Macro Accuracy
            'val/epoch_dice': float(val_mean_dice),  # 紀錄驗證平均 Dice
            'trainer/epoch_time_sec': float(epoch_time_sec),  # 紀錄當前 epoch 訓練耗時
            'trainer/epoch': int(epoch),  # 紀錄 epoch 數字
        }, step=global_step)  # 使用目前步數對齊時間軸

        # Log per-class Dice to wandb (if evaluate returned them)
        if isinstance(val_per_class_dice, (list, tuple)) and len(val_per_class_dice) > 0:
            per_class_log = {}
            for idx, dice_val in enumerate(val_per_class_dice):
                class_name = class_names.get(idx, f'class_{idx}')
                try:
                    per_class_log[f'val/dice/{class_name}'] = float(dice_val) if dice_val is not None else None
                except Exception:
                    per_class_log[f'val/dice/{class_name}'] = None
            # also log mDice (mean across classes excluding background if that was used in evaluate)
            per_class_log['val/mDice'] = float(val_mDice)
            wandb.log(per_class_log, step=global_step)

        # 上傳少量驗證圖片（Input | GroundTruth | Prediction）到 wandb，方便視覺化檢查
        try:
            # 從驗證資料集中隨機抽取少量樣本（最多 4 張）以利視覺化，而非總是使用第一個 batch
            ds_len = len(val_loader.dataset)
            n_vis = min(4, ds_len)
            # 以 numpy 隨機抽取不重複索引
            idxs = np.random.choice(ds_len, size=n_vis, replace=False)
            imgs = []
            masks = []
            for id_ in idxs:
                item = val_loader.dataset[id_]
                imgs.append(item['image'])
                masks.append(item['mask'])

            images_vis = torch.stack(imgs, dim=0).to(device)
            masks_vis = torch.stack(masks, dim=0).to(device)

            with torch.inference_mode():
                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits_vis = model(images_vis)
                    if logits_vis.dim() == 4:
                        _, _, h_logits, w_logits = logits_vis.shape
                        try:
                            h_mask, w_mask = masks_vis.shape[1], masks_vis.shape[2]
                        except Exception:
                            h_mask = masks_vis.size(-2)
                            w_mask = masks_vis.size(-1)
                        if (h_logits != h_mask) or (w_logits != w_mask):
                            logits_vis = F.interpolate(logits_vis, size=(h_mask, w_mask), mode='bilinear', align_corners=False)
                    if model.n_classes == 1:
                        preds_vis = (torch.sigmoid(logits_vis).squeeze(1) > 0.5).long()
                    else:
                        preds_vis = F.softmax(logits_vis, dim=1).argmax(dim=1)

            # simple color map (背景=黑, 陸地=綠, 油汙=藍, 相似物=紅, 其他保留)
            # 索引對應請參考訓練時的 class index
            class_colors = [
                (0, 0, 0),      # background (黑)
                (0, 255, 0),    # Land (綠)
                (0, 0, 255),    # OilSpill (藍)
                (255, 0, 0),    # Lookalike (紅)
                (255, 255, 0),  # Ship (黃)
            ]

            images_to_log = []
            for i in range(n_vis):
                img = images_vis[i].cpu().numpy()
                # C,H,W -> H,W,C
                if img.shape[0] == 1:
                    img = np.repeat(img, 3, axis=0)
                img = np.transpose(img, (1, 2, 0))
                # scale to 0-255 for visualization
                if img.max() <= 1.0:
                    img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                else:
                    img = np.clip(img, 0, 255).astype(np.uint8)

                true_mask = masks_vis[i].cpu().numpy()
                pred_mask = preds_vis[i].cpu().numpy()
                h, w = true_mask.shape
                true_color = np.zeros((h, w, 3), dtype=np.uint8)
                pred_color = np.zeros((h, w, 3), dtype=np.uint8)
                for cls_idx in range(min(len(class_colors), model.n_classes)):
                    color = class_colors[cls_idx]
                    true_color[true_mask == cls_idx] = color
                    pred_color[pred_mask == cls_idx] = color

                combined = np.concatenate([img, true_color, pred_color], axis=1)
                # create PIL image and add legend below
                combined_pil = PILImage.fromarray(combined)

                # build legend image
                try:
                    font = ImageFont.truetype("arial.ttf", 14)
                except Exception:
                    font = ImageFont.load_default()

                pad = 6
                box_size = 16
                spacing = 8
                x = pad
                legend_height = box_size + pad * 2
                legend = PILImage.new('RGB', (combined_pil.width, legend_height), (255, 255, 255))
                draw = ImageDraw.Draw(legend)

                for cls_idx in range(min(model.n_classes, len(class_colors))):
                    color = tuple(class_colors[cls_idx])
                    label = class_names.get(cls_idx, f'class_{cls_idx}')
                    # draw color box
                    y0 = pad
                    draw.rectangle([x, y0, x + box_size, y0 + box_size], fill=color)
                    # draw text
                    text_x = x + box_size + 4
                    text_y = y0
                    draw.text((text_x, text_y), label, fill=(0, 0, 0), font=font)
                    # advance x
                    try:
                        bbox = draw.textbbox((0, 0), label, font=font)
                        text_w = bbox[2] - bbox[0]
                    except Exception:
                        # fallback for older Pillow versions
                        text_w = draw.textsize(label, font=font)[0]
                    x += box_size + 4 + text_w + spacing
                    if x > combined_pil.width - 50:
                        break

                # compose final image
                final_h = combined_pil.height + legend.height
                final_img = PILImage.new('RGB', (combined_pil.width, final_h), (255, 255, 255))
                final_img.paste(combined_pil, (0, 0))
                final_img.paste(legend, (0, combined_pil.height))

                # include dataset index in caption so it's clear which sample is shown
                try:
                    data_index = int(idxs[i])
                except Exception:
                    data_index = idxs[i]
                caption = f'Epoch {epoch} — idx {data_index} — GT | Pred'
                images_to_log.append(wandb.Image(final_img, caption=caption))

            if images_to_log:
                wandb.log({'val/predictions': images_to_log}, step=global_step)
        except Exception as e:
            logging.debug(f'Could not log prediction images to wandb: {e}')

        # 追加至本地歷史（用於自製一張合併收斂圖）
        history_epochs.append(epoch)  # 儲存 epoch 序列
        history_train_loss.append(train_loss_epoch)  # 儲存訓練損失序列
        history_val_loss.append(val_loss_epoch)  # 儲存驗證損失序列
        history_train_macro_acc.append(train_macro_acc_epoch)  # 儲存訓練 Macro Accuracy 序列
        history_val_macro_acc.append(val_macro_acc_epoch)  # 儲存驗證 Macro Accuracy 序列
        history_val_dice.append(val_mDice)  # 儲存驗證 mDice

        # 在文字日誌輸出摘要
        logging.info(
            f"[Epoch {epoch}/{epochs}] "
            f"TrainLoss={train_loss_epoch:.4f} | ValLoss={val_loss_epoch:.4f} | "
            f"TrainMacroAcc={train_macro_acc_epoch:.4f} | ValMacroAcc={val_macro_acc_epoch:.4f} | "
            f"ValmDice={val_mDice:.4f} | "
            f"Time={epoch_time_sec:.2f}s"
        )  # 將本輪訓練與驗證摘要寫入日誌

        # 儲存每輪權重（亦可加條件：若 ValLoss 下降或 Dice/Iou 改善才存）
        if save_checkpoint:  # 若啟用儲存
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)  # 確保檢查點路徑存在
            ckpt = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_metrics': best_metrics,
            }
            torch.save(ckpt, str(dir_checkpoint / f'checkpoint_epoch{epoch}.pth'))  # 儲存完整 checkpoint
            logging.info(f'Checkpoint {epoch} saved!')  # 記錄已儲存訊息

        # 以驗證 Macro F1 或 Dice 亦可更新 best（此處示範仍沿用題目中的 best_dice_total）
        # 實務上，建議改以 val_loss 最佳作為 best 模型依據
        # 若有 dice 與 iou 的 evaluate，可在 evaluate 回傳內擴充並在此比較

    # 生成一張合併收斂圖（使用 wandb.plot.line_series 自製圖表）
    # 準備表格資料：每列包含 epoch、train_loss、val_loss、train_macro_acc、val_macro_acc
    table = wandb.Table(columns=["epoch", "train_loss", "val_loss", "train_macro_acc", "val_macro_acc"])  # 建立表格欄位
    for e, tl, vl, ta, va in zip(history_epochs, history_train_loss, history_val_loss, history_train_macro_acc, history_val_macro_acc):  # 逐行加入資料
        table.add_data(int(e), float(tl), float(vl), float(ta), float(va))  # 加到表格

    # 建立「Loss 收斂圖」：同一張圖顯示 Train 與 Val Loss
    loss_plot = wandb.plot.line_series(
        xs=history_epochs,  # X 軸為 epoch
        ys=[history_train_loss, history_val_loss],  # Y 軸為兩條序列：Train 與 Val Loss
        keys=["train_loss", "val_loss"],  # 曲線名稱
        title="Convergence - Train vs Val Loss",  # 圖表標題
        xname="epoch"  # X 軸名稱
    )  # 回傳可直接 wandb.log 的視覺化物件

    # 建立「Macro Accuracy 收斂圖」
    acc_plot = wandb.plot.line_series(
        xs=history_epochs,  # X 軸為 epoch
        ys=[history_train_macro_acc, history_val_macro_acc],  # Y 軸為兩條序列：Train 與 Val Macro Accuracy
        keys=["train_macro_acc", "val_macro_acc"],  # 曲線名稱
        title="Convergence - Train vs Val Macro Accuracy",  # 圖表標題
        xname="epoch"  # X 軸名稱
    )  # 產生視覺化

    # 將表格與兩張圖上傳至 wandb（可在面板的「Media」或「Charts」檢視）
    wandb.log({
        "convergence/table": table,  # 上傳資料表
        "convergence/loss_plot": loss_plot,  # 上傳 Loss 收斂圖
        "convergence/macro_accuracy_plot": acc_plot,  # 上傳 Macro Accuracy 收斂圖
    })  # 一次性上傳多個可視化

def get_args() -> argparse.Namespace:
    """解析命令列參數並回傳 Namespace。"""  # 函式說明
    parser = argparse.ArgumentParser(description='Train the Attention UNet on images and target masks')  # 建立參數解析器
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=100, help='Number of epochs')  # 設定 epochs
    parser.add_argument('--batch-size', '-b', dest='batchsize', metavar='B', type=int, default=2, help='Batch size')  # 設定批次大小
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=1e-4, help='Learning rate', dest='lr')  # 設定學習率
    parser.add_argument('--load', '-f', type=str, default=None, help='Load model from a .pth file')  # 選擇載入既有權重
    parser.add_argument('--scale', '-s', dest='scale', type=float, default=1.0, help='Downscaling factor of the images')  # 設定影像縮放
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0, help='Percent of the data that is used as validation (0-100)')  # 設定驗證比例
    parser.add_argument('--amp', action='store_true', default=False, help='Use automatic mixed precision')  # 啟用 AMP
    parser.add_argument('--accumulation-steps', dest='accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--use-checkpoint', dest='use_checkpoint', action='store_true', default=False, help='Use activation checkpointing in model')
    parser.add_argument('--weights-decay', '-w', dest='weight_decay', type=float, default=1e-8, help='Weight decay for optimizer')  # 權重衰減
    parser.add_argument('--momentum', '-m', metavar='Momentum', type=float, default=0.999, help='Momentum for optimizer')  # 動量參數（若改用 SGD）
    return parser.parse_args()  # 回傳解析結果

if __name__ == '__main__':  # 入口點判斷
    logging.basicConfig(level=logging.INFO)  # 設定 logging 基本等級為 INFO
    args = get_args()  # 解析命令列參數
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 優先使用 CUDA，否則退回 CPU

    # 建立模型並移動至裝置（此處示範使用 ResUNetWithShuffle；可依需求替換）
    model = ResUNetWithShuffle(n_channels=3, n_classes=5)  # 指定輸入通道與輸出類別數
    # 若使用者要求 activation checkpointing，嘗試啟用模型的 checkpointing 方法（若模型支援）
    if getattr(args, 'use_checkpoint', False):
        try:
            # 部分模型提供 `use_checkpointing()` 或類似方法
            if hasattr(model, 'use_checkpointing'):
                model.use_checkpointing()
            elif hasattr(model, 'enable_checkpointing'):
                model.enable_checkpointing()
            else:
                logging.warning('Model does not expose a checkpointing enable method; skipping activation checkpointing.')
        except Exception as e:
            logging.warning(f'Failed to enable activation checkpointing on model: {e}')
    model.to(device=device)  # 將模型放到指定裝置

    # 執行訓練主流程（內含 wandb 收斂圖紀錄）
    train_model(
        model=model,  # 傳入模型
        device=device,  # 指定裝置
        epochs=args.epochs,  # 訓練輪數
        batch_size=args.batchsize,  # 批次大小
        accumulation_steps=args.accumulation_steps,
        learning_rate=args.lr,  # 學習率
        val_percent=args.val / 100,  # 驗證比例轉為 0~1
        save_checkpoint=True,  # 啟用權重儲存
        img_scale=args.scale,  # 影像縮放倍率
        amp=args.amp  # 是否啟用 AMP
        , load=args.load
    )  # 開始訓練
