# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from typing import Tuple, Optional, List, Dict

# 只保留需要的匯入，避免重複
from utils.dice_score import dice_coeff, multiclass_dice_coeff

@torch.inference_mode()
def evaluate(
    net: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    amp: bool,
    criterion,                              # 新增：用於計算 validation loss（例如 CE 或 BCE）
    ignore_background: bool = True,         # 選項：多分類是否忽略背景計算 Dice
) -> Tuple[float, float, float, List[Optional[float]], float]:
    """
    回傳：(avg_val_loss, avg_macro_acc, avg_mean_dice)
    - 二分類：loss = BCE（若訓練端有加 Dice，可視需求在此加總，但通常用 CE/BCE 當收斂觀察）
    - 多分類：loss = CrossEntropy
    - Macro Accuracy：以整體正確率近似，若需更嚴謹可改用 per-class 再取平均
    - Dice：二分類用 soft Dice；多分類用 soft Dice（可選擇忽略背景）
    """
    net.eval()

    num_batches = len(dataloader)
    loss_sum = 0.0
    macro_acc_sum = 0.0
    dice_sum = 0.0

    # 為每個類別準備累加器（包含 background），稍後會視 ignore_background 做處理
    if net.n_classes == 1:
        per_class_dice_sum = torch.zeros(1, device=device)
    else:
        per_class_dice_sum = torch.zeros(net.n_classes, device=device)

    # 使用 autocast 包住整個驗證迴圈
    autocast_device = device.type if device.type != 'mps' else 'cpu'
    with torch.autocast(autocast_device, enabled=amp):
        for batch in dataloader:
            images, masks_true = batch['image'], batch['mask']

            # 搬移到裝置
            images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
            masks_true = masks_true.to(device=device, dtype=torch.long)

            # 前向傳播
            logits = net(images)

            if net.n_classes == 1:
                # 二分類：BCE + soft Dice（Dice 僅做指標，不一定要加到 loss）
                probs = torch.sigmoid(logits)
                # BCE 作為 validation loss（與訓練一致）
                loss = criterion(logits.squeeze(1), masks_true.float())

                # MacroAcc 近似（pixel-level）
                preds_bin = (probs.squeeze(1) > 0.5).long()
                macro_acc = (preds_bin == masks_true).float().mean().item()

                # soft Dice（使用機率與二值標註）
                dice = float(dice_coeff(probs.squeeze(1), masks_true.float()))

            else:
                # 多分類：CE + soft Dice
                probs = F.softmax(logits, dim=1).float()
                loss = criterion(logits, masks_true)

                # MacroAcc（簡化版：整體正確率）
                preds = probs.argmax(dim=1)
                macro_acc = (preds == masks_true).float().mean().item()

                # 構造 one-hot 標註
                masks_one_hot = F.one_hot(masks_true, num_classes=net.n_classes).permute(0, 3, 1, 2).float()

                # 為了能取得每個類別的 Dice，我們在 batch 維度上計算每個類別的 soft-dice
                # probs: (B, C, H, W), masks_one_hot: (B, C, H, W)
                probs_for_dice = probs
                target_for_dice = masks_one_hot

                eps = 1e-6
                # per-class intersection and sums across batch and spatial dims -> shape (C,)
                inter = 2.0 * (probs_for_dice * target_for_dice).sum(dim=(0, 2, 3))
                sets_sum = probs_for_dice.sum(dim=(0, 2, 3)) + target_for_dice.sum(dim=(0, 2, 3))
                sets_sum = torch.where(sets_sum == 0, inter, sets_sum)
                per_class_dice_batch = (inter + eps) / (sets_sum + eps)

                # 若指定忽略背景，將 background 的值保留但之後不計入 mDice
                # 將 per_class_dice_batch（torch tensor, C）加到累加器
                if per_class_dice_sum.shape[0] == per_class_dice_batch.shape[0]:
                    per_class_dice_sum += per_class_dice_batch.to(device)
                else:
                    # 這個分支通常不會發生，但保險處理：若 per_class_dice_sum 包含 background 而 per_class_dice_batch 不含
                    # （或反之），則對齊索引後相加
                    min_c = min(per_class_dice_sum.shape[0], per_class_dice_batch.shape[0])
                    per_class_dice_sum[:min_c] += per_class_dice_batch.to(device)[:min_c]

                # 使用 multiclass_dice_coeff 作為整體平均（與之前回傳相容）
                dice = float(multiclass_dice_coeff(probs_for_dice if not ignore_background else probs_for_dice[:, 1:],
                                                   target_for_dice if not ignore_background else target_for_dice[:, 1:]))

            loss_sum += float(loss.item())
            macro_acc_sum += float(macro_acc)
            dice_sum += float(dice)

    avg_loss = loss_sum / max(1, num_batches)
    avg_macro_acc = macro_acc_sum / max(1, num_batches)
    avg_mean_dice = dice_sum / max(1, num_batches)

    # 將每類的累加值除以 batches 得到每類的平均 Dice
    if per_class_dice_sum.numel() > 0:
        per_class_dice_avg = (per_class_dice_sum / max(1, num_batches)).cpu().tolist()
    else:
        per_class_dice_avg = []

    # 若忽略 background，使用者通常也想看到 mDice（mean Dice）不含 background
    if net.n_classes > 1 and ignore_background:
        mDice = float(sum(per_class_dice_avg[1:]) / max(1, len(per_class_dice_avg) - 1))
    else:
        mDice = float(sum(per_class_dice_avg) / max(1, len(per_class_dice_avg))) if len(per_class_dice_avg) > 0 else 0.0

    # 還原訓練模式給外部主程式
    net.train()
    # 回傳：avg_loss, avg_macro_acc, avg_mean_dice (歷史相容), per_class_dice_avg, mDice
    return avg_loss, avg_macro_acc, avg_mean_dice, per_class_dice_avg, mDice
