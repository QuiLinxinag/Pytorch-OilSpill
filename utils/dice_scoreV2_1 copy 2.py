"""Metric helpers (Dice / IoU / Accuracy) with robust multi-class handling.

主要改進:
1. 自動 one-hot: 若 target 形狀為 (B,H,W) 且輸入為 (B,C,H,W) 則自動 one-hot。
2. Vectorized: 一次性計算所有類別的 intersection / union，避免 Python 迴圈重複 softmax。
3. 忽略空類別: ignore_empty=True 時，不將整個 batch 中完全沒出現的類別納入 macro 平均。
4. 可回傳 per-class 指標: 設 per_class=True 取得 tensor。
5. 與原有介面兼容: dice_coeff / multiclass_dice_coeff / dice_loss 仍可使用。
"""

from typing import Tuple, Optional, Dict
import torch
import torch.nn.functional as F
from torch import Tensor


def _to_one_hot(target: Tensor, num_classes: int, dtype=None) -> Tensor:
    """Convert (B,H,W) integer mask to one-hot (B,C,H,W)."""
    if target.dim() != 3:
        raise ValueError("Expected target with shape (B,H,W) for one-hot conversion")
    oh = F.one_hot(target.long(), num_classes)  # (B,H,W,C)
    oh = oh.permute(0, 3, 1, 2).contiguous()
    if dtype is not None:
        oh = oh.to(dtype)
    return oh


def dice_per_class(
    probs: Tensor,
    target: Tensor,
    epsilon: float = 1e-6,
    ignore_empty: bool = True,
    per_class: bool = False,
) -> Tensor:
    """Compute per-class Dice given soft probabilities and integer or one-hot targets.

    probs:  (B,C,H,W) soft probabilities or binary mask (float)
    target: (B,H,W) integer labels or (B,C,H,W) one-hot
    return: macro-average (scalar) or per-class dice (C) if per_class=True
    """
    if probs.dim() == 3:  # treat as binary single-channel (B,H,W)
        probs = probs.unsqueeze(1)
    if target.dim() == 3:  # integer labels -> one hot
        target = _to_one_hot(target, probs.size(1), dtype=probs.dtype)
    if target.shape != probs.shape:
        raise ValueError(f"Shape mismatch: probs={probs.shape} target={target.shape}")

    # (B,C,H,W) -> sum over spatial dims and batch
    dims = (0, 2, 3)
    inter = (probs * target).sum(dim=dims)
    denom = probs.sum(dim=dims) + target.sum(dim=dims)

    dice_c = (2 * inter + epsilon) / (denom + epsilon)

    if ignore_empty:
        # A class is empty if both prediction and target are zero everywhere -> denom == 0
        valid = denom > 0
        if valid.any():
            macro = dice_c[valid].mean()
        else:
            macro = torch.tensor(1.0, device=probs.device)
    else:
        macro = dice_c.mean()

    return dice_c if per_class else macro


def iou_per_class(
    probs: Tensor,
    target: Tensor,
    epsilon: float = 1e-6,
    ignore_empty: bool = True,
    per_class: bool = False,
) -> Tensor:
    """Compute per-class IoU given soft probabilities and integer or one-hot targets.

    We convert probs to hard labels for IoU (argmax) to keep consistency with most definitions.
    probs:  (B,C,H,W) or (B,H,W)
    target: (B,H,W) or one-hot (B,C,H,W)
    """
    if probs.dim() == 3:  # (B,H,W) -> assume binary predictions already
        hard = probs.unsqueeze(1) > 0.5
        C = 1
    else:
        C = probs.size(1)
        hard = probs.argmax(dim=1)  # (B,H,W)
        hard = F.one_hot(hard, C).permute(0, 3, 1, 2).bool()

    if target.dim() == 3:
        target_oh = _to_one_hot(target, C).bool()
    else:
        target_oh = target.bool()
        if target_oh.size(1) != C:
            raise ValueError("Target channel count mismatch")

    dims = (0, 2, 3)
    inter = (hard & target_oh).sum(dim=dims).float()
    union = (hard | target_oh).sum(dim=dims).float()

    iou_c = (inter + epsilon) / (union + epsilon)

    if ignore_empty:
        valid = union > 0
        if valid.any():
            macro = iou_c[valid].mean()
        else:
            macro = torch.tensor(1.0, device=probs.device)
    else:
        macro = iou_c.mean()

    return iou_c if per_class else macro


# Backward compatible wrappers -------------------------------------------------
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):  # noqa
    """保持原介面; 現在會自動處理 (B,H,W) target.
    注意: reduce_batch_first 參數保留但不再使用。
    """
    return dice_per_class(input, target, epsilon=epsilon, ignore_empty=False, per_class=False)


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):  # noqa
    return dice_per_class(input, target, epsilon=epsilon, ignore_empty=False, per_class=False)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # input 已假設為 soft probabilities (after sigmoid/softmax)
    return 1.0 - dice_per_class(input, target, ignore_empty=False)


def accuracy(preds: Tensor, targets: Tensor, n_classes: Optional[int] = None, ignore_empty: bool = True) -> float:
    """Macro accuracy over classes. Only classes that appear in targets are averaged if ignore_empty.

    preds: (B,C,H,W) probabilities/logits 或 (B,H,W) 已離散
    targets: (B,H,W) 整數標籤
    """
    if preds.dim() == 4:
        if preds.dtype.is_floating_point:
            preds_labels = preds.argmax(dim=1)
        else:
            preds_labels = preds  # unlikely path
        C = preds.size(1)
    elif preds.dim() == 3:
        preds_labels = preds
        C = n_classes if n_classes is not None else int(preds_labels.max().item() + 1)
    else:
        raise ValueError("Unsupported preds shape")

    if n_classes is not None:
        C = n_classes

    accs = []
    for cls in range(C):
        mask = (targets == cls)
        total = mask.sum()
        if total == 0 and ignore_empty:
            continue
        correct = (preds_labels == cls) & mask
        accs.append(correct.sum().float() / (total + 1e-6))
    if not accs:
        return 1.0
    return torch.stack(accs).mean().item()


def iou_score(preds: Tensor, targets: Tensor, n_classes: Optional[int] = None, epsilon: float = 1e-6, ignore_empty: bool = True):  # noqa
    """Backward compatible: returns macro IoU.
    preds: (B,C,H,W) soft or logits; targets: (B,H,W)
    """
    if preds.dim() == 4:
        probs = F.softmax(preds, dim=1) if preds.dtype.is_floating_point else preds.float()
    elif preds.dim() == 3:
        probs = preds.unsqueeze(1).float()
    else:
        raise ValueError("Unsupported preds shape")
    if n_classes is not None and probs.size(1) != n_classes:
        # allow resizing if needed (rare)
        raise ValueError("n_classes mismatch with preds shape")
    return iou_per_class(probs, targets, epsilon=epsilon, ignore_empty=ignore_empty, per_class=False).item()


__all__ = [
    'dice_coeff', 'multiclass_dice_coeff', 'dice_loss',
    'accuracy', 'iou_score',
    'dice_per_class', 'iou_per_class'
]
