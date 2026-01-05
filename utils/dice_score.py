import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Dice Coefficient and Loss
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    assert input.size() == target.size(), "Input 和 Target 的尺寸必須相同"
    if input.dim() == 4:
        sum_dim = (-1, -2)
    elif input.dim() == 3:
        sum_dim = (-1, -2)
    else:
        raise ValueError("不支持的張量維度")

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()

def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    if multiclass:
        return 1 - multiclass_dice_coeff(input, target, reduce_batch_first=True)
    else:
        return 1 - dice_coeff(input, target, reduce_batch_first=True)

# Accuracy Calculation
def accuracy(preds, targets):
    preds = preds.argmax(dim=1)
    correct = (preds == targets).float().sum()
    total = targets.numel()
    return correct / total


def iou_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    計算 IOU (Intersection over Union) 系數
    :param input: 預測張量 (B, H, W) 或 (B, C, H, W)
    :param target: 目標張量 (B, H, W) 或 (B, C, H, W)
    :param epsilon: 防止除零的微小值
    :return: IOU 系數
    """
    assert input.size() == target.size(), "Input 和 Target 的尺寸必須相同"
    if input.dim() == 4:
        sum_dim = (-1, -2)
    elif input.dim() == 3:
        sum_dim = (-1, -2)
    else:
        raise ValueError("不支援的張量維度")

    inter = (input * target).sum(dim=sum_dim)
    union = input.sum(dim=sum_dim) + target.sum(dim=sum_dim) - inter
    union = torch.where(union == 0, inter, union)

    iou = (inter + epsilon) / (union + epsilon)
    return iou.mean()

def multiclass_iou_coeff(input: Tensor, target: Tensor, epsilon: float = 1e-6):
    """
    計算多類別 IOU 系數
    :param input: 預測張量 (B, C, H, W)
    :param target: 目標張量 (B, C, H, W)
    :param epsilon: 防止除零的微小值
    :return: 多類別 IOU 系數
    """
    return iou_coeff(input.flatten(0, 1), target.flatten(0, 1), epsilon)