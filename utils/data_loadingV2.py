import logging
import numpy as np
import torch
from PIL import Image
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 載入影像（支援多種格式）
def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


# 獲取遮罩的唯一值（用於分類）
def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


# 數據增強函數
def get_transforms(augment=False):
    if augment:
        return A.Compose([
            A.RandomCrop(width=512, height=512),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.OneOf([
                A.GaussianBlur(p=1),
                A.MotionBlur(p=1)
            ], p=0.5),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(height=650, width=1250),
            A.Normalize(mean=(0.5,), std=(0.5,)),
            ToTensorV2()
        ])


# 基本數據集類別
class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', augment: bool = False):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        self.augment = augment
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.transforms = get_transforms(augment)

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')

        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'

        mask = np.array(load_image(mask_file[0]))
        img = np.array(load_image(img_file[0]))

        # 確保影像和遮罩大小一致
        assert img.shape[:2] == mask.shape[:2], \
            f'Image and mask {name} should be the same size, but are {img.shape} and {mask.shape}'

        # 套用增強操作
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']

        # 確保遮罩為單通道
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        return {
            'image': img.float(),
            'mask': torch.tensor(mask, dtype=torch.long)
        }



# 特殊數據集類別
class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1, augment=False):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', augment=augment)
