import argparse
import logging
import os
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from torchvision import transforms
from unet import AttentionUNet, UNet,SiameseAttentionUNet,EnhancedAttentionUNet,UNetWithHybridPooling,MultiScaleAttentionUNet
from unet import ResUNet,ResUNetWithShuffle
from utils.data_loading import BasicDataset
from utils.utils import plot_img_and_mask
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.ndimage import binary_opening, binary_closing  # 引入形態學操作

font_path = 'C:\Windows\Fonts\MSJH.TTC'  # 設定字體路徑
font_prop = font_manager.FontProperties(fname=font_path)
plt.rcParams['font.family'] = font_prop.get_name()

class_names = {
    0: 'SeaSurface',  # 黑色 [0, 0, 0]
    1: 'Land',        # 綠色 [0, 153, 0]
    2: 'OilSpill',    # 青色 [0, 255, 255]
    3: 'Lookalike',   # 紅色 [255, 0, 0]
    4: 'Ship'         # 棕色 [153, 76, 0]
}

# 預測函數，支援多類別預測並加入後處理
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    net.eval()
    # 圖像預處理：將圖片增強以提升預測效果
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    mask_np = mask[0].long().squeeze().numpy()

    # 後處理：形態學操作來去除噪點和平滑邊界
    mask_np = binary_closing(mask_np, structure=np.ones((5,5)))  # 閉運算，去除小區域
    mask_np = binary_opening(mask_np, structure=np.ones((3,3)))  # 開運算，平滑邊界

    print(f"Predicted mask unique values: {np.unique(mask_np)}")
    return mask_np

# 將預測的遮罩轉換為彩色圖像
def mask_to_image(mask: np.ndarray, mask_values):
    colors = np.array([
        [0, 0, 0],       # 類別 0 (背景): 黑色
        [0, 255, 255],     # 類別 1: 深綠色
        [0, 153, 0],   # 類別 2: 青色
        [153, 76, 0],    # 類別 3: 棕色
        [255, 0, 0]      # 類別 4: 紅色
    ])
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    for i in range(len(colors)):
        mask_rgb[mask == i] = colors[i]
        print(f"Class {i} pixel count: {np.sum(mask == i)}")

    return Image.fromarray(mask_rgb)


def load_model(model_path, device, args):
    net = ResUNetWithShuffle(n_channels=3, n_classes=args.classes, bilinear=args.bilinear)
    net.to(device=device)
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    net.load_state_dict(state_dict)
    return net, mask_values


# 解析命令行參數
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default='D:\\Pytorch-UNet-master\\checkpoints\\checkpoints_CAR-UNet\\checkpoint_epoch20.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--models', '-M', metavar='FILE', nargs='+', help='Multiple checkpoint paths; overrides --model when provided')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--save-compare', '-C', action='store_true', help='Save comparison image across models for each input')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')

    return parser.parse_args()


# 自動生成輸出檔名
def get_output_filenames(args, model_tag):
    def _generate_name(fn):
        return f"{os.path.splitext(fn)[0]}_{model_tag}_OUT.png"

    if args.output:
        if len(args.output) != len(args.input):
            raise ValueError('When specifying --output, its length must match --input')
        out_files = []
        for fn in args.output:
            stem, ext = os.path.splitext(fn)
            out_files.append(f"{stem}_{model_tag}{ext or '.png'}")
        return out_files

    return list(map(_generate_name, args.input))


def save_comparison_figure(base_img, comparisons, save_path):
    if not comparisons:
        return
    cols = len(comparisons) + 1
    fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))
    axes[0].imshow(base_img)
    axes[0].set_title('Input')
    axes[0].axis('off')
    for idx, (model_tag, mask, mask_values) in enumerate(comparisons, start=1):
        axes[idx].imshow(np.array(mask_to_image(mask, mask_values)))
        axes[idx].set_title(model_tag)
        axes[idx].axis('off')
    plt.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)

# 主函數
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    in_files = args.input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_paths = args.models if args.models else [args.model]
    model_tags = [os.path.splitext(os.path.basename(p))[0] for p in model_paths]

    logging.info(f'Using device {device}')

    # 預先載入輸入圖片避免重複 IO
    images_cache = {fn: Image.open(fn).convert('RGB') for fn in in_files}
    comparisons = {fn: [] for fn in in_files}

    for model_path, model_tag in zip(model_paths, model_tags):
        logging.info(f'Loading model {model_path}')
        net, mask_values = load_model(model_path, device, args)
        logging.info('Model loaded!')

        out_files = get_output_filenames(args, model_tag)

        for i, filename in enumerate(in_files):
            logging.info(f'Predicting image {filename} with {model_tag} ...')
            img = images_cache[filename].copy()

            # 圖像增強（例如亮度和對比度調整）
            enhancer = ImageEnhance.Contrast(img)
            img = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(1.2)

            mask = predict_img(net=net, full_img=img, scale_factor=args.scale, out_threshold=args.mask_threshold, device=device)

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'Mask saved to {out_filename}')

            comparisons[filename].append((model_tag, mask, mask_values))

            if args.viz:
                logging.info(f'Visualizing results for image {filename} ({model_tag}), close to continue...')
                plot_img_and_mask(img, mask)

    if args.save_compare:
        for filename in in_files:
            compare_path = f"{os.path.splitext(filename)[0]}_comparison.png"
            save_comparison_figure(images_cache[filename], comparisons[filename], compare_path)
            logging.info(f'Comparison saved to {compare_path}')
