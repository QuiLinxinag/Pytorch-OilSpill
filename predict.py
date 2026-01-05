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

font_path = r'C:\Windows\Fonts\MSJH.TTC'  # 設定字體路徑
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
def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5, use_morph=True):
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

    # 後處理：形態學操作來去除噪點和平滑邊界（可選）
    if use_morph:
        mask_np = binary_closing(mask_np, structure=np.ones((5,5)))  # 閉運算，去除小區域
        mask_np = binary_opening(mask_np, structure=np.ones((3,3)))  # 開運算，平滑邊界

    print(f"Predicted mask unique values: {np.unique(mask_np)}")
    return mask_np

# 遮罩標準化：確保海洋背景顯示為黑色
def normalize_binary_mask(mask_np: np.ndarray):
    uniq = np.unique(mask_np)
    
    # 計算每個類別的像素數量
    class_counts = {cls: np.sum(mask_np == cls) for cls in uniq}
    
    # 找出佔最多的類別（應該是海洋背景）
    background_class = max(class_counts, key=class_counts.get)
    
    # 二值遮罩：0=海洋，1=油汙
    if set(uniq.tolist()) <= {0, 1}:
        if background_class == 1:  # 若1是背景則翻轉
            return 1 - mask_np
        return mask_np
    
    # 多類別遮罩：將佔多數的類別重新映射為0（黑色海洋背景）
    if background_class != 0:
        mask_remapped = np.zeros_like(mask_np)
        # 背景映射為 0
        mask_remapped[mask_np == background_class] = 0
        
        # 其他類別保持或映射為油汙相關類別
        # 類別1,2,3,4 中，類別2是OilSpill（青色），優先映射非背景為2
        other_classes = sorted([c for c in uniq if c != background_class])
        for cls in other_classes:
            # 統一映射為類別2（OilSpill 青色）
            mask_remapped[mask_np == cls] = 2
        
        return mask_remapped
    
    return mask_np

# 將預測的遮罩轉換為彩色圖像
def mask_to_image(mask: np.ndarray, mask_values):
    uniq = np.unique(mask)
    
    # 二值遮罩：0=海洋(黑)，1=油汙(青)
    if set(uniq.tolist()) <= {0, 1}:
        colors = np.array([
            [0, 0, 0],       # 0: 海洋背景 黑色
            [0, 255, 255],   # 1: 油汙 青色
        ])
    else:
        # 多類別：依照原始定義
        # 0: SeaSurface 黑色
        # 1: Land 綠色  
        # 2: OilSpill 青色
        # 3: Lookalike 紅色
        # 4: Ship 棕色
        colors = np.array([
            [0, 0, 0],       # 0: SeaSurface 黑色
            [0, 153, 0],     # 1: Land 綠色
            [0, 255, 255],   # 2: OilSpill 青色
            [255, 0, 0],     # 3: Lookalike 紅色
            [153, 76, 0]     # 4: Ship 棕色
        ])

    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    
    # 映射顏色
    for i in range(min(len(colors), int(mask.max()) + 1)):
        mask_rgb[mask == i] = colors[i]
        count = np.sum(mask == i)
        if count > 0:
            print(f"Class {i} ({class_names.get(i, 'Unknown')}) pixel count: {count}")

    return Image.fromarray(mask_rgb)


# 載入模型函數
def load_model(model_path, n_classes, bilinear, device):
    """載入指定路徑的模型"""
    # 載入 state_dict 檢查模型架構
    state_dict = torch.load(model_path, map_location=device)
    mask_values = state_dict.pop('mask_values', [0, 1])
    
    # 檢查是否使用 bilinear（通過檢查是否存在 up.weight 鍵）
    use_bilinear = not any('up.weight' in key for key in state_dict.keys())
    
    # 檢查模型的 feature map 大小來判斷架構
    # 檢查 down4 的輸出通道數
    down4_channels = None
    if 'down4.maxpool_conv.1.double_conv.0.weight' in state_dict:
        down4_channels = state_dict['down4.maxpool_conv.1.double_conv.0.weight'].shape[0]
    
    # 檢查是否有 attention 層
    has_attention = any('attention' in key for key in state_dict.keys())
    
    # 檢查是否為 ResBlock 架構（有 conv1、shortcut 等特徵）
    has_resblock = any('conv1.weight' in key or 'shortcut.weight' in key for key in state_dict.keys())
    
    # 根據特徵判斷模型類型
    if has_resblock and ('Resunet' in model_path or 'ResUnet' in model_path):
        net = ResUNet(n_channels=3, n_classes=5, bilinear=use_bilinear)
    elif 'CAR-UNet' in model_path:
        # 注意：CAR-UNet 的 checkpoint 可能與本專案的 MultiScaleAttentionUNet 架構不同
        # 先建立相近的模型並嘗試部分載入相符權重，以允許推論執行
        net = ResUNetWithShuffle(n_channels=3, n_classes=5, bilinear=use_bilinear)
    elif has_attention:
        # 根據 down4 的通道數判斷是 EnhancedAttentionUNet 還是 AttentionUNet
        if down4_channels and down4_channels > 512:
            net = EnhancedAttentionUNet(n_channels=3, n_classes=5, bilinear=use_bilinear)
        else:
            net = AttentionUNet(n_channels=3, n_classes=5, bilinear=use_bilinear)
    else:
        # 預設為標準 UNet（包括錯誤命名為 Resunet 但實際是 UNet 的情況）
        net = UNet(n_channels=3, n_classes=5, bilinear=use_bilinear)
    
    net.to(device=device)
    
    try:
        if 'CAR-UNet' in model_path:
            model_sd = net.state_dict()
            filtered_sd = {k: v for k, v in state_dict.items() if k in model_sd and v.shape == model_sd[k].shape}
            missing = [k for k in model_sd.keys() if k not in filtered_sd]
            unexpected = [k for k in state_dict.keys() if k not in model_sd]
            if filtered_sd:
                model_sd.update(filtered_sd)
                net.load_state_dict(model_sd)
                logging.warning(f'  CAR-UNet 部分載入：匹配 {len(filtered_sd)} 項，缺少 {len(missing)}，忽略 {len(unexpected)}')
            else:
                net.load_state_dict(state_dict, strict=False)
                logging.warning('  CAR-UNet 無匹配權重，使用隨機初始化與非嚴格載入')
        else:
            net.load_state_dict(state_dict)
        logging.info(f'  模型類型: {net.__class__.__name__}, Bilinear: {use_bilinear}, Channels: {down4_channels}')
    except RuntimeError as e:
        logging.error(f'  載入模型失敗: {e}')
        logging.error(f'  請檢查模型架構是否正確')
        raise
    
    return net, mask_values

# 比較多個模型的預測結果
def compare_predictions(models_info, img, device, scale_factor, out_threshold, use_morph=True):
    """對多個模型進行預測並比較結果"""
    predictions = {}
    
    for model_name, (net, mask_values) in models_info.items():
        # CAR-UNet 使用更高閾值和強制形態學處理來減少雜訊
        if 'CAR-UNet' in model_name or 'CAR' in model_name:
            threshold = max(0.7, out_threshold)  # 至少 0.7
            morph = True  # 強制使用形態學
            logging.info(f'  {model_name}: 使用 threshold={threshold:.1f}, morph={morph}')
        else:
            threshold = out_threshold
            morph = use_morph
        
        mask = predict_img(net=net, full_img=img, scale_factor=scale_factor, 
                          out_threshold=threshold, device=device, use_morph=morph)
        predictions[model_name] = mask
    
    return predictions

# 可視化多個模型的比較結果
def visualize_comparison(img, predictions, save_path=None, ground_truth_mask=None):
    """顯示原圖和多個模型的預測結果"""
    n_models = len(predictions)
    # 如果有 ground truth，多一格顯示
    n_cols = n_models + 2 if ground_truth_mask is not None else n_models + 1
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # 顯示原圖
    axes[0].imshow(img)
    axes[0].set_title('原始圖像', fontproperties=font_prop)
    axes[0].axis('off')
    
    # 如果有 ground truth，顯示在第二格
    start_idx = 1
    if ground_truth_mask is not None:
        gt_img = mask_to_image(ground_truth_mask, [0, 1])
        axes[1].imshow(gt_img)
        
        # 計算 ground truth 油汙比例
        total_pixels = ground_truth_mask.size
        uniq = np.unique(ground_truth_mask)
        if set(uniq.tolist()) <= {0, 1}:
            oil_pixels = np.sum(ground_truth_mask == 1)
        else:
            oil_pixels = np.sum(ground_truth_mask == 2)
        oil_percentage = (oil_pixels / total_pixels) * 100
        
        title = f'Ground Truth\n油汙: {oil_percentage:.2f}%'
        axes[1].set_title(title, fontproperties=font_prop, fontsize=10)
        axes[1].axis('off')
        start_idx = 2
    
    # 顯示每個模型的預測結果
    for idx, (model_name, mask) in enumerate(predictions.items(), start_idx):
        mask_img = mask_to_image(mask, [0, 1])
        axes[idx].imshow(mask_img)
        
        # 計算油汙像素比例（二值時為類別1，多類別時為類別2）
        total_pixels = mask.size
        uniq = np.unique(mask)
        if set(uniq.tolist()) <= {0, 1}:
            oil_pixels = np.sum(mask == 1)
        else:
            oil_pixels = np.sum(mask == 2)  # 多類別中類別2是OilSpill
        oil_percentage = (oil_pixels / total_pixels) * 100
        
        title = f'{model_name}\n油汙: {oil_percentage:.2f}%'
        axes[idx].set_title(title, fontproperties=font_prop, fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logging.info(f'比較圖已儲存至 {save_path}')
    
    plt.show()

# 解析命令行參數
def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default=r'D:\python\Unet_test\Pytorch-UNet-master\checkpoints_RESUSffle\checkpoint_epoch40.pth', metavar='FILE',
                        help='Specify the file in which the model is stored')
    parser.add_argument('--models', metavar='MODELS', nargs='+', help='多個模型路徑用於比較')
    parser.add_argument('--input', '-i', metavar='INPUT', nargs='+', help='Filenames of input images', required=True)
    parser.add_argument('--ground-truth', '-g', metavar='GT', nargs='+', help='Ground truth mask filenames (對應 input 的正確答案遮罩)')
    parser.add_argument('--output', '-o', metavar='OUTPUT', nargs='+', help='Filenames of output images')
    parser.add_argument('--viz', '-v', action='store_true', help='Visualize the images as they are processed')
    parser.add_argument('--no-save', '-n', action='store_true', help='Do not save the output masks')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5, help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--scale', '-s', type=float, default=1.0, help='Scale factor for the input images')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=5, help='Number of classes')
    parser.add_argument('--save-compare', action='store_true', help='儲存模型比較圖')
    parser.add_argument('--no-morph', action='store_true', help='停用形態學後處理（避免清除細小油汙）')

    return parser.parse_args()

# 自動生成輸出檔名
def get_output_filenames(args):
    def _generate_name(fn):
        return f'{os.path.splitext(fn)[0]}_OUT.png'

    return args.output or list(map(_generate_name, args.input))

# 主函數
if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    in_files = args.input
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # 多模型比較模式
    if args.models:
        logging.info(f'多模型比較模式：載入 {len(args.models)} 個模型')
        models_info = {}
        
        for model_path in args.models:
            model_name = os.path.basename(os.path.dirname(model_path))
            logging.info(f'載入模型: {model_name} from {model_path}')
            try:
                net, mask_values = load_model(model_path, args.classes, args.bilinear, device)
                models_info[model_name] = (net, mask_values)
            except Exception as e:
                logging.error(f'無法載入模型 {model_name}: {str(e)}')
                logging.warning(f'跳過模型 {model_name}')
                continue
        
        if not models_info:
            logging.error('沒有成功載入任何模型，程式結束')
            exit(1)
        
        logging.info(f'成功載入 {len(models_info)} 個模型！')
        
        # 對每張圖片進行預測
        for i, filename in enumerate(in_files):
            logging.info(f'預測圖片 {filename} ...')
            img = Image.open(filename)
            
            # 圖像增強
            enhancer = ImageEnhance.Contrast(img)
            img_enhanced = enhancer.enhance(1.5)
            enhancer = ImageEnhance.Brightness(img_enhanced)
            img_enhanced = enhancer.enhance(1.2)
            
            # 使用所有模型進行預測
            use_morph = not args.no_morph
            predictions = compare_predictions(models_info, img_enhanced, device, 
                                            args.scale, args.mask_threshold, use_morph=use_morph)
            # 標準化二值遮罩方向
            for k, m in list(predictions.items()):
                predictions[k] = normalize_binary_mask(m)
            
            # 儲存各模型的預測結果
            if not args.no_save:
                base_name = os.path.splitext(os.path.basename(filename))[0]
                for model_name, mask in predictions.items():
                    out_filename = f'output/{base_name}_{model_name}.png'
                    os.makedirs('output', exist_ok=True)
                    result = mask_to_image(mask, [0, 1])
                    result.save(out_filename)
                    logging.info(f'{model_name} 遮罩已儲存至 {out_filename}')
            
            # 視覺化比較
            if args.viz or args.save_compare:
                # 載入 ground truth（如果有提供）
                gt_mask = None
                if args.ground_truth and i < len(args.ground_truth):
                    gt_path = args.ground_truth[i]
                    try:
                        gt_img = Image.open(gt_path).convert('L')  # 轉灰階
                        gt_mask = np.array(gt_img)
                        # 標準化 ground truth（假設 0=背景，255=油汙 或已是 0/1）
                        if gt_mask.max() > 1:
                            gt_mask = (gt_mask > 127).astype(np.uint8)
                        gt_mask = normalize_binary_mask(gt_mask)
                        logging.info(f'  載入 ground truth: {gt_path}')
                    except Exception as e:
                        logging.warning(f'  無法載入 ground truth {gt_path}: {e}')
                
                save_path = None
                if args.save_compare:
                    base_name = os.path.splitext(os.path.basename(filename))[0]
                    save_path = f'output/{base_name}_comparison.png'
                    os.makedirs('output', exist_ok=True)
                visualize_comparison(img, predictions, save_path, ground_truth_mask=gt_mask)
    
    # 單一模型模式
    else:
        out_files = get_output_filenames(args)
        
        logging.info(f'單一模型模式：載入模型 {args.model}')
        net, mask_values = load_model(args.model, args.classes, args.bilinear, device)
        logging.info('模型載入完成！')

        for i, filename in enumerate(in_files):
            logging.info(f'預測圖片 {filename} ...')
            img = Image.open(filename)

            # 圖像增強（例如亮度和對比度調整）
            enhancer = ImageEnhance.Contrast(img)
            img_enhanced = enhancer.enhance(1.5)  # 增加對比度
            enhancer = ImageEnhance.Brightness(img_enhanced)
            img_enhanced = enhancer.enhance(1.2)  # 增加亮度

            # 預測並進行後處理
            use_morph = not args.no_morph
            mask = predict_img(net=net, full_img=img_enhanced, scale_factor=args.scale, 
                             out_threshold=args.mask_threshold, device=device, use_morph=use_morph)
            mask = normalize_binary_mask(mask)

            if not args.no_save:
                out_filename = out_files[i]
                result = mask_to_image(mask, mask_values)
                result.save(out_filename)
                logging.info(f'遮罩已儲存至 {out_filename}')

            if args.viz:
                logging.info(f'視覺化圖片 {filename} 的結果，關閉視窗以繼續...')
                plot_img_and_mask(img, mask)
