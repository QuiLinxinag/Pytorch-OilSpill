# 🛢️ Oil Spill Semantic Segmentation (SAR Images)

[![Python](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13+-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.8-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/license-MIT-lightgrey.svg)](LICENSE)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-enabled-yellow.svg)](https://wandb.ai/)

This repository provides a **deep learning–based semantic segmentation framework for oil spill detection in SAR imagery**, built upon **U-Net and its advanced variants** (Attention U-Net, Enhanced Attention U-Net, CAR-UNet, etc.).

The project focuses on **multi-class segmentation** of SAR images into:
- Oil Spill
- Look-alike Objects
- Ships
- Land
- Sea Surface (background)

---

## 📌 Project Overview

<img width="5235" height="461" alt="img_0016_comparison" src="https://github.com/user-attachments/assets/35a4bb6b-c123-41e6-af1b-85d18b0f510f" />

> *Example SAR image segmentation results for oil spill detection.*

---

## 🧠 Model Architecture

<img width="3840" height="2160" alt="ResUnetShuffle3" src="https://github.com/user-attachments/assets/8a7dfacf-e55c-439c-96a8-6c3322b5be2f" />

The framework supports multiple encoder–decoder architectures:
- **U-Net**
- **Attention U-Net**
- **Enhanced Attention U-Net**
- **CAR-UNet (Channel Attention + Residual learning)**

All models follow a unified training and inference pipeline.

---

## 📑 Table of Contents

- [Quick Start](#quick-start)
  - [Windows 10](#windows-10)
- [Usage](#usage)
  - [Training](#training)
  - [Prediction](#prediction)
- [Weights & Biases](#weights--biases)
- [Data Description](#data-description)
- [References](#references)

---

## 🚀 Quick Start

### Windows 10

**Python version:** `3.9`

#### 1️⃣ Install CUDA & cuDNN
- Download **CUDA 11.8**  
  👉 https://developer.nvidia.com/cuda-11-8-0-download-archive  
- Install the corresponding **cuDNN**

#### 2️⃣ Install PyTorch (CUDA 11.8)
```bash
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

#### 3️⃣ Install dependencies
```bash
pip install -r requirements.txt
```

#### 4️⃣ Install TensorFlow (GPU version)
```bash
pip install tensorflow-gpu==2.10.1
```

#### 5️⃣ Download data & start training
```bash
python train.py --amp
```
⚙️ Usage
🔧 Training
```bash
python train.py -h
```

```bash
usage: train.py [-h] [--epochs E] [--batch-size B] [--learning-rate LR]
                [--load LOAD] [--scale SCALE] [--validation VAL] [--amp]

Train the UNet on images and target masks

optional arguments:
  -h, --help            show this help message and exit
  --epochs E, -e E      Number of epochs
  --batch-size B, -b B  Batch size
  --learning-rate LR, -l LR
                        Learning rate
  --load LOAD, -f LOAD  Load model from a .pth file
  --scale SCALE, -s SCALE
                        Downscaling factor of the images
  --validation VAL, -v VAL
                        Percentage of data used for validation (0–100)
  --amp                 Use automatic mixed precision
```

📌 Notes

Default --scale=0.5

Use --scale=1 for higher accuracy (requires more GPU memory)

--amp is highly recommended for faster training and lower memory usage
👉 Mixed Precision Training

🔍 Prediction

After training, the model will be saved as MODEL.pth.

Predict a single image
```bash
python predict.py -i image.jpg -o output.jpg
```
Predict multiple images (visualization only)
```bash
python predict.py -i image1.jpg image2.jpg --viz --no-save
```
Compare multiple models
```bash
python predict.py \
  --models \
  checkpoints/checkpoints_unet/checkpoint_epoch20.pth \
  checkpoints/checkpoints_attention/checkpoint_epoch20.pth \
  checkpoints/checkpoints_EnhancedAttentionUNet/checkpoint_epoch20.pth \
  checkpoints/checkpoints_CAR-UNet/checkpoint_epoch20.pth \
  --input data/imgs/img_0016.jpg data/imgs/img_0662.jpg \
  --save-compare
```
```bash
python predict.py -h
```
```bash
optional arguments:
  --model FILE, -m FILE     Path to model file (.pth)
  --input INPUT [INPUT ...]
  --output INPUT [INPUT ...]
  --viz                     Visualize predictions
  --no-save                 Do not save output masks
  --mask-threshold T        Probability threshold
  --scale SCALE             Image scale factor
```
📊 Weights & Biases

Training progress is logged using Weights & Biases (W&B):

Training & validation loss curves

Dice score metrics

Model weights & gradients

Prediction visualizations

A dashboard link will be printed in the console when training starts.

🔑 If you have a W&B account:
```bash
export WANDB_API_KEY=your_key
```

Otherwise, an anonymous run will be created and automatically deleted after 7 days.

🗂️ Data Description
```bash
data/
├── imgs/    # SAR images
└── masks/   # Corresponding segmentation masks
```
Dataset Background

A major challenge in oil spill detection is the lack of fully annotated SAR datasets.
In 2019, K. Krestenitis et al. constructed a comprehensive labeled dataset using SAR imagery collected from the Copernicus Open Access Hub (ESA).
- Satellite: Sentinel-1
- Frequency band: C-band
- Polarization: VV
- Time span: 2015/09/28 – 2017/10/31
- Image size: 1250 × 650
- Preprocessing:
  - 7×7 median filtering
  - dB-to-linear intensity conversion

Classes (5)
1. Oil Spill
2. Look-alike Objects
3. Ships
4. Land
5. Sea Surface (background)

Dataset Split
Purpose	Images
Training	890
Validation	112
Testing	110
📚 References

Krestenitis et al., Oil Spill Identification from Satellite Images Using Deep Neural Networks

Sentinel-1 SAR Data – ESA Copernicus

Pytorch-UNet
👉 https://github.com/milesial/Pytorch-UNet
