# 100-Class Image Classification with ResNet Ensemble

This project implements a high-performance image classification pipeline for a 100-category dataset. By combining modern regularization techniques and a multi-scale ensemble inference strategy, the model achieves a top-tier accuracy of **0.95** on the test leaderboard.

---

## 1. Introduction

The core of this method is built on a modified ResNet architecture. Key features include:

* **Custom Classification Head**
  Replaces the vanilla linear layer with a Bottleneck MLP (Linear → BN → ReLU → Dropout) to prevent overfitting on the 100-class subset.

* **Advanced Regularization**
  Integration of:

  * MixUp (α = 0.4)
  * CutMix (α = 1.0)
  * Label Smoothing (0.1)

* **Inference Strategy**
  A robust ensemble of ResNet-50 and ResNet-101 using Multi-scale Test-Time Augmentation (TTA).

---

## 2. Environment Setup

The code is developed and tested on the **TWCC (Taiwan Computing Cloud)** environment.

### Prerequisites

* OS: Linux (Ubuntu)
* Python: 3.13+
* GPU: NVIDIA A100 / V100 / H100 (CUDA 12.x recommended)

### Installation

```bash
conda create -n vision_hw1 python=3.13 -y
conda activate vision_hw1

pip install torch torchvision torchaudio
pip install pandas Pillow tqdm
```

---

## 3. Usage

### Data Preparation

Ensure the dataset is organized as follows:

```plaintext
data/
├── train/
│   ├── 0/
│   ├── 1/
│   └── ... (up to 99)
└── test/
    ├── image_1.jpg
    └── ...
```

### Training

To train the ResNet-50 model with default hyperparameters:

```bash
python train.py --model resnet50 --lr 0.0001 --batch_size 64
```

### Inference (Final Submission)

To generate `prediction.csv` using the **0.95 Ensemble + TTA strategy**:

```bash
python super_test_v2.py
```

---

## 4. Performance Snapshot

| Backbone(s)     | Strategy               | Val Acc | Test Acc |
| --------------- | ---------------------- | ------- | -------- |
| ResNet-50       | Baseline               | 84.2%   | -        |
| ResNet-50       | Custom Head + Aug      | 91.0%   | 0.94     |
| ResNet-50 + 101 | Ensemble + 3-Scale TTA | -       | **0.95** |
