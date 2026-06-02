# HW4 — All-in-One Image Restoration (PromptIR)

Single model for joint rain and snow removal using **PromptIR** (NeurIPS 2023).  
Competition metric: PSNR (dB) on CodaBench.

## Results

| Model | Val PSNR | Val PSNR (8-way TTA) |
|-------|----------|----------------------|
| PromptIR dim=48 (baseline) | 24.75 dB | 24.93 dB |
| PromptIR dim=64 | **25.81 dB** | **26.01 dB** |

Best submission: **26.01 dB** (dim=64, 200 epochs, 8-way TTA)

## Method

### Architecture

PromptIR uses a Restormer-style encoder-decoder with **PromptBlocks** injected at every level. Each PromptBlock maintains K=5 learnable prompt components that are adaptively weighted via global average pooling of the input features — allowing a single model to handle multiple degradation types without knowing which degradation is present at test time.

Key building blocks:
- **MDTA** (Multi-Dconv Head Transposed Attention): channel-wise attention with O(C²·HW) complexity instead of O((HW)²)
- **GDFN** (Gated Depth-wise Feed-forward Network): gated convolution for local feature mixing

### Modifications

**1. SE Channel Attention in GDFN**  
A Squeeze-and-Excitation block was added after the depth-wise convolution inside GDFN. Global average pooling produces a channel descriptor, which is recalibrated through two FC layers and a sigmoid gate. This helps the network selectively emphasise channels that carry degradation-relevant information.

**2. Frequency-Domain Loss**  
The training loss combines three terms:
```
L = L1 + 0.1 × FFT_L1 + 0.1 × SSIM
```
`FFT_L1` is an L1 loss computed on the magnitude spectrum of the prediction and ground truth. Rain and snow are quasi-periodic textures that are more directly penalised in the frequency domain than in pixel space.

### Training Details

| Hyperparameter | Value |
|----------------|-------|
| Base channel dim | 64 |
| Prompt components K | 5 |
| Patch size | 128 × 128 |
| Batch size | 32 (8 per GPU × 4 GPUs) |
| Epochs | 200 |
| Optimizer | AdamW, weight decay 1e-4 |
| Learning rate | 2e-4 → 1e-6 cosine + 5-epoch warmup |
| AMP | bfloat16 |
| Hardware | 4× NVIDIA H100 80GB |

Dataset split: 3040 train / 160 val (5% holdout, seed=42).

### Test-Time Augmentation (TTA)

8-way TTA averages predictions across all combinations of horizontal flip, vertical flip, and 90° rotation. Each augmentation is applied to the input, the model is run, then the reverse transform is applied to the output before averaging. Free +0.2 dB gain.

## Repository Structure

```
hw4/
├── models/
│   ├── __init__.py
│   └── promptir.py       # PromptIR model (MDTA + GDFN + PromptBlock)
├── dataset.py            # PairDataset (train/val) + TestDataset
├── losses.py             # CombinedLoss: L1 + FFT_L1 + SSIM
├── train.py              # Training script (DataParallel, AMP, cosine LR)
├── inference.py          # Basic inference → pred.npz
├── inference_tta.py      # 8-way TTA inference → pred.npz
├── evaluate_val.py       # Evaluate val PSNR ± TTA
└── evaluate_ensemble.py  # Ensemble multiple checkpoints with TTA
```

## Usage

**Train from scratch:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --dim 64 --epochs 200 --batch_size 8 --patch_size 128 \
  --lr 2e-4 --save_dir checkpoints/run
```

**Fine-tune from checkpoint:**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python train.py \
  --dim 64 --epochs 260 --batch_size 6 --patch_size 192 \
  --lr 5e-5 --resume checkpoints/run/best.pth --reset_best_psnr \
  --save_dir checkpoints/run_ft
```

**TTA inference (generates pred.npz for submission):**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python inference_tta.py \
  --checkpoint checkpoints/run/best.pth \
  --dim 64 --output pred.npz
```

**Evaluate val PSNR:**
```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_val.py \
  --checkpoint checkpoints/run/best.pth \
  --dim 64 --tta
```

## References

- Potlapalli et al., *PromptIR: Prompting for All-in-One Blind Image Restoration*, NeurIPS 2023. [arXiv:2306.13090](https://arxiv.org/abs/2306.13090)
- Zamir et al., *Restormer: Efficient Transformer for High-Resolution Image Restoration*, CVPR 2022.
- Hu et al., *Squeeze-and-Excitation Networks*, CVPR 2018.
