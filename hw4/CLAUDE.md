# HW4 Visual Recognition — Image Restoration (PromptIR)

## Project State (last updated: 2026-05-31)

### Deliverables
- `pred.npz` — **ready for CodaBench submission** (run-4 + TTA, ~26.01 dB val PSNR)
- `pred_run1_backup.npz` — fallback from run-1 + TTA (24.93 dB)

### Key Paths
```
/share/sean/visual_recog/hw4/
├── models/promptir.py        — PromptIR model implementation
├── dataset.py                — PairDataset + TestDataset
├── losses.py                 — CombinedLoss (L1 + FFT + SSIM)
├── train.py                  — training script (multi-GPU, cosine LR, AMP)
├── inference.py              — basic inference → pred.npz
├── inference_tta.py          — TTA inference (8-way augmentation)
├── evaluate_val.py           — eval val PSNR ± TTA
├── auto_pipeline.py          — autonomous fine-tune pipeline
├── pred.npz                  — FINAL SUBMISSION FILE ✅
├── pred_run1_backup.npz      — run-1 backup
└── checkpoints/
    ├── best.pth              — run-1 best (24.75 dB, dim=48)
    ├── run4/best.pth         — run-4 best (25.81 dB, dim=64) ← BEST MODEL
    ├── run5/                 — run-5 in progress (fine-tune run-4 at 192px)
    └── */train_log.csv       — per-epoch CSV logs for each run
```

### Training Runs Summary
| Run | Model | Patch | Epochs | Best Val PSNR | TTA PSNR | Notes |
|-----|-------|-------|--------|--------------|----------|-------|
| run-1 | dim=48 | 128px | 150 | 24.75 dB | 24.93 dB | baseline, from scratch |
| run-2 | dim=48 | 192px | 100 | 24.66 dB | 24.83 dB | finetune run-1, slightly worse |
| run-3 | dim=48 | 256px | 30 | 24.56 dB | — | finetune run-1, no ckpt saved (bug) |
| **run-4** | **dim=64** | 128px | 200 | **25.81 dB** | **26.01 dB** | **best model, from scratch** |
| run-5 | dim=64 | 192px | 80 | in progress | — | finetune run-4, `--reset_best_psnr` |

### Environment
- Python: 3.13, conda env `vr` → `/share/sean/miniconda3/envs/vr/bin/python3`
- PyTorch: 2.6.0+cu124, 4× H100 80GB GPUs
- Run all training with: `CUDA_VISIBLE_DEVICES=0,1,2,3`

### Key Implementation Details
- **Model**: PromptIR — Restormer encoder-decoder + PromptBlock at every level
  - dim=64 → ~61M params (best); dim=48 → ~34M params (baseline)
  - K=5 prompt components per level, attention-weighted from GAP of features
  - **SE channel attention added to GDFN** (modification #1 for report)
- **Loss**: L1 + 0.1×FFT_L1 + 0.1×SSIM (modification #2 for report)
- **TTA**: 8-way (all combinations of H-flip, V-flip, rot90) → +~0.2 dB free
- **Dataset split**: 3040 train / 160 val (5% holdout, seed=42)
- **LR**: cosine annealing base→1e-6, 5-epoch linear warmup

### How to Generate pred.npz from Best Model
```bash
cd /share/sean/visual_recog/hw4

# TTA inference (best quality, ~2 min on 4× H100):
CUDA_VISIBLE_DEVICES=0,1,2,3 /share/sean/miniconda3/envs/vr/bin/python3 \
  inference_tta.py \
  --checkpoint checkpoints/run4/best.pth \
  --dim 64 --gpus 0,1,2,3 --output pred.npz
```

### How to Check / Use Run-5 Results
```bash
# Check progress:
tail -f /share/sean/visual_recog/hw4/checkpoints/run5/train_log.csv

# Evaluate run-5 with TTA (run after training completes):
CUDA_VISIBLE_DEVICES=0 /share/sean/miniconda3/envs/vr/bin/python3 \
  evaluate_val.py --checkpoint checkpoints/run5/best.pth --dim 64 --tta

# If run-5 TTA PSNR > 26.01 dB, update pred.npz:
CUDA_VISIBLE_DEVICES=0,1,2,3 /share/sean/miniconda3/envs/vr/bin/python3 \
  inference_tta.py \
  --checkpoint checkpoints/run5/best.pth \
  --dim 64 --gpus 0,1,2,3 --output pred.npz
```

### How to Start a New Training Run
```bash
cd /share/sean/visual_recog/hw4
mkdir -p checkpoints/runN

CUDA_VISIBLE_DEVICES=0,1,2,3 nohup /share/sean/miniconda3/envs/vr/bin/python3 train.py \
  --data_root hw4_realse_dataset/train \
  --save_dir checkpoints/runN \
  --gpus 0,1,2,3 --epochs 200 --batch_size 6 --patch_size 128 \
  --lr 2e-4 --dim 64 --num_prompts 5 \
  --lambda_fft 0.1 --lambda_ssim 0.1 \
  > checkpoints/runN/train.log 2>&1 &

# For fine-tuning from an existing checkpoint, add:
#   --resume checkpoints/runX/best.pth --reset_best_psnr
```

### Verify pred.npz Before Submission
```bash
python3 -c "
import numpy as np
d = np.load('pred.npz')
keys = sorted(d.keys(), key=lambda x: int(x.split('.')[0]))
print(f'{len(d)} images | first={keys[0]} last={keys[-1]}')
arr = d[keys[0]]
print(f'shape={arr.shape} dtype={arr.dtype} range=[{arr.min()},{arr.max()}]')
"
# Expected: 100 images, shape=(3,256,256), dtype=uint8, range=[0,255]
```

### Submission Checklist
- [ ] Verify pred.npz format (see above)
- [ ] Submit pred.npz to CodaBench
- [ ] Push all code to GitHub with README.md
- [ ] Write PDF report in English (ECCV 2026 template)
- [ ] Zip code (.py files) + report → submit to E3
- Deadline: **23:59, 2026-06-02**

### Report Outline (key content)
1. **Intro**: all-in-one image restoration, single model for rain+snow
2. **Method**:
   - PromptIR architecture: encoder-decoder, MDTA attention, GDFN, PromptBlock
   - Modification 1 — SE channel attention in GDFN: *hypothesis* recalibrates channels that carry rain/snow artifact info
   - Modification 2 — FFT loss: *hypothesis* rain/snow are quasi-periodic → frequency-domain L1 penalises structured artifacts directly
   - Hyperparams: dim=64, K=5 prompts, L1+0.1FFT+0.1SSIM, AdamW 2e-4, cosine decay 200ep
3. **Results**:
   - Training curve: `checkpoints/run4/train_log.csv`
   - dim=48 vs dim=64: +1.06 dB
   - No TTA vs TTA: +0.2 dB
   - Larger patches (192px) fine-tune did NOT help (report this as a negative result with explanation)
4. **References**: PromptIR (NeurIPS 2023), Restormer (CVPR 2022)

### Known Issues
- `run3/best.pth` missing: fine-tune at 256px resumed run-1's best_psnr=24.75, val PSNR peaked at 24.56 so no checkpoint was saved. Fixed with `--reset_best_psnr` flag.
- Python 3.13 + triton: `pkg_resources.ImpImporter` removed in 3.13 → fixed by `pip install --upgrade setuptools`
- Val PSNR oscillates ±0.5 dB (only 160 val images) — always use `best.pth` not `last.pth`
