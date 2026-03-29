#!/bin/bash
# Task: Train resnet101 with lr=0.0003 and bs=32

# 1. 啟動環境 (根據你在 TWCC 上的環境修改，如果你預設就有 PyTorch 就不需要這行)
# source /home/sean910526/miniconda3/bin/activate base

# 2. 切換到你的專案目錄
cd /home/sean910526/visual_recog/hw1

# 3. 執行訓練 (傳入超參數)
python train.py \
    --model resnet101 \
    --batch_size 32 \
    --lr 0.0003