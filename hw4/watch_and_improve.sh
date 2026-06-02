#!/bin/bash
# Waits for run-1 training to finish, then launches the auto improvement pipeline.
# Run this in the background: nohup bash watch_and_improve.sh > checkpoints/pipeline.log 2>&1 &

HW4="/share/sean/visual_recog/hw4"
PYTHON="/share/sean/miniconda3/envs/vr/bin/python3"
LOG="$HW4/checkpoints/pipeline.log"
TRAIN_PID_FILE="$HW4/checkpoints/train.pid"

echo "[watch] Starting watcher at $(date)" | tee -a "$LOG"

# Wait for run-1 training to finish by polling the CSV log for epoch 149
while true; do
    if grep -q "^149," "$HW4/checkpoints/train_log.csv" 2>/dev/null; then
        echo "[watch] Run-1 training complete at $(date)" | tee -a "$LOG"
        break
    fi
    # Also exit if the training process died early (no more python3 training)
    if ! pgrep -f "python3 train.py" > /dev/null 2>&1; then
        LAST_EPOCH=$(tail -1 "$HW4/checkpoints/train_log.csv" | cut -d, -f1)
        echo "[watch] Training process ended at epoch $LAST_EPOCH ($(date))" | tee -a "$LOG"
        break
    fi
    sleep 30
done

echo "[watch] Launching auto_pipeline.py ..." | tee -a "$LOG"
cd "$HW4"
CUDA_VISIBLE_DEVICES=0,1,2,3 "$PYTHON" auto_pipeline.py \
    --run1_ckpt "$HW4/checkpoints/best.pth" \
    >> "$LOG" 2>&1

echo "[watch] Pipeline complete at $(date)" | tee -a "$LOG"
