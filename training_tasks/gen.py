#!/usr/bin/env python3
"""
Generate training task scripts for Image Classification Assignment.

Usage:
    python gen.py
"""

from pathlib import Path
from jinja2 import Template

# 定義 Bash 腳本的模板
TEMPLATE = Template("""\
#!/bin/bash
# Task: Train {{ model }} with lr={{ lr }} and bs={{ bs }}

# 1. 啟動環境 (根據你在 TWCC 上的環境修改，如果你預設就有 PyTorch 就不需要這行)
# source /home/sean910526/miniconda3/bin/activate base

# 2. 切換到你的專案目錄
cd /home/sean910526/visual_recog/hw1

# 3. 執行訓練 (傳入超參數)
python train.py \\
    --model {{ model }} \\
    --batch_size {{ bs }} \\
    --lr {{ lr }}
""")

# 設定輸出資料夾
GEN_DIR = Path(__file__).parent / "0_gen"
PENDING_DIR = Path(__file__).parent / "1_pending"

def generate_task(model: str, lr: float, bs: int) -> None:
    """產生單一訓練任務的 bash 腳本"""
    # 確保資料夾存在
    GEN_DIR.mkdir(parents=True, exist_ok=True)
    PENDING_DIR.mkdir(parents=True, exist_ok=True)

    # 命名腳本檔案
    filename = f"train_{model}_lr{lr}_bs{bs}.sh"
    content = TEMPLATE.render(model=model, lr=lr, bs=bs)
    
    file_path = GEN_DIR / filename
    with open(file_path, "w") as f:
        f.write(content)
    print(f"Generated {file_path}")

if __name__ == "__main__":
    # ==========================================
    # 在這裡定義你想要跑的實驗參數組合
    # ==========================================
    # 測試 ResNet-50 在不同學習率下的表現 (Batch Size 設 64)
    generate_task("resnet50", 0.0003, 64)
    generate_task("resnet50", 0.0001, 64)
    
    # (可選) 測試 ResNet-101，因為模型變大，Batch Size 降為 32 避免 OOM
    generate_task("resnet101", 0.0003, 32)
    generate_task("resnet101", 0.0001, 32)

    print(f"\n✅ 任務腳本已產生至 {GEN_DIR}")
    print(f"下一步驟: mv {GEN_DIR}/*.sh {PENDING_DIR}/")
    print(f"最後執行: bash launch.sh")