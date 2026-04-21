import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from dataset import get_dataloaders
from tqdm import tqdm

def train_resume():
    # ==========================================
    # 1. 基本設定
    # ==========================================
    DATA_DIR = '/share/sean/vr/hw2/nycu-hw2-data'
    BATCH_SIZE = 4
    EPOCHS_TO_ADD = 50              # 再多跑 50 個 Epoch
    LR = 5e-5                       # 因為是接續微調，我們把 Transformer LR 稍微調小一點點
    LR_BACKBONE = 1e-5
    WEIGHT_PATH = 'best_detr_model.pth' # 你上一波跑完的權重
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動【接續訓練】，使用設備: {device}")

    train_loader, val_loader, processor = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # ==========================================
    # 2. 載入上一波的最強模型，並修改 Loss 權重！
    # ==========================================
    print(f"正在載入你剛剛練成的權重: {WEIGHT_PATH}")
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=10, 
        ignore_mismatched_sizes=True
    )
    
    # 🌟 核心修改 1：載入你訓練好的權重
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    
    # 🌟 核心修改 2：調整 Loss 權重 (策略三)
    # 讓模型「先求認出數字對不對，框框沒那麼準沒關係」
    model.config.class_cost = 2  # 預設 1.0 -> 提高分類的權重
    model.config.bbox_cost = 2   # 預設 5.0 -> 降低 L1 框框的權重
    model.config.giou_cost = 2   # 預設 2.0 -> 保持交集比例的權重

    model.to(device)

    # ==========================================
    # 3. Optimizer & AMP 設定
    # ==========================================
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
    ]
    optimizer = AdamW(param_dicts, lr=LR, weight_decay=1e-4)
    scaler = torch.cuda.amp.GradScaler()

    # ==========================================
    # 4. 接續訓練迴圈
    # ==========================================
    best_val_loss = float('inf') # 我們重新計算這 50 輪的 best

    for epoch in range(EPOCHS_TO_ADD):
        model.train()
        train_loss = 0.0
        
        # 標示為 Epoch 51~100，讓你好辨識
        loop = tqdm(train_loader, leave=True, desc=f"Phase 2 Epoch [{epoch+51}/{EPOCHS_TO_ADD+50}]")
        
        for batch_encoded, targets in loop:
            pixel_values = batch_encoded["pixel_values"].to(device)
            pixel_mask = batch_encoded["pixel_mask"].to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
                loss = outputs.loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ==========================================
        # 5. 驗證階段
        # ==========================================
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_encoded, targets in val_loader:
                pixel_values = batch_encoded["pixel_values"].to(device)
                pixel_mask = batch_encoded["pixel_mask"].to(device)
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                with torch.cuda.amp.autocast():
                    outputs = model(pixel_values=pixel_values, pixel_mask=pixel_mask, labels=targets)
                    val_loss += outputs.loss.item()
                    
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\n✅ Phase 2 Epoch {epoch+51} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 覆蓋存檔：只要這輪比 Phase 2 之前跑得好，就更新 best_detr_model.pth
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # 我們存成另外一個檔名，以免覆蓋掉你 0.3 的保底分數！
            torch.save(model.state_dict(), "best_detr_phase2.pth")
            print(f"🌟 Phase 2 發現更好的模型！已儲存至 best_detr_phase2.pth")

if __name__ == '__main__':
    train_resume()