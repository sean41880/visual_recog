import os
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import DetrForObjectDetection
from dataset import get_dataloaders
from tqdm import tqdm

def train_model():
    # ==========================================
    # 1. 基本設定與超參數
    # ==========================================
    DATA_DIR = '/share/sean/vr/hw2/nycu-hw2-data' # 請確認路徑
    BATCH_SIZE = 4                  # 如果 OOM 請降到 2
    EPOCHS = 50
    LR = 1e-4                       # Transformer 的學習率
    LR_BACKBONE = 1e-5              # Backbone (ResNet) 的學習率
    NUM_CLASSES = 10                # 數字 0~9 (或 1~10，共 10 類)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 啟動訓練，使用設備: {device}")

    # ==========================================
    # 2. 載入資料 (我們剛寫好的 Dataset)
    # ==========================================
    train_loader, val_loader, processor = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)

    # ==========================================
    # 3. 初始化模型 (站在巨人的肩膀上)
    # ==========================================
    print("正在載入 Hugging Face DETR 預訓練模型...")
    # ignore_mismatched_sizes=True 是因為我們把 91 類的 COCO 改成了 10 類的數字任務
    model = DetrForObjectDetection.from_pretrained(
        "facebook/detr-resnet-50",
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True 
    )
    model.to(device)

    # ==========================================
    # 4. Optimizer: 分離學習率設定 (極度重要！)
    # ==========================================
    param_dicts = [
        # 不包含 backbone 的參數 (Transformer Encoder/Decoder) 使用較大的 LR
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        # 包含 backbone 的參數 (ResNet) 使用較小的 LR
        {"params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad], "lr": LR_BACKBONE},
    ]
    optimizer = AdamW(param_dicts, lr=LR, weight_decay=1e-4)

    # 啟用 AMP (自動混合精度) 預防 OOM 並加速
    scaler = torch.cuda.amp.GradScaler()

    # ==========================================
    # 5. 開始訓練迴圈
    # ==========================================
    best_val_loss = float('inf')

    for epoch in range(EPOCHS):
        model.train()
        train_loss = 0.0
        
        # 使用 tqdm 顯示進度條
        loop = tqdm(train_loader, leave=True, desc=f"Epoch [{epoch+1}/{EPOCHS}]")
        
        for batch_encoded, targets in loop:
            # 將資料推上 GPU
            pixel_values = batch_encoded["pixel_values"].to(device)
            pixel_mask = batch_encoded["pixel_mask"].to(device)
            
            # target 是一個 list of dict，裡面的 tensor 也要推上 GPU
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()

            # 使用 AMP 進行 forward pass
            with torch.cuda.amp.autocast():
                # Hugging Face DETR 會自動幫我們計算超複雜的 Bipartite Matching Loss
                outputs = model(
                    pixel_values=pixel_values, 
                    pixel_mask=pixel_mask, 
                    labels=targets
                )
                loss = outputs.loss
                
                # loss_dict 裡面有詳細的分類、Bbox、GIoU loss，你可以印出來看
                # loss_dict = outputs.loss_dict 

            # Backward pass & 梯度更新
            scaler.scale(loss).backward()
            
            # 梯度裁剪 (防止梯度爆炸，DETR 必備)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1)
            
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = train_loss / len(train_loader)

        # ==========================================
        # 6. 驗證階段 (觀察 Overfitting)
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
        
        print(f"\n✅ Epoch {epoch+1} Summary | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        # 存檔邏輯
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), "best_detr_model.pth")
            print(f"🌟 發現更好的模型！已儲存至 best_detr_model.pth")

if __name__ == '__main__':
    train_model()