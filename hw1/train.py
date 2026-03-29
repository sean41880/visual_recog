import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import argparse

# 匯入我們自己寫的模組
from dataset import get_dataloaders, get_mixup_cutmix
from model import CustomResNet

def train_model():
    # ==========================================
    # 1. 讀取命令列超參數 (支援從 bash script 傳入)
    # ==========================================
    parser = argparse.ArgumentParser(description="Image Classification Training")
    parser.add_argument('--model', type=str, default='resnet50', help='Model backbone')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    args = parser.parse_args()

    DATA_DIR = '/home/sean910526/visual_recog/data'
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = 100
    WEIGHT_DECAY = 1e-4
    NUM_CLASSES = 100
    
    # 讓存檔名稱自動跟著參數改變，避免覆蓋！
    SAVE_PATH = f'best_{args.model}_lr{args.lr}_bs{args.batch_size}.pth'
    
    print(f"🚀 開始訓練: Model={args.model}, LR={LEARNING_RATE}, Batch Size={BATCH_SIZE}")

    # 設定設備 (自動偵測是否有 GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"目前使用的運算設備: {device}")

    # ==========================================
    # 2. 準備資料與模型
    # ==========================================
    print("正在載入資料...")
    train_loader, val_loader, class_names = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    cutmix_or_mixup = get_mixup_cutmix(num_classes=NUM_CLASSES)

    print("正在載入模型...")
    model = CustomResNet(backbone_name=args.model, num_classes=NUM_CLASSES, dropout_rate=0.5)
    model = model.to(device)

    # ==========================================
    # 3. 定義 Loss, Optimizer 與 Scheduler
    # ==========================================
    # 加入 label_smoothing=0.1，這能防止模型對自己的預測過度自信，是非常好的加分項！
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 使用 AdamW 取代傳統 SGD，收斂更快且泛化更好
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 餘弦退火學習率：讓學習率先平穩下降，最後階段變得非常小以精細微調
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # ==========================================
    # 4. 開始訓練迴圈
    # ==========================================
    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        start_time = time.time()
        
        # --- 訓練階段 ---
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # 【關鍵】應用 Mixup 或 CutMix (這會把 labels 變成機率分佈)
            images, labels = cutmix_or_mixup(images, labels)
            
            optimizer.zero_grad()
            outputs = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * images.size(0)
            
        train_loss = train_loss / len(train_loader.dataset)
        
        # --- 驗證階段 ---
        model.eval()
        val_loss = 0.0
        corrects = 0
        total = 0
        
        with torch.no_grad(): # 驗證時不計算梯度，節省記憶體與時間
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * images.size(0)
                
                # 計算準確率 (找機率最高的那一類)
                _, preds = torch.max(outputs, 1)
                corrects += torch.sum(preds == labels.data)
                total += labels.size(0)
                
        val_loss = val_loss / len(val_loader.dataset)
        val_acc = corrects.double() / total
        
        # 更新學習率
        scheduler.step()
        
        epoch_time = time.time() - start_time
        
        # 輸出進度
        print(f"Epoch [{epoch+1}/{EPOCHS}] | "
              f"Time: {epoch_time:.0f}s | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"  🌟 發現更好的模型！已儲存至 {SAVE_PATH} (準確率: {best_val_acc:.4f})")

    print(f"\n訓練完成！最高驗證集準確率: {best_val_acc:.4f}")

if __name__ == '__main__':
    train_model()