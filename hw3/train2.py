import os
import torch
import torchvision.transforms.v2 as T
from torch.utils.data import DataLoader

# 引入你自己寫的模組
from dataset import CellDataset
from model2 import get_model_instance_segmentation

def collate_fn(batch):
    return tuple(zip(*batch))

def main():
    # 1. 基礎設定
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")
    
    num_classes = 5 
    num_epochs = 100 
    batch_size = 2  
    
    # 2. 資料擴增 (Data Augmentation)
    transform = T.Compose([
        T.ToDtype(torch.float32, scale=True),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.5),
        T.RandomRotation(degrees=(0, 180)), # <--- 加入隨機旋轉，細胞不分正反面！
        T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
    ])
    
    dataset = CellDataset(root_dir='train', transforms=transform)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn)

    # 3. 載入模型 (從你寫的 model.py)
    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    # 4. 優化器與學習率排程
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=2e-4, weight_decay=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

    # 5. 訓練迴圈
    print("開始百煉成鋼...")
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0
        
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            optimizer.zero_grad()
            loss_dict = model(images, targets)
            # --- 加入這行：強制加重 Mask 預測的懲罰力度 (例如放大 2 倍) ---
            loss_dict['loss_mask'] = loss_dict['loss_mask'] * 2.0

            losses = sum(loss for loss in loss_dict.values())
            
            losses.backward()
            optimizer.step()
            epoch_loss += losses.item()
            
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch: {epoch+1}/{num_epochs}, LR: {current_lr:.6f}, Total Loss: {epoch_loss/len(data_loader):.4f}")

        # 每 10 個 Epoch 存檔
        # 在 train_B.py 底部存檔的地方，把路徑改成 checkpoints_B
        if (epoch + 1) % 10 == 0:
            save_path = f'checkpoints_B/mask_rcnn_epoch_{epoch+1}.pth'
            torch.save(model.state_dict(), save_path)

    torch.save(model.state_dict(), 'checkpoints_B/mask_rcnn_final.pth')

if __name__ == '__main__':
    main()