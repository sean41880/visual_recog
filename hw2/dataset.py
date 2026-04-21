import os
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection
from transformers import DetrImageProcessor
import random

class DigitDetectionDataset(CocoDetection):
    def __init__(self, root, annFile, processor, train=True):
        super().__init__(root, annFile)
        self.processor = processor
        self.train = train

    def __getitem__(self, idx):
        # 1. 讀取圖片與原始 target (COCO 格式)
        img, target = super().__getitem__(idx)
        
        # 2. 處理 target
        # image_id 必須是 tensor
        image_id = self.ids[idx]
        image_id = torch.tensor([image_id])

        # 整理 Bounding Boxes 和 Labels
        boxes = []
        labels = []
        area = []
        iscrowd = []
        
        for t in target:
            # COCO 預設 bbox 是 [x_min, y_min, width, height]
            # Hugging Face DETR 預期的是 [x_min, y_min, x_max, y_max] 或保持 COCO 格式交給 processor 處理
            # 我們直接把原始格式傳給 processor，它會幫我們轉換並 Normalize
            boxes.append(t['bbox'])
            labels.append(t['category_id'])
            area.append(t['area'] if 'area' in t else t['bbox'][2] * t['bbox'][3])
            iscrowd.append(t['iscrowd'] if 'iscrowd' in t else 0)

        # 構建 DETR 需要的 target 字典
        target = {
            'image_id': image_id,
            'annotations': [
                {'bbox': b, 'category_id': l, 'area': a, 'iscrowd': c}
                for b, l, a, c in zip(boxes, labels, area, iscrowd)
            ]
        }

        # 3. 使用 Processor 進行圖片增強與 Bbox 正規化
        if self.train:
            # 🌟 黑科技：Multi-scale Training 
            # 隨機挑選一個短邊長度 (480 到 800 之間)
            random_size = random.choice([480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800])
            
            encoding = self.processor(
                images=img, 
                annotations=target, 
                return_tensors="pt",
                # 動態覆蓋預設大小，最長邊限制在 1333 避免顯存炸掉
                size={"shortest_edge": random_size, "longest_edge": 1333} 
            )
        else:
            # 驗證集與測試集必須保持固定大小 (通常是 800)，這樣分數才準確！
            encoding = self.processor(
                images=img, 
                annotations=target, 
                return_tensors="pt",
                size={"shortest_edge": 800, "longest_edge": 1333}
            )
        
        pixel_values = encoding["pixel_values"].squeeze()
        target = encoding["labels"][0]

        return pixel_values, target

# ==========================================
# Collate Function: 處理 Batch 中不同大小的圖片
# ==========================================
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # 1. 找出這個 Batch 裡面最大的高 (H) 和寬 (W)
    max_h = max([pv.shape[1] for pv in pixel_values])
    max_w = max([pv.shape[2] for pv in pixel_values])
    
    # 2. 建立全 0 的畫布 (代表圖片) 和 遮罩 (代表有效區域)
    batch_size = len(batch)
    padded_pixel_values = torch.zeros((batch_size, 3, max_h, max_w), dtype=torch.float32)
    
    # pixel_mask 裡面：0 代表是黑邊(Padding)，1 代表是真實圖片
    pixel_mask = torch.zeros((batch_size, max_h, max_w), dtype=torch.long)
    
    # 3. 把圖片一張一張貼上畫布的左上角
    for i, pv in enumerate(pixel_values):
        _, h, w = pv.shape
        padded_pixel_values[i, :, :h, :w] = pv
        pixel_mask[i, :h, :w] = 1 # 標示這塊區域是真實圖片
        
    # 4. 包裝成 DETR 喜歡的字典格式
    batch_encoded = {
        "pixel_values": padded_pixel_values,
        "pixel_mask": pixel_mask
    }
    
    return batch_encoded, targets

def get_dataloaders(data_dir, batch_size=4):
    # 初始化 Hugging Face 的 Image Processor (這會處理所有的 Resize, Normalize)
    # 我們使用基礎的 resnet50 processor
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")


    # 設定路徑 (請依據你實際解壓縮的資料夾結構調整)
    train_img_dir = os.path.join(data_dir, 'train')
    train_ann_file = os.path.join(data_dir, 'train.json')
    
    val_img_dir = os.path.join(data_dir, 'valid')
    val_ann_file = os.path.join(data_dir, 'valid.json')
    
    print("正在載入訓練集...")
    train_dataset = DigitDetectionDataset(train_img_dir, train_ann_file, processor=processor, train=True)
    print("正在載入驗證集...")
    val_dataset = DigitDetectionDataset(val_img_dir, val_ann_file, processor=processor, train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    return train_loader, val_loader, processor