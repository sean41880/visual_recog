import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from utils import read_maskfile
from torchvision import tv_tensors

class CellDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.img_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])

    def __getitem__(self, idx):
        # 1. 取得影像路徑
        img_dir_path = os.path.join(self.root_dir, self.img_dirs[idx])
        img_path = os.path.join(img_dir_path, "image.tif")
        
        img = Image.open(img_path).convert("RGB")
        w, h = img.size
        
        # 2. 讀取並處理 Masks
        masks = []
        labels = []
        
        for class_id in range(1, 5): 
            mask_path = os.path.join(img_dir_path, f"class{class_id}.tif")
            if os.path.exists(mask_path):
                mask_img = read_maskfile(mask_path) 
                obj_ids = np.unique(mask_img)
                obj_ids = obj_ids[obj_ids != 0] 
                
                for obj_id in obj_ids:
                    binary_mask = mask_img == obj_id 
                    masks.append(binary_mask)
                    labels.append(class_id)
        
        num_objs = len(labels)
        if num_objs == 0:
            masks = torch.zeros((0, h, w), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
            labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # 3. 將資料包裝為 tv_tensors 格式以支援 v2 Transform
        img = tv_tensors.Image(img)
        target = {}
        # ⚠️ 注意：這裡我們「不」把 boxes 丟進去，避免它被 Transform 算壞
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # 4. 執行 Transform (影像與 Mask 會同時旋轉、翻轉)
        if self.transforms is not None:
            img, target = self.transforms(img, target)
            
        # 5. Transform 結束後，重新根據「旋轉後的 Mask」計算 Bounding Box
        transformed_masks = target["masks"]
        transformed_labels = target["labels"]
        
        valid_boxes = []
        valid_masks = []
        valid_labels = []
        
        for i in range(len(transformed_labels)):
            # 如果這個細胞剛好在邊緣，旋轉後整顆被切掉不見了，就過濾掉它
            if not torch.any(transformed_masks[i]):
                continue
                
            pos = torch.where(transformed_masks[i])
            xmin = torch.min(pos[1]).float()
            xmax = torch.max(pos[1]).float()
            ymin = torch.min(pos[0]).float()
            ymax = torch.max(pos[0]).float()
            
            # 強制保證寬高 > 0 (防止 1 pixel 的細胞被壓扁成 0 導致報錯)
            if xmax == xmin: xmax += 1.0
            if ymax == ymin: ymax += 1.0
                
            valid_boxes.append([xmin, ymin, xmax, ymax])
            valid_masks.append(transformed_masks[i])
            valid_labels.append(transformed_labels[i])
            
        # 6. 更新 target (處理極端情況與正常情況)
        if len(valid_boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            masks = torch.zeros((0, img.shape[-2], img.shape[-1]), dtype=torch.uint8)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(valid_boxes, dtype=torch.float32)
            masks = torch.stack(valid_masks)
            labels = torch.stack(valid_labels)
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        # 把絕對正確的 Box 放進 target
        target["boxes"] = tv_tensors.BoundingBoxes(boxes, format="XYXY", canvas_size=(img.shape[-2], img.shape[-1]))
        target["masks"] = tv_tensors.Mask(masks)
        target["labels"] = labels
        target["area"] = area
        target["iscrowd"] = iscrowd
            
        return img, target

    def __len__(self):
        return len(self.img_dirs)