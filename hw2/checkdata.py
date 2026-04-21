import torch
from dataset import get_dataloaders

def test_dataloader():
    DATA_DIR = '/share/sean/vr/hw2/nycu-hw2-data' 
    
    train_loader, _, _ = get_dataloaders(DATA_DIR, batch_size=2)
    
    # 拿出一把 batch (注意現在 images 是一個包含了 pixel_values 和 pixel_mask 的字典)
    encoded_images, targets = next(iter(train_loader))
    
    # 印出 padded 後的圖片 Tensor 大小
    print(f"Padded Pixel Values Shape: {encoded_images['pixel_values'].shape}")
    
    # 印出 Mask 大小 (這應該跟圖片大小一樣，用 1 和 0 來標示有效區域)
    print(f"Pixel Mask Shape: {encoded_images['pixel_mask'].shape}")
    
    print(f"First target bounding boxes format: {targets[0]['boxes']}") 
    print(f"First target classes: {targets[0]['class_labels']}") 

if __name__ == '__main__':
    test_dataloader()