import os
import torch
import torch.nn.functional as F
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from model import CustomResNet

class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_names = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 原圖與翻轉圖同時處理 (TTA)
        if self.transform:
            image_orig = self.transform(image)
            image_flip = self.transform(v2.functional.hflip(image))
            
        pure_name = os.path.splitext(img_name)[0]
        return image_orig, image_flip, pure_name

def super_inference():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = '/home/sean910526/visual_recog/data'
    TEST_DIR = '/home/sean910526/visual_recog/data/test'
    
    # 指向你的兩個最強權重
    R50_PATH = 'best_resnet50_lr0.0001_bs64.pth'
    R101_PATH = 'best_resnet101_lr0.0001_bs32.pth'

    # 1. 載入類別對照
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
    class_names = train_dataset.classes

    # 2. 測試預處理
    test_transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)

    # 3. 載入兩個模型
    print("正在準備 ResNet-50...")
    model50 = CustomResNet(backbone_name='resnet50', num_classes=100).to(DEVICE)
    model50.load_state_dict(torch.load(R50_PATH))
    model50.eval()

    print("正在準備 ResNet-101...")
    model101 = CustomResNet(backbone_name='resnet101', num_classes=100).to(DEVICE)
    model101.load_state_dict(torch.load(R101_PATH))
    model101.eval()

    results = []
    print("🚀 啟動 Ensemble + TTA 終極推論系統...")
    
    with torch.no_grad():
        for img_orig, img_flip, names in test_loader:
            img_orig, img_flip = img_orig.to(DEVICE), img_flip.to(DEVICE)
            
            # 獲取機率分佈 (Softmax)
            # R50 兩次預測
            out50_orig = F.softmax(model50(img_orig), dim=1)
            out50_flip = F.softmax(model50(img_flip), dim=1)
            
            # R101 兩次預測
            out101_orig = F.softmax(model101(img_orig), dim=1)
            out101_flip = F.softmax(model101(img_flip), dim=1)
            
            # 四個預測機率求平均 (Soft Voting)
            final_probs = (out50_orig + out50_flip + out101_orig + out101_flip) / 4
            
            _, preds = torch.max(final_probs, 1)
            
            for name, pred in zip(names, preds):
                results.append({
                    'image_name': name, 
                    'pred_label': int(class_names[pred.item()])
                })

    df = pd.DataFrame(results)
    df.to_csv('prediction.csv', index=False)
    print(f"✅ 黑科技推論完成！")

if __name__ == '__main__':
    super_inference()