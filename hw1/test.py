import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import v2
from model import CustomResNet  # 確保與訓練時的模型架構一致

# 1. 定義測試集讀取類別
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 只讀取圖片檔案，並排序確保順序穩定
        self.image_names = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # 回傳圖片以及不帶副檔名的檔名 (符合作業要求的 image_name 格式)
        pure_name = os.path.splitext(img_name)[0]
        return image, pure_name

def inference():
    # --- 設定 ---
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    TEST_DIR = '/home/sean910526/visual_recog/data/test'
    # 指向你表現最好的那個權重檔案
    MODEL_PATH = 'best_resnet50_lr0.0001_bs64.pth' 
    BATCH_SIZE = 64

    # 2. 測試集 Transform (必須與訓練時的驗證集完全一致)
    test_transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 載入資料
    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. 載入模型架構與權重
    print(f"正在載入模型與權重: {MODEL_PATH}")
    model = CustomResNet(backbone_name='resnet50', num_classes=100)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    # 5. 開始推論
    results = []
    print("開始推論測試集...")
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 將結果存入清單
            for name, pred in zip(names, preds):
                results.append({'image_name': name, 'pred_label': pred.item()})

    # 6. 儲存為 CSV
    df = pd.DataFrame(results)
    df.to_csv('prediction.csv', index=False)
    print(f"✅ 推論完成！結果已儲存至 prediction.csv (共 {len(df)} 筆資料)")

if __name__ == '__main__':
    inference()