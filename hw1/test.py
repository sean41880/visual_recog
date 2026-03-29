import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets
from torchvision.transforms import v2
from model import CustomResNet  # 確保導入的是我們修改後的通用模型類別

# 1. 定義測試集 Dataset (處理沒有子資料夾的情況)
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # 取得所有圖片檔名並排序，確保輸出順序穩定
        self.image_names = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.png', '.jpeg'))])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = self.image_names[idx]
        img_path = os.path.join(self.root_dir, img_name)
        # 讀取圖片並轉為 RGB
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
            
        # image_name 不應包含副檔名 (.jpg)
        pure_name = os.path.splitext(img_name)[0]
        return image, pure_name

def inference():
    # ==========================================
    # 設定區塊
    # ==========================================
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    DATA_DIR = '/home/sean910526/visual_recog/data'     # 包含 train/ 的目錄
    TEST_DIR = '/home/sean910526/visual_recog/data/test' # 測試集圖片目錄
    MODEL_PATH = 'best_resnet50_lr0.0001_bs64.pth'      # 你的最強模型權重
    BATCH_SIZE = 64
    
    # --- 關鍵修正：取得訓練時的類別映射表 ---
    # ImageFolder 會按字串排序資料夾: ['0', '1', '10', '11'...]
    train_dataset = datasets.ImageFolder(os.path.join(DATA_DIR, 'train'))
    class_names = train_dataset.classes  # 這是 index -> folder_name 的對照表
    print(f"✅ 已載入類別映射表，範例：Index 2 對應資料夾名稱 '{class_names[2]}'")

    # 2. 測試集預處理 (必須與訓練時的驗證集一致)
    test_transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 3. 準備 DataLoader
    test_dataset = TestDataset(TEST_DIR, transform=test_transform)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. 載入模型
    print(f"正在載入模型與權重: {MODEL_PATH}")
    model = CustomResNet(backbone_name='resnet50', num_classes=100)
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.to(DEVICE)
    model.eval()

    # 5. 開始推論
    results = []
    print("🚀 開始執行測試集推論...")
    
    with torch.no_grad():
        for images, names in test_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            for name, pred in zip(names, preds):
                # 【重要修正】
                # pred.item() 是模型輸出的索引 (0~99)
                # class_names[idx] 拿到的是真正的資料夾名稱 (如 "2")
                # 最後轉成 int 填入 CSV
                folder_name = class_names[pred.item()]
                results.append({
                    'image_name': name, 
                    'pred_label': int(folder_name)
                })

    # 6. 輸出 CSV
    df = pd.DataFrame(results)
    df.to_csv('prediction.csv', index=False)
    print(f"\n🎉 推論完成！結果已儲存至 prediction.csv")
    print(f"檢查前 5 行：\n{df.head()}")

if __name__ == '__main__':
    inference()