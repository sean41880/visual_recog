import os
import torch
from torchvision import datasets
from torchvision.transforms import v2
from torch.utils.data import DataLoader

def get_transforms():
    """
    定義資料增強與預處理管線
    注意：我們使用了最新的 torchvision.transforms.v2，效能更好且原生支援 Mixup/CutMix
    """
    # ImageNet 的標準正規化參數 (使用預訓練權重必備)
    normalize = v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    # 訓練集：基礎增強 (進階的 Mixup/CutMix 會獨立在 Batch 階段處理)
    train_transform = v2.Compose([
        v2.RandomResizedCrop(size=224, antialias=True),
        v2.RandomHorizontalFlip(p=0.5),
        # 即使我們後面要用 Mixup，這裡加一點色彩隨機變化(ColorJitter)效果會更好
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        
        # v2 版本的標準轉換寫法 (取代舊版的 ToTensor)
        v2.ToImage(), 
        v2.ToDtype(torch.float32, scale=True),
        normalize,
    ])

    # 驗證集/測試集：標準裁切，絕對不能有任何隨機變化
    val_transform = v2.Compose([
        v2.Resize(256, antialias=True),
        v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        normalize
    ])

    return train_transform, val_transform


def get_dataloaders(data_dir, batch_size=64, num_workers=4):
    """
    建立並回傳 DataLoader
    :param data_dir: dataset 的根目錄路徑
    :param batch_size: 批次大小 (建議 32 或 64)
    :param num_workers: 讀取資料的 CPU 執行緒數 (TWCC 上可以設 4 或 8)
    """
    train_transform, val_transform = get_transforms()

    # 定義路徑 (假設你的資料夾裡面有 train 和 val 兩個子資料夾)
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')

    # 使用 ImageFolder 自動讀取 (它會把子資料夾名稱當作類別標籤)
    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    # 建立 DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,       # 訓練集必須打亂
        num_workers=num_workers,
        pin_memory=True,    # 加速傳輸到 GPU
        drop_last=True      # Mixup/CutMix 建議捨棄最後一個不滿的 Batch，避免維度報錯
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,      # 驗證集不需要打亂
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, train_dataset.classes


def get_mixup_cutmix(num_classes=100):
    """
    回傳 Mixup 和 CutMix 增強器
    這個函數會在你的「訓練迴圈 (Train Loop)」中被呼叫，因為它們是作用在 Batch 上的！
    """
    # alpha 值控制混合的強度，這是文獻中最常用的預設值
    mixup = v2.MixUp(num_classes=num_classes, alpha=0.4)
    cutmix = v2.CutMix(num_classes=num_classes, alpha=1.0)
    
    # 每次隨機選擇使用 Mixup 還是 CutMix (50% / 50%)
    cutmix_or_mixup = v2.RandomChoice([mixup, cutmix])
    
    return cutmix_or_mixup