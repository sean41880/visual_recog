import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights, resnet101, ResNet101_Weights

class CustomResNet(nn.Module):
    def __init__(self, backbone_name='resnet50', num_classes=100, dropout_rate=0.5):
        super(CustomResNet, self).__init__()
        
        # 1. 根據傳入的名稱，動態載入對應的 ResNet
        if backbone_name == 'resnet50':
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        elif backbone_name == 'resnet101':
            self.backbone = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"不支援的模型名稱: {backbone_name}")
            
        # 2. 取得原本全連接層 (fc) 的輸入維度 (ResNet50 和 101 都是 2048)
        in_features = self.backbone.fc.in_features
        
        # 3. 拔掉原本的全連接層
        self.backbone.fc = nn.Identity()
        
        # 4. 自定義分類頭
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        features = self.backbone(x)
        out = self.classifier(features)
        return out