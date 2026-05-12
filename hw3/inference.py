import os
import json
import numpy as np
import tifffile
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from pycocotools import mask as mask_util
from tqdm import tqdm
from model import get_model_instance_segmentation

# 1. 重新定義與訓練時完全相同的模型架構
def get_model_instance_segmentation(num_classes):
    # 推論時不需要重新下載預訓練權重
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
    return model

def main():
    # ================== 1. 路徑設定 ==================
    test_dir = 'test_release/'  # 測試集資料夾
    json_mapping_path = 'test_image_name_to_ids.json'
    weights_path = 'checkpoints_B/mask_rcnn_epoch_100.pth' # 這裡請改成你訓練好的權重檔路徑
    output_file = 'test-results.json' # 規定必須叫這個名字
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    # ================== 2. 載入 Image ID 對應表 ==================
    print("載入 Image ID mapping...")
    with open(json_mapping_path, 'r') as f:
        mapping_data = json.load(f)
        name_to_id = {item['file_name']: item['id'] for item in mapping_data}

    # ================== 3. 載入你訓練好的模型 ==================
    print(f"載入模型權重從 {weights_path} ...")
    num_classes = 5 # 1 背景 + 4 種類別的細胞
    model = get_model_instance_segmentation(num_classes)
    
    # 載入剛剛訓練好的權重
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval() # 記得切換到評估模式！

    predictions = []

    # ================== 4. 進行推論並轉換 RLE ==================
    test_images = [f for f in os.listdir(test_dir) if f.endswith('.tif')]
    import torchvision.transforms.functional as F

    for img_name in tqdm(test_images, desc="Generating Predictions"):
        img_path = os.path.join(test_dir, img_name)
        image_id = name_to_id.get(img_name)
        
        if image_id is None:
            continue
            
        # 讀取 .tif 影像並轉為 Tensor
        img_array = tifffile.imread(img_path)
        # 如果影像不是 RGB 格式，可能需要轉換。假設讀進來是 HxWxC
        # 先轉 PIL Image 再轉 Tensor 是最穩的做法
        from PIL import Image
        img_pil = Image.fromarray(img_array).convert("RGB")
        img_tensor = F.to_tensor(img_pil).unsqueeze(0).to(device) # 增加 batch 維度
        
        # 進行推論
        with torch.no_grad():
            outputs = model(img_tensor)[0] # 取出 batch 裡第一筆的結果
        
        masks = outputs['masks'].cpu().numpy()      # shape: [N, 1, H, W]
        labels = outputs['labels'].cpu().numpy()    # shape: [N]
        scores = outputs['scores'].cpu().numpy()    # shape: [N]
        
        # 設定信心度門檻 (例如 0.5，低於這個分數的預測就不要)
        confidence_threshold = 0.5
        
        for i in range(len(scores)):
            if scores[i] < confidence_threshold:
                continue
                
            # Mask R-CNN 輸出的 mask 是 0~1 之間的機率值，大於 0.5 視為前景
            binary_mask = masks[i, 0] > 0.5
            
            # pycocotools 規定 Mask 必須是 Fortran order 的 uint8 array
            fortran_mask = np.asfortranarray(binary_mask, dtype=np.uint8)
            
            # 進行 RLE 編碼[cite: 3]
            rle = mask_util.encode(fortran_mask)
            rle['counts'] = rle['counts'].decode('utf-8')
            
            predictions.append({
                "image_id": image_id,
                "category_id": int(labels[i]),
                "segmentation": rle,
                "score": float(scores[i])
            })

    # ================== 5. 儲存結果 ==================
    with open(output_file, 'w') as f:
        json.dump(predictions, f)
        
    print(f"\n✅ 成功！已將 {len(predictions)} 個實體預測儲存至 {output_file}")

if __name__ == '__main__':
    main()