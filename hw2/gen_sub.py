import os
import json
import torch
from tqdm import tqdm
from mmdet.apis import init_detector, inference_detector

def generate_pred():
    # 路徑設定
    config_file = '/share/sean/vr/hw2/mmdetection/my_dino.py'
    checkpoint_file = '/share/sean/vr/hw2/mmdetection/work_dirs/my_dino/epoch_6.pth'
    test_dir = '/share/sean/vr/hw2/nycu-hw2-data/test/'
    
    # 初始化模型 (MMDetection 自動幫你處理 device 和 fp16)
    model = init_detector(config_file, checkpoint_file, device='cuda:0')
    
    test_images = sorted([f for f in os.listdir(test_dir) if f.endswith(('.png', '.jpg'))])
    predictions = []

    for img_name in tqdm(test_images, desc="Generating Predictions"):
        image_id = int(os.path.splitext(img_name)[0])
        img_path = os.path.join(test_dir, img_name)
        
        # MMDetection 內建推論，自動處理 Resize 和 Normalization
        result = inference_detector(model, img_path)
        
        # 取得預測結果 (MMDetection 預設已經依分數高低排序好了)
        pred_instances = result.pred_instances
        
        # 極限低門檻 (0.001)，並且每張圖最多只交前 100 個框 (符合 COCO mAP 規範)
        scores = pred_instances.scores
        keep_idx = scores > 0.5
        
        # 加上 [:100] 強制切斷，避免交出太多垃圾框反而被扣分
        filtered_boxes = pred_instances.bboxes[keep_idx][:100]
        filtered_scores = scores[keep_idx][:100]
        filtered_labels = pred_instances.labels[keep_idx][:100]     
        
        if len(filtered_boxes) > 0:
            for box, score, label in zip(filtered_boxes, filtered_scores, filtered_labels):
                # MMDetection 輸出的 box 是 [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box.tolist()
                
                # 轉換為 Kaggle 要求的 [xmin, ymin, width, height]
                w = xmax - xmin
                h = ymax - ymin
                
                predictions.append({
                    "image_id": image_id,
                    "bbox": [xmin, ymin, w, h],
                    "score": float(score.item()),
                    # 直接使用 label (0~9)，不需要 +1！ 
                    # dino的要 +1
                    "category_id": int(label.item()) + 1
                })
        else:
            # 保底機制
            predictions.append({
                "image_id": image_id,
                "bbox": [0, 0, 1, 1],
                "score": 0.0,
                "category_id": 1
            })

    # 輸出最終 JSON
    with open('pred.json', 'w') as f:
        json.dump(predictions, f)
    print(f"🎉 預測完成！產出 {len(predictions)} 個框。請提交 pred.json")

if __name__ == '__main__':
    generate_pred()