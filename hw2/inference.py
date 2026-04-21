import os
import json
import torch
from PIL import Image
from tqdm import tqdm
from transformers import DetrForObjectDetection, DetrImageProcessor
from torchvision.ops import batched_nms

def final_push():
    DATA_DIR = '/share/sean/vr/hw2/nycu-hw2-data'
    TEST_DIR = os.path.join(DATA_DIR, 'test')
    WEIGHT_PATH = 'best_detr_phase2.pth'
    
    device = torch.device("cuda")
    # 如果不確定是 NGC 還是 PyTorch Image，我們直接載入 Processor，DETR 對這不太挑剔
    processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")
    model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=10, ignore_mismatched_sizes=True)
    model.load_state_dict(torch.load(WEIGHT_PATH, map_location=device))
    model.to(device).eval()

    test_images = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(('.png', '.jpg'))])
    predictions = []

    with torch.no_grad():
        for img_name in tqdm(test_images, desc="Final Submission Build"):
            image_id = int(os.path.splitext(img_name)[0])
            img_path = os.path.join(TEST_DIR, img_name)
            
            # 1. 讀取原始圖片並記錄原始尺寸
            original_image = Image.open(img_path).convert("RGB")
            orig_w, orig_h = original_image.size
            
            # 🌟 策略 1：推論前手動放大 2 倍 (增加對小數字的辨識力)
            resized_image = original_image.resize((orig_w*2, orig_h*2), Image.LANCZOS)
            
            # 將放大後的圖片送入 processor
            inputs = processor(images=resized_image, return_tensors="pt").to(device)
            outputs = model(**inputs)
            
            # 🌟 關鍵修正：target_sizes 用原始尺寸，Processor 會自動幫你縮小坐標回原圖大小
            # 這樣 BBox 就不會因為放大兩倍而全部飛飛去圖片外面
            target_sizes = torch.tensor([[orig_h, orig_w]]).to(device) 
            
            # 提高信心門檻至 0.3，過濾掉那些 0.1、0.2 的垃圾框
            results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.3)[0]

            if len(results["boxes"]) > 0:
                # 🌟 策略 2：batched_nms (Class-Specific NMS)
                # 這樣對於 $1.png$ (67 靠很近)，數字 6 (ID 7) 就不會把靠很近的數字 7 (ID 8) 給排擠剔除
                keep = batched_nms(
                    results["boxes"], 
                    results["scores"], 
                    results["labels"], 
                    iou_threshold=0.45 
                )
                
                final_boxes = results["boxes"][keep]
                final_scores = results["scores"][keep]
                final_labels = results["labels"][keep]

                for score, label, box in zip(final_scores, final_labels, final_boxes):
                    xmin, ymin, xmax, ymax = box.tolist()
                    predictions.append({
                        "image_id": image_id,
                        "bbox": [xmin, ymin, xmax - xmin, ymax - ymin],
                        "score": float(score.item()),
                        # 🌟 策略 3：不加 1 (根據你 0.32 分的成功經驗)
                        "category_id": int(label.item()) 
                    })
            else:
                # 保底機制：若沒偵測到，也要塞一個空框，確保 13068 張圖一張都不漏
                predictions.append({
                    "image_id": image_id,
                    "bbox": [0, 0, 1, 1],
                    "score": 0.0,
                    "category_id": 1
                })

    with open('pred_final_nms_v2.json', 'w') as f:
        json.dump(predictions, f)
    print(f"🎉 預測完成！產出 {len(predictions)} 個框。請提交 pred.json")

if __name__ == '__main__':
    final_push()