import torch
from PIL import Image, ImageDraw, ImageFont
from transformers import DetrImageProcessor, DetrForObjectDetection
import os

def draw_predictions(img_path, output_path, model, processor, device):
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    target_sizes = torch.tensor([image.size[::-1]])
    results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.5)[0]
    
    draw = ImageDraw.Draw(image)
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        # 畫框
        draw.rectangle(box, outline="red", width=3)
        # 寫字 (label + score)
        draw.text((box[0], box[1]), f"{int(label.item())}: {round(score.item(), 2)}", fill="red")
    
    image.save(output_path)
    print(f"✅ 視覺化結果已存至: {output_path}")

# 使用範例
device = torch.device("cuda")
model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", num_labels=10, ignore_mismatched_sizes=True)
model.load_state_dict(torch.load("best_detr_phase2.pth"))
model.to(device).eval()
processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50")

# 隨機挑一張驗證集或測試集的圖來看看
draw_predictions("/share/sean/vr/hw2/nycu-hw2-data/test/24.png", "test_result.png", model, processor, device)