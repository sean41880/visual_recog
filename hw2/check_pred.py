import json
with open('pred.json', 'r') as f:
    data = json.load(f)
print(f"總預測框數: {len(data)}")
print(f"平均每張圖預測框數: {len(data) / 13068:.2f}")