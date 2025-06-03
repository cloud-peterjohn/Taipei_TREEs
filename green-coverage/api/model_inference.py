import os
import cv2
import torch
from ultralytics import YOLO
from typing import Tuple
import uuid

# 初始化模型
model_ckpt_path = "model.pt"
model = YOLO(model_ckpt_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

def analyze_image(file_path: str, save_dir: str) -> Tuple[str, float]:
    os.makedirs(save_dir, exist_ok=True)
    img = cv2.imread(file_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    with torch.no_grad():
        results = model.predict(img_rgb, conf=0.005, iou=0.5, device=device)
        boxes = results[0].boxes.xyxy.cpu().numpy()

    # 計算綠覆蓋率（假設 box 區域代表綠色植栽）
    h, w, _ = img.shape
    total_area = h * w
    green_area = sum((x2 - x1) * (y2 - y1) for x1, y1, x2, y2 in boxes)
    green_coverage = round((green_area / total_area) * 100, 1)

    # 繪製標記
    for box in boxes:
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (0, 255, 0), 2)

    output_filename = f"result_{uuid.uuid4().hex}.png"
    output_path = os.path.join(save_dir, output_filename)
    cv2.imwrite(output_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))

    return output_path, green_coverage