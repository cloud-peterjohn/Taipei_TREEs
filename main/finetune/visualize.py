from ultralytics import YOLO
import os
import random
import cv2
import matplotlib.pyplot as plt
import torch

model_ckpt_path = "path/to/your/yolo_final.pt"  # Update with your model path
conf = 0.25  # Confidence threshold for detection
model = YOLO(model_ckpt_path)

images_dir = "/kaggle/input/tree-yolo-taipei/yolo_dataset_finetune/images"
img_files = [
    f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
img_name = random.choice(img_files)

output_dir = "output_images"
os.makedirs(output_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
model.to(device)
model.eval()

with torch.no_grad():
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model.predict(img_rgb, conf=conf, iou=0.5, device=device)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()

    tree_count = len(boxes)
    print(f"Detected {tree_count} trees with confidence > {conf}")

    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)

    save_path = os.path.join(output_dir, f"result_{img_name}.png")
    plt.figure(figsize=(6, 6))
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.show()
