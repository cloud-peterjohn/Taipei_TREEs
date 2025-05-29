from ultralytics import YOLO
import os
import random
import cv2
import matplotlib.pyplot as plt

model_ckpt_path = "runs/detect/train/weights/best.pt"
model = YOLO(model_ckpt_path)

images_dir = "/kaggle/input/tree-yolo/yolo_dataset/images"
img_files = [
    f for f in os.listdir(images_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))
]
selected_imgs = random.sample(img_files, 18)

plt.figure(figsize=(18, 9))
for idx, img_name in enumerate(selected_imgs):
    img_path = os.path.join(images_dir, img_name)
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    results = model(img_rgb)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    confs = results[0].boxes.conf.cpu().numpy()
    clss = results[0].boxes.cls.cpu().numpy()

    for box, conf, cls in zip(boxes, confs, clss):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), (255, 0, 0), 2)
        label = f"{int(cls)}:{conf:.2f}"
        cv2.putText(
            img_rgb, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
        )

    plt.subplot(3, 6, idx + 1)
    plt.imshow(img_rgb)
    plt.axis("off")
    plt.title(img_name)

plt.tight_layout()
plt.savefig("visualization.svg", format="svg")
