import os
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt

def rename_csv(csv_dir = "Taipei/csv/"):
    for filename in os.listdir(csv_dir):
        if filename.endswith(".csv") and not filename.endswith("-640x640m.csv"):
            name, ext = os.path.splitext(filename)
            new_filename = f"{name}-640x640m{ext}"
            old_path = os.path.join(csv_dir, filename)
            new_path = os.path.join(csv_dir, new_filename)
            os.rename(old_path, new_path)
            print(f"Rename: {filename} -> {new_filename}")


def draw_bboxes_on_image(image_path, csv_path, box_size=20):
    img = cv2.imread(image_path)
    if img is None:
        print(f"Can not load: {image_path}")
        return None, 0
    df = pd.read_csv(csv_path)
    print(f"Image shape: {img.shape}")
    print(
        f"x min/max: {df['x'].min()}/{df['x'].max()}, y min/max: {df['y'].min()}/{df['y'].max()}"
    )
    tree_count = 0
    for _, row in df.iterrows():
        x, y = int(row["x"]), int(row["y"])
        if 0 <= x < img.shape[1] and 0 <= y < img.shape[0]:
            top_left = (x - box_size // 2, y - box_size // 2)
            bottom_right = (x + box_size // 2, y + box_size // 2)
            cv2.rectangle(img, top_left, bottom_right, color=(0, 255, 0), thickness=2)
            tree_count += 1
        else:
            print(f"Skip out-of-bounds: ({x},{y})")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, tree_count


def visualize_random_image(images_dir="Taipei/images/", csv_dir="Taipei/csv/"):
    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    if not image_files:
        print("File not found in images directory.")
        return
    img_file = random.choice(image_files)
    # img_file = "25.0810876989846,121.51784315043032-640x640m.jpg"
    img_path = os.path.join(images_dir, img_file)
    csv_file = os.path.splitext(img_file)[0] + ".csv"
    csv_path = os.path.join(csv_dir, csv_file)
    if not os.path.exists(csv_path):
        print(f"Csv not found: {csv_path}")
        return
    img_with_bboxes, tree_count = draw_bboxes_on_image(img_path, csv_path)
    if img_with_bboxes is not None:
        plt.figure(figsize=(8, 8))
        plt.imshow(img_with_bboxes)
        plt.title(f"Image: {img_file}")
        plt.axis("off")
        plt.show()
        print(f"We have {tree_count} trees in this image.")

visualize_random_image()
