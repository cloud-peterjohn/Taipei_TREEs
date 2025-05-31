import os
import random
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import numpy as np


def parse_image_name(image_name):
    base_name = os.path.splitext(image_name)[0]
    base_name = base_name.replace("-640x640m", "")
    coords = base_name.split(",")
    center_lat = float(coords[0])
    center_lon = float(coords[1])
    return center_lat, center_lon


def calculate_image_bounds(center_lat, center_lon):
    lat_offset = 0.00288288
    lon_offset = 0.003198

    bounds = {
        "min_lat": center_lat - lat_offset,
        "max_lat": center_lat + lat_offset,
        "min_lon": center_lon - lon_offset,
        "max_lon": center_lon + lon_offset,
    }
    return bounds


def geo_to_pixel(lat, lon, center_lat, center_lon, image_size=1920):
    lat_offset = 0.00288288
    lon_offset = 0.003198

    lat_ratio = (lat - center_lat) / (2 * lat_offset)
    lon_ratio = (lon - center_lon) / (2 * lon_offset)

    pixel_x = int(image_size / 2 + lon_ratio * image_size)
    pixel_y = int(image_size / 2 - lat_ratio * image_size)

    return pixel_x, pixel_y


def is_tree_in_image(tree_lat, tree_lon, bounds):
    return (
        bounds["min_lat"] <= tree_lat <= bounds["max_lat"]
        and bounds["min_lon"] <= tree_lon <= bounds["max_lon"]
    )


def has_green_area(img, pixel_x, pixel_y, bbox_size=15):
    h, w = img.shape[:2]

    x1 = max(0, pixel_x - bbox_size // 2)
    y1 = max(0, pixel_y - bbox_size // 2)
    x2 = min(w, pixel_x + bbox_size // 2)
    y2 = min(h, pixel_y + bbox_size // 2)

    bbox_region = img[y1:y2, x1:x2]

    if bbox_region.size == 0:
        return False

    hsv = cv2.cvtColor(bbox_region, cv2.COLOR_BGR2HSV)

    lower_green_ultra_wide = np.array([15, 3, 3])
    upper_green_ultra_wide = np.array([120, 255, 255])

    mask_green = cv2.inRange(hsv, lower_green_ultra_wide, upper_green_ultra_wide)

    green_pixel_count = cv2.countNonZero(mask_green)
    total_pixels = bbox_region.shape[0] * bbox_region.shape[1]
    green_ratio = green_pixel_count / total_pixels

    return green_ratio > 0.35


def create_yolo_label(pixel_x, pixel_y, bbox_size=22, image_size=1920):
    center_x_norm = pixel_x / image_size
    center_y_norm = pixel_y / image_size
    width_norm = bbox_size / image_size
    height_norm = bbox_size / image_size

    return (
        f"0 {center_x_norm:.6f} {center_y_norm:.6f} {width_norm:.6f} {height_norm:.6f}"
    )


def process_image(image_path, tree_data, output_labels_dir):
    image_name = os.path.basename(image_path)
    print(f"Processing: {image_name}")
    center_lat, center_lon = parse_image_name(image_name)
    print(f"Center coordinates: ({center_lat}, {center_lon})")
    bounds = calculate_image_bounds(center_lat, center_lon)
    print(
        f"Image bounds: lat({bounds['min_lat']:.6f}, {bounds['max_lat']:.6f}), "
        f"lon({bounds['min_lon']:.6f}, {bounds['max_lon']:.6f})"
    )
    trees_in_image = []
    for _, tree in tree_data.iterrows():
        tree_lat = tree["Latitude"]
        tree_lon = tree["Longitude"]
        if is_tree_in_image(tree_lat, tree_lon, bounds):
            trees_in_image.append((tree_lat, tree_lon))
    print(f"Found {len(trees_in_image)} trees in this image")
    if len(trees_in_image) == 0:
        print("No trees found in this image")
        return
    img = cv2.imread(image_path)
    if img is None:
        print(f"Cannot load image: {image_path}")
        return
    initial_bboxes = []
    filtered_bboxes = []
    for tree_lat, tree_lon in trees_in_image:
        pixel_x, pixel_y = geo_to_pixel(tree_lat, tree_lon, center_lat, center_lon)
        if 0 <= pixel_x < 1920 and 0 <= pixel_y < 1920:
            initial_bboxes.append((pixel_x, pixel_y))
            if has_green_area(img, pixel_x, pixel_y):
                filtered_bboxes.append((pixel_x, pixel_y))
    print(
        f"Initial bboxes: {len(initial_bboxes)}, After green filtering: {len(filtered_bboxes)}"
    )

    yolo_labels = []
    for pixel_x, pixel_y in filtered_bboxes:
        yolo_label = create_yolo_label(pixel_x, pixel_y)
        yolo_labels.append(yolo_label)

    label_name = os.path.splitext(image_name)[0] + ".txt"
    label_path = os.path.join(output_labels_dir, label_name)

    os.makedirs(output_labels_dir, exist_ok=True)
    with open(label_path, "w") as f:
        for label in yolo_labels:
            f.write(label + "\n")

    print(f"Saved {len(yolo_labels)} labels to {label_path}")


def calculate_bounding_boxes():
    images_dir = "source_datasets_finetuning/images/"
    tree_csv_path = "source_datasets_finetuning/TaipeiTree.csv"
    labels_dir = "source_datasets_finetuning/labels/"

    print("Loading tree data...")
    tree_data = pd.read_csv(tree_csv_path)
    print(f"Loaded {len(tree_data)} trees from CSV")

    image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg")]
    print(f"Found {len(image_files)} images")

    if not image_files:
        print("No images found!")
        return

    print(f"\nProcessing all {len(image_files)} images...")
    for i, image_file in enumerate(image_files):
        image_path = os.path.join(images_dir, image_file)
        process_image(image_path, tree_data, labels_dir)
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(image_files)} images")

    print("All images processed!")


def split_patches():
    input_images_dir = "./source_datasets_finetuning/images/"
    input_labels_dir = "./source_datasets_finetuning/labels/"
    output_images_dir = "./yolo_dataset_finetune/images/"
    output_labels_dir = "./yolo_dataset_finetune/labels/"
    os.makedirs(output_images_dir, exist_ok=True)
    os.makedirs(output_labels_dir, exist_ok=True)
    original_size = 1920
    patch_size = 480
    final_size = 640
    patches_per_row = original_size // patch_size
    scale_factor = final_size / patch_size
    print(
        f"Patching: {original_size}x{original_size} -> {patches_per_row}x{patches_per_row} patches of {patch_size}x{patch_size} -> resize to {final_size}x{final_size}"
    )
    image_files = [f for f in os.listdir(input_images_dir) if f.endswith(".jpg")]
    print(f"Find {len(image_files)} images in {input_images_dir}")
    total_patches = 0
    valid_patches = 0
    discarded_green = 0
    for img_idx, image_file in enumerate(image_files):
        print(f"Processing {img_idx + 1}/{len(image_files)}: {image_file}")
        img_path = os.path.join(input_images_dir, image_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Cannot read image: {img_path}")
            continue
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(input_labels_dir, label_file)
        bboxes = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split()
                        if len(parts) == 5:
                            class_id = int(parts[0])
                            center_x = float(parts[1]) * original_size
                            center_y = float(parts[2]) * original_size
                            width = float(parts[3]) * original_size
                            height = float(parts[4]) * original_size
                            bboxes.append([class_id, center_x, center_y, width, height])
        for row in range(patches_per_row):
            for col in range(patches_per_row):
                total_patches += 1
                x_start = col * patch_size
                y_start = row * patch_size
                x_end = x_start + patch_size
                y_end = y_start + patch_size
                patch = img[y_start:y_end, x_start:x_end]
                patch_bboxes = []
                for bbox in bboxes:
                    class_id, center_x, center_y, width, height = bbox
                    bbox_x1 = center_x - width / 2
                    bbox_y1 = center_y - height / 2
                    bbox_x2 = center_x + width / 2
                    bbox_y2 = center_y + height / 2
                    intersect_x1 = max(bbox_x1, x_start)
                    intersect_y1 = max(bbox_y1, y_start)
                    intersect_x2 = min(bbox_x2, x_end)
                    intersect_y2 = min(bbox_y2, y_end)
                    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
                        intersect_area = (intersect_x2 - intersect_x1) * (
                            intersect_y2 - intersect_y1
                        )
                        bbox_area = width * height
                        if intersect_area >= bbox_area / 3:
                            new_x1 = max(0, intersect_x1 - x_start)
                            new_y1 = max(0, intersect_y1 - y_start)
                            new_x2 = min(patch_size, intersect_x2 - x_start)
                            new_y2 = min(patch_size, intersect_y2 - y_start)
                            new_center_x = (new_x1 + new_x2) / 2
                            new_center_y = (new_y1 + new_y2) / 2
                            new_width = new_x2 - new_x1
                            new_height = new_y2 - new_y1
                            new_center_x *= scale_factor
                            new_center_y *= scale_factor
                            new_width *= scale_factor
                            new_height *= scale_factor
                            yolo_center_x = new_center_x / final_size
                            yolo_center_y = new_center_y / final_size
                            yolo_width = new_width / final_size
                            yolo_height = new_height / final_size
                            yolo_center_x = max(0, min(1, yolo_center_x))
                            yolo_center_y = max(0, min(1, yolo_center_y))
                            yolo_width = max(0, min(1, yolo_width))
                            yolo_height = max(0, min(1, yolo_height))
                            patch_bboxes.append(
                                [
                                    class_id,
                                    yolo_center_x,
                                    yolo_center_y,
                                    yolo_width,
                                    yolo_height,
                                ]
                            )
                if len(patch_bboxes) > 0:
                    bbox_mask = np.zeros((patch_size, patch_size), dtype=np.uint8)
                    for bbox in patch_bboxes:
                        _, cx, cy, w, h = bbox
                        patch_cx = cx * patch_size / scale_factor
                        patch_cy = cy * patch_size / scale_factor
                        patch_w = w * patch_size / scale_factor
                        patch_h = h * patch_size / scale_factor
                        x1 = int(max(0, patch_cx - patch_w / 2))
                        y1 = int(max(0, patch_cy - patch_h / 2))
                        x2 = int(min(patch_size, patch_cx + patch_w / 2))
                        y2 = int(min(patch_size, patch_cy + patch_h / 2))
                        bbox_mask[y1:y2, x1:x2] = 255
                    non_bbox_mask = cv2.bitwise_not(bbox_mask)
                    hsv_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
                    lower_green_ultra_wide = np.array([15, 3, 3])
                    upper_green_ultra_wide = np.array([120, 255, 255])
                    green_mask = cv2.inRange(
                        hsv_patch, lower_green_ultra_wide, upper_green_ultra_wide
                    )
                    non_bbox_green_mask = cv2.bitwise_and(green_mask, non_bbox_mask)
                    non_bbox_pixels = cv2.countNonZero(non_bbox_mask)
                    non_bbox_green_pixels = cv2.countNonZero(non_bbox_green_mask)
                    if non_bbox_pixels > 0:
                        green_ratio = non_bbox_green_pixels / non_bbox_pixels
                        if green_ratio > 0.5:
                            discarded_green += 1
                            print(
                                f"  Discarding patch {row}_{col}: too much green outside bboxes ({green_ratio:.2%})"
                            )
                            continue
                    valid_patches += 1
                    patch_resized = cv2.resize(
                        patch, (final_size, final_size), interpolation=cv2.INTER_CUBIC
                    )
                    base_name = os.path.splitext(image_file)[0]
                    patch_name = f"{base_name}_patch_{row}_{col}.jpg"
                    patch_img_path = os.path.join(output_images_dir, patch_name)
                    cv2.imwrite(patch_img_path, patch_resized)
                    patch_label_path = os.path.join(
                        output_labels_dir, os.path.splitext(patch_name)[0] + ".txt"
                    )
                    with open(patch_label_path, "w") as f:
                        for bbox in patch_bboxes:
                            f.write(
                                f"{bbox[0]} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f} {bbox[4]:.6f}\n"
                            )
                    print(f"  Saving patch {row}_{col}: {len(patch_bboxes)} bboxes")
    print(f"Total patches created: {total_patches}")
    print(f"Valid patches with bboxes: {valid_patches}")
    print(f"Discarded due to excessive green: {discarded_green}")
    print(f"Valid patch ratio: {valid_patches / total_patches:.2%}")

def split_train_val():
    dataset_dir = "./yolo_dataset_finetune/"
    images_dir = os.path.join(dataset_dir, "images")
    image_files = [f for f in os.listdir(images_dir) if f.endswith('.jpg')]
    print(f"Found {len(image_files)} images for train/val split")
    if len(image_files) == 0:
        print("No images found for splitting!")
        return
    random.seed(42)
    random.shuffle(image_files)
    split_point = int(len(image_files) * 0.9)
    train_files = image_files[:split_point]
    val_files = image_files[split_point:]
    print(f"Train set: {len(train_files)} images")
    print(f"Val set: {len(val_files)} images")
    train_txt_path = os.path.join(dataset_dir, "train.txt")
    with open(train_txt_path, 'w') as f:
        for img_file in train_files:
            f.write(f"./images/{img_file}\n")
    val_txt_path = os.path.join(dataset_dir, "val.txt")
    with open(val_txt_path, 'w') as f:
        for img_file in val_files:
            f.write(f"./images/{img_file}\n")
    print(f"Created train.txt with {len(train_files)} entries")
    print(f"Created val.txt with {len(val_files)} entries")
    print(f"Train/Val split ratio: {len(train_files)/len(image_files):.1%}/{len(val_files)/len(image_files):.1%}")

if __name__ == "__main__":
    calculate_bounding_boxes()
    # split_patches()
    # split_train_val()