# Original Data Files are downloaded from: https://zenodo.org/records/5914554#.YfRhcPXMKHE
# Ben Weinstein, Sergio Marconi, & Ethan White. (2022). Data for the NeonTreeEvaluation Benchmark (0.2.2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.5914554

import os
import xml.etree.ElementTree as ET
import shutil
from typing import Dict, List, Tuple, Optional
import random


def convert_single_voc_to_yolo(
    xml_file_path: str, output_label_path: str, class_mapping: Dict[str, int]
) -> bool:
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()

        size_node = root.find("size")
        if size_node is None:
            print(
                f"Warning: <size> tag not found in {xml_file_path}. Skipping this file."
            )
            return False

        img_width_node = size_node.find("width")
        img_height_node = size_node.find("height")

        if (
            img_width_node is None
            or img_height_node is None
            or img_width_node.text is None
            or img_height_node.text is None
        ):
            print(
                f"Warning: <width> or <height> not found or empty in <size> tag of {xml_file_path}. Skipping this file."
            )
            return False

        img_width = int(img_width_node.text)
        img_height = int(img_height_node.text)

        if img_width == 0 or img_height == 0:
            print(
                f"Warning: Image width or height is 0 in {xml_file_path}. Skipping this file."
            )
            return False

        yolo_annotations: List[str] = []
        for obj_node in root.findall("object"):
            class_name_node = obj_node.find("name")
            if class_name_node is None or class_name_node.text is None:
                print(
                    f"Warning: <name> tag not found in an object in {xml_file_path}. Skipping this object."
                )
                continue
            class_name = class_name_node.text

            if class_name not in class_mapping:
                print(
                    f"Warning: Unknown class '{class_name}' found in {xml_file_path}. Skipping this object."
                )
                continue

            class_id = class_mapping[class_name]

            bndbox_node = obj_node.find("bndbox")
            if bndbox_node is None:
                print(
                    f"Warning: <bndbox> not found for object '{class_name}' in {xml_file_path}. Skipping this object."
                )
                continue

            try:
                xmin = float(bndbox_node.findtext("xmin", "0"))
                ymin = float(bndbox_node.findtext("ymin", "0"))
                xmax = float(bndbox_node.findtext("xmax", "0"))
                ymax = float(bndbox_node.findtext("ymax", "0"))
            except ValueError:
                print(
                    f"Warning: Invalid coordinate values in <bndbox> in {xml_file_path}. Skipping this object."
                )
                continue

            # Convert to YOLO format
            x_center = (xmin + xmax) / 2.0
            y_center = (ymin + ymax) / 2.0
            width = xmax - xmin
            height = ymax - ymin

            # Normalize
            norm_x_center = x_center / img_width
            norm_y_center = y_center / img_height
            norm_width = width / img_width
            norm_height = height / img_height

            yolo_annotations.append(
                f"{class_id} {norm_x_center:.6f} {norm_y_center:.6f} {norm_width:.6f} {norm_height:.6f}"
            )

        if yolo_annotations:
            with open(output_label_path, "w") as f:
                for ann in yolo_annotations:
                    f.write(ann + "\n")
            return True  # Label file created
        return False  # No objects found, label file not created

    except ET.ParseError:
        print(f"Error: Could not parse XML file {xml_file_path}. Skipping.")
        return False
    except Exception as e:
        print(
            f"An unexpected error occurred while processing {xml_file_path}: {e}. Skipping."
        )
        return False


def get_image_filename_from_xml(xml_file_path: str) -> Optional[str]:
    try:
        tree = ET.parse(xml_file_path)
        root = tree.getroot()
        filename_node = root.find("filename")
        if filename_node is not None and filename_node.text:
            return filename_node.text
        else:
            print(f"Warning: <filename> tag not found or empty in {xml_file_path}.")
            return None
    except ET.ParseError:
        print(
            f"Error: Could not parse XML file {xml_file_path} to get image filename. Skipping."
        )
        return None
    except Exception as e:
        print(f"Error getting image filename from {xml_file_path}: {e}. Skipping.")
        return None


def create_yolo_dataset_yaml(yolo_dataset_dir: str, class_names: List[str]) -> None:
    for i, name in enumerate(class_names):
        yaml_content += f"  {i}: {name}\n"

    yaml_file_path = os.path.join(yolo_dataset_dir, "dataset.yaml")
    with open(yaml_file_path, "w") as f:
        f.write(yaml_content)
    print(f"dataset.yaml file created at: {yaml_file_path}")


def process_dataset(
    base_dir: str, class_mapping: Dict[str, int], class_names_list: List[str]
) -> None:
    images_input_dir = os.path.join(base_dir, "images")
    annotations_input_dir = os.path.join(base_dir, "annotations")

    yolo_dataset_dir = os.path.join(base_dir, "yolo_dataset")
    yolo_images_dir = os.path.join(yolo_dataset_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dataset_dir, "labels")

    # Create output directories
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)

    if not os.path.exists(annotations_input_dir):
        print(f"Error: Annotations directory '{annotations_input_dir}' does not exist.")
        return
    if not os.path.exists(images_input_dir):
        print(f"Error: Images directory '{images_input_dir}' does not exist.")
        return

    processed_xml_files = 0
    label_files_created = 0

    for xml_filename in os.listdir(annotations_input_dir):
        if not xml_filename.lower().endswith(".xml"):
            continue

        xml_file_path = os.path.join(annotations_input_dir, xml_filename)
        original_image_filename = get_image_filename_from_xml(xml_file_path)

        if not original_image_filename:
            continue  # Skip if filename couldn't be extracted

        source_image_path = os.path.join(images_input_dir, original_image_filename)
        dest_image_path = os.path.join(yolo_images_dir, original_image_filename)

        base_name_for_label = os.path.splitext(original_image_filename)[0]
        output_label_path = os.path.join(yolo_labels_dir, base_name_for_label + ".txt")

        if os.path.exists(source_image_path):
            shutil.copy(source_image_path, dest_image_path)
            if convert_single_voc_to_yolo(
                xml_file_path, output_label_path, class_mapping
            ):
                label_files_created += 1
            processed_xml_files += 1
        else:
            print(
                f"Warning: Image file '{source_image_path}' (referenced in {xml_filename}) not found. Skipping this annotation."
            )

    print(f"\nProcessing complete.")
    print(f"Total XML files processed: {processed_xml_files}")
    print(f"YOLO label files created: {label_files_created}")
    print(f"YOLO format images saved in: {yolo_images_dir}")
    print(f"YOLO format labels saved in: {yolo_labels_dir}")

    create_yolo_dataset_yaml(yolo_dataset_dir, class_names_list)


def spilit_train_val():
    dataset_root = "./yolo_dataset"
    images_dir = os.path.join(dataset_root, "images")
    labels_dir = os.path.join(dataset_root, "labels")

    image_files = [
        f
        for f in os.listdir(images_dir)
        if os.path.isfile(os.path.join(images_dir, f))
        and f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
    ]

    valid_image_files = []
    for img_file in image_files:
        label_file_name = os.path.splitext(img_file)[0] + ".txt"
        label_file_path = os.path.join(labels_dir, label_file_name)
        if os.path.exists(label_file_path):
            valid_image_files.append(img_file)
        else:
            print(
                f"Warning: Could not find label file '{label_file_name}' for image '{img_file}'. Skipping this image."
            )

    image_files = valid_image_files

    random.shuffle(image_files)
    train_ratio = 0.925
    num_train = int(len(image_files) * train_ratio)
    train_files = image_files[:num_train]
    val_files = image_files[num_train:]

    def write_paths_to_file(file_path, file_list):
        with open(file_path, "w") as f:
            for file_name in file_list:
                f.write(f"./images/{file_name}\n")

    train_txt_path = os.path.join(dataset_root, "train.txt")
    val_txt_path = os.path.join(dataset_root, "val.txt")

    write_paths_to_file(train_txt_path, train_files)
    write_paths_to_file(val_txt_path, val_files)

    print(f"train.txt: {len(train_files)}")
    print(f"val.txt: {len(val_files)}")
    print(
        "Please update the dataset.yaml file with the correct paths for train.txt and val.txt."
    )


def main():
    base_dir = os.getcwd()

    # Define class mapping
    class_mapping: Dict[str, int] = {"Tree": 0}
    # Ensure the order matches the indices in class_mapping
    class_names_list: List[str] = ["Tree"]

    process_dataset(base_dir, class_mapping, class_names_list)
    spilit_train_val()


if __name__ == "__main__":
    main()
