# Original datasets are downloaded from https://huggingface.co/datasets/IGNF/PureForest
# Gaydon, C., & Roche, F. (2025, February). Pureforest: A large-scale aerial lidar and aerial imagery dataset for tree species classification in monospecific forests. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV) (pp. 5895-5904). IEEE.

import os
import shutil
import pandas as pd

CURRENT_WORKING_DIRECTORY = os.getcwd()
PUREFOREST_DIR_NAME = "PureForest"
DATASETS_DIR_NAME = "datasets"
PATCHES_CSV_NAME = "PureForest-patches.csv"

PUREFOREST_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, PUREFOREST_DIR_NAME)
DATASETS_PATH = os.path.join(CURRENT_WORKING_DIRECTORY, DATASETS_DIR_NAME)
PATCHES_CSV_PATH = os.path.join(PUREFOREST_PATH, PATCHES_CSV_NAME)


def organize_image_folders(base_dir):
    print(f"Organizing image folders in: {base_dir}")
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        imagery_train_dir = os.path.join(folder_path, "imagery", "train")
        if os.path.isdir(imagery_train_dir):
            for file_name in os.listdir(imagery_train_dir):
                src_file = os.path.join(imagery_train_dir, file_name)
                dst_file = os.path.join(folder_path, file_name)

                # Handle potential name conflicts
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file_name)
                    i = 1
                    while os.path.exists(os.path.join(folder_path, f"{base}_{i}{ext}")):
                        i += 1
                    dst_file = os.path.join(folder_path, f"{base}_{i}{ext}")
                shutil.move(src_file, dst_file)

            # Remove the 'imagery' directory structure
            imagery_dir_to_remove = os.path.join(folder_path, "imagery")
            if os.path.exists(imagery_dir_to_remove):
                shutil.rmtree(imagery_dir_to_remove)
    print("Image folder organization complete.")


def rename_tiff_files_in_pureforest(base_dir):
    print(f"Renaming TIFF files in: {base_dir}")
    prefixes_to_remove = ["TEST-", "TRAIN-", "VAL-"]
    for folder_name in os.listdir(base_dir):
        folder_path = os.path.join(base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".tiff"):
                original_file_name = file_name
                for prefix in prefixes_to_remove:
                    if file_name.startswith(prefix):
                        new_name = file_name[len(prefix) :]
                        src_path = os.path.join(folder_path, original_file_name)
                        dst_path = os.path.join(folder_path, new_name)
                        os.rename(src_path, dst_path)
                        print(
                            f"Renamed: {original_file_name} -> {new_name} in {folder_name}"
                        )
                        break
    print("TIFF file renaming complete.")


def copy_tiff_files_to_datasets(source_base_dir, destination_base_dir):
    print(f"Copying TIFF files from {source_base_dir} to {destination_base_dir}")
    if not os.path.exists(destination_base_dir):
        os.makedirs(destination_base_dir)

    for folder_name in os.listdir(source_base_dir):
        folder_path = os.path.join(source_base_dir, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in os.listdir(folder_path):
            if file_name.lower().endswith(".tiff"):
                src_file = os.path.join(folder_path, file_name)
                dst_file = os.path.join(destination_base_dir, file_name)

                # Handle potential name conflicts in destination
                if os.path.exists(dst_file):
                    base, ext = os.path.splitext(file_name)
                    i = 1
                    while os.path.exists(
                        os.path.join(destination_base_dir, f"{base}_{i}{ext}")
                    ):
                        i += 1
                    dst_file = os.path.join(destination_base_dir, f"{base}_{i}{ext}")

                shutil.copy2(src_file, dst_file)
                # print(f"Copied: {src_file} -> {dst_file}")
    print("TIFF file copying complete.")


def generate_labels_csv(datasets_dir, patches_csv_path):
    print(f"Generating labels.csv in: {datasets_dir}")
    if not os.path.exists(patches_csv_path):
        print(f"Error: Patches CSV file not found at {patches_csv_path}")
        return

    try:
        patches_df = pd.read_csv(patches_csv_path, dtype=str)
        # Ensure 'patch_id' column exists before setting it as index
        if "patch_id" not in patches_df.columns:
            print(f"Error: 'patch_id' column not found in {patches_csv_path}")
            return
        patches_df.set_index("patch_id", inplace=True)
    except Exception as e:
        print(f"Error reading or processing patches CSV {patches_csv_path}: {e}")
        return

    labels_data = []
    files_to_remove = []

    for file_name in os.listdir(datasets_dir):
        if file_name.lower().endswith(".tiff"):
            patch_id = os.path.splitext(file_name)[0]
            if patch_id in patches_df.index:
                row = patches_df.loc[patch_id]
                class_name = row.get("class_name_en", row.get("class_name"))
                class_index = (
                    int(row.get("class_index")) + 33
                )  # IDTRees has 0-32 classes, so let's begin from 33.

                if class_name is not None and class_index is not None:
                    labels_data.append([file_name, class_index, class_name])
                else:
                    print(
                        f"Warning: Missing 'class_name' or 'class_index' for patch_id {patch_id} in {patches_csv_path}. Removing file: {file_name}"
                    )
                    files_to_remove.append(os.path.join(datasets_dir, file_name))
            else:
                print(
                    f"Patch ID {patch_id} not found in patches CSV. Removing file: {file_name}"
                )
                files_to_remove.append(os.path.join(datasets_dir, file_name))

    for file_to_remove in files_to_remove:
        try:
            os.remove(file_to_remove)
            print(f"Removed: {file_to_remove}")
        except OSError as e:
            print(f"Error removing file {file_to_remove}: {e}")

    if labels_data:
        labels_df = pd.DataFrame(
            labels_data, columns=["filename", "class_id", "scientificName"]
        )
        labels_output_path = os.path.join(datasets_dir, "labels.csv")
        labels_df.to_csv(labels_output_path, index=False, encoding="utf-8")
        print(f"labels.csv generated at: {labels_output_path}")
    else:
        print("No valid labeled data found to generate labels.csv.")


if __name__ == "__main__":
    # Step 1: Organize image folders within PureForest
    organize_image_folders(PUREFOREST_PATH)

    # Step 2: Rename TIFF files within PureForest
    rename_tiff_files_in_pureforest(PUREFOREST_PATH)

    # Step 3: Copy TIFF files from PureForest to datasets directory
    copy_tiff_files_to_datasets(PUREFOREST_PATH, DATASETS_PATH)

    # Step 4: Generate labels.csv for files in datasets directory
    generate_labels_csv(DATASETS_PATH, PATCHES_CSV_PATH)

    print("All operations complete.")
