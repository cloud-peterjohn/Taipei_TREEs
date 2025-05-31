from ultralytics import YOLO
import os
import torch

YOLO_MODEL_SIZE = "/kaggle/input/adaption-model/best.pt"  # yolo11n.pt, yolo11s.pt, yolo11m.pt, yolo11l.pt, yolo11x.pt
NUM_COUNTING_CLASSES = 5
CONFIDENCE_THRESHOLD = 0.5
NMS_IOU_THRESHOLD = 0.7
TOTAL_EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 2
NUM_WORKERS = 4 if os.name != "nt" else 0
SEED = 42
DROPOUT = 0.05
MULTI_SCALE = True
HSV_H = 0.015
HSV_S = 0.70
HSV_V = 0.40
ROTATION_DEGREE = 3.0
TRANSLATE = 0.02
RANDOM_SCALE = 0.2
SHEAR = 0.0
FLIP_UD = 0.2
FLIP_LR = 0.3
PERSPECTIVE = 0.0005
MOSAIC = 0.0
COPY_PASTE = 0.08
ERASE = 0.08
MIXUP = 0.04


def train_yolo(data_path):
    model = YOLO(YOLO_MODEL_SIZE)
    model.train(
        data=data_path,
        epochs=TOTAL_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=NUM_WORKERS,
        save=True,
        save_period=1,
        cache=False,
        seed=SEED,
        cos_lr=True,
        amp=True,
        dropout=DROPOUT,
        multi_scale=MULTI_SCALE,
        iou=NMS_IOU_THRESHOLD,
        plots=True,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=ROTATION_DEGREE,
        translate=TRANSLATE,
        scale=RANDOM_SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIP_UD,
        fliplr=FLIP_LR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASE,
    )
    model.save("yolo_final.pt")


def resume_train(data_path, last_ckpt_path):
    model = YOLO(last_ckpt_path)
    start_epoch = torch.load(last_ckpt_path, map_location="cpu").get("epoch", 0)
    print(f"Resuming from epoch {start_epoch} ...")
    model.train(
        data=data_path,
        epochs=TOTAL_EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        workers=NUM_WORKERS,
        save=True,
        save_period=1,
        cache=False,
        seed=SEED,
        cos_lr=True,
        amp=True,
        dropout=DROPOUT,
        multi_scale=MULTI_SCALE,
        iou=NMS_IOU_THRESHOLD,
        plots=True,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        degrees=ROTATION_DEGREE,
        translate=TRANSLATE,
        scale=RANDOM_SCALE,
        shear=SHEAR,
        perspective=PERSPECTIVE,
        flipud=FLIP_UD,
        fliplr=FLIP_LR,
        mosaic=MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        erasing=ERASE,
        resume=True,  # Resume training from the last checkpoint
    )


if __name__ == "__main__":
    DATASET_PATH = "/kaggle/input/tree-yolo-taipei/yolo_dataset_finetune/dataset.yaml"
    train_yolo(data_path=DATASET_PATH)
    # resume_train(
    #     data_path=DATASET_PATH,
    #     last_ckpt_path="/kaggle/input/sealion-epoch34/epoch34.pt",
    # )
