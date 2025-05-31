# Modified from official documentation: https://docs.ultralytics.com/models/yolo11/

from ultralytics import YOLO


def main(
    dataset_yaml_path="./yolo_dataset/dataset.yaml",
    epochs=100,
    image_size=640,
    batch_size=2,
    num_workers=4,
    device="cuda",
):

    # Load a pre-trained YOLO model
    model = YOLO("yolo11x.pt")  # n, s, m, l, x versions available

    # Train the model on our custom dataset
    model.train(
        data=dataset_yaml_path,
        epochs=epochs,
        imgsz=image_size,
        batch_size=batch_size,
        device=device,
        workers=num_workers,
    )


if __name__ == "__main__":
    main()
