import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


class PadTileToMinSize:
    def __init__(self, min_size=112):
        self.min_size = min_size

    def __call__(self, img):
        w, h = img.size
        if w < self.min_size:
            n = (self.min_size + w - 1) // w
            imgs = [img] * n
            new_img = Image.new(img.mode, (w * n, h))
            for i in range(n):
                new_img.paste(imgs[i], (i * w, 0))
            img = new_img.crop((0, 0, self.min_size, h))
        w, h = img.size
        if h < self.min_size:
            n = (self.min_size + h - 1) // h
            imgs = [img] * n
            new_img = Image.new(img.mode, (w, h * n))
            for i in range(n):
                new_img.paste(imgs[i], (0, i * h))
            img = new_img.crop((0, 0, w, self.min_size))
        return img


class TreeDataset(Dataset):
    def __init__(self, df, transform=None, strong_transform=None, rare_classes=None):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.strong_transform = strong_transform
        self.rare_classes = set(rare_classes) if rare_classes is not None else set()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["source_img_dir"], row["filename"])
        image = Image.open(img_path).convert("RGB")
        label = int(row["class_id"])
        if self.strong_transform and label in self.rare_classes:
            image = self.strong_transform(image)
        elif self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train=True):
    if train:
        return T.Compose(
            [
                PadTileToMinSize(112),
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.RandomRotation(30),
                T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
    else:
        return T.Compose(
            [
                PadTileToMinSize(112),
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def get_strong_transforms():
    return T.Compose(
        [
            PadTileToMinSize(112),
            T.Resize((224, 224)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomRotation(45),
            T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.2),
            T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


class LitMobileNet(pl.LightningModule):
    def __init__(self, num_classes, lr=1e-3, class_weights=None):
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])
        self.model = mobilenet_v2(weights="DEFAULT")
        self.model.classifier[1] = torch.nn.Linear(self.model.last_channel, num_classes)
        if class_weights is not None:
            self.criterion = torch.nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.hparams.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
        return {"optimizer": optimizer, "lr_scheduler": scheduler}


def check_chinese_in_path():
    path = os.path.abspath(os.getcwd())
    if any("\u4e00" <= ch <= "\u9fff" for ch in path):
        raise RuntimeError(
            f"The current working directory contains Chinese characters: {path}. Please change the directory to one that does not contain Chinese characters."
        )


def main(base_dir=r"classification_dataset"):
    check_chinese_in_path()
    
    print("Begin to load dataset...")
    # img_dir1 = os.path.join(base_dir, "IDTREES_classification/images")
    # label_csv1 = os.path.join(base_dir, "IDTREES_classification/labels.csv")
    # df1 = pd.read_csv(label_csv1)
    # df1["source_img_dir"] = img_dir1

    img_dir2 = os.path.join(base_dir, "PureForest_classification/images")
    label_csv2 = os.path.join(base_dir, "PureForest_classification/labels.csv")
    df2 = pd.read_csv(label_csv2)
    df2["source_img_dir"] = img_dir2

    # df = pd.concat([df1, df2], ignore_index=True)
    df = df2

    cls_counts = df["class_id"].value_counts()
    valid_classes = cls_counts[cls_counts >= 2].index
    df = df[df["class_id"].isin(valid_classes)].reset_index(drop=True)
    num_classes = df["class_id"].nunique()

    class_sample_count = df["class_id"].value_counts().sort_index().values
    class_weights = 1.0 / torch.tensor(class_sample_count, dtype=torch.float)
    class_weights = class_weights / class_weights.sum() * num_classes

    train_df, val_df = train_test_split(
        df, test_size=0.1, stratify=df["class_id"], random_state=42
    )

    rare_classes = cls_counts[cls_counts < 10].index.tolist()

    train_class_counts = train_df["class_id"].value_counts().sort_index()
    sample_weights = 1.0 / train_class_counts
    sample_weights = sample_weights / sample_weights.sum() * len(sample_weights)
    weights = train_df["class_id"].map(lambda x: sample_weights[x]).values
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_ds = TreeDataset(
        train_df,
        transform=get_transforms(train=True),
        strong_transform=get_strong_transforms(),
        rare_classes=rare_classes,
    )
    val_ds = TreeDataset(val_df, transform=get_transforms(train=False))

    num_workers = 4 if os.name != "nt" else 0
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    print("Dataset loaded successfully.")

    model = LitMobileNet(num_classes=num_classes, class_weights=class_weights)

    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc", mode="max", save_top_k=1, filename="mobilenetv2-best"
    )

    trainer = pl.Trainer(
        max_epochs=30,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
