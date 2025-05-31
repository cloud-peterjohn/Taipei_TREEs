import os
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
from torchvision.models import mobilenet_v2
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint


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
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )


def get_strong_transforms():
    return T.Compose(
        [
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
        self.model = MobileNetV2(num_classes=num_classes)
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


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        # dw
        layers.append(
            nn.Conv2d(
                hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False
            )
        )
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        # pw-linear
        layers.append(nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False))
        layers.append(nn.BatchNorm2d(oup))
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=10, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 1],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        input_channel = int(input_channel * width_mult)
        self.last_channel = (
            int(last_channel * width_mult) if width_mult > 1.0 else last_channel
        )
        self.features = [
            nn.Conv2d(3, input_channel, 3, stride=1, padding=1, bias=False),  # stride=1
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True),
        ]
        for t, c, n, s in interverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(
                    block(input_channel, output_channel, stride, expand_ratio=t)
                )
                input_channel = output_channel
        self.features.append(
            nn.Conv2d(input_channel, self.last_channel, 1, 1, 0, bias=False)
        )
        self.features.append(nn.BatchNorm2d(self.last_channel))
        self.features.append(nn.ReLU6(inplace=True))
        self.features = nn.Sequential(*self.features)
        self.classifier = nn.Linear(self.last_channel, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(x.shape[0], -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


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
        max_epochs=100,
        accelerator="auto",
        devices=1,
        callbacks=[checkpoint_callback],
        log_every_n_steps=1,
    )

    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
