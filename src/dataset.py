from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LABEL_ORDER = [
    "pen",
    "paper",
    "book",
    "clock",
    "phone",
    "laptop",
    "chair",
    "desk",
    "bottle",
    "keychain",
    "backpack",
    "calculator",
]
VALID_LABELS = set(LABEL_ORDER)
LABEL_TO_IDX = {label: idx for idx, label in enumerate(LABEL_ORDER)}


class MultiLabelImageFolder(Dataset):
    """Dataset for directories named by underscore-separated labels."""

    def __init__(self, root, transform=None, separator="_", classes=LABEL_ORDER):
        self.root = Path(root)
        self.transform = transform
        self.separator = separator
        self.classes = list(classes)
        self.samples = []

        if not self.root.exists():
            raise FileNotFoundError(f"Dataset directory does not exist: {self.root}")

        for subdir in sorted(self.root.iterdir()):
            if not subdir.is_dir():
                continue

            labels = subdir.name.split(self.separator)
            if not labels or any(label not in VALID_LABELS for label in labels):
                continue
            if len(labels) != len(set(labels)):
                continue

            target = torch.zeros(len(self.classes), dtype=torch.float32)
            for label in labels:
                target[LABEL_TO_IDX[label]] = 1.0

            for img_path in sorted(subdir.glob("*.png")):
                if img_path.is_file():
                    self.samples.append((img_path, target.clone()))

        if not self.samples:
            raise ValueError(f"No PNG samples found in valid label folders under {self.root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, target = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, target


def build_train_transform(image_size=128):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def build_eval_transform(image_size=128):
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
