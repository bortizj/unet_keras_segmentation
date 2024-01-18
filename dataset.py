from torch.utils.data import Dataset
import numpy as np
import cv2

from pathlib import Path


class CostumeDataset(Dataset):
    def __init__(self, source_folder: Path, labels_folder: Path, transform=None):
        self.source_path = source_folder
        self.labels_path = labels_folder
        self.transform = transform

        self.source_img_path_list = self.source_path.glob("*.png")

    def __len__(self):
        return len(self.source_img_path_list)

    def __getitem__(self, idx):
        source_img_path = self.source_img_path_list[idx]
        label_img_path = self.labels_path.joinpath(source_img_path.name)

        # Reading and normalizing the source images to 0 - 1 as 32bit float
        source_img = cv2.imread(str(source_img_path), cv2.IMREAD_UNCHANGED)
        source_img = source_img.astype("float32") / 255.0

        # Reading from one hot encoding images
        label_img = cv2.imread(str(label_img_path), cv2.IMREAD_UNCHANGED).astype(
            "float32"
        )

        if self.transform is not None:
            augmentations = self.transform(image=source_img, mask=label_img)
            source_img = augmentations["image"]
            label_img = augmentations["mask"]

        return source_img, label_img
