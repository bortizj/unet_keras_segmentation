from torch.utils.data import Dataset
import numpy as np
import cv2

from pathlib import Path


class CostumeDataset(Dataset):
    def __init__(self, source_list: Path, labels_list: Path, transform=None):
        self.transform = transform

        self.source_img_path_list = source_list
        self.labels_img_path_list = labels_list

    def __len__(self):
        return len(self.source_img_path_list)

    def __getitem__(self, idx):
        # The list of images must be sorted and correspond to each other
        source_img_path = self.source_img_path_list[idx]
        label_img_path = self.labels_img_path_list[idx]

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
