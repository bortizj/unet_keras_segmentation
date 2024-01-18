from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import cv2

from pathlib import Path


class CostumeDataset(Dataset):
    def __init__(self, source_list: Path, labels_list: Path, transform=None):
        self.transform = transform

        # Define a transform to convert the image to tensor
        self.tensor_transform = transforms.ToTensor()

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
        label_img = cv2.imread(str(label_img_path), cv2.IMREAD_UNCHANGED)
        label_img = label_img.astype("float32") / 255.0

        # Converting the images into tensors
        source_img = self.tensor_transform(source_img)
        label_img = self.tensor_transform(label_img)

        if self.transform is not None:
            source_img = self.transform(source_img)
            label_img = self.transform(label_img)

        return source_img, label_img
