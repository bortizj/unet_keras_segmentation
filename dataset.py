from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2

import numpy as np


class CostumeDataset(Dataset):
    """
    Convenient class to create a costume database that reads images
    from list of image files. This is used by pytorch data generators
    """

    def __init__(self, source_list: list, labels_list: list, transform=None):
        self.transform = transform

        # Define a transform to convert the image to torch tensor
        self.tensor_transform = transforms.ToTensor()

        # These are lists of pathlib Paths
        self.source_img_path_list = source_list
        self.labels_img_path_list = labels_list

    def __len__(self):
        return len(self.source_img_path_list)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        # The list of images must correspond to each other
        source_img_path = self.source_img_path_list[idx]
        label_img_path = self.labels_img_path_list[idx]

        # Reading and normalizing the source images to 0 - 1 as 32bit float
        source_img = cv2.imread(str(source_img_path), cv2.IMREAD_UNCHANGED)
        source_img = source_img.astype("float32") / 255.0

        # Reading and normalizing the one hot encoding images
        label_img = cv2.imread(str(label_img_path), cv2.IMREAD_UNCHANGED)
        label_img = label_img.astype("float32") / 255.0

        # Converting the images into torch tensors
        source_img = self.tensor_transform(source_img)
        label_img = self.tensor_transform(label_img)

        # If there are extra image transformers apply it
        if self.transform is not None:
            source_img = self.transform(source_img)
            label_img = self.transform(label_img)

        # Return the source image and the one hot encoding labelled image
        return source_img, label_img
