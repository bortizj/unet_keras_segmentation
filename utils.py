import torch
import torchvision
from dataset import CostumeDataset
from torch.utils.data import DataLoader

import numpy as np


def save_checkpoint(state, filename):
    """
    Stores in disk the current state of the NN
    """
    print("Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    """
    Restores in memory the current state of the NN
    """
    print("Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def divide_train_validate(path_data, percentage_training):
    # Reading the images and the labels in the same order
    source_filenames = list(path_data.joinpath("source").glob(r"*.png"))
    label_filenames = []
    for ii in source_filenames:
        filename = path_data.joinpath("labels", ii.name)
        label_filenames.append(filename)

    # Getting random samples for the training and validation
    np.random.seed(2023)
    n_files = len(source_filenames)
    idx_training = np.random.choice(
        n_files, int(percentage_training * n_files / 100), replace=False
    ).tolist()
    idx_validation = np.setdiff1d(np.arange(n_files), idx_training).tolist()

    # Getting the file list from the random samples for training
    source_training = list(map(source_filenames.__getitem__, idx_training))
    labels_training = list(map(label_filenames.__getitem__, idx_training))

    # Getting the file list from the random samples for testing
    source_validation = list(map(source_filenames.__getitem__, idx_validation))
    labels_validation = list(map(label_filenames.__getitem__, idx_validation))

    return source_training, labels_training, source_validation, labels_validation


def get_data_loaders(
    path_data, percentage_training, batch_size, num_workers, pin_memory
):
    # Dividing in training and validation
    files_tuple = divide_train_validate(path_data, percentage_training)
    source_training, labels_training, source_validation, labels_validation = files_tuple

    # Creating the data loader
    train_ds = CostumeDataset(
        source_folder=source_training, labels_folder=labels_training, transform=None
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )
