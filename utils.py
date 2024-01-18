import torch
import torchvision
from dataset import CostumeDataset
from torch.utils.data import DataLoader

import cv2

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
    source_training_list = list(map(source_filenames.__getitem__, idx_training))
    labels_training_list = list(map(label_filenames.__getitem__, idx_training))

    # Getting the file list from the random samples for testing
    source_validation_list = list(map(source_filenames.__getitem__, idx_validation))
    labels_validation_list = list(map(label_filenames.__getitem__, idx_validation))

    return (
        source_training_list,
        labels_training_list,
        source_validation_list,
        labels_validation_list,
    )


def get_data_loaders(
    path_data, percentage_training, batch_size, num_workers, pin_memory
):
    """
    Convenient function to get data loaders from a data path
    """
    # Dividing in training and validation
    files_tuple = divide_train_validate(path_data, percentage_training)
    (
        source_training_list,
        labels_training_list,
        source_validation_list,
        labels_validation_list,
    ) = files_tuple

    # Creating the training data loader
    train_ds = CostumeDataset(
        source_folder=source_training_list,
        labels_folder=labels_training_list,
        transform=None,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # Creating the validation data loader
    validation_ds = CostumeDataset(
        source_folder=source_validation_list,
        labels_folder=labels_validation_list,
        transform=None,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, validation_loader


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds(preds > 0.5).astype("float32")
            num_correct += (preds == y).sum(axis=[0, 1])
            num_pixels += np.sum(preds.shape[2::])
            dice_score += (2 * preds * y).sum(axis=[0, 1]) / (
                (preds + y).sum(axis=[0, 1]) + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with accuracy {100 * num_correct / num_pixels:.2f}"
    )
    print(f"Dice score {dice_score / len(loader):.6f}")

    model.train()


def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            preds = torch.sigmoid(model(x))
            preds(preds > 0.5).astype("float32")

            # Store here the images Perhaps encode them with RGBY as in the original images

    model.train()
