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
    """
    Convenient function to list all pngs in a folder and divide into validation and training
    """
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


def get_data_loaders_divide(
    path_data, percentage_training, batch_size, num_workers, pin_memory, transform=None
):
    """
    Convenient function to get data loaders from a data path dividing into validation and training
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
        source_list=source_training_list,
        labels_list=labels_training_list,
        transform=transform,
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
        source_list=source_validation_list,
        labels_list=labels_validation_list,
        transform=transform,
    )
    validation_loader = DataLoader(
        validation_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return train_loader, validation_loader


def get_data_loaders(path_data, batch_size, num_workers, pin_memory, transform=None):
    """
    Gets a data loader of the given folder path
    """
    # Reading the images and the labels in the same order
    source_list = list(path_data.joinpath("source").glob(r"*.png"))
    labels_list = []
    for ii in source_list:
        filename = path_data.joinpath("labels", ii.name)
        labels_list.append(filename)

    # Creating the training data loader
    ds = CostumeDataset(
        source_list=source_list,
        labels_list=labels_list,
        transform=transform,
    )
    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    return data_loader, ds


def check_accuracy(loader, model, device="cuda"):
    """
    Convenient function to compute accuracy of a given dataloader
    """
    flag = True

    # Tells pytorch that we will evaluate the model
    model.eval()
    # Does not calculate the gradient
    with torch.no_grad():
        for x, y in loader:
            # sending the data to the device for computations
            x = x.to(device)
            y = y.to(device)

            # Predicts the output using the given model
            preds = torch.sigmoid(model(x))

            preds = (preds > 0.5).type(torch.float32)
            # Shape of the tensor is NBatch, Channels, Rows, Cols
            shape_tensor = preds.shape
            # Computes the metrics for each class individually
            if flag:
                num_correct = (preds == y).sum(axis=[0, 2, 3])
                num_pixels = shape_tensor[0] * shape_tensor[2] * shape_tensor[3]
                dice_score = (2 * preds * y).sum(axis=[0, 2, 3]) / (
                    (preds + y).sum(axis=[0, 2, 3]) + 1e-8
                )
                flag = False
            else:
                num_correct += (preds == y).sum(axis=[0, 2, 3])
                num_pixels += shape_tensor[0] * shape_tensor[2] * shape_tensor[3]
                dice_score += (2 * preds * y).sum(axis=[0, 2, 3]) / (
                    (preds + y).sum(axis=[0, 2, 3]) + 1e-8
                )

    # Print per class the accuracy
    print("------------------Accuracy------------------")
    for ii in range(model.out_channels):
        print(
            f"Class {ii} Got {num_correct[ii]}/{num_pixels} with accuracy {100 * num_correct[ii] / num_pixels:.2f}"
        )
    print("--------------------Dice--------------------")
    for ii in range(model.out_channels):
        print(f"Class {ii} Dice score {100 * dice_score[ii] / len(loader):.2f}")

    # Tells pytorch that the model still training
    model.train()


def save_predictions_as_imgs(loader, model, folder, device="cuda"):
    """
    Convenient function to store the predictions of each epoch
    """
    # Tells pytorch that we will evaluate the model
    model.eval()
    # Does not calculate the gradient
    with torch.no_grad():
        for idx, (x, y) in enumerate(loader):
            # Very simple threshold to estimate the labels
            x = x.to(device=device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # Storing the images using pytorch as a mosaic of the batch
            for ii in range(model.out_channels):
                # Storing each channel individually
                path_out = folder.joinpath(f"{idx}_{ii}_pred.png")
                torchvision.utils.save_image(preds[:, [ii]], str(path_out))
                path_out = folder.joinpath(f"{idx}_{ii}_y.png")
                torchvision.utils.save_image(y[:, [ii]], str(path_out))

    # Tells pytorch that the model still training
    model.train()


def save_individual_prediction(
    source_tensor, labels_tensor, model, filename, device="cuda"
):
    """
    Convenient function to store individual predictions using the original filename
    It is a wrapper for testing trained models in individual images
    """
    # Sending the data to the device
    source_tensor = source_tensor.to(device=device)
    # Tells pytorch that we will evaluate the model
    model.eval()
    # Does not calculate the gradient
    with torch.no_grad():
        # The unsqueeze is used to add the batch dimension
        preds_tensor = torch.sigmoid(model(source_tensor.unsqueeze(0)))

    # Moving to CPU if needed
    if device == "cuda":
        preds_tensor = preds_tensor.squeeze(0).to("cpu")
        source_tensor = source_tensor.to("cpu")

    # Converting from tensor to image size
    out_top = np.array(255 * source_tensor.permute(1, 2, 0), dtype="uint8")
    out_bottom = out_top.copy()

    # Making a mosaic where top is manual annotation and bottom is probability map
    for ii in range(4):
        # Top image is the manual segmentation
        label_ii = np.array(255 * labels_tensor[ii, ::, ::], dtype="uint8")
        pseudo_label_ii = cv2.applyColorMap(label_ii, cv2.COLORMAP_PARULA)
        out_top = np.hstack((out_top, pseudo_label_ii))

        # bottom image is the automatic segmentation
        prob_ii = np.array(
            255 * torch.clip(preds_tensor[ii, ::, ::], min=0, max=1), dtype="uint8"
        )
        pseudo_label_ii = cv2.applyColorMap(prob_ii, cv2.COLORMAP_PARULA)
        out_bottom = np.hstack((out_bottom, pseudo_label_ii))

    # Storing the mosaic image
    out_img = np.vstack((out_top, out_bottom))
    cv2.imwrite(str(filename), out_img)
