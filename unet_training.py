import torch

import torch.nn as nn
import torch.optim as optim
from unet import UNet
from utils import load_checkpoint
from utils import save_checkpoint
from utils import get_data_loaders_divide
from utils import check_accuracy
from utils import save_predictions_as_imgs
from transforms import CostumeAffineTransform

import tqdm
from pathlib import Path


# Parameters for the trainer
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
NUM_EPOCHS = 500
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False


# +/- 20% scale change [ratio]
scales = [0.8, 1.2]
# Allowing any angle since it is potentially possible depending of microscope orientation [deg]
angles = [-90, 90]
# Small translations allowed since it is expected to be more or less centered [px] (+/-50 [px])
txs = [-3.125, 3.125]
tys = [-3.125, 3.125]
TRANSFORM = CostumeAffineTransform(scales, angles, txs, tys)


# Data paths
TRAINING_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data\training")
CHECKPOINT_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data")
PERCENTAGE_TRAINING = 70


def train_fun(loader, model, optimizer, loss_fn, scaler):
    """
    Will do one epoch of the training
    """
    tqdm_loop = tqdm.tqdm(loader)

    for batch_idx, (data, targets) in enumerate(tqdm_loop):
        # Better practice to send the data to device in case that is running in GPU
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward step
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward step (optimization step)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tqdm_loop.set_postfix(loss=loss.item())


def main():
    """
    Convenient function to start the model training for a number of epochs
    """
    # Getting the model, lost function and the optimizer for training
    model = UNet(in_channels=3, out_channels=4).to(DEVICE)
    # Loss function recommended for multi-class problems
    loss_fn = nn.CrossEntropyLoss()
    # Most used optimizer
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Avoids gradients flush to zero due to numerical precision
    scaler = torch.cuda.amp.GradScaler()

    # Getting the data loaders for training
    train_loader, val_loader = get_data_loaders_divide(
        TRAINING_DIR,
        PERCENTAGE_TRAINING,
        BATCH_SIZE,
        NUM_WORKERS,
        PIN_MEMORY,
        transform=TRANSFORM,
    )
    # Model filename checkpoint
    filename = str(CHECKPOINT_DIR.joinpath("unet_checkpoint.tar"))

    # Folder out for storing temporal outputs
    path_out = TRAINING_DIR.joinpath("predictions")

    # Loads the checkpoint in case of continue training the model
    if LOAD_MODEL:
        checkpoint = torch.load(filename)
        load_checkpoint(checkpoint, model)

    # Checks the accuracy of the loaded / initial model
    check_accuracy(val_loader, model, device=DEVICE)

    # Trains the model for a number of epochs
    for epoch in range(NUM_EPOCHS):
        train_fun(train_loader, model, optimizer, loss_fn, scaler)

        # saving current status of model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename)

        # check accuracy for the current version of the model
        check_accuracy(val_loader, model, device=DEVICE)

        # stores a mosaic of images for the current validation loader to a folder
        folder_epoch = path_out.joinpath(f"epoch_{epoch}")
        folder_epoch.mkdir(parents=True, exist_ok=True)
        save_predictions_as_imgs(val_loader, model, folder_epoch, device=DEVICE)

        print(f"Epochs processed {epoch + 1} / {NUM_EPOCHS}")


if __name__ == "__main__":
    main()
