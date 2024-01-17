import torch

import torch.nn as nn
import torch.optim as optim
from unet import UNet
# from utils import (
#     load_checkpoint,
#     save_checkpoint,
#     get_loaders,
#     check_accuracy,
#     save_predictions_as_imgs,
# )

import tqdm


def get_data_loaders():
    pass



# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 3
NUM_WORKERS = 2
IMAGE_HEIGHT = 68  # 1080 originally
IMAGE_WIDTH = 120  # 1920 originally
PIN_MEMORY = True
LOAD_MODEL = False


def train_fun(loader, model, optimizer, loss_fn, scaler):
    """
    Will do one epoch of the training
    """
    tqdm_loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(tqdm_loop):
        data = data.to(device=DEVICE)
        targets = targets.to(device=DEVICE)

        # Forward step
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward step
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        tqdm_loop.set_postfix(loss=loss.item())


def main():
    # Getting the model, lost function and the optimizer for training
    model = UNet(in_channels=3, out_channels=4).to(DEVICE)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Getting the data loaders for training
    train_loader, val_loader = get_data_loaders(training_dir, validation_dir, BATCH_SIZE)