import torch

import torch.nn as nn
import torch.optim as optim
from unet import UNet
from utils import load_checkpoint
from utils import save_checkpoint
from utils import get_data_loaders
from utils import check_accuracy
from utils import save_predictions_as_imgs

import tqdm
from pathlib import Path


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

TRAINING_DIR = Path("D:\gitProjects\segmentation_unet\data_set\data\training")
PERCENTAGE_VALIDATION = 30


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
    train_loader, val_loader = get_data_loaders(
        TRAINING_DIR, PERCENTAGE_VALIDATION, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fun(train_loader, model, optimizer, loss_fn, scaler)


if __name__ == "__main__":
    pass
