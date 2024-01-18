import torch

import torch.nn as nn
import torch.optim as optim
from unet import UNet
from utils import load_checkpoint
from utils import get_data_loaders
from utils import check_accuracy

from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAINING_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data\training")
TESTING_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data\testing")
CHECKPOINT_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data")


def main():
    # Getting the model, lost function and the optimizer for training
    model = UNet(in_channels=3, out_channels=4).to(DEVICE)

    # Model filename checkpoint
    filename = str(CHECKPOINT_DIR.joinpath("unet_checkpoint.tar"))

    # Folder out for storing temporal outputs
    path_out = TRAINING_DIR.joinpath("predictions")
    path_out.mkdir(parents=True, exist_ok=True)

    load_checkpoint(torch.load(filename), model)

    check_accuracy(val_loader, model, device=DEVICE)

    for epoch in range(NUM_EPOCHS):
        train_fun(train_loader, model, optimizer, loss_fn, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint, filename)

        # check accuracy
        check_accuracy(val_loader, model, device=DEVICE)

        # print some examples to a folder
        save_predictions_as_imgs(val_loader, model, path_out, device=DEVICE)

        print(f"Epochs processed {epoch + 1} / {NUM_EPOCHS}")


if __name__ == "__main__":
    main()
