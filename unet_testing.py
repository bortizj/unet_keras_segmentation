import torch

import torch.nn as nn
import torch.optim as optim
from unet import UNet
from utils import load_checkpoint
from utils import get_data_loaders
from utils import check_accuracy
from utils import save_individual_prediction

from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

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

    loader_train_val, ds_train_val = get_data_loaders(
        TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )
    loader_test, ds_test = get_data_loaders(
        TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    print("Performance training and validation set")
    check_accuracy(loader_train_val, model, device=DEVICE)

    print("Performance test set")
    check_accuracy(loader_test, model, device=DEVICE)

    # Checking predictions in the training and validation data
    path_out = TRAINING_DIR.joinpath("predictions", "prediction")
    path_out.mkdir(parents=True, exist_ok=True)
    for ii in range(len(ds_train_val)):
        out_file = path_out.joinpath(ds_train_val.source_img_path_list[ii].name)
        source_tensor, labels_tensor = ds_train_val[ii]
        save_individual_prediction(
            source_tensor, labels_tensor, model, out_file, device=DEVICE
        )

    # Checking predictions in the testing data
    path_out = TESTING_DIR.joinpath("predictions", "prediction")
    path_out.mkdir(parents=True, exist_ok=True)
    for ii in range(len(ds_test)):
        out_file = path_out.joinpath(ds_test.source_img_path_list[ii].name)
        source_tensor, labels_tensor = ds_test[ii]
        save_individual_prediction(
            source_tensor, labels_tensor, model, out_file, device=DEVICE
        )


if __name__ == "__main__":
    main()
