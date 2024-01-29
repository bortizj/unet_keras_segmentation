import torch

from unet import UNet
from utils import load_checkpoint
from utils import get_data_loaders
from utils import check_accuracy
from utils import store_predictions

from pathlib import Path


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_WORKERS = 2
PIN_MEMORY = True

TRAINING_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data\training")
TESTING_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data\testing")
CHECKPOINT_DIR = Path(r"D:\gitProjects\segmentation_unet\data_set\data")


def main():
    # Getting the model for testing
    model = UNet(in_channels=3, out_channels=4).to(DEVICE)

    # Model filename checkpoint
    filename = str(CHECKPOINT_DIR.joinpath("unet_checkpoint.tar"))

    # Folder out for storing temporal outputs
    path_out = TRAINING_DIR.joinpath("predictions")
    path_out.mkdir(parents=True, exist_ok=True)

    # Loads the current model checkpoint
    load_checkpoint(torch.load(filename), model)

    loader_train_val, ds_train_val = get_data_loaders(
        TRAINING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )
    loader_test, ds_test = get_data_loaders(
        TESTING_DIR, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY
    )

    # Computing the performance for the training set
    print("Performance training and validation set")
    check_accuracy(loader_train_val, model, device=DEVICE)

    # Computing the performance for the testing set
    print("Performance test set")
    check_accuracy(loader_test, model, device=DEVICE)

    # Checking predictions in the training and validation data
    store_predictions(TRAINING_DIR, ds_train_val, model, DEVICE)

    # Checking predictions in the testing data
    store_predictions(TESTING_DIR, ds_test, model, DEVICE)


if __name__ == "__main__":
    main()
