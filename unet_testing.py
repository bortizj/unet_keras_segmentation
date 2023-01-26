import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from unet import UNet


# Create a new model with the same architecture
model = UNet(in_channels=3, out_channels=4)

# Load the trained model parameters
model.load_state_dict(torch.load('unet_model.pth'))

# Set the model to evaluation mode
model.eval()

# Test the model on some images
with torch.no_grad():
    outputs = model(test_images)
