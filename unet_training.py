import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
from unet import UNet

class YourDataset(Dataset):
    def __init__(self, images_folder, segments_folder, transform=None):
        self.images_folder = datasets.ImageFolder(images_folder, transform=transform)
        self.segments_folder = datasets.ImageFolder(segments_folder, transform=transform)

    def __len__(self):
        return len(self.images_folder)

    def __getitem__(self, idx):
        image, _ = self.images_folder[idx]
        segment, _ = self.segments_folder[idx]

        return image, segment

# Define your U-Net model
model = UNet(in_channels=3, out_channels=4)

# Define a loss function and optimizer
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters())

# Define the data transform
transform = transforms.Compose([transforms.ToTensor()])

# Create the dataset
dataset = YourDataset(images_folder='D:\gitProjects\segmentation_unet\data_set\images', segments_folder='D:\gitProjects\segmentation_unet\data_set\labels', transform=transform)

# Create the dataloader with a batch size of 32
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

num_epochs = 10
# Train the model
for epoch in range(num_epochs):
    for i, (images, segments) in enumerate(dataloader):
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, segments)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(model.state_dict(), 'unet_model.pth')
