import torch
import torch.nn as nn
import torchvision.transforms.functional as tf
import torchvision.transforms as transforms

from utils import load_checkpoint

import numpy as np
import cv2

from pathlib import Path

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_CHECKPOINT_DIR = Path(
    r"D:\gitProjects\segmentation_unet\data_set\data\unet_checkpoint.tar"
)


class DoubleConv(nn.Module):
    """
    Basic building block for UNet which is double convolution per resolution level
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        # Two convolutions per level and no bias because it is normalized
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: np.ndarray):
        return self.conv(x)


class UNet(nn.Module):
    """
    UNet model
    "U-Net: Convolutional Networks for Biomedical Image Segmentation"
    by Olaf Ronneberger, Philipp Fischer, and Thomas Brox
    Great explanation at https://www.youtube.com/watch?v=IHq1t7NxS8k
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 4,
        features: list = [64, 128, 256, 512],
    ):
        """
        Initializes the building block of the UNet model
        """
        super().__init__()

        # The number of channels for the input and output data tensors
        self.in_channels = in_channels
        self.out_channels = out_channels

        # For the down-sampling side of UNet (encoder)
        self.downs = nn.ModuleList()

        # For the up-sampling side of UNet (decoder)
        self.ups_deconv = nn.ModuleList()
        self.ups_double = nn.ModuleList()

        #  The pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder of UNet as proposed in the original paper
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder of UNet as proposed in the original paper
        for feature in reversed(features):
            self.ups_deconv.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups_double.append(DoubleConv(feature * 2, feature))

        # The bottle neck of the network or connection between encoder and decoder
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # The final convolution to set the number of output channels or labels
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: np.ndarray):
        """
        Computes the UNet model in x
        """
        skip_connections = []

        # computing encoder steps
        for down in self.downs:
            # Applies the down-sampling layer
            x = down(x)
            # Stores the connection to be applied in the up-sampling
            skip_connections.append(x)
            x = self.pooling(x)

        # Computing bottle neck step
        x = self.bottleneck(x)

        # skip connections starts from the lowest to the highest resolution
        skip_connections = skip_connections[::-1]

        # Computing the decoder step
        for ii in range(0, len(self.ups_deconv)):
            # Applies ConvTranspose2d
            x = self.ups_deconv[ii](x)

            # Adding the data from the down-sampling phase
            skip_connection = skip_connections[ii]

            # Resizing in case that image size is not divisible by two
            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=False)

            # Adds the layers of same resolution
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Applies DoubleConv
            x = self.ups_double[ii](concat_skip)

        return self.final_conv(x)


class UNetModel:
    """
    Convenient class to wrap a UNet model.
    Make easy the evaluation for any given image
    """

    def __init__(
        self,
        path_model: Path = DEFAULT_CHECKPOINT_DIR,
        in_channels: int = 3,
        out_channels: int = 4,
        sampling_factor: int = 8,
        device: str = DEVICE,
    ):
        # Getting the model for testing
        self.model = UNet(in_channels=in_channels, out_channels=out_channels).to(DEVICE)

        # Preprocessing to match the size of the training images
        self.sampling_factor = sampling_factor

        # To convert to a valid tensor
        self.tensor_transform = transforms.ToTensor()

        # Device used to compute the results
        self.device = device

        # List of colors for pseudo color storage
        self.colors = np.array(
            [[0, 255, 255], [0, 255, 0], [0, 0, 255], [255, 0, 0]], dtype="uint8"
        )

        # Loads the current model checkpoint
        load_checkpoint(torch.load(str(path_model)), self.model)

    def _preprocessing(self, img):
        # Removing the interlacing
        source_img = img[::2, ::2]

        # down-sampling the source image
        source_img = cv2.GaussianBlur(source_img, ksize=(7, 7), sigmaX=5)
        source_img = source_img[:: self.sampling_factor, :: self.sampling_factor]

        # Converting the source image to a torch tensor
        source_img = source_img.astype("float32") / 255.0
        source_tensor = self.tensor_transform(source_img)

        return source_tensor

    def evaluate_image(self, img: np.ndarray):
        source_tensor = self._preprocessing(img)

        # Sending the data to the device
        source_tensor = source_tensor.to(device=self.device)

        # Tells pytorch that we will evaluate the model
        self.model.eval()
        # Does not calculate the gradient
        with torch.no_grad():
            # The unsqueeze is used to add the batch dimension
            preds_tensor = torch.sigmoid(self.model(source_tensor.unsqueeze(0)))
            preds_tensor = preds_tensor > 0.5

        # Moving to CPU if needed
        if self.device == "cuda":
            preds_tensor = preds_tensor.squeeze(0).to("cpu")
            source_tensor = source_tensor.to("cpu")

        # Making a mosaic where from left to right is original and automatic annotations
        out_img = np.array(255 * source_tensor.permute(1, 2, 0), dtype="uint8")
        labels = np.zeros_like(out_img)
        for ii in range(4):
            idx = np.where(preds_tensor[ii, ::, ::])
            labels[idx] = self.colors[ii, ::].reshape(1, 1, -1)

        return np.hstack((out_img, labels))


def test():
    """
    Making sure that the input output sizes are equal after going through the network
    """
    x = torch.randn((16, 3, 68, 120))
    model = UNet(in_channels=3, out_channels=3)
    preds = model(x)
    print(preds.shape)
    print(x.shape)

    assert preds.shape == x.shape


if __name__ == "__main__":
    print(torch.cuda.is_available())
    test()
