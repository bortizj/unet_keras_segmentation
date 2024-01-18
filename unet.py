import torch
import torch.nn as nn
import torchvision.transforms.functional as tf

import numpy as np


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

        # For the up-sampling side of UNet (decoder)
        self.ups = nn.ModuleList()
        # For the down-sampling side of UNet (encoder)
        self.downs = nn.ModuleList()
        #  The pooling layer
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder of UNet as proposed in the original paper
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Decoder of UNet as proposed in the original paper
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feature * 2, feature))

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
            x = down(x)
            skip_connections.append(x)
            x = self.pooling(x)

        # Computing bottle neck step
        x = self.bottleneck(x)

        # skip connections starts from the lowest to the highest resolution
        skip_connections = skip_connections[::-1]

        # Computing the decoder step
        for ii in range(0, len(self.ups), 2):
            # Applies ConvTranspose2d
            x = self.ups[ii](x)
            skip_connection = skip_connections[ii // 2]

            # Resizing in case that image size is not divisible by two
            if x.shape != skip_connection.shape:
                x = tf.resize(x, size=skip_connection.shape[2:], antialias=False)

            # Adds the layers of same resolution
            concat_skip = torch.cat((skip_connection, x), dim=1)

            # Applies DoubleConv
            x = self.ups[ii + 1](concat_skip)

        return self.final_conv(x)


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
