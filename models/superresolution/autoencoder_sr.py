# E:\models\superresolution\autoencoder_sr.py
import torch
import torch.nn as nn


class AutoencoderSR(nn.Module):
    """Autoencoder-based SR model from the Keras notebook.

    Assumes input spatial sizes divisible by 4.
    """

    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()

        # Encoder
        self.enc_conv1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc_conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, stride=2)
        self.drop1 = nn.Dropout2d(0.3)

        self.enc_conv3 = nn.Conv2d(base_channels, base_channels * 2, 3, padding=1)
        self.enc_conv4 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, stride=2)
        self.drop2 = nn.Dropout2d(0.5)

        self.enc_conv5 = nn.Conv2d(base_channels * 2, base_channels * 4, 3, padding=1)

        # Decoder
        self.up1 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv1 = nn.Conv2d(base_channels * 4, base_channels * 2, 3, padding=1)
        self.dec_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, padding=1)

        self.up2 = nn.Upsample(scale_factor=2, mode="nearest")
        self.dec_conv3 = nn.Conv2d(base_channels * 2, base_channels, 3, padding=1)
        self.dec_conv4 = nn.Conv2d(base_channels, base_channels, 3, padding=1)

        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, padding=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        l1 = self.relu(self.enc_conv1(x))
        l2 = self.relu(self.enc_conv2(l1))
        l3 = self.pool1(l2)
        l3 = self.drop1(l3)

        l4 = self.relu(self.enc_conv3(l3))
        l5 = self.relu(self.enc_conv4(l4))
        l6 = self.pool2(l5)
        _ = self.drop2(l3)  # mirrors the original notebook (l3 reuse)

        l7 = self.relu(self.enc_conv5(l6))

        # Decoder
        l8 = self.up1(l7)
        l9 = self.relu(self.dec_conv1(l8))
        l10 = self.relu(self.dec_conv2(l9))

        l11 = l5 + l10
        l12 = self.up2(l11)

        l13 = self.relu(self.dec_conv3(l12))
        l14 = self.relu(self.dec_conv4(l13))

        l15 = l14 + l2
        decoded = self.out_conv(l15)

        return decoded
