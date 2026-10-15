# E:\models\superresolution\srgan.py
import math
import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.prelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return x + out


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, scale: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels * (scale ** 2), 3, padding=1)
        self.pixel_shuffle = nn.PixelShuffle(scale)
        self.prelu = nn.PReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.prelu(x)
        return x


class SRGANGenerator(nn.Module):
    """Standard SRGAN-like generator for single-input LR images."""

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        num_residual_blocks: int = 16,
        scale: int = 4,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        if scale not in {2, 4, 8}:
            raise ValueError("scale must be one of {2,4,8}")

        self.conv1 = nn.Conv2d(in_channels, base_channels, 9, padding=4)
        self.prelu = nn.PReLU()

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_channels) for _ in range(num_residual_blocks)]
        )
        self.conv2 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(base_channels)

        up_blocks = []
        if scale in {2, 4, 8}:
            for _ in range(int(math.log2(scale))):
                up_blocks.append(UpsampleBlock(base_channels, 2))
        self.upsample = nn.Sequential(*up_blocks)

        self.conv3 = nn.Conv2d(base_channels, out_channels, 9, padding=4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.prelu(self.conv1(x))
        x2 = self.res_blocks(x1)
        x3 = self.bn2(self.conv2(x2))
        x = x1 + x3
        x = self.upsample(x)
        x = self.conv3(x)
        return x


class SRGANDiscriminator(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 64) -> None:
        super().__init__()
        layers = []
        channels = base_channels

        def block(in_c, out_c, stride, bn=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1)]
            if bn:
                layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        layers += block(in_channels, channels, stride=1, bn=False)
        layers += block(channels, channels, stride=2)
        layers += block(channels, channels * 2, stride=1)
        layers += block(channels * 2, channels * 2, stride=2)
        layers += block(channels * 2, channels * 4, stride=1)
        layers += block(channels * 4, channels * 4, stride=2)
        layers += block(channels * 4, channels * 8, stride=1)
        layers += block(channels * 8, channels * 8, stride=2)

        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels * 8, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
