# E:\models\superresolution\srcnn.py
import torch
import torch.nn as nn


class SRCNN(nn.Module):
    """SRCNN as defined in the Keras notebook.

    Notes:
    - Uses VALID padding for conv1 and conv3 (kernel 9 and 5), so output is smaller.
    - Input/output channels are configurable (default 1 for Y channel).
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 128, kernel_size=9, padding=0, bias=True)
        self.conv2 = nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=True)
        self.conv3 = nn.Conv2d(64, out_channels, kernel_size=5, padding=0, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x
