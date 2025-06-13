import torch
import torch.nn as nn

class CombineCNN(nn.Module):
    """Simple convolutional network with three layers for combining stain predictions."""

    def __init__(self, in_channels: int, out_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
