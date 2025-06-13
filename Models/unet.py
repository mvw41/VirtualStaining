# -*- coding: utf-8 -*-
"""Small UNet architecture with residual blocks.

This network serves as a base model for reconstructing individual stains
such as DAPI from brightfield images.
"""

import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    """Classic residual block with two convolutions."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        return self.relu(out)


class UNet(nn.Module):
    """UNet with three levels and skip connections."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1, base_feat: int = 32):
        super().__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(in_channels, base_feat, kernel_size=3, padding=1),
            ResidualBlock(base_feat),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.down2 = nn.Sequential(
            nn.Conv2d(base_feat, base_feat * 2, kernel_size=3, padding=1),
            ResidualBlock(base_feat * 2),
        )
        self.pool2 = nn.MaxPool2d(2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(base_feat * 2, base_feat * 4, kernel_size=3, padding=1),
            ResidualBlock(base_feat * 4),
        )

        self.up2 = nn.ConvTranspose2d(base_feat * 4, base_feat * 2, kernel_size=2, stride=2)
        self.up_block2 = nn.Sequential(
            nn.Conv2d(base_feat * 4, base_feat * 2, kernel_size=3, padding=1),
            ResidualBlock(base_feat * 2),
        )

        self.up1 = nn.ConvTranspose2d(base_feat * 2, base_feat, kernel_size=2, stride=2)
        self.up_block1 = nn.Sequential(
            nn.Conv2d(base_feat * 2, base_feat, kernel_size=3, padding=1),
            ResidualBlock(base_feat),
        )

        self.final_conv = nn.Conv2d(base_feat, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        p1 = self.pool1(d1)
        d2 = self.down2(p1)
        p2 = self.pool2(d2)
        b = self.bottleneck(p2)
        u2 = self.up2(b)
        u2 = torch.cat([u2, d2], dim=1)
        u2 = self.up_block2(u2)
        u1 = self.up1(u2)
        u1 = torch.cat([u1, d1], dim=1)
        u1 = self.up_block1(u1)
        return self.final_conv(u1)
