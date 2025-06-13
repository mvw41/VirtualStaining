# -*- coding: utf-8 -*-
"""Network definitions for virtual staining.

This module contains PyTorch implementations for per-stain networks and
for a network that combines multiple stains into a final fluorescence
prediction.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class StainNet(nn.Module):
    """Simple convolutional network for a single stain."""

    def __init__(self, in_channels: int = 3, out_channels: int = 1):
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


class CombineNet(nn.Module):
    """Network that combines predictions from multiple stain networks."""

    def __init__(self, num_stains: int, out_channels: int = 3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(num_stains, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DummyDataset(torch.utils.data.Dataset):
    """Simple dataset returning random brightfield and stain tensors."""

    def __init__(self, num_samples: int = 10, image_shape: tuple[int, int] = (64, 64)):
        super().__init__()
        self.num_samples = num_samples
        self.image_shape = image_shape

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        brightfield = torch.randn(3, *self.image_shape)
        stains = {
            "stain1": torch.randn(1, *self.image_shape),
            "stain2": torch.randn(1, *self.image_shape),
        }
        return brightfield, stains


def train_epoch(
    dataloader: torch.utils.data.DataLoader,
    stain_nets: dict[str, StainNet],
    combine_net: nn.Module,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> None:
    for brightfield, stains in dataloader:
        preds = []
        for name, net in stain_nets.items():
            pred = net(brightfield)
            loss = loss_fn(pred, stains[name])
            loss.backward()
            preds.append(pred)
        stacked = torch.cat(preds, dim=1)
        combined = combine_net(stacked)
        combined_loss = combined.abs().mean()
        combined_loss.backward()
        optimizer.step()
        optimizer.zero_grad()


def main() -> None:
    from combine_cnn import CombineCNN

    dataset = DummyDataset()
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=2)

    stain_nets = {
        "stain1": StainNet(),
        "stain2": StainNet(),
    }
    combine_net = CombineCNN(in_channels=len(stain_nets), out_channels=3)

    params = list(combine_net.parameters())
    for net in stain_nets.values():
        params += list(net.parameters())

    optimizer = torch.optim.Adam(params, lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(2):
        train_epoch(dataloader, stain_nets, combine_net, loss_fn, optimizer)
    print("training finished")


if __name__ == "__main__":
    main()
