"""Training script for reconstructing DAPI from brightfield images using UNet."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataloader import create_dataloader
from unet import UNet


def train_epoch(dataloader: DataLoader, model: UNet, loss_fn: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    model.train()
    for bf, dapi in dataloader:
        bf = bf.to(device)
        dapi = dapi.to(device)
        pred = model(bf)
        loss = loss_fn(pred, dapi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UNet to predict DAPI from brightfield images")
    parser.add_argument("--bf_dir", required=True, help="Directory with brightfield images (1024x1024)")
    parser.add_argument("--dapi_dir", required=True, help="Directory with corresponding DAPI images")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    dataloader = create_dataloader(args.bf_dir, args.dapi_dir, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        train_epoch(dataloader, model, loss_fn, optimizer, device)

    print("DAPI training finished")


if __name__ == "__main__":
    main()
