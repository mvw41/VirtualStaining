"""Train UNet to reconstruct DAPI images from brightfield inputs."""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data_handling.dataloader import BFDAPIDataset
from Models.unet import UNet

BF_DIR = "Data_handling/Trainings_Daten/8_bit/BF"
DAPI_DIR = "Data_handling/Trainings_Daten/8_bit/DAPI"
MODEL_PATH = "dapi_model.pth"


def get_loaders(bf_dir: str, dapi_dir: str, batch_size: int):
    files = [f for f in os.listdir(bf_dir) if f.lower().endswith(".tif")]
    split = int(0.8 * len(files))
    train_ds = BFDAPIDataset(bf_dir, dapi_dir, files[:split])
    val_ds = BFDAPIDataset(bf_dir, dapi_dir, files[split:])
    kwargs = dict(batch_size=batch_size, num_workers=4, shuffle=True)
    return DataLoader(train_ds, **kwargs), DataLoader(val_ds, **kwargs)


def train_epoch(loader: DataLoader, model: UNet, loss_fn: nn.Module,
                optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    model.train()
    for bf, dapi in loader:
        bf, dapi = bf.to(device), dapi.to(device)
        loss = loss_fn(model(bf), dapi)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train UNet for DAPI reconstruction")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    train_loader, _ = get_loaders(BF_DIR, DAPI_DIR, args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(in_channels=1, out_channels=1).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        train_epoch(train_loader, model, loss_fn, optimizer, device)

    torch.save(model.state_dict(), MODEL_PATH)
    print("DAPI training finished")


if __name__ == "__main__":
    main()
