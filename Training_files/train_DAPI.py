"""Training script for reconstructing DAPI from brightfield images using UNet."""

from __future__ import annotations

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from Data_handling.dataloader import BFDAPIDataset
from Models.unet import UNet

BF_DIR       = "Data_handling/Trainings_Daten/8_bit/BF"
DAPI_DIR     = "Data_handling/Trainings_Daten/8_bit/DAPI"
FILE_LIST   = [f for f in os.listdir(BF_DIR) if f.lower().endswith(".tif")]
N_EPOCHS    = 20
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_loaders(batch_size):
    # Split in Train/Val
    split = int(0.8 * len(FILE_LIST))
    train_files, val_files = FILE_LIST[:split], FILE_LIST[split:]

    train_ds = BFDAPIDataset(
        bf_dir=BF_DIR,
        dapi_dir=DAPI_DIR,
        file_list=train_files,
        tile_size=256,
        num_tiles_per_image=100
    )
    val_ds = BFDAPIDataset(
        bf_dir=BF_DIR,
        dapi_dir=DAPI_DIR,
        file_list=val_files,
        tile_size=256,
        num_tiles_per_image=100
    )
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False, num_workers=4)
    return train_loader, val_loader


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

    dataloader = BFDAPIDataset(args.bf_dir, args.dapi_dir, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()

    for _ in range(args.epochs):
        train_epoch(dataloader, model, loss_fn, optimizer, device)

    print("DAPI training finished")


if __name__ == "__main__":
    main()
