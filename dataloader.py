# -*- coding: utf-8 -*-
"""Dataset and DataLoader utilities for DAPI training."""

from __future__ import annotations

import os
import random
from typing import List

from PIL import Image
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class BFDAPIDataset(Dataset):
    """Loads matching brightfield and DAPI images and returns random tiles."""

    def __init__(self, bf_dir: str, dapi_dir: str, tile_size: int = 256):
        self.bf_paths: List[str] = sorted(
            [
                os.path.join(bf_dir, f)
                for f in os.listdir(bf_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff"))
            ]
        )
        self.dapi_dir = dapi_dir
        self.tile_size = tile_size

    def __len__(self) -> int:
        return len(self.bf_paths)

    def _load_image(self, path: str, mode: str = "RGB") -> np.ndarray:
        img = Image.open(path).convert(mode)
        return np.array(img)

    def __getitem__(self, idx: int):
        bf_path = self.bf_paths[idx]
        filename = os.path.basename(bf_path)
        dapi_path = os.path.join(self.dapi_dir, filename)

        bf = self._load_image(bf_path, mode="RGB")
        dapi = self._load_image(dapi_path, mode="L")

        h, w = bf.shape[:2]
        if h < self.tile_size or w < self.tile_size:
            raise ValueError("Input images are smaller than the tile size")

        x = random.randint(0, w - self.tile_size)
        y = random.randint(0, h - self.tile_size)

        bf_tile = bf[y : y + self.tile_size, x : x + self.tile_size]
        dapi_tile = dapi[y : y + self.tile_size, x : x + self.tile_size]

        bf_tensor = torch.from_numpy(bf_tile.transpose(2, 0, 1)).float() / 255.0
        dapi_tensor = torch.from_numpy(dapi_tile).unsqueeze(0).float() / 255.0

        return bf_tensor, dapi_tensor


def create_dataloader(
    bf_dir: str,
    dapi_dir: str,
    batch_size: int = 4,
    tile_size: int = 256,
    shuffle: bool = True,
) -> DataLoader:
    dataset = BFDAPIDataset(bf_dir, dapi_dir, tile_size=tile_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
