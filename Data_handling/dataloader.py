# -*- coding: utf-8 -*-
"""Dataset utilities for training a DAPI reconstruction network."""

import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class BFDAPIDataset(Dataset):
    """Yields random 256×256 tiles from brightfield and DAPI images."""

    def __init__(
        self,
        bf_dir: str,
        dapi_dir: str,
        file_list: list[str],
        tile_size: int = 256,
        tiles_per_image: int = 100,
    ) -> None:
        self.bf_dir = bf_dir
        self.dapi_dir = dapi_dir
        self.file_list = file_list
        self.tile_size = tile_size
        self.tiles_per_image = tiles_per_image
        self.n = len(file_list) * tiles_per_image

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        fname = self.file_list[idx // self.tiles_per_image]
        bf = Image.open(os.path.join(self.bf_dir, fname)).convert("L")
        dapi = Image.open(os.path.join(self.dapi_dir, fname)).convert("L")

        w, h = bf.size
        x0 = random.randint(0, w - self.tile_size)
        y0 = random.randint(0, h - self.tile_size)
        box = (x0, y0, x0 + self.tile_size, y0 + self.tile_size)
        bf = bf.crop(box)
        dapi = dapi.crop(box)

        angle = random.choice([0, 90, 180, 270])
        bf = bf.rotate(angle)
        dapi = dapi.rotate(angle)
        if random.random() < 0.5:
            bf = ImageOps.mirror(bf)
            dapi = ImageOps.mirror(dapi)
        if random.random() < 0.5:
            bf = ImageOps.flip(bf)
            dapi = ImageOps.flip(dapi)

        b = random.uniform(0.9, 1.1)
        c = random.uniform(0.9, 1.1)
        bf = TF.adjust_contrast(TF.adjust_brightness(bf, b), c)
        dapi = TF.adjust_contrast(TF.adjust_brightness(dapi, b), c)

        bf = TF.to_tensor(bf)
        bf = TF.normalize(bf, [0.5], [0.5])
        dapi = TF.to_tensor(dapi)
        dapi = TF.normalize(dapi, [0.5], [0.5])

        return bf, dapi
