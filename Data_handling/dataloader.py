# -*- coding: utf-8 -*-
"""Dataset and DataLoader utilities for DAPI training."""

import os
import random
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class BFDAPIDataset(Dataset):
    """
    Liefert 256×256-Tiles aus BF und DAPI (Graustufen).
    Wendet Rotation (Vielfache von 90°), Flip und ColorJitter an.
    Rückgabe: (bf_tensor, dapi_tensor)
    """
    def __init__(
            self,
            bf_dir: str,
            dapi_dir: str,
            file_list: list,
            tile_size: int = 256,
            num_tiles_per_image: int = 100,
            transform=None
    ):
        self.bf_dir = bf_dir
        self.dapi_dir = dapi_dir
        self.file_list = file_list
        self.tile_size = tile_size
        self.num_tiles_per_image = num_tiles_per_image
        self.n = len(file_list) * num_tiles_per_image

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
      # 1) Bestimme Bild und Datei
      image_idx = idx // self.num_tiles_per_image
      fname = self.file_list[image_idx]

      # 2) Lade beide Kanäle als Graustufen
      bf = Image.open(os.path.join(self.bf_dir, fname)).convert("L")
      dapi = Image.open(os.path.join(self.dapi_dir, fname)).convert("L")

      # 3) Identischer Crop
      w, h = bf.size
      x0 = random.randint(0, w - self.tile_size)
      y0 = random.randint(0, h - self.tile_size)
      crop_bf = bf.crop((x0, y0, x0 + self.tile_size, y0 + self.tile_size))
      crop_dapi = dapi.crop((x0, y0, x0 + self.tile_size, y0 + self.tile_size))

      # 4) Gemeinsame Rotation um Vielfache von 90°
      angle = random.choice([0, 90, 180, 270])
      crop_bf = crop_bf.rotate(angle)
      crop_dapi = crop_dapi.rotate(angle)

      # 5) Gemeinsames horizontales Flip?
      if random.random() < 0.5:
        crop_bf = ImageOps.mirror(crop_bf)
        crop_dapi = ImageOps.mirror(crop_dapi)

      # 6) Gemeinsames vertikales Flip?
      if random.random() < 0.5:
        crop_bf = ImageOps.flip(crop_bf)
        crop_dapi = ImageOps.flip(crop_dapi)

      # 7) Gemeinsames ColorJitter (Helligkeit & Kontrast)
      b_factor = random.uniform(0.9, 1.1)
      c_factor = random.uniform(0.9, 1.1)
      crop_bf = TF.adjust_brightness(crop_bf, b_factor)
      crop_dapi = TF.adjust_brightness(crop_dapi, b_factor)
      crop_bf = TF.adjust_contrast(crop_bf, c_factor)
      crop_dapi = TF.adjust_contrast(crop_dapi, c_factor)

      # 8) ToTensor & Normalize
      # bf_tensor = TF.to_tensor(crop_bf)
      # bf_tensor = TF.normalize(bf_tensor, [0.5], [0.5])
      # dapi_tensor = TF.to_tensor(crop_dapi)
      # dapi_tensor = TF.normalize(dapi_tensor, [0.5], [0.5])

      #return bf_tensor, dapi_tensor
      return crop_bf, crop_dapi