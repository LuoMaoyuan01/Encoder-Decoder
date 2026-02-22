import os, glob
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
from anomalib.data.dataclasses import ImageBatch
import numpy as np
import torch.nn.functional as F


class CoarsePredictDataset(Dataset):
    """
    Coarse stream for multi-scale inference:
    - Reads the full 512x512 image
    - Downscales to 256x256 (or any target size)
    - Returns ImageBatch so anomalib Engine can apply its transforms
    """
    def __init__(self, img_dir: str, transform=None, out_size: int = 256, expect_hw=(512, 512)):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.transform = transform if transform is not None else T.ToTensor()

        self.out_size = int(out_size)
        self.resize = T.Resize((self.out_size, self.out_size), antialias=True)

        self.H, self.W = expect_hw  # your original SEM image resolution

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]

        # Ground-truth label from filename (same convention)
        fname = os.path.basename(path).lower()
        label = 1 if "anomalous" in fname else 0

        pil = Image.open(path).convert("RGB")
        img = self.transform(pil)  # (C,H,W)

        # Safety check: ensure you didn't accidentally resize in transform
        if img.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Expected transformed image to be {self.H}x{self.W}, got {tuple(img.shape[-2:])} for {path}. "
                f"Check your transform pipeline."
            )

        # Downscale full image to model input resolution
        coarse = self.resize(img)  # (C,out_size,out_size)

        return ImageBatch(
            image=coarse,
            gt_mask=None,
            gt_label=torch.tensor([label], dtype=torch.long),
            image_path=[path],
            mask_path=None,
            explanation=["coarse"],
        )
