import os, glob, math
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from anomalib.data.dataclasses import ImageBatch
import torchvision.transforms as T

def split_into_n_patches(img_tensor_1chw: torch.Tensor, patch=256, n=4):
    """
    img_tensor_1chw: torch tensor (1,C,H,W) with H=W=512
    Returns:
      patches: (16, C, 128, 128)
      coords: list of (y, x) top-left coords
    """
    assert img_tensor_1chw.ndim == 4 and img_tensor_1chw.shape[-1] == 512 and img_tensor_1chw.shape[-2] == 512
    _, C, H, W = img_tensor_1chw.shape

    patches = []
    coords = []
    idx = 0
    for gy in range(int(math.sqrt(n))):
        for gx in range(int(math.sqrt(n))):
            y = gy * patch
            x = gx * patch
            p = img_tensor_1chw[0, :, y:y+patch, x:x+patch]
            patches.append(p)
            coords.append((y, x))
            idx += 1

    patches = torch.stack(patches, dim=0)  # (4, C, 128, 128)
    return patches, coords

# ------------------------ HELPER FUNCTION FOR NOTEBOOK, CREATING DATASET FOR PREDICTION ----------------------- #

class PatchifyPredictDataset(torch.utils.data.Dataset):
    """
    Patchify each 512x512 image into sliding-window patches using (patch_size, stride).
    - No `n` parameter.
    - Number of patches is derived from image size, patch_size, stride.
    - Returns one patch per __getitem__ along with:
        * image_path = [original_path]
        * explanation = [f"g{row}{col}"]  (grid id for stitching/pooling)
    """
    def __init__(self, img_dir: str, transform=None, patch_size: int = 256, stride: int = 128, require_exact_grid: bool = True):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.transform = transform if transform is not None else T.ToTensor()
        self.patch_size = int(patch_size)
        self.stride = int(stride)
        self.require_exact_grid = bool(require_exact_grid)

        # Your images are fixed-size 512x512
        self.H = 512
        self.W = 512

        if self.require_exact_grid:
            if (self.H - self.patch_size) % self.stride != 0 or (self.W - self.patch_size) % self.stride != 0:
                raise ValueError(
                    f"Stride {self.stride} doesn't tile {self.H}x{self.W} with patch {self.patch_size}. "
                    f"(H-patch)%stride={(self.H-self.patch_size)%self.stride}, "
                    f"(W-patch)%stride={(self.W-self.patch_size)%self.stride}"
                )

        self.n_rows = (self.H - self.patch_size) // self.stride + 1
        self.n_cols = (self.W - self.patch_size) // self.stride + 1
        self.patches_per_image = self.n_rows * self.n_cols

    def __len__(self):
        return len(self.img_paths) * self.patches_per_image

    def __getitem__(self, idx: int):
        img_idx = idx // self.patches_per_image
        patch_linear_idx = idx % self.patches_per_image

        r = patch_linear_idx // self.n_cols
        c = patch_linear_idx % self.n_cols
        y = r * self.stride
        x = c * self.stride

        path = self.img_paths[img_idx]

        # Ground-truth label from filename (your convention)
        fname = os.path.basename(path).lower()
        label = 1 if "anomalous" in fname else 0

        pil = Image.open(path).convert("RGB")
        img = self.transform(pil)  # (C,512,512)

        # Safety check (helps catch accidental resizing)
        if img.shape[-2:] != (self.H, self.W):
            raise ValueError(f"Expected transformed image to be {self.H}x{self.W}, got {tuple(img.shape[-2:])} for {path}")

        patch = img[:, y:y+self.patch_size, x:x+self.patch_size]  # (C,patch,patch)

        # Grid id (2D) for later grouping/stitching
        grid_id = f"g{r}{c}"  # for stride=128,patch=256 => r,c in 0..2 (3x3)

        return ImageBatch(
            image=patch,
            gt_mask=None,
            gt_label=torch.tensor([label], dtype=torch.long),  # 1 anomalous, 0 normal
            image_path=[path],                                 # ORIGINAL path only (no ::p)
            mask_path=None,
            explanation=[grid_id],                             # carry patch location here
        )