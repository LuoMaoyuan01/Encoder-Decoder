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
    def __init__(self, img_dir: str, transform=None, patch=256, n=4):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        self.transform = transform if transform is not None else T.ToTensor()
        self.patch = patch
        self.n = n

    def __len__(self):
        return len(self.img_paths) * self.n

    def __getitem__(self, idx: int):
        img_idx = idx // self.n
        patch_idx = idx % self.n

        path = self.img_paths[img_idx]

        # Ground truth label from path based on file name
        fname = os.path.basename(path).lower()
        label = 1 if "anomalous" in fname else 0

        pil = Image.open(path).convert("RGB")
        x = self.transform(pil)     # (C,512,512)
        x = x.unsqueeze(0)          # (1,C,512,512) for your patchifier

        patches, _coords = split_into_n_patches(x, patch=self.patch, n=self.n)
        patch = patches[patch_idx]  # (C,patch,patch)
        
        fake_path = f"{path}::p{patch_idx}"
        return ImageBatch(
          image=patch,
          gt_mask=None,
          gt_label=torch.tensor([label], dtype=torch.long), # 1 for anomalous, 0 for normal
          image_path=[path],       # <-- REAL path only
          mask_path=None,
          explanation=[f"p{patch_idx}"],  # <-- carry patch id here
        )