import torch
import math

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
