import torch

def stitch_n_maps(patch_maps: torch.Tensor, coords, out_size=512):
     """
        patch_maps: (N, H, W)
        coords: list of (y, x) for each patch
        out_size: final image size (512)

        Works for:
        N=16, H=W=128
        N=4,  H=W=256
    """
     
     device = patch_maps.device
     patch = patch_maps.shape[-1]

     full = torch.zeros((out_size, out_size), device=device)
     count = torch.zeros((out_size, out_size), device=device)

     for i, (y, x) in enumerate(coords):
        full[y:y+patch, x:x+patch] += patch_maps[i]
        count[y:y+patch, x:x+patch] += 1.0

     return full / count.clamp_min(1.0)
