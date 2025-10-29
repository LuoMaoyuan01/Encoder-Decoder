import os
import torch
import cv2
import numpy as np
import torch.nn.functional as F
import numpy as np
from PIL import Image
from pathlib import Path

def coerce_path(x):
    # pred.image_path might be a list[str] or str; take the first if it's a list
    if isinstance(x, (list, tuple)):
        x = x[0]
    return Path(x)

# Normalize each heatmap to [0,1] using min-max normalization
def minmax_norm(t: torch.Tensor, eps=1e-8):
    # Works for HxW or 1xHxW or Nx1xHxW
    tmin = t.amin(dim=(-2, -1), keepdim=True)
    tmax = t.amax(dim=(-2, -1), keepdim=True)
    return (t - tmin) / (tmax - tmin + eps)

# Expand (“dilate”) the detected anomaly pixels by one-pixel neighborhood.
# If any pixel in the 3x3 neighborhood is 1, the center becomes 1.
def dilate_3x3(binary_mask: torch.Tensor) -> torch.Tensor:
    """
    binary_mask: torch float/bool tensor of shape (H,W) or (1,H,W) or (N,1,H,W)
    returns: same shape, dilated (any neighbor in 3x3 turns center to 1)
    """
    added_dim = False
    if binary_mask.dim() == 2:  # HxW
        binary_mask = binary_mask.unsqueeze(0).unsqueeze(0)
        added_dim = True
    elif binary_mask.dim() == 3:  # 1xHxW or CxHxW
        binary_mask = binary_mask.unsqueeze(0)  # -> 1xCxHxW

    # max-pool with kernel=3, padding=1 is binary dilation for {0,1}
    dilated = F.max_pool2d(binary_mask.float(), kernel_size=3, stride=1, padding=1)

    if added_dim:
        dilated = dilated.squeeze(0).squeeze(0)  # back to HxW
    elif dilated.shape[0] == 1 and dilated.shape[1] == 1:
        dilated = dilated.squeeze(0)  # -> 1xHxW
    return (dilated > 0.5)

def postprocess_heatmap(
    anomaly_map: torch.Tensor,
    thresh: float = 0.5,
    normalize: bool = True,
    require_any: bool = True
):
    """
    anomaly_map: torch tensor HxW or 1xHxW (dtype float)
    thresh: threshold in [0,1] if normalized; otherwise in heatmap's scale
    normalize: min-max normalize per image before thresholding
    require_any: if True, return empty mask if nothing crosses threshold
    """
    if anomaly_map.dim() == 3 and anomaly_map.shape[0] == 1:
        heat = anomaly_map.squeeze(0)
    else:
        heat = anomaly_map  # HxW

    heat = heat.detach()
    if normalize:
        heat = minmax_norm(heat)

    binary = (heat >= thresh)
    if require_any and not binary.any():
        return binary  # all False

    # 3x3 neighborhood activation
    dilated = dilate_3x3(binary)
    return dilated  # bool HxW

# Saves mask to disk as PNG
def save_mask_png(mask_bool: torch.Tensor, out_path: str):
    mask_np = (mask_bool.cpu().numpy().astype(np.uint8) * 255)
    Image.fromarray(mask_np).save(out_path)

# Overlays heatmap on image and saves to disk
def overlay_heatmap_on_image(
    img_pil: Image.Image,
    heat: torch.Tensor,          # HxW or 1xHxW, any range
    out_path: str,
    threshold: float = 0.6,      # show overlay only where heat >= threshold (after norm)
    alpha: float = 0.6           # opacity of visible (hot) regions
):
    # expects heat normalized to [0,1], shape HxW
    h = heat.detach().cpu().numpy()
    h = (h - h.min()) / (h.max() - h.min() + 1e-8)
    mask = (h >= threshold)
    h_color = cv2.applyColorMap((h*255).astype(np.uint8), cv2.COLORMAP_JET)[:, :, ::-1]  # to RGB
    img = np.array(img_pil.convert("RGB"))
    img[mask] = (alpha * h_color[mask] + (1 - alpha) * img[mask]).astype(np.uint8)
    Image.fromarray(img).save(out_path)