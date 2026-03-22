import cv2
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os, glob
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as T
from anomalib.data.dataclasses import ImageBatch

class CoarsePredictDataset(Dataset):
    """
    Coarse stream for a DSR model trained on FULL 512x512 images.
    - Loads each image
    - Applies the SAME transform pipeline you used for the 512-model training
      (at minimum: ToTensor; optionally Resize/Normalize if you used them)
    - Returns an anomalib ImageBatch with image shape (C,512,512)
    """
    def __init__(
        self,
        img_dir: str,
        transform=None,
        expect_hw=(512, 512),
        glob_pattern="*.*",
        label_from_filename=True,
        anomalous_token="anomalous",
    ):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, glob_pattern)))
        self.transform = transform if transform is not None else T.ToTensor()
        self.H, self.W = expect_hw
        self.label_from_filename = bool(label_from_filename)
        self.anomalous_token = anomalous_token.lower()

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]

        # Label (optional; useful for your own AUROC/AUPR outside anomalib)
        if self.label_from_filename:
            fname = os.path.basename(path).lower()
            label = 1 if self.anomalous_token in fname else 0
        else:
            label = 0  # default if you don't use filename labeling

        pil = Image.open(path).convert("RGB")
        img = self.transform(pil)  # expected (C,512,512)

        # Safety check (catches accidental resizing)
        if img.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Expected transformed image to be {self.H}x{self.W}, got {tuple(img.shape[-2:])} for {path}. "
                f"Make sure your transform matches training (no extra Resize)."
            )

        return ImageBatch(
            image=img,
            gt_mask=None,
            gt_label=torch.tensor([label], dtype=torch.long),
            image_path=[path],
            mask_path=None,
            explanation=["coarse_full512"],
        )

def extract_maps(preds):
    """
     Extract 1-channel anomaly maps from preds, keyed by image_path.
    """
    maps = {}
    for p in preds:
        path = p.image_path[0]

        if hasattr(p, "anomaly_map"):
            am = p.anomaly_map
        elif isinstance(p, dict) and "anomaly_map" in p:
            am = p["anomaly_map"]
        else:
            raise KeyError("No anomaly_map found")

        if am.ndim == 2:
            am = am.unsqueeze(0).unsqueeze(0)
        elif am.ndim == 3:
            am = am.unsqueeze(1)

        maps[path] = am.cpu()

    return maps

def coarse_preds_to_mapdict(coarse_preds):
    """
    Returns dict: path -> torch tensor of shape (1,H,W) or (H,W)
    """
    out = {}
    for p in coarse_preds:
        # support dict or attribute predictions
        path = p["image_path"][0] if hasattr(p, "__getitem__") else p.image_path[0]
        am = p["anomaly_map"] if hasattr(p, "__getitem__") else p.anomaly_map

        if not torch.is_tensor(am):
            am = torch.tensor(am)

        # normalize to (1,H,W)
        if am.ndim == 2:
            am = am.unsqueeze(0)
        elif am.ndim == 3 and am.shape[0] != 1:
            am = am[:1]

        out[path] = am.detach().cpu()
    return out

def _to_tensor_map(x, device="cpu"):
    if torch.is_tensor(x):
        return x.detach().to(device).float()
    return torch.tensor(x, dtype=torch.float32, device=device)


def _gaussian_blur_numpy(x, sigma=1.2, ksize=5):
    if sigma <= 0:
        return x
    return cv2.GaussianBlur(x, (ksize, ksize), sigma)


def _dilate_mask_numpy(mask, kernel_size=5, iterations=1):
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    return cv2.dilate(mask.astype(np.uint8), kernel, iterations=iterations)

def _remove_small_components(mask, min_area=0):
    """
    mask: uint8 binary mask
    """
    if min_area <= 0:
        return mask.astype(np.uint8)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )

    out = np.zeros_like(mask, dtype=np.uint8)
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == i] = 1
    return out

def fuse_stitched_with_coarse(
    fine_stitched,
    coarse_mapdict,
    out_size=512,
    threshold=0.49,
    update_pred_mask=True,
    beta=0.80,               # unsupported coarse regions keep beta strength
    q_coarse=0.90,           # kept for compatibility
    q_fine_protect=0.95,     # fine support threshold for coarse suppression
    show_suppression=False,
    Tcoarse=None,            # optional dataset-level coarse threshold
    k=12.0,                  # kept for compatibility
    blur_sigma=1.2,
    use_residual=True,       # now used for optional mild fine residual
    Tfine=None,              # optional dataset-level fine threshold
    fine_blur_sigma=0.0,
    dilate_kernel=5,
    dilate_iter=1,
    use_binary_fine_support=True,

    # -------------------------------
    # fine-only rescue settings
    # -------------------------------
    enable_fine_rescue=True,
    q_fine_rescue=0.99,      # stronger threshold than q_fine_protect
    Tfine_rescue=None,       # optional fixed threshold for rescue
    q_coarse_weak=0.80,      # pixel/region considered coarse-weak below this
    gamma=0.75,              # rescue boost strength
    rescue_dilate_kernel=3,
    rescue_dilate_iter=1,
    rescue_min_area=8,       # remove tiny rescued speckles
):
    """
    Adaptive coarse-fine fusion with fine-only rescue path.

    Output format remains the same:
      - list of dicts
      - new_d["anomaly_map"]
      - new_d["pred_mask"] if update_pred_mask=True
    """

    fused = []

    for d in fine_stitched:
        path = d["image_path"][0] if isinstance(d["image_path"], (list, tuple)) else d["image_path"]

        fine_am = d["anomaly_map"]
        if fine_am is None or path not in coarse_mapdict:
            fused.append(d)
            continue

        # -----------------------------------
        # 1) Convert fine map
        # -----------------------------------
        A_f = _to_tensor_map(fine_am)

        if A_f.ndim == 3:
            A_f = A_f.squeeze(0)

        # -----------------------------------
        # 2) Get / resize coarse map
        # -----------------------------------
        coarse_t = coarse_mapdict[path]

        if coarse_t.ndim == 3:
            coarse_t = coarse_t.squeeze(0)

        if coarse_t.shape[-2:] != (out_size, out_size):
            coarse_up = F.interpolate(
                coarse_t.unsqueeze(0).unsqueeze(0),
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False
            )[0, 0]
        else:
            coarse_up = coarse_t

        A_c = coarse_up.float()

        if A_f.shape[-2:] != (out_size, out_size):
            A_f = F.interpolate(
                A_f.unsqueeze(0).unsqueeze(0),
                size=(out_size, out_size),
                mode="bilinear",
                align_corners=False
            )[0, 0]

        # -----------------------------------
        # 3) Optional smoothing
        # -----------------------------------
        A_c_np = A_c.detach().cpu().numpy()
        A_c_np = _gaussian_blur_numpy(A_c_np, sigma=blur_sigma, ksize=5)
        A_c_s = torch.tensor(A_c_np, dtype=torch.float32)

        A_f_np = A_f.detach().cpu().numpy()
        if fine_blur_sigma > 0:
            A_f_np = _gaussian_blur_numpy(A_f_np, sigma=fine_blur_sigma, ksize=5)
        A_f_s = torch.tensor(A_f_np, dtype=torch.float32)

        # -----------------------------------
        # 4) Fine support mask for suppressing coarse
        # -----------------------------------
        if Tfine is None:
            tf = torch.quantile(A_f_s.flatten(), q_fine_protect)
        else:
            tf = torch.tensor(Tfine, dtype=torch.float32)

        if use_binary_fine_support:
            fine_support = (A_f_s > tf).detach().cpu().numpy().astype(np.uint8)
            fine_support = _dilate_mask_numpy(
                fine_support,
                kernel_size=dilate_kernel,
                iterations=dilate_iter
            )
            fine_support_t = torch.tensor(fine_support, dtype=torch.float32)
        else:
            support_np = (A_f_s / (A_f_s.max() + 1e-8)).detach().cpu().numpy()
            support_np = cv2.GaussianBlur(support_np, (5, 5), 1.0)
            fine_support_t = torch.tensor(np.clip(support_np, 0.0, 1.0), dtype=torch.float32)

        # -----------------------------------
        # 5) Soft suppression of coarse by fine support
        # -----------------------------------
        W = beta + (1.0 - beta) * fine_support_t
        A_c_supp = A_c * W

        # -----------------------------------
        # 6) Fine-only rescue path
        # -----------------------------------
        A_f_rescue = torch.zeros_like(A_f)

        if enable_fine_rescue:
            # strong fine threshold
            if Tfine_rescue is None:
                tf_rescue = torch.quantile(A_f_s.flatten(), q_fine_rescue)
            else:
                tf_rescue = torch.tensor(Tfine_rescue, dtype=torch.float32)

            # weak coarse threshold
            if Tcoarse is None:
                tc_weak = torch.quantile(A_c_s.flatten(), q_coarse_weak)
            else:
                tc_weak = torch.tensor(Tcoarse, dtype=torch.float32)

            fine_strong = (A_f_s > tf_rescue).detach().cpu().numpy().astype(np.uint8)
            coarse_weak = (A_c_s < tc_weak).detach().cpu().numpy().astype(np.uint8)

            rescue_mask = fine_strong * coarse_weak

            # allow small spatial tolerance
            rescue_mask = _dilate_mask_numpy(
                rescue_mask,
                kernel_size=rescue_dilate_kernel,
                iterations=rescue_dilate_iter
            )

            # remove tiny rescued speckles
            rescue_mask = _remove_small_components(rescue_mask, min_area=rescue_min_area)

            rescue_mask_t = torch.tensor(rescue_mask, dtype=torch.float32)

            # boosted fine in rescue regions
            A_f_rescue = A_f * (1.0 + gamma * rescue_mask_t)

        # -----------------------------------
        # 7) Optional mild fine residual path
        # -----------------------------------
        A_f_res = torch.zeros_like(A_f)
        if use_residual:
            # only keep fine content that rises above smoothed coarse
            A_f_res = torch.relu(A_f - A_c_s)

        # -----------------------------------
        # 8) Final fusion
        # -----------------------------------
        A_out = torch.maximum(A_c_supp, A_f_rescue)
        if use_residual:
            A_out = torch.maximum(A_out, A_f_res)

        # -----------------------------------
        # 9) Optional visualization
        # -----------------------------------
        if show_suppression:
            removed = (A_c - A_c_supp).clamp_min(0)

            plt.figure(figsize=(12, 4))

            plt.subplot(1, 3, 1)
            plt.imshow(A_c.detach().cpu().numpy(), cmap="inferno")
            plt.title("Original Coarse")
            plt.axis("off")

            plt.subplot(1, 3, 2)
            plt.imshow(removed.detach().cpu().numpy(), cmap="inferno")
            plt.title("Suppressed Coarse")
            plt.axis("off")

            plt.subplot(1, 3, 3)
            plt.imshow(A_out.detach().cpu().numpy(), cmap="inferno")
            plt.title("Final Fused")
            plt.axis("off")

            plt.tight_layout()
            plt.show()

        # -----------------------------------
        # 10) Preserve output format
        # -----------------------------------
        new_d = dict(d)
        new_d["anomaly_map"] = A_out.detach().cpu().numpy()

        if update_pred_mask:
            new_d["pred_mask"] = (A_out > threshold).detach().cpu().numpy()

        fused.append(new_d)

    return fused