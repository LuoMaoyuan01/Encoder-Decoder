import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
import os, glob
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


def fuse_stitched_with_coarse(fine_stitched, coarse_mapdict, out_size=512, threshold=0.49, update_pred_mask=True, beta=0.22, q_coarse = 0.90, q_fine_protect=0.995, show_suppression=False):
    """
    Fuse stitched fine anomaly maps with coarse anomaly maps.

    For each image, the function:

    1) Takes the stitched fine anomaly map (full resolution).
    2) Resizes the corresponding coarse anomaly map to the same resolution if needed.
    3) Applies structure-aware suppression, where strong coarse responses attenuate
       the fine anomaly scores while protecting the strongest fine activations.
    4) Returns fused predictions in the same format as the stitched inputs.

    The output anomaly map preserves defect-sensitive fine responses while reducing
    coarse-induced structural artifacts.
    """
    fused = []

    for d in fine_stitched:
        path = d["image_path"][0]

        # fine anomaly map (numpy HxW) -> torch (1,H,W)
        fine_am = d["anomaly_map"]
        if fine_am is None:
            fused.append(d)
            continue

        fine_t = torch.tensor(fine_am, dtype=torch.float32).unsqueeze(0)  # (1,512,512)

        # coarse anomaly map -> upsample to (1,512,512)
        coarse_t = coarse_mapdict[path]  # (1,256,256) typically
        if coarse_t.ndim == 2:
            coarse_t = coarse_t.unsqueeze(0)

        if coarse_t.shape[-2:] != (out_size, out_size):
            coarse_up = F.interpolate(
                coarse_t.unsqueeze(0), size=(out_size, out_size),
                mode="bilinear", align_corners=False
            )[0]
        else:
            coarse_up = coarse_t

        # --- Structure suppression fusion (fine-protect variant) ---
        use_mask = True

        A_f = fine_t.squeeze(0)     # (512,512)
        A_c = coarse_up.squeeze(0)  # (512,512)

        # --- NEW: define protection & suppression masks ---

        coarse_strong = A_c > torch.quantile(A_c.flatten(), q_coarse)
        fine_strong   = A_f > torch.quantile(A_f.flatten(), q_fine_protect)

        supp_mask = coarse_strong & (~fine_strong)

        # --- Suppression ---
        if use_mask:
            A_out = torch.clamp(A_f - beta * supp_mask.float() * A_c, min=0.0)
        else:
            A_out = torch.clamp(A_f - beta * A_c, min=0.0)

        final_t = A_out.unsqueeze(0)  # back to (1,512,512)

        # Optional debug: what got removed
        if show_suppression:
            delta = (A_f - A_out).clamp_min(0)
            plt.figure(); plt.imshow(delta.detach().cpu(), cmap="inferno")
            plt.colorbar(); plt.title("Suppression Removed (fine - final)")
            plt.axis("off"); plt.show()

        # write back in same format your plotter expects: numpy (H,W)
        new_d = dict(d)
        new_d["anomaly_map"] = final_t.squeeze(0).numpy()

        if update_pred_mask:
            new_d["pred_mask"] = (final_t.squeeze(0) > threshold).numpy()

        fused.append(new_d)
    return fused