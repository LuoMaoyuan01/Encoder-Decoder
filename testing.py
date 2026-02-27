
import os, re, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

from sklearn.metrics import roc_auc_score, average_precision_score

from anomalib.engine import Engine
from anomalib.models.image import Dsr
from anomalib.data.dataclasses import ImageBatch


# ----------------- HELPERS -----------------
def fetch(pred, key):
    if hasattr(pred, "__getitem__"):
        try:
            return pred[key]
        except Exception:
            pass
    return getattr(pred, key, None)

def to_scalar(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return float(x.item()) if x.numel() == 1 else float(x.flatten()[0].item())
    if isinstance(x, (list, tuple, np.ndarray)):
        if len(x) == 0:
            return None
        return float(x[0])
    return float(x)

def get_gt_label_scalar(p):
    gt = fetch(p, "gt_label")
    if gt is None:
        gt = fetch(p, "label")
    if gt is None:
        gt = fetch(p, "target")
    return int(to_scalar(gt)) if gt is not None else None

def get_image_path_str(p):
    ip = fetch(p, "image_path")
    if isinstance(ip, (list, tuple)) and len(ip) > 0:
        return ip[0]
    return ip

def collate_first(batch):
    # dataset returns ImageBatch already
    return batch[0]


# ----------------- COARSE DATASET (returns ImageBatch, NOT dict) -----------------
class CoarsePredictDataset(torch.utils.data.Dataset):
    """
    One FULL 512x512 image per item, returned as ImageBatch.
    Label rule matches your patch dataset: filename contains 'anomalous' -> 1 else 0.
    """
    def __init__(self, img_dir: str, transform=None, H: int = 512, W: int = 512):
        self.img_paths = sorted(glob.glob(os.path.join(img_dir, "*.*")))
        if len(self.img_paths) == 0:
            raise FileNotFoundError(f"No images found in {img_dir}")

        self.transform = transform if transform is not None else T.ToTensor()
        self.H = int(H)
        self.W = int(W)

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx: int):
        path = self.img_paths[idx]

        # Ground-truth label from filename (your convention)
        fname = os.path.basename(path).lower()
        label = 1 if "anomalous" in fname else 0

        pil = Image.open(path).convert("RGB")   # change to "L" if model trained grayscale
        img = self.transform(pil)               # (C,H,W)

        # Safety check
        if img.shape[-2:] != (self.H, self.W):
            raise ValueError(
                f"Expected transformed image to be {self.H}x{self.W}, got {tuple(img.shape[-2:])} for {path}. "
                f"Did you resize during training? If yes, add Resize((512,512)) here."
            )

        return ImageBatch(
            image=img,                                  # FULL image
            gt_mask=None,
            gt_label=torch.tensor([label], dtype=torch.long),
            image_path=[path],
            mask_path=None,
            explanation=["coarse"],
        )


# ----------------- SELF-COMPUTED SCORE FROM ANOMALY MAP -----------------
def anomaly_map_to_score(pred, mode="topk_mean", topk=200, remove_baseline=False):
    """
    Compute your OWN image-level anomaly score from anomaly_map.

    Recommended defaults for your case:
      mode="topk_mean"  (more stable if anomaly_map saturates at 1)
      topk=200

    Other options:
      mode="q999"       (99.9th percentile of pixels)
      mode="max"        (often saturates; not recommended)

    remove_baseline:
      subtract median (ONLY if you see global bright offset in normals)
    """
    am = fetch(pred, "anomaly_map")
    if am is None:
        raise KeyError("Prediction has no 'anomaly_map' field.")

    if torch.is_tensor(am):
        am = am.detach().cpu()
    else:
        am = torch.tensor(am)

    am = am.squeeze()
    if am.ndim != 2:
        raise ValueError(f"Expected anomaly_map 2D after squeeze, got {tuple(am.shape)}")

    flat = am.flatten()

    if remove_baseline:
        flat = flat - flat.median()

    if mode == "max":
        return float(flat.max().item())

    if mode.startswith("q"):
        # "q999" -> 0.999, "q99" -> 0.99
        q_str = mode[1:]
        q = float(q_str) / (1000.0 if len(q_str) >= 3 else 100.0)
        return float(torch.quantile(flat, q).item())

    if mode == "topk_mean":
        k = min(int(topk), flat.numel())
        return float(torch.topk(flat, k).values.mean().item())

    raise ValueError("mode must be one of: max, q999, q99, topk_mean")


# ----------------- METRICS (coarse, uses SELF score) -----------------
def compute_metrics_from_coarse_preds_selfscore(preds, score_mode="topk_mean", topk=200, q=0.99, remove_baseline=False):
    y_true, y_score = [], []
    debug = {}

    for p in preds:
        gt = get_gt_label_scalar(p)
        if gt is None:
            continue

        sc = anomaly_map_to_score(p, mode=score_mode, topk=topk, remove_baseline=remove_baseline)

        y_true.append(gt)
        y_score.append(sc)

        path = get_image_path_str(p)
        if path is not None:
            debug[path] = {"gt": int(gt), "score": float(sc)}

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    if len(y_true) == 0:
        raise ValueError("No usable labels found. Check your dataset label rule.")

    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Need both normal(0) and anomalous(1). Got labels: {np.unique(y_true)}")

    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)
    q99_norm = float(np.quantile(y_score[y_true == 0], q))

    meta = {
        "n_used": int(len(y_true)),
        "score_mode": score_mode,
        "topk": int(topk),
        "remove_baseline": bool(remove_baseline),
        "score_min": float(y_score.min()),
        "score_max": float(y_score.max()),
        "score_mean": float(y_score.mean()),
    }

    return float(auroc), float(aupr), float(q99_norm), y_true, y_score, debug, meta


# ---------------- CONFIG ----------------
img_dir = "data/temp_infer_configA_synthetic/images"   # FULL images directory
CKPT_DIR  = "checkpoints"
RUN_NAME  = "dsr-semicon-configA_coarse512_selfscore"

EVAL_BATCH_SIZE = 1      # keep 1 with collate_first (same as patch pipeline)
Q_NORM = 0.99

# Self-score settings
SCORE_MODE = "topk_mean"   # try: "topk_mean" (best if maps saturate), or "q999"
TOPK = 200
REMOVE_BASELINE = False    # set True only if normals have global bright offset

transform = T.Compose([
    # T.Resize((512, 512)),  # uncomment ONLY if you resized during training
    T.ToTensor(),
])

def parse_epoch(path: str) -> int:
    base = os.path.basename(path)
    m = re.search(r"-([0-9]{2,4})\.ckpt$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch=([0-9]{1,4})", base)
    return int(m.group(1)) if m else -1


# ---------------- ENGINE ----------------
engine = Engine(
    logger=True,
    accelerator="gpu",   # change to "cpu" if needed
    log_every_n_steps=10,
)


# ---------------- MAIN LOOP ----------------
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")), key=parse_epoch)
rows = []

ds = CoarsePredictDataset(img_dir, transform=transform, H=512, W=512)
dl = DataLoader(ds, batch_size=EVAL_BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_first)

# sanity check: ensure the batch is ImageBatch (not dict)
b = next(iter(dl))
print("Sanity batch type:", type(b), "has .image:", hasattr(b, "image"), "shape:", tuple(b.image.shape))

for ckpt in ckpts:
    epoch = parse_epoch(ckpt)
    if epoch < 0:
        continue

    MODEL_PATH = ckpt

    model = Dsr.load_from_checkpoint(
        MODEL_PATH,
        evaluator=False
    )

    preds = engine.predict(model=model, dataloaders=dl, ckpt_path=MODEL_PATH)

    auroc, aupr, q99_norm, y_true, y_score, debug, meta = compute_metrics_from_coarse_preds_selfscore(
        preds,
        score_mode=SCORE_MODE,
        topk=TOPK,
        q=Q_NORM,
        remove_baseline=REMOVE_BASELINE
    )

    rows.append({"epoch": epoch, "q99_norm": q99_norm, "AUROC": auroc, "AUPR": aupr})
    print(
        f"epoch={epoch:03d}  q99_norm={q99_norm:.6f}  AUROC={auroc:.4f}  AUPR={aupr:.4f}  "
        f"score_mode={SCORE_MODE} topk={TOPK} meta={meta}"
    )

df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
out_csv = f"{RUN_NAME}_epoch_metrics.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)


# ---------------- PLOTS ----------------
plt.figure()
plt.plot(df["epoch"], df["q99_norm"])
plt.xlabel("Epoch")
plt.ylabel("q99(normal image score)")
plt.title("Normal stability (q99) vs epoch")
plt.show()

plt.figure()
plt.plot(df["epoch"], df["AUROC"])
plt.xlabel("Epoch")
plt.ylabel("Image AUROC")
plt.title("Image AUROC vs epoch")
plt.ylim(0, 1)
plt.show()

plt.figure()
plt.plot(df["epoch"], df["AUPR"])
plt.xlabel("Epoch")
plt.ylabel("Image AUPR")
plt.title("Image AUPR vs epoch")
plt.ylim(0, 1)
plt.show()

# ---------------- COMBINED OVERLAY PLOT ----------------
fig, ax1 = plt.subplots(figsize=(8, 5))

# Left axis: AUROC + AUPR
line1, = ax1.plot(df["epoch"], df["AUROC"], label="AUROC")
line2, = ax1.plot(df["epoch"], df["AUPR"], label="AUPR")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("AUROC / AUPR")
ax1.set_ylim(0, 1)
ax1.grid(True)

# Right axis: q99_norm
ax2 = ax1.twinx()
line3, = ax2.plot(df["epoch"], df["q99_norm"], linestyle="--", label="q99_norm")
ax2.set_ylabel("q99_norm")

# Combine legends
lines = [line1, line2, line3]
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, loc="best")

plt.title("AUROC, AUPR and q99_norm vs Epoch (Overlay)")
plt.tight_layout()
plt.show()