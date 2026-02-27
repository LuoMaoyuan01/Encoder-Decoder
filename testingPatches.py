import os, re, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from utils.split_into_patches import *
from utils.visualize import *
from sklearn.metrics import roc_auc_score, average_precision_score

from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import ValSplitMode
from anomalib.models.image import Dsr
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# ----------------- YOUR HELPERS (reuse exactly) -----------------
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

def get_image_path_str(p):
    ip = fetch(p, "image_path")
    if isinstance(ip, (list, tuple)) and len(ip) > 0:
        return ip[0]
    return ip

def get_gt_label_scalar(p):
    gt = fetch(p, "gt_label")
    return int(to_scalar(gt)) if gt is not None else None

def get_pred_score_scalar(p):
    # anomalib commonly uses pred_score; fallbacks included
    s = fetch(p, "pred_score")
    if s is None:
        s = fetch(p, "score")
    if s is None:
        s = fetch(p, "image_score")
    return to_scalar(s)

def collate_first(batch):
    return batch[0]

transform = T.Compose([
    # T.Resize((512, 512)),  # uncomment if you resized during training
    T.ToTensor(),
])

# ---------------- CONFIG ----------------
img_dir = "data/temp_infer_configA_synthetic/images"
CKPT_DIR  = "checkpoints"
RUN_NAME  = "dsr-semicon-configA_p4"
EVAL_BATCH_SIZE = 8
Q = 0.99


def parse_epoch(path: str) -> int:
    # expects "...-{epoch:03d}.ckpt" OR "...epoch=123.ckpt"
    base = os.path.basename(path)
    m = re.search(r"-([0-9]{2,4})\.ckpt$", base)
    if m:
        return int(m.group(1))
    m = re.search(r"epoch=([0-9]{1,4})", base)
    return int(m.group(1)) if m else -1


# ---------------- EVAL DATAMODULE ----------------

engine = Engine(
    logger = True,  # enable logging
    accelerator="gpu",  # use available accelerator (GPU/CPU)
    log_every_n_steps=10,
)


# ---------------- MAIN LOOP ----------------
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")), key=parse_epoch)
rows = []

for ckpt in ckpts:
    epoch = parse_epoch(ckpt)
    if epoch < 0:
        continue

    MODEL_PATH = ckpt

    model = Dsr.load_from_checkpoint(
        MODEL_PATH,
        evaluator=False
    )

    ds = PatchifyPredictDataset(img_dir, transform=transform, patch_size=256, stride=256)
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_first)

    preds = engine.predict(model=model, dataloaders=dl, ckpt_path=MODEL_PATH)

    # ---------------- TEST METRICS FROM PATCH PREDICTIONS ----------------
    auroc, aupr, debug, y_true, y_score = compute_image_metrics_from_patch_preds(
        preds,
        patch_score_mode="q999",   # try: "q999" then "topk_mean" then "max"
        pool="top2",               # try: "top2" then "max"
        # method="Overlap"
    )

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    q99_norm, _ = compute_q99_norm_from_patch_preds(
        preds,
        patch_score_mode="q999",
        pool="top2",
        method="NoOverlap",
        q=0.99
    )

    rows.append({"epoch": epoch, "q99_norm": q99_norm, "AUROC": auroc, "AUPR": aupr})
    print(f"epoch={epoch:03d}  q99_norm={q99_norm:.6f}  AUROC={auroc:.4f}  AUPR={aupr:.4f}")

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