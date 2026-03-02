import os, re, glob
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, average_precision_score
from anomalib.engine import Engine
from anomalib.models.image import Dsr
from torch.utils.data import DataLoader
import torchvision.transforms as T

# your existing utils:
# - PatchifyPredictDataset (returns ImageBatch with explanation grid_id etc.)
# - patch_score_from_anomaly_map, pool_patch_scores
# - compute_image_metrics_from_patch_preds, compute_q99_norm_from_patch_preds
from utils.split_into_patches import *
from utils.visualize import *


# ---------------- CONFIG ----------------
img_dir = "data/temp_infer_configA_synthetic/images"
CKPT_DIR = "checkpoints"
RUN_NAME = "dsr-semicon-configA_p128_s64_overlap"

Q = 0.99

PATCH_SIZE = 256
STRIDE = 128                 # 50% overlap
METHOD = "Overlap"          # IMPORTANT for your baseline subtraction logic

PATCH_SCORE_MODE = "q999"   # good default; if saturated then try "topk_mean"
POOL = "top10"              # with many patches, top2 is too spiky; try top5/top10/top20

transform = T.Compose([
    # T.Resize((512, 512)),  # only if you resized during training
    T.ToTensor(),
])

def collate_first(batch):
    return batch[0]

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
    accelerator="gpu",
    log_every_n_steps=10,
)


# ---------------- MAIN LOOP ----------------
ckpts = sorted(glob.glob(os.path.join(CKPT_DIR, "*.ckpt")), key=parse_epoch)
rows = []

# Build dataloader once (same images for all epochs)
ds = PatchifyPredictDataset(
    img_dir,
    transform=transform,
    patch_size=PATCH_SIZE,
    stride=STRIDE,
    require_exact_grid=True
)
dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_first)

for ckpt in ckpts:
    epoch = parse_epoch(ckpt)
    if epoch < 0:
        continue

    MODEL_PATH = ckpt
    model = Dsr.load_from_checkpoint(MODEL_PATH, evaluator=False)

    preds = engine.predict(model=model, dataloaders=dl, ckpt_path=MODEL_PATH)

    # Image-level AUROC/AUPR using patch preds (group -> patch score -> pool)
    auroc, aupr, debug, y_true, y_score = compute_image_metrics_from_patch_preds(
        preds,
        patch_score_mode="q999",  # or "q999" if not saturated
        pool="top2",
        method="Overlap",
    )

    q99_norm, _ = compute_q99_norm_from_patch_preds(
        preds,
        patch_score_mode="q999",
        pool="top2",
        method="Overlap",
        q=0.99,
    )

    rows.append({"epoch": epoch, "q99_norm": q99_norm, "AUROC": auroc, "AUPR": aupr})
    print(f"epoch={epoch:03d}  q99_norm={q99_norm:.6f}  AUROC={auroc:.4f}  AUPR={aupr:.4f}")

df = pd.DataFrame(rows).sort_values("epoch").reset_index(drop=True)
out_csv = f"{RUN_NAME}_epoch_metrics.csv"
df.to_csv(out_csv, index=False)
print("Saved:", out_csv)


# ---------------- PICK BEST EPOCH ----------------
# You have 3 objectives. A practical selection rule is:
# 1) maximize AUPR (most sensitive for imbalanced anomaly datasets)
# 2) tie-break by AUROC
# 3) prefer LOWER q99_norm (more stable normals)
#
# We implement this with a sortable key:
#   (-AUPR, -AUROC, +q99_norm)

best_row = df.sort_values(by=["AUPR", "AUROC", "q99_norm"], ascending=[False, False, True]).iloc[0]
print("\n=== BEST EPOCH (rank by AUPR, then AUROC, then lowest q99_norm) ===")
print(best_row.to_dict())


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

line1, = ax1.plot(df["epoch"], df["AUROC"], label="AUROC")
line2, = ax1.plot(df["epoch"], df["AUPR"], label="AUPR")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("AUROC / AUPR")
ax1.set_ylim(0, 1)
ax1.grid(True)

ax2 = ax1.twinx()
line3, = ax2.plot(df["epoch"], df["q99_norm"], linestyle="--", label="q99_norm")
ax2.set_ylabel("q99_norm")

ax1.legend([line1, line2, line3], ["AUROC", "AUPR", "q99_norm"], loc="best")
plt.title("AUROC, AUPR and q99_norm vs Epoch (Overlay)")
plt.tight_layout()
plt.show()