import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import glob, os
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score
from collections import defaultdict

def visualize_prediction_scores(predictions):
    "Visualize prediction scores as a histogram."

    image_scores = []
    image_labels = []
    image_paths = []

    for pred in predictions:
        # adjust keys if needed depending on your anomalib version
        image_scores.append(torch.nan_to_num(pred["pred_score"], nan=0.0)) # Prevent nan score issue
        image_labels.append(pred["pred_label"])
        image_paths.append(pred["image_path"])
    
    image_scores = np.array(image_scores)
    image_labels = np.array(image_labels)  # 0 = normal, 1 = anomalous

    # Obtain scores for normal and anomalous images as predicted by model
    normal_scores   = image_scores[image_labels == 0]
    anomaly_scores  = image_scores[image_labels == 1]

    # Visualize as histogram, the distributions of anomaly scores for normal and anomalous images
    plt.figure(figsize=(8, 5))
    plt.hist(normal_scores,  bins=30, alpha=0.6, label="Normal",  density=True)
    plt.hist(anomaly_scores, bins=30, alpha=0.6, label="Anomalous", density=True)
    plt.xlabel("Anomaly score")
    plt.ylabel("Density")
    plt.title("Distribution of anomaly scores")
    plt.legend()
    plt.tight_layout()
    plt.show()

def obtain_best_f1(predictions):
    "Try all thresholds between 0 and 1 (inclusive) and compute the F1 for each. Return the best F1 score obtained."
    image_scores = []
    image_labels = []
    image_paths = []

    for pred in predictions:
        # adjust keys if needed depending on your anomalib version
        image_scores.append(pred["pred_score"])
        image_labels.append(pred["pred_label"])
        image_paths.append(pred["image_path"])
    
    for thr in np.linspace(0, 1, 101):
        preds = (image_scores >= thr).astype(int)  # classify based on threshold
        f1 = f1_score(image_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    print(f"Best threshold: {best_thr:.3f}, F1 score: {best_f1:.4f}")

def plot_full_anomaly_map(original_img, full_map, title="Full Anomaly Map", percentile=96.0):
    """
    full_map: torch.Tensor of shape (512, 512)
    percentile: try 99, 99.5, 99.8, 99.9
    """
    fm = full_map.detach().cpu().numpy().astype(np.float32)

    # Optional smoothing (makes thresholding more stable)
    fm_s = cv2.GaussianBlur(fm, (5,5), 0)

    # Percentile threshold on ORIGINAL scale
    p = np.percentile(fm_s, percentile)
    mask = (fm_s >= p).astype(np.uint8)

    # Normalize for heatmap visualization only
    disp = np.clip(fm_s / (p + 1e-8), 0, 1)

    # Plot heatmap + mask
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(disp, cmap="jet")
    ax[0].set_title(f"Heatmap (normalized by p{percentile})")
    ax[0].axis("off")

    ax[1].imshow(mask, cmap="gray")
    ax[1].set_title(f"Binary mask (>{percentile}th percentile)")
    ax[1].axis("off")

    ax[2].imshow(original_img, cmap="jet")
    ax[2].imshow(mask, cmap="winter", alpha=0.2)
    ax[2].set_title("Overlay")
    ax[2].axis("off")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

    return mask

# -------------------------- GENERAL HELPER FUNCTIONS -------------------------- #
def fetch(pred, key):
    # Try dict-style first
    if hasattr(pred, "__getitem__"):
        try:
            return pred[key]
        except Exception:
            pass
    # Then attribute-style
    return getattr(pred, key, None)

def norm_image_path(p):
    ip = fetch(p, "image_path")
    return ip[0] if isinstance(ip, (list, tuple)) else ip

def get_patch_idx_from_pred(p):
    ex = fetch(p, "explanation")
    # ex can be ["p0"] or "p0"
    if isinstance(ex, (list, tuple)) and len(ex) > 0:
        ex = ex[0]
    if isinstance(ex, str) and ex.startswith("p"):
        return int(ex[1:])
    return 0  # fallback

def to_scalar(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        x = x.detach().cpu()
        return float(x.item()) if x.numel() == 1 else float(x.flatten()[0].item())
    if isinstance(x, (list, tuple, np.ndarray)):
        # if it's a single-element list/array
        if len(x) == 0:
            return None
        return float(x[0])
    return float(x)

def get_image_key(pred):
    ip = pred["image_path"]
    return ip[0] if isinstance(ip, (list, tuple)) else ip

def get_image_label(pred):
    # Image-level label: True=anomalous, False=normal
    gt = pred["gt_label"]
    if torch.is_tensor(gt):
        return int(bool(gt.detach().cpu().item()))
    return int(bool(gt))

def get_image_path_str(p):
    ip = fetch(p, "image_path")
    if isinstance(ip, (list, tuple)) and len(ip) > 0:
        return ip[0]
    return ip

def _to_numpy(x):
    if x is None: return None
    if torch.is_tensor(x): return x.detach().cpu().numpy()
    return np.array(x)

def _unwrap_first(x):
    if x is None: return None
    if x.ndim == 4: return x[0]  # (B,C,H,W)->(C,H,W)
    if x.ndim == 3 and x.shape[0] != x.shape[1] and x.shape[0] != x.shape[2]:
        return x[0]              # (B,H,W)->(H,W)
    return x

# -------------------------- STITCHING HELPER FUNCTIONS -------------------------- #

def stitch_4_by_order(patch_preds_4, H=512, W=512, patch=256):
    # patch_preds_4 is length 4 in TL,TR,BL,BR order
    # stitch image (CHW)
    img0 = _unwrap_first(_to_numpy(fetch(patch_preds_4[0], "image")))
    full_img = None
    if img0 is not None:
        if img0.ndim == 2: img0 = img0[None, ...]
        C = img0.shape[0]
        full_img = np.zeros((C, H, W), dtype=img0.dtype)
        for pi, o in enumerate(patch_preds_4):
            gy, gx = divmod(pi, 2)
            y0, x0 = gy*patch, gx*patch
            im = _unwrap_first(_to_numpy(fetch(o, "image")))
            if im is None: continue
            if im.ndim == 2: im = im[None, ...]
            full_img[:, y0:y0+patch, x0:x0+patch] = im

    # stitch anomaly_map (HW)
    am0 = _unwrap_first(_to_numpy(fetch(patch_preds_4[0], "anomaly_map")))
    full_am = None
    if am0 is not None:
        if am0.ndim == 3 and am0.shape[0] == 1: am0 = am0[0]
        full_am = np.zeros((H, W), dtype=am0.dtype)
        for pi, o in enumerate(patch_preds_4):
            gy, gx = divmod(pi, 2)
            y0, x0 = gy*patch, gx*patch
            am = _unwrap_first(_to_numpy(fetch(o, "anomaly_map")))
            if am is None: continue
            if am.ndim == 3 and am.shape[0] == 1: am = am[0]
            full_am[y0:y0+patch, x0:x0+patch] = am

    # stitch pred_mask (HW)
    pm0 = _unwrap_first(_to_numpy(fetch(patch_preds_4[0], "pred_mask")))
    full_pm = None
    if pm0 is not None:
        if pm0.ndim == 3 and pm0.shape[0] == 1: pm0 = pm0[0]
        full_pm = np.zeros((H, W), dtype=pm0.dtype)
        for pi, o in enumerate(patch_preds_4):
            gy, gx = divmod(pi, 2)
            y0, x0 = gy*patch, gx*patch
            pm = _unwrap_first(_to_numpy(fetch(o, "pred_mask")))
            if pm is None: continue
            if pm.ndim == 3 and pm.shape[0] == 1: pm = pm[0]
            full_pm[y0:y0+patch, x0:x0+patch] = pm

    # image_path: take first
    ip = fetch(patch_preds_4[0], "image_path")
    ip_list = ip if isinstance(ip, (list, tuple)) else [ip]

    # pooled score
    scores = []
    for o in patch_preds_4:
        s = fetch(o, "pred_score")
        if s is None: continue
        if torch.is_tensor(s): s = float(s.detach().cpu())
        scores.append(float(s))
    pooled = max(scores) if scores else None

    return {"image_path": ip_list, "image": full_img, "anomaly_map": full_am, "pred_mask": full_pm, "pred_score": pooled}

def stitch_preds(preds, H=512, W=512, patch=256, expected_patches=4):
    grouped = defaultdict(list)

    for p in preds:
        base = get_image_path_str(p)            # real path
        pi = get_patch_idx_from_pred(p)         # 0..3 from explanation
        grouped[base].append((pi, p))

    stitched = []
    for base, items in grouped.items():
        items.sort(key=lambda t: t[0])
        patch_list = [p for _, p in items]

        if len(patch_list) != expected_patches:
            print(f"[WARN] {base}: expected {expected_patches}, got {len(patch_list)}")

        stitched.append(stitch_4_by_order(patch_list, H=H, W=W, patch=patch))
        stitched[-1]["image_path"] = [base]     # keep original

    return stitched

# -------------------------- SCORING HELPER FUNCTIONS -------------------------- #
def patch_score_from_anomaly_map(pred, mode="q999"):
    """
    Robust patch score from anomaly_map.
    Use q999 instead of max if max saturates/clips.
    """
    am = pred["anomaly_map"]
    if torch.is_tensor(am):
        am = am.detach().cpu()
    else:
        am = torch.tensor(am)

    # am usually shape (1,H,W) or (H,W); normalize to (H,W)
    if am.ndim == 3 and am.shape[0] == 1:
        am = am[0]
    elif am.ndim == 4 and am.shape[0] == 1 and am.shape[1] == 1:
        am = am[0, 0]

    flat = am.flatten()

    if mode == "max":
        return float(flat.max().item())
    if mode == "q999":
        return float(torch.quantile(flat, 0.999).item())
    if mode == "topk_mean":
        k = min(200, flat.numel())
        return float(torch.topk(flat, k).values.mean().item())
    raise ValueError("mode must be one of: max, q999, topk_mean")

def pool_patch_scores(scores, pool="max"):
    """
    Pool patch scores -> one image score.
    For localized defects: max or top2 is best.
    """
    scores = [s for s in scores if s is not None and not np.isnan(s)]
    if len(scores) == 0:
        return None
    scores = sorted(scores, reverse=True)

    if pool == "max":
        return float(scores[0])
    if pool == "top2":
        return float(np.mean(scores[:2])) if len(scores) >= 2 else float(scores[0])
    if pool == "mean":
        return float(np.mean(scores))
    raise ValueError("pool must be one of: max, top2, mean")

def compute_image_metrics_from_patch_preds(
    preds,
    patch_score_mode="q999",   # 'q999' recommended if max saturates
    pool="top2",               # 'max' or 'top2' recommended for 4 patches
):
    # Group patches by original image path
    grouped = defaultdict(list)
    for p in preds:
        grouped[get_image_key(p)].append(p)

    y_true, y_score = [], []
    debug = {}

    for path, patch_list in grouped.items():
        gt = get_image_label(patch_list[0])  # image-level
        patch_scores = [patch_score_from_anomaly_map(pp, mode=patch_score_mode) for pp in patch_list]
        img_score = pool_patch_scores(patch_scores, pool=pool)
        if img_score is None:
            continue

        y_true.append(gt)
        y_score.append(img_score)
        debug[path] = {"gt": gt, "img_score": img_score, "patch_scores": patch_scores}

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Need both normal(0) and anomalous(1) images. Got: {np.unique(y_true)}")

    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)  # positive class = 1 (anomalous)

    return auroc, aupr, debug, y_true, y_score