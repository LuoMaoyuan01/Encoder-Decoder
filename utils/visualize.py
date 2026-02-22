import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
import glob, os
import re
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
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    if isinstance(x, np.ndarray) and x.ndim >= 1 and x.shape[0] == 1:
        return x[0]
    return x

def parse_grid_from_explanation(p):
    """
    expects explanation like ["g12"] or "g12"
    returns (row, col) as ints
    """
    expl = fetch(p, "explanation")
    if isinstance(expl, (list, tuple)) and len(expl) > 0:
        expl = expl[0]
    m = re.search(r"g(\d+)(\d+)", str(expl))
    if not m:
        raise ValueError(f"Cannot parse grid id from explanation={expl}. Expected like 'g12'.")
    return int(m.group(1)), int(m.group(2))

# -------------------------- WEIGHT MAP HELPER FUNCTIONS -------------------------- #

def uniform_weight(patch):
    return np.ones((patch, patch), dtype=np.float32)

def linear_weight(patch):
    """
    Triangular / bilinear weighting.
    Center weighted more than edges.
    """
    ramp = np.linspace(0, 1, patch, dtype=np.float32)
    ramp = np.minimum(ramp, ramp[::-1])
    w = np.outer(ramp, ramp)
    w /= (w.max() + 1e-8)
    return w

def gaussian_weight(h: int, w: int, sigma: float = None):
    """
    Create a 2D gaussian weight map normalized to max=1.
    sigma defaults to patch/4 (good starting point).
    """
    if sigma is None:
        sigma = min(h, w) / 4.0

    yy = np.arange(h, dtype=np.float32) - (h - 1) / 2.0
    xx = np.arange(w, dtype=np.float32) - (w - 1) / 2.0
    Y, X = np.meshgrid(yy, xx, indexing="ij")
    W = np.exp(-(X**2 + Y**2) / (2.0 * sigma**2)).astype(np.float32)
    W /= (W.max() + 1e-8)
    return W  # (h,w), max=1

# -------------------------- STITCHING HELPER FUNCTIONS -------------------------- #

def stitch_preds(
    preds,
    H=512, W=512,
    patch=256,
    stride=128,
    mode="gaussian",      # "uniform" | "linear" | "gaussian"
    sigma=None,
    stitch_pred_mask=True,
    vote_threshold=1,   # majority vote in overlap regions
):
    # ---- Select blending weight ----
    if mode == "uniform":
        w2d = np.ones((patch, patch), dtype=np.float32)

    elif mode == "linear":
        ramp = np.linspace(0, 1, patch, dtype=np.float32)
        ramp = np.minimum(ramp, ramp[::-1])
        w2d = np.outer(ramp, ramp)
        w2d /= (w2d.max() + 1e-8)

    elif mode == "gaussian":
        w2d = gaussian_weight(patch, patch, sigma=sigma)

    else:
        raise ValueError("mode must be 'uniform', 'linear', or 'gaussian'")

    w2d_3 = w2d[None, :, :]  # for CHW images

    grouped = defaultdict(list)
    for p in preds:
        grouped[get_image_path_str(p)].append(p)

    stitched = []

    for base, patch_list in grouped.items():
        img_acc = None
        img_wgt = np.zeros((H, W), dtype=np.float32)

        am_acc = np.zeros((H, W), dtype=np.float32)
        am_wgt = np.zeros((H, W), dtype=np.float32)

        # pred_mask voting accumulators (optional)
        pm_acc = np.zeros((H, W), dtype=np.float32) if stitch_pred_mask else None
        pm_cnt = np.zeros((H, W), dtype=np.float32) if stitch_pred_mask else None

        for p in patch_list:
            r, c = parse_grid_from_explanation(p)
            y0, x0 = r * stride, c * stride

            # ---- image ----
            im = _unwrap_first(_to_numpy(fetch(p, "image")))
            if im is not None:
                if im.ndim == 2:
                    im = im[None, ...]
                if img_acc is None:
                    C = im.shape[0]
                    img_acc = np.zeros((C, H, W), dtype=np.float32)

                img_acc[:, y0:y0+patch, x0:x0+patch] += im.astype(np.float32) * w2d_3
                img_wgt[y0:y0+patch, x0:x0+patch] += w2d

            # ---- anomaly_map ----
            am = _unwrap_first(_to_numpy(fetch(p, "anomaly_map")))
            if am is not None:
                if am.ndim == 3 and am.shape[0] == 1:
                    am = am[0]
                am_acc[y0:y0+patch, x0:x0+patch] += am.astype(np.float32) * w2d
                am_wgt[y0:y0+patch, x0:x0+patch] += w2d

            # ---- pred_mask (vote, not gaussian) ----
            if stitch_pred_mask:
                pm = _unwrap_first(_to_numpy(fetch(p, "pred_mask")))
                if pm is not None:
                    if pm.ndim == 3 and pm.shape[0] == 1:
                        pm = pm[0]
                    # vote uses counts (0/1), not gaussian weights
                    pm_acc[y0:y0+patch, x0:x0+patch] += pm.astype(np.float32)
                    pm_cnt[y0:y0+patch, x0:x0+patch] += 1.0

        # ---- finalize ----
        full_img = None
        if img_acc is not None:
            full_img = img_acc / np.clip(img_wgt[None, ...], 1e-6, None)

        full_am = None
        if am_wgt.max() > 0:
            full_am = am_acc / np.clip(am_wgt, 1e-6, None)

        full_pm = None
        if stitch_pred_mask and pm_cnt.max() > 0:
            pm_mean = pm_acc / np.clip(pm_cnt, 1e-6, None)
            full_pm = (pm_mean >= vote_threshold)

        stitched.append({
            "image_path": [base],
            "image": full_img,
            "anomaly_map": full_am,
            "pred_mask": full_pm,
        })

    return stitched

# -------------------------- PATCH SCORING HELPER FUNCTIONS -------------------------- #
def patch_score_from_anomaly_map(pred, mode="q999", method="NoOverlap"):
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
    # --- FIX 1: remove global baseline (handles "everything bright" normals), penalizes even distribution
    if method == "Overlap":
        flat = flat - flat.median()

    # --- FIX 2 (optional but strong): suppress extreme spikes (padding/borders)
    # cap = torch.quantile(flat, 0.99)
    # flat = torch.clamp(flat, max=cap)

    if mode == "max":
        return float(flat.max().item())
    if mode[0] == "q":
        return float(torch.quantile(flat, float(mode[1:])/1000.0).item())
    if mode == "topk_mean":
        k = min(200, flat.numel())
        return float(torch.topk(flat, k).values.mean().item())
    raise ValueError("mode must be one of: max, q999, topk_mean")

def pool_patch_scores(scores, pool="max"):
    """
    Pool patch scores -> one image score.
    For localized defects: max or topn is best.
    """
    scores = [s for s in scores if s is not None and not np.isnan(s)]
    if len(scores) == 0:
        return None
    scores = sorted(scores, reverse=True)

    if pool == "max":
        return float(scores[0])
    if pool[0:3] == "top":
        n = int(pool[3:])
        return float(np.mean(scores[:n])) if len(scores) >= n else float(scores[0])
    if pool == "mean":
        return float(np.mean(scores))
    raise ValueError("pool must be one of: max, top2, mean")

def compute_image_metrics_from_patch_preds(
    preds,
    patch_score_mode="q999",   # 'q999' recommended if max saturates
    pool="top2",               # 'max' or 'top2' recommended for 4 patches
    method="NoOverlap",           # "NoOverlap" or "Overlap" (if patches overlap)
):
    # Group patches by original image path
    grouped = defaultdict(list)
    for p in preds:
        grouped[get_image_key(p)].append(p)

    y_true, y_score = [], []
    debug = {}

    for path, patch_list in grouped.items():
        gt = get_image_label(patch_list[0])  # image-level
        patch_scores = [patch_score_from_anomaly_map(pp, mode=patch_score_mode, method=method) for pp in patch_list]
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

# -------------------------- MULTI SCALE SCORING HELPER FUNCTIONS -------------------------- #

def image_score_from_anomaly_map(am, mode="q999"):
    """
    am: anomaly map (H,W) or (1,H,W) torch/numpy
    mode:
      - "max"
      - "mean"
      - "q99", "q995", "q999"
    """
    # torch -> numpy
    if hasattr(am, "detach"):
        am = am.detach().cpu().numpy()
    else:
        am = np.array(am)

    if am.ndim == 3:   # (1,H,W)
        am = am[0]
    am = am.astype(np.float32)

    if mode == "max":
        return float(am.max())
    if mode == "mean":
        return float(am.mean())

    if mode.startswith("q"):
        q = float(mode[1:]) / 1000.0  # q999 -> 0.999
        return float(np.quantile(am.reshape(-1), q))

    raise ValueError(f"Unknown mode: {mode}")


def compute_image_metrics_from_fused_preds(
    fused_preds,
    score_mode="q999",
    label_from="gt_label",   # "gt_label" or "filename"
    anomalous_token="anomalous",
):
    """
    fused_preds: list of dicts like:
      {'image_path':[path], 'anomaly_map': (512,512), ... , optionally 'gt_label'}
    """
    y_true, y_score = [], []
    debug = {}

    for d in fused_preds:
        path = d["image_path"][0]

        # label
        if label_from == "gt_label" and "gt_label" in d and d["gt_label"] is not None:
            gt = d["gt_label"]
            # handle shapes like [1] or np array
            if hasattr(gt, "__len__") and not isinstance(gt, (str, bytes)):
                gt = int(np.array(gt).reshape(-1)[0])
            else:
                gt = int(gt)
        else:
            # fallback to filename convention
            gt = 1 if anomalous_token in path.lower() else 0

        # score from fused anomaly map
        am = d["anomaly_map"]
        score = image_score_from_anomaly_map(am, mode=score_mode)

        y_true.append(gt)
        y_score.append(score)
        debug[path] = {"gt": gt, "img_score": score}

    y_true = np.array(y_true, dtype=int)
    y_score = np.array(y_score, dtype=float)

    if len(np.unique(y_true)) < 2:
        raise ValueError(f"Need both normal(0) and anomalous(1) images. Got: {np.unique(y_true)}")

    auroc = roc_auc_score(y_true, y_score)
    aupr  = average_precision_score(y_true, y_score)

    return auroc, aupr, debug, y_true, y_score