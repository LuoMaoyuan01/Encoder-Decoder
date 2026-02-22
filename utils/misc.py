import torch
import torch.nn.functional as F

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


def fuse_stitched_with_coarse(fine_stitched, coarse_mapdict, out_size=512, threshold=0.49, update_pred_mask=True):
    """
    fine_stitched: list of dicts from stitch_preds (anomaly_map is numpy HxW)
    coarse_mapdict: dict path -> torch (1,256,256) or (1,H,W)
    Returns fused list of dicts, same format as fine_stitched.
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

        coarse_up = F.interpolate(
            coarse_t.unsqueeze(0), size=(out_size, out_size),
            mode="bilinear", align_corners=False
        )[0]  # (1,512,512)

        final_t = torch.maximum(fine_t, coarse_up)  # (1,512,512)

        # write back in same format your plotter expects: numpy (H,W)
        new_d = dict(d)
        new_d["anomaly_map"] = final_t.squeeze(0).numpy()

        if update_pred_mask:
            new_d["pred_mask"] = (final_t.squeeze(0) > threshold).numpy()

        fused.append(new_d)

    return fused