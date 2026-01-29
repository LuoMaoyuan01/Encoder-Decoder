import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from anomalib.models.image import Dsr

from utils.split_into_patches import split_into_n_patches
from utils.stitch_patches import stitch_n_maps
from utils.visualize import plot_full_anomaly_map

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT_PATH = "checkpoints/last.ckpt"

# Load model
model = Dsr.load_from_checkpoint(CKPT_PATH)
model = model.to(DEVICE).eval()

# ---------- HELPER FUNCTION ----------
def get_field(out, name):
    # dict-like
    if hasattr(out, "__getitem__"):
        try:
            return out[name]
        except Exception:
            pass
    # attribute-like
    return getattr(out, name, None)

# --------- INFERENCE FUNCTION ----------
@torch.no_grad()
def predict_full_image(model, img_1chw):
    patches, coords = split_into_n_patches(img_1chw, patch=128, n=4)
    patches = patches.to(DEVICE)

    out = model(patches)

    patch_maps = get_field(out, "anomaly_map")
    flat = patch_maps.view(patch_maps.shape[0], -1)
    k = max(1, int(0.01 * flat.shape[1]))
    patch_scores = flat.topk(k=k, dim=1).values.mean(dim=1)

    # patch_maps: (16,256,256) -> (16,128,128) since DSR upsamples internally to 256 x 256
    if patch_maps.shape[-1] != 128:
        patch_maps = F.interpolate(
            patch_maps.unsqueeze(1),  # (16,1,256,256)
            size=(128, 128),
            mode="bilinear",
            align_corners=False
        ).squeeze(1)  # (16,128,128)

    # Some builds may name it slightly differently:
    if patch_maps is None:
        patch_maps = get_field(out, "anomaly_maps")
    if patch_scores is None:
        patch_scores = get_field(out, "pred_scores")

    if patch_maps is None or patch_scores is None:
        # Debug: show what's available
        avail = [a for a in dir(out) if "pred" in a or "anomaly" in a]
        raise RuntimeError(f"Could not find anomaly_map/pred_score. Available related attrs: {avail}")

    # patch_maps: (16, H, W) or (16,1,H,W)
    if patch_maps.ndim == 4 and patch_maps.shape[1] == 1:
        patch_maps = patch_maps[:, 0]  # -> (16,H,W)

    patch_scores = patch_scores.detach().cpu().flatten()  # -> (16,)
    patch_maps_cpu = patch_maps.detach().cpu()

    full_map = stitch_n_maps(patch_maps, coords, out_size=512).detach().cpu()

    image_score_max = float(patch_scores.max().item())
    image_score_top3 = float(patch_scores.topk(k=3).values.mean().item())

    return full_map, patch_maps_cpu, patch_scores, image_score_max, image_score_top3


if __name__ == "__main__":
    img_path = "data/temp_infer_configA/images/anomalous1.png"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),   # keep [0,1], no ImageNet normalization
    ])

    img = Image.open(img_path).convert("RGB")
    img_tensor_1chw = transform(img).unsqueeze(0)  # (1,3,512,512)

    full_map, patch_maps, patch_scores, smax, stop3 = predict_full_image(model, img_tensor_1chw)

    # Visualize
    plot_full_anomaly_map(full_map, title=f"Anomaly Map for {img_path}")

    print("Patch scores:", patch_scores.tolist())
    print("Image score (max):", smax)
    print("Image score (top3 mean):", stop3)
