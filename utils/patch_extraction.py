import os
import math
import cv2
from pathlib import Path

SRC_ROOT = Path("data/configA")
DST_ROOT = Path("data/configA_p4_stride50")

PATCH = 256
STRIDE = 128 

def iter_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in folder.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

def save_patches_stride(
    src_path: Path,
    dst_dir: Path,
    patch_size: int = 256,
    stride: int = 128,
    require_exact_grid: bool = True,
):
    """
    Saves patches using sliding window (patch_size, stride).
    Names patches by grid location: g{row}{col}
      - stride=256 => 2x2 => g00..g11
      - stride=128 => 3x3 => g00..g22

    If require_exact_grid=True, enforces that the grid lands exactly on the bottom/right edge.
    """
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0

    H, W = img.shape
    dst_dir.mkdir(parents=True, exist_ok=True)

    if require_exact_grid:
        # Ensure last patch lands exactly at H-patch_size and W-patch_size
        if (H - patch_size) % stride != 0 or (W - patch_size) % stride != 0:
            raise ValueError(
                f"Stride {stride} doesn't tile image {H}x{W} with patch {patch_size}. "
                f"(H-patch)%stride={(H-patch_size)%stride}, (W-patch)%stride={(W-patch_size)%stride}"
            )

    n_rows = (H - patch_size) // stride + 1
    n_cols = (W - patch_size) // stride + 1

    count = 0
    for r in range(n_rows):
        y = r * stride
        for c in range(n_cols):
            x = c * stride
            patch = img[y:y+patch_size, x:x+patch_size]

            out_name = f"{src_path.stem}_g{r}{c}{src_path.suffix}"
            cv2.imwrite(str(dst_dir / out_name), patch)
            count += 1

    return count

def process_split(split_rel: str):
    src_dir = SRC_ROOT / split_rel
    dst_dir = DST_ROOT / split_rel
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in iter_images(src_dir):
        total += save_patches_stride(img_path, dst_dir, patch_size=PATCH, stride=STRIDE)

    print(f"{split_rel}: wrote {total} patches")

print("SRC_ROOT:", SRC_ROOT.resolve(), SRC_ROOT.exists())
print("train/normal exists:", (SRC_ROOT/"train/normal").exists())
print("test/normal exists:", (SRC_ROOT/"test/normal").exists())
print("test/anomalous exists:", (SRC_ROOT/"test/anomalous").exists())

process_split("train/normal")
process_split("test/normal")
process_split("test/anomalous")
