import os
import math
from pathlib import Path
import cv2
from pathlib import Path

SRC_ROOT = Path("data/configA")
DST_ROOT = Path("data/configA_p4")

PATCH = 256
STRIDE = 256  # no overlap -> 4 patches for 512x512

def iter_images(folder: Path):
    exts = {".png", ".jpg", ".jpeg", ".bmp"}
    for p in folder.rglob("*"):
        if p.suffix.lower() in exts:
            yield p

def save_n_patches(src_path: Path, dst_dir: Path, n=4):
    img = cv2.imread(str(src_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    h, w = img.shape
    assert h == 512 and w == 512, f"Expected 512x512, got {h}x{w} for {src_path}"

    count = 0
    for gy in range(int(math.sqrt(n))):
        for gx in range(int(math.sqrt(n))):
            y = gy * STRIDE
            x = gx * STRIDE
            patch = img[y:y+PATCH, x:x+PATCH]

            out_name = f"{src_path.stem}_g{gy}{gx}{src_path.suffix}"
            out_path = dst_dir / out_name
            cv2.imwrite(str(out_path), patch)
            count += 1
    return count

def process_split(split_rel: str):
    src_dir = SRC_ROOT / split_rel
    dst_dir = DST_ROOT / split_rel
    dst_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for img_path in iter_images(src_dir):
        total += save_n_patches(img_path, dst_dir, n=4)

    print(f"{split_rel}: wrote {total} patches")

print("SRC_ROOT:", SRC_ROOT.resolve(), SRC_ROOT.exists())
print("train/normal exists:", (SRC_ROOT/"train/normal").exists())
print("test/normal exists:", (SRC_ROOT/"test/normal").exists())
print("test/anomalous exists:", (SRC_ROOT/"test/anomalous").exists())

process_split("train/normal")
process_split("test/normal")
process_split("test/anomalous")
