import torch
import yaml
import os
import numpy as np
from PIL import Image
from utils.mask_augmentation import *
from utils.visualize import *
from anomalib.models import Draem
from anomalib.models.image import Dsr
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.metrics import F1Score, AUROC, Evaluator, AUPR


torch.set_float32_matmul_precision("high")

# ------------------------ LOAD CONFIGS --------------------------------
# load variables from config file into the environment
with open("config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

DATA_ROOT = "data"
MODEL_PATH = cfg.get("model", "weights/dsr.ckpt")  # path to the trained model checkpoint
INPUT_DIR = cfg.get("test_path", "data/test/anomalous")  # path to the test images
OUTPUT_DIR = cfg.get("test_output_path", "output/test")  # path to save test outputs
THRESHOLD = cfg.get("threshold", 0.99)  # global threshold for mask post-processing
BATCH_SIZE = cfg.get("batch_size", 8)
ALPHA = cfg.get("overlay_alpha", 0.5)  # alpha for overlay heatmap

# ---- Ensure GPU Used When Available ----
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set to the first GPU if available

# Configure Evaluator
f1_score = F1Score(fields=["pred_label", "gt_label"])
auroc = AUROC(fields=["pred_score", "gt_label"])
aupr = AUPR(fields=["pred_score", "gt_label"])
evaluator = Evaluator(test_metrics=[f1_score, auroc, aupr])

# --- load trained model ---
model = Dsr.load_from_checkpoint(
    MODEL_PATH,
    evaluator=evaluator,
)

# ---------------- DATAMODULE ----------------
# Keep transforms simple (no normalization) to satisfy DSR requirements.
dm = Folder(
    name="semicon_dsr",
    root=DATA_ROOT,
    normal_dir="train/normal",
    normal_test_dir="test/normal",
    abnormal_dir="test/anomalous",
    # batch sizes / workers
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    test_split_mode=TestSplitMode.FROM_DIR,
    val_split_mode=ValSplitMode.FROM_TEST,
    val_split_ratio=0.4,
)

# --- create engine for testing ---
engine = Engine(
    logger = True,  # enable logging
    default_root_dir=OUTPUT_DIR,  # directory to save outputs
    accelerator="gpu",  # use available accelerator (GPU/CPU)
    log_every_n_steps=10,
)

# --- Image Post Processing Controller Function ---
def image_postprocess_controller(preds):
    # Iterate through predictions
    for pred in preds:
        PRED_IMAGE_PATH = coerce_path(pred.image_path) # path to the image
        PRED_SCORE = pred.pred_score # image-level anomaly score (scalar)
        ANOMALY_MAP = pred.anomaly_map # pixel-wise anomaly heatmap

        # 1) Post-process (normalize->threshold->3x3 dilate)
        # Resize heatmap to 512x512 here
        ANOMALY_MAP = torch.nn.functional.interpolate(
            ANOMALY_MAP.unsqueeze(0), size=(512, 512), mode="bilinear", align_corners=False
        ).squeeze(0)

        mask = postprocess_heatmap(ANOMALY_MAP, thresh=THRESHOLD, normalize=True, require_any=True)

        # 2) Save mask
        os.makedirs(os.path.join(OUTPUT_DIR, "masks"), exist_ok=True) # Ensure mask directory exists
        base = os.path.splitext(os.path.basename(PRED_IMAGE_PATH))[0]
        save_mask_png(mask, os.path.join(OUTPUT_DIR, "masks", f"{base}_mask.png"))

        # 3) (Optional) Save overlay for visual QA
        try:
            img_pil = Image.open(PRED_IMAGE_PATH)
            # Use the normalized heat used for thresholding/overlay
            heat_norm = minmax_norm(ANOMALY_MAP.squeeze(0) if ANOMALY_MAP.dim()==3 and ANOMALY_MAP.size(0)==1 else ANOMALY_MAP)
            os.makedirs(os.path.join(OUTPUT_DIR, "overlays"), exist_ok=True) # Ensure overlay directory
            overlay_heatmap_on_image(img_pil, heat_norm, os.path.join(OUTPUT_DIR, "overlays", f"{base}_overlay.png"), threshold=THRESHOLD, alpha=ALPHA)
        except Exception as e:
            print(f"Overlay failed for {PRED_IMAGE_PATH}: {e}")


# --- run test ---
if __name__ == "__main__":

    # Get predictions
    # preds = engine.predict(model=model, data_path=INPUT_DIR)
    # print(preds[0]) # print first prediction for debugging

    # Visualize prediction scores and distributions
    # visualize_prediction_scores(preds)

    # Compute optimal F1 score over all thresholds
    # obtain_best_f1(preds)

    # Post-process and save masks + overlays
    # image_postprocess_controller(preds)

    # Evaluate model
    test_results = engine.test(model=model, datamodule=dm)
    print(test_results)

    