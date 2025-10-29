import os
from pathlib import Path
import torch
import yaml
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.models.image import Dsr

# ------------------------ LOAD CONFIGS --------------------------------
# load variables from config file into the environment
with open("config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

DATA_ROOT = "data"
TRAINING_PATH = cfg.get("training_path", "data/train/normal")  # path to the training images
TRAINING_OUTPUT_PATH = cfg.get("training_output_path", "output/training")  # path to save training outputs
EPOCHS = cfg.get("epochs", 20)
BATCH_SIZE = cfg.get("batch_size", 8)
LEARNING_RATE = cfg.get("learning_rate", 0.0001)
IMAGE_SIZE = cfg.get("image_size", 512)

# ---- Ensure GPU Used When Available ----
if torch.cuda.is_available():
    torch.cuda.set_device(0)  # Set to the first GPU if available


# ---------------- DATAMODULE ----------------
# Keep transforms simple (no normalization) to satisfy DSR requirements.
dm = Folder(
    name="semicon_dsr",
    root=DATA_ROOT,
    normal_dir="train/normal",
    abnormal_dir="test/anomalous",
    normal_test_dir="test/normal",
    # batch sizes / workers
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    # Let the model's preprocessor handle resizing; if your Folder version
    # exposes augmentations, avoid Normalize().
)

# ---------------- MODEL ----------------
model = Dsr()

# ---------------- TRAINER / ENGINE ----------------
engine = Engine(
    accelerator="gpu" if torch.cuda.is_available() else "cpu",
    devices=1,
    precision="16-mixed" if torch.cuda.is_available() else "32-true",
    max_epochs=EPOCHS,
    default_root_dir=TRAINING_OUTPUT_PATH,
    logger=True,
    log_every_n_steps=10,
)

# ---------------- RUN TRAINING ----------------
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    engine.fit(model=model, datamodule=dm)