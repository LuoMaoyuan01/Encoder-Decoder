import os
from pathlib import Path
import numpy as np
import torch
import yaml
from anomalib.engine import Engine
from anomalib.callbacks import ModelCheckpoint, GraphLogger
from anomalib.loggers import AnomalibTensorBoardLogger
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.models.image import Dsr

# ------------------------ LOAD CONFIGS --------------------------------
# load variables from config file into the environment
with open("config.yaml", "r") as f:   # adjust path if config.yaml is elsewhere
    cfg = yaml.safe_load(f)

DATA_ROOT = "data/configA"
TRAINING_PATH = cfg.get("training_path", "data/train/normal")  # path to the training images
# TRAINING_OUTPUT_PATH = cfg.get("training_output_path", "output/training")  # path to save training outputs
CHECKPOINT_PATH = cfg.get("checkpoint_path", "checkpoints")  # path to save model checkpoints
MODEL = cfg.get("model", "weights/dsr.ckpt")  # path to the trained model checkpoint
EPOCHS = cfg.get("epochs", 90)
BATCH_SIZE = cfg.get("batch_size", 8)
LEARNING_RATE = cfg.get("learning_rate", 0.00005)
IMAGE_SIZE = cfg.get("image_size", 512)


# ---------------- DATAMODULE ----------------
# Keep transforms simple (no normalization) to satisfy DSR requirements.
dm = Folder(
    name="semicon_dsr_configA",
    root=DATA_ROOT,
    normal_dir="train/normal",
    normal_test_dir="test/normal",
    abnormal_dir="test/anomalous",
    # normal-only validation split from training normals
    val_split_mode=ValSplitMode.FROM_TRAIN,
    val_split_ratio=0.15,   # 10% of training images used for val
    train_batch_size=BATCH_SIZE,
    eval_batch_size=BATCH_SIZE,
    num_workers=2,
)

# --------- Checkpoint callback ----------------
checkpoint_cb = ModelCheckpoint(
    dirpath=CHECKPOINT_PATH,       # folder where ckpts are saved
    filename="dsr-semicon-configA-{epoch:02d}",  # name pattern
    monitor="train_loss_epoch",      # metric to track (from evaluator logs)
    mode="min",                      # higher AUROC is better
    save_top_k=-1,                    # save all checkpoint
    save_last=True,                  # ALSO save last.ckpt
    every_n_epochs=1,               # save every 5 epochs
)

# dm.setup()
# print("Train loader:", dm.train_dataloader() is not None)
# print("Val loader:", dm.val_dataloader() is not None)

# ---------------- MODEL ----------------

model = Dsr(evaluator=False)

# ---------------- TRAINER / ENGINE ----------------
logger = AnomalibTensorBoardLogger(save_dir="logs/", name="dsr-semicon-configA")

engine = Engine(
    accelerator="gpu",
    devices=1,
    precision="16-mixed",
    max_epochs=EPOCHS,
    logger=logger,
    log_every_n_steps=20,
    callbacks=[checkpoint_cb, GraphLogger()],
)

# ---------------- RUN TRAINING ----------------
if __name__ == "__main__":

    # print(torch.version.cuda)
    # print(torch.cuda.is_available())
    # print(torch.cuda.get_device_name(0))

    # Train + Validate
    # ---- Ensure GPU Used When Available ----
    if torch.cuda.is_available():
        # torch.cuda.set_device(0)  # Set to the first GPU if available
        torch.set_float32_matmul_precision("medium")
        torch.backends.cudnn.benchmark = True

    # Clear CUDA cache before training
    import gc, torch
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    engine.fit(model=model, datamodule=dm)

    # Evaluate best checkpoint
    # engine.test(model=model, datamodule=dm, ckpt_path=MODEL)
