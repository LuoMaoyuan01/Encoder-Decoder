import glob
import re
import matplotlib.pyplot as plt
from anomalib.engine import Engine
from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.metrics import AUROC, F1Score, Evaluator, AUPR
from anomalib.models.image.dsr import Dsr

# ------------------------ LOAD CONFIGS --------------------------------
DATA_ROOT = "data/configA"
CHECKPOINT_PATH = "checkpoints"
OUTPUT_DIR = "output/test"
BATCH_SIZE = 8

# Metrics for TEST
# AUROC uses continuous anomaly score
auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
aupr  = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
# F1 uses binary pred_label (depends on thresholding/post-processing inside model)
f1 = F1Score(fields=["pred_label", "gt_label"], prefix="image_")

evaluator = Evaluator(test_metrics=[auroc, aupr])

engine = Engine(accelerator="gpu", devices=1, logger=True, precision="16-mixed", default_root_dir=OUTPUT_DIR, log_every_n_steps=5)

# Sort checkpoints by epoch number
def extract_epoch(path):
    # Extract number after "epoch="
    return int(re.search(r"epoch=(\d+)", path).group(1))
ckpts = sorted(glob.glob(f"{CHECKPOINT_PATH}/dsr-semicon-configA-epoch*.ckpt"))
ckpts_sorted = sorted(ckpts, key=extract_epoch)


# Test Checkpoints to find checkop
dm_test = Folder(
    name="semicon_dsr_test",
    root=DATA_ROOT,
    normal_dir="test/normal",
    normal_test_dir="test/normal",
    abnormal_dir="test/anomalous",
    test_split_mode=TestSplitMode.FROM_DIR,
    val_split_mode=ValSplitMode.NONE,
    eval_batch_size=BATCH_SIZE,
    num_workers=0,
)

auroc = AUROC(fields=["pred_score", "gt_label"], prefix="image_")
aupr  = AUPR(fields=["pred_score", "gt_label"], prefix="image_")
f1 = F1Score(fields=["pred_label", "gt_label"], prefix="image_")

def evaluate_checkpoints(ckpts, dm_test, engine, evaluator):
    results = []

    for ckpt in ckpts:
        print(f"Evaluating {ckpt}")

        model = Dsr.load_from_checkpoint(ckpt, evaluator=evaluator)
        test_metrics = engine.test(model=model, datamodule=dm_test)[0]

        auroc = test_metrics.get("image_AUROC", None)
        aupr  = test_metrics.get("image_AUPR", None)

        results.append({
            "ckpt": ckpt,
            "AUROC": auroc,
            "AUPR": aupr
        })

        print(f"AUROC={auroc:.4f}, AUPR={aupr:.4f}")
    
    return results

# Find the best checkpoint
def best_checkpoint(results):
    best_ckpt_auroc = None
    best_ckpt_aupr = None
    for result in results:
        if result["AUROC"] == max(r["AUROC"] for r in results):
            best_ckpt_auroc = result
        if result["AUPR"] == max(r["AUPR"] for r in results):
            best_ckpt_aupr = result

    print("\nBest Checkpoint by AUROC:")
    print(f"Checkpoint: {best_ckpt_auroc}") 
    print("\nBest Checkpoint by AUPR:")
    print(f"Checkpoint: {best_ckpt_aupr}") 

def plot_results(results):
    # Sort results by epoch
    results_sorted = sorted(results, key=lambda x: extract_epoch(x["ckpt"]))

    epochs = [extract_epoch(r["ckpt"]) for r in results_sorted]
    aurocs = [r["AUROC"] for r in results_sorted]
    auprs  = [r["AUPR"] for r in results_sorted]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, aurocs, marker="o", label="AUROC")
    plt.plot(epochs, auprs, marker="x", label="AUPR")

    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("DSR Performance vs Training Epoch")
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    print(ckpts)
    results = evaluate_checkpoints(ckpts_sorted, dm_test, engine, evaluator)
    best_checkpoint(results)
    plot_results(results)

    # dm_test.setup()
    # dl = dm_test.test_dataloader()
    # print("num test batches:", len(dl))
    # print("estimated num test images:", len(dl.dataset))
