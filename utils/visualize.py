import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score

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