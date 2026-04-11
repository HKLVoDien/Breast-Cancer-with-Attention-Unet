# Mission: Đánh giá các chỉ số hiệu suất của mô hình.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, confusion_matrix


def evaluate_metrics(model, dataloader, device):
    """
    Evaluate binary classification metrics.
    Returns dict: accuracy, precision, recall, f1
    """

    model.eval()

    all_preds = []
    all_labels = []
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)  # (B,) or (B,1)
            probs = torch.sigmoid(logits)  # (0,1)
            preds = (probs >= 0.6).long()  # threshold = 0.6

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
    # ===== concat all batches =====
    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_labels).numpy().flatten()
    # ===== compute metrics =====
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm,
    }
