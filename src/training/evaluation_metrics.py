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
    all_probs = [] # Cần mảng này cho ROC-AUC
    
    with torch.no_grad():
        for batch in dataloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            logits = model(images)              # (B,) or (B,1)
            probs = torch.sigmoid(logits)       # (0,1)
            preds = (probs >= 0.5).long()       # threshold = 0.5

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu())

    # ===== concat all batches =====
    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_labels).numpy().flatten()
    y_probs = torch.cat(all_probs).numpy().flatten() # Dùng cho ROC-AUC

    # ===== compute metrics =====
    TP = np.sum((y_pred == 1) & (y_true == 1))
    TN = np.sum((y_pred == 0) & (y_true == 0))
    FP = np.sum((y_pred == 1) & (y_true == 0))
    FN = np.sum((y_pred == 0) & (y_true == 1))

    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
    precision = TP / (TP + FP + 1e-8)
    recall = TP / (TP + FN + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    # Tính ROC-AUC và Confusion Matrix
    try:
        auc = roc_auc_score(y_true, y_probs)
    except ValueError:
        auc = 0.0 # Xử lý trường hợp batch chỉ có 1 class
        
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "auc": auc,
        "confusion_matrix": cm
    }
