# Mission: Entry point huấn luyện RGB Soft Voting Model.
# Author: Lê Văn Hoàn
# Version: 1.0
# Chạy: python run_rgb.py --epochs 30 --batch-size 32 --lr 1e-4

import torch
import torch.nn as nn
from torch.optim import Adam
import os, csv, json, argparse, wandb, numpy as np
from sklearn.metrics import confusion_matrix

from configs.default_configs import Config
from src.datasets.breast_cancer_dataframe import build_metadata_csv, get_pos_weight
from src.datasets.split_dataset import main as split_main
from src.datasets.breast_cancer_dataloader_rgb import create_dataloader_rgb
from src.utils.soft_voting import get_soft_voting_preds
from src.models.rgb_soft_voting import RGBSoftVotingModel
from src.training.train_rgb import Train_model_RGB
from src.utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description="Train RGB Soft Voting Model")

    parser.add_argument(
        "--build-data", action="store_true", help="Rebuild metadata CSV before training"
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--epochs", type=int, default=30, help="Number of training epochs"
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="Breast-Cancer-IDC",
        help="WandB project name for logging",
    )
    return parser.parse_args()


def evaluate_metrics_rgb(model, dataloader, device):
    """
    Đánh giá RGBSoftVotingModel trên dataloader RGB.
    Nhận batch["image_r"], batch["image_g"], batch["image_b"].
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            image_r = batch["image_r"].to(device)
            image_g = batch["image_g"].to(device)
            image_b = batch["image_b"].to(device)
            labels = batch["label"].to(device)

            logit_r, logit_g, logit_b = model(image_r, image_g, image_b)
            preds, _ = get_soft_voting_preds(logit_r, logit_g, logit_b, threshold=0.6)

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

    y_pred = torch.cat(all_preds).numpy().flatten()
    y_true = torch.cat(all_labels).numpy().flatten()

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
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm,
    }


def main(args):
    MODEL_NAME = "rgb_soft_voting"
    set_seed(Config.SEED)

    # === KHỞI TẠO WANDB ===
    if Config.Turn_WandB_Off:
        print("[INFO] WandB logging is turned OFF.")
    else:
        wandb.init(
            project=args.wandb_project,
            name=f"{MODEL_NAME}_epochs{args.epochs}_bs{args.batch_size}_lr{args.lr}",
            config=vars(args),
        )

    # ===== Results directories & log file =====
    RESULTS_DIR = Config.get_exp_dir(MODEL_NAME)
    LOG_DIR = os.path.join(RESULTS_DIR, "logs")
    CKPT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    LOG_PATH = os.path.join(LOG_DIR, "train_log.csv")
    BEST_MODEL_PATH = os.path.join(CKPT_DIR, "best.pth")
    LAST_MODEL_PATH = os.path.join(CKPT_DIR, "last.pth")

    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "train_loss_r",
                    "train_loss_g",
                    "train_loss_b",
                    "train_accuracy",
                    "train_precision",
                    "train_recall",
                    "train_f1",
                    "val_loss",
                    "val_f1",
                    "val_accuracy",
                    "val_recall",
                    "val_precision",
                ]
            )

    # ===== DataLoader RGB =====
    train_loader = create_dataloader_rgb(
        dataframe_csv_path=Config.TRAIN_CSV,
        split="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.TRAIN_SHUFFLE,
        drop_last=Config.TRAIN_DROP_LAST,
    )
    val_loader = create_dataloader_rgb(
        dataframe_csv_path=Config.VAL_CSV,
        split="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.VAL_SHUFFLE,
        drop_last=Config.VAL_DROP_LAST,
    )
    test_loader = create_dataloader_rgb(
        dataframe_csv_path=Config.TEST_CSV,
        split="test",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.TEST_SHUFFLE,
        drop_last=Config.TEST_DROP_LAST,
    )

    # ===== Model =====
    model = RGBSoftVotingModel().to(Config.DEVICE)

    # ===== Pos_weight =====
    pos_weight, pos_weight_value = get_pos_weight(Config.TRAIN_CSV, Config.DEVICE)
    print(f"[INFO] Using pos_weight = {pos_weight_value:.4f}")
    if not Config.Turn_WandB_Off:
        wandb.config.update({"pos_weight": float(pos_weight_value)})

    # ===== Loss & Optimizer =====
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=Config.BASE_LR)

    # ===== Trainer RGB =====
    trainer = Train_model_RGB(
        model, optimizer, criterion, Config.DEVICE, scheduler=None
    )
    best_epoch_num = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience= None,  # Không dùng Early Stopping
        log_path=LOG_PATH,
        best_model_path=BEST_MODEL_PATH,
        last_model_path=LAST_MODEL_PATH,
        model_name=MODEL_NAME,
    )

    # ===== Load best model and evaluate =====
    print("[INFO] Bắt đầu quá trình Đánh giá trên tập Test...")
    print("[INFO] Loading best model for evaluation...")
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    metrics = evaluate_metrics_rgb(
        model=model, dataloader=test_loader, device=Config.DEVICE
    )

    print("[INFO] Evaluation results:")
    for k, v in metrics.items():
        if k != "confusion_matrix":
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    if not Config.Turn_WandB_Off:
        wandb.log(
            {
                "Test Accuracy": metrics["accuracy"],
                "Test Recall": metrics["recall"],
                "Test Precision": metrics["precision"],
                "Test F1": metrics["f1"],
            }
        )

    metrics_to_save = {
        "model": MODEL_NAME,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.BASE_LR,
        "pos_weight": float(pos_weight_value),
        "best_epoch": best_epoch_num,
        "accuracy": metrics["accuracy"],
        "recall": metrics["recall"],
        "precision": metrics["precision"],
        "f1": metrics["f1"],
        "confusion_matrix": metrics["confusion_matrix"],
    }
    if not Config.Turn_WandB_Off:
        wandb.finish()

    METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)
    print(f"[INFO] Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    args = parse_args()
    Config.update_from_args(args)

    if args.build_data:
        print("[INFO] Building metadata CSV...")
        build_metadata_csv(Config.DATA_ROOT, Config.METADATA_CSV)
        split_main()
        print("[INFO] Metadata CSV built successfully.")
        exit(0)

    print(f"[INFO] Training model: rgb_soft_voting")
    print("[INFO] Starting training...")
    main(args)
