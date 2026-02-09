# Mission: Chạy luồng xử lý chính: xây dựng metadata và huấn luyện mô hình UNet.
# Author: Lê Văn Hoàn
# Version: 2.0
import torch
import torch.nn as nn
from torch.optim import Adam
import time
import os
import csv
import json
import argparse
from src.datasets.breast_cancer_dataframe import build_metadata_csv
from src.datasets.split_dataset import main as split_main
from src.models.build_models import build_model
from src.training.train import Train_model
from src.training.evaluation_metrics import evaluate_metrics
from src.datasets.breast_cancer_dataloader import create_dataloader

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-data",
        action="store_true",
        help="Rebuild metadata CSV before training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "attention_unet", "resnet"],
        help="Model to train"
    )
    return parser.parse_args()

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV = "data/metadata/train.csv"
VAL_CSV   = "data/metadata/val.csv"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 20
# DataLoader parameters
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TEST_SHUFFLE = False
TRAIN_DROP_LAST = True
VAL_DROP_LAST = True
# ==================
def load_best_model(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model
# ==================
def main(model_name: str):
    # ===== DataLoader =====
    train_loader = create_dataloader(
        dataframe_csv_path=TRAIN_CSV,
        split="train",
        batch_size=BATCH_SIZE,
        shuffle=TRAIN_SHUFFLE,
        drop_last=TRAIN_DROP_LAST
    )

    val_loader = create_dataloader(
        dataframe_csv_path=VAL_CSV,
        split="val",
        batch_size=BATCH_SIZE,
        shuffle=VAL_SHUFFLE,
        drop_last=VAL_DROP_LAST
    )

    # ===== Model =====
    model = build_model(
        name=model_name,
        in_channels=3
    ).to(DEVICE)

    # ===== Loss & Optimizer =====
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    trainer = Train_model(model, optimizer, criterion, DEVICE)

    # ===== Results directories & log file =====
    RESULTS_DIR = os.path.join("results", model_name)
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
                ["epoch", "train_loss", "val_loss", "epoch_time"]
            )   
    best_val_loss = float("inf")
    training_start_time = time.time()
    # ===== Training loop =====
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        train_loss = trainer.train_one_epoch(train_loader)
        val_loss = trainer.evaluate(val_loader)
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        print(
            f"[{model_name}] "
            f"Epoch {epoch+1} | "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        # ===== CSV log =====
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [epoch + 1, train_loss, val_loss, epoch_time]
            )

        # ===== best checkpoint  =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model_name,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH
            )

        # ===== last checkpoint  =====
        torch.save(
            {
                "epoch": epoch + 1,
                "model": model_name,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            },
            LAST_MODEL_PATH
        )
    print(
        f"[INFO] Training completed in {total_time/60:.2f} minutes."
    )
    # ===== Load best model and evaluate =====
    print("[INFO] Loading best model for evaluation...")

    model = load_best_model(model, BEST_MODEL_PATH, DEVICE)

    metrics = evaluate_metrics(
        model=model,
        dataloader=val_loader,
        device=DEVICE
    )

    print("[INFO] Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    # ===== Save metrics to JSON =====
    METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
    metrics_to_save = {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_to_save, f, indent=4)

    print(f"[INFO] Metrics saved to {METRICS_PATH}")



if __name__ == "__main__":
    args = parse_args()

    if args.build_data:
        print("[INFO] Building metadata CSV...")
        build_metadata_csv()
        split_main()
        print("[INFO] Metadata CSV built successfully.")
        exit(0)   # ⬅ DÒNG QUAN TRỌNG
    #Chọn mô hình để huấn luyện: --model "attention_unet, unet, resnet"
    print(f"[INFO] Training model: {args.model}")
    print("[INFO] Starting training...")
    main(args.model)
