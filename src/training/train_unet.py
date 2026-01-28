# Mission: Huấn luyện mô hình UNet trên dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm
import time
import os
import csv
from src.models.unet  import UNET
from src.datasets.breast_cancer_dataloader import create_dataloader

# ===== CONFIG =====
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TRAIN_CSV = "data/metadata/train.csv"
VAL_CSV   = "data/metadata/val.csv"
BATCH_SIZE = 32
LR = 1e-4
EPOCHS = 3
TRAIN_SHUFFLE = True
VAL_SHUFFLE = False
TEST_SHUFFLE = False
TRAIN_DROP_LAST = True
VAL_DROP_LAST = True
# ===== LOG & CHECKPOINT CONFIG =====
RESULTS_DIR = "results"
CHECKPOINT_DIR = os.path.join(RESULTS_DIR, "checkpoints")
LOG_DIR = os.path.join(RESULTS_DIR, "logs")

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "best_model_unet.pth")
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, "last_model_unet.pth")
LOG_PATH = os.path.join(LOG_DIR, "train_log_unet.csv")
# ==================
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)   # (B,)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


def evaluate(model, loader, criterion):
    model.eval()
    running_loss = 0.0

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(DEVICE)
            labels = batch["label"].float().to(DEVICE)

            outputs = model(images).squeeze(1)
            loss = criterion(outputs, labels)

            running_loss += loss.item()

    return running_loss / len(loader)

#Dùng để debug DataLoader
def debug_dataloader(loader, name="loader"):
    print(f"\n[DEBUG] Testing {name}")
    print(f"Type: {type(loader)}")
    print(f"Length (num batches): {len(loader)}")

    batch = next(iter(loader))
    print("Batch keys:", batch.keys())

    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"{k}: shape={v.shape}, dtype={v.dtype}")
        else:
            print(f"{k}: {type(v)}")


def main():
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
    model = UNET(in_channels=3).to(DEVICE)

    # ===== Loss & Optimizer =====
    criterion = nn.BCEWithLogitsLoss()
    optimizer = Adam(model.parameters(), lr=LR)
    # ===== Results =====
    best_val_loss = float("inf")
    training_start_time = time.time()

    # ===== Training =====
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss   = evaluate(model, val_loader, criterion)
        epoch_time = time.time() - epoch_start_time
        total_time = time.time() - training_start_time
        # ===== Console log =====
        print(
            f"Epoch [{epoch+1}/{EPOCHS}] "
            f"Train: {train_loss:.4f} | "
            f"Val: {val_loss:.4f} | "
            f"Epoch time: {epoch_time:.1f}s"
        )
        # ===== CSV log =====
        with open(LOG_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, train_loss, val_loss, epoch_time])

        # ===== Checkpoint  =====
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                BEST_MODEL_PATH
            )
        torch.save(
        {
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        LAST_MODEL_PATH
        )
    print(
        f"Training completed in {total_time/60:.2f} minutes."
        )


if __name__ == "__main__":
    # ===== Prepare folders =====
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    # ===== Init log file =====
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["epoch", "train_loss", "val_loss", "epoch_time"])

    main()
