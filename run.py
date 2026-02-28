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
# Các module dữ liệu
from configs.default_configs import Config
from src.training.find_lr import find_lr 
from src.datasets.breast_cancer_dataframe import build_metadata_csv, get_pos_weight
from src.datasets.split_dataset import main as split_main
from src.datasets.breast_cancer_dataloader import create_dataloader
# Các module mô hình và huấn luyện
from src.models.build_models import build_model
from src.training.train import Train_model
from src.training.evaluation_metrics import evaluate_metrics
from src.utils.callbacks import EarlyStopping
from src.utils.model_utils import load_best_model

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-data",
        action="store_true",
        help="Rebuild metadata CSV before training"
    )
    parser.add_argument(
        "--find-lr",
        action="store_true",
        help="Run learning rate finder before training"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="unet",
        choices=["unet", "attention_unet", "resnet"],
        help="Model to train"
    )
    parser.add_argument("--max-lr", type=float, default=1e-3, help="Maximum learning rate")
    parser.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for CyclicLR")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    return parser.parse_args()

# ==================
def main(args):
    model_name = args.model
    
    # ===== Results directories & log file =====
    RESULTS_DIR = Config.get_exp_dir(model_name)
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
                ["epoch", "train_loss", "val_loss", "F1_score"]
            ) 
   
    # ===== DataLoader =====
    train_loader = create_dataloader(
        dataframe_csv_path=Config.TRAIN_CSV,
        split="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.TRAIN_SHUFFLE,
        drop_last=Config.TRAIN_DROP_LAST
    )
    val_loader = create_dataloader(
        dataframe_csv_path=Config.VAL_CSV,
        split="val",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.VAL_SHUFFLE,
        drop_last=Config.VAL_DROP_LAST
    )
    test_loader = create_dataloader(
        dataframe_csv_path=Config.TEST_CSV,  # Sử dụng test.csv làm test để đánh giá cuối cùng
        split="test",
        batch_size=Config.BATCH_SIZE,
        shuffle=Config.TEST_SHUFFLE,
        drop_last=Config.TEST_DROP_LAST
    )
    
    # ===== Model =====
    model = build_model(
        name=model_name,
        in_channels=3
    ).to(Config.DEVICE)
    
    # ===== Pos_weight =====
    pos_weight, pos_weight_value = get_pos_weight(Config.TRAIN_CSV, Config.DEVICE)
    print(f"[INFO] Using pos_weight = {pos_weight_value:.4f}")
    
    # ===== Loss & Optimizer =====
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=Config.LR)
    # Đặt lr ban đầu bằng base_lr
    base_lr = args.lr
    max_lr = args.max_lr
    # Tính số bước (steps) cho nửa chu kỳ (thường bằng 2-8 lần số batch trong 1 epoch)
    steps_per_epoch = len(train_loader)
    step_size_up = 2 * steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CyclicLR(
        optimizer, 
        base_lr=base_lr, 
        max_lr=max_lr, 
        step_size_up=step_size_up,
        mode="triangular", # Mô hình lượn sóng hình tam giác giống tác giả notebook
        cycle_momentum=False # Vì Adam không dùng tham số momentum như SGD
    ) 
    # ===== Trainer =====
    trainer = Train_model(model, optimizer, criterion, Config.DEVICE, scheduler=scheduler)
    best_epoch_num = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=Config.EPOCHS,
        patience=5,
        log_path=LOG_PATH,
        best_model_path=BEST_MODEL_PATH,
        last_model_path=LAST_MODEL_PATH,
        model_name=model_name
        
    )
    # ===== Load best model and evaluate =====
    print("[INFO] Loading best model for evaluation...")
    model = load_best_model(model, BEST_MODEL_PATH, Config.DEVICE)
    metrics = evaluate_metrics(
        model=model,
        dataloader=test_loader,
        device=Config.DEVICE
    )

    print("[INFO] Evaluation results:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
        
    # ===== Save metrics to JSON =====
    METRICS_PATH = os.path.join(RESULTS_DIR, "metrics.json")
    metrics_to_save = {
        "model": model_name,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.LR,
        "pos_weight": float(pos_weight_value), 
        "best_epoch": best_epoch_num,
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
    # 1. Cập nhật Config từ tham số Command Line (LR, BATCH_SIZE)
    Config.update_from_args(args)
    
    # 2. Nếu người dùng chọn --build-data, thực hiện xây dựng metadata và chia dataset rồi thoát
    if args.build_data:
        print("[INFO] Building metadata CSV...")
        build_metadata_csv(Config.DATA_ROOT, Config.METADATA_CSV)
        split_main()
        print("[INFO] Metadata CSV built successfully.")
        exit(0)
    if args.find_lr:
        print("[INFO] Running learning rate finder...")
        find_lr(model_name=args.model)
        print("[INFO] Learning rate finder completed.")
        exit(0)
    #Chọn mô hình để huấn luyện: --model "attention_unet, unet, resnet"
    print(f"[INFO] Training model: {args.model}")
    print("[INFO] Starting training...")
    main(args)
