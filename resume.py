# Mission: Cứu hộ và huấn luyện nối tiếp mô hình bị ngắt giữa chừng.
# Author: Lê Văn Hoàn

import os, argparse, json, csv
import torch
import torch.nn as nn
from torch.optim import Adam
import wandb

from configs.default_configs import Config
from src.datasets.breast_cancer_dataloader import create_dataloader
from src.datasets.breast_cancer_dataframe import get_pos_weight
from src.models.build_models import build_model
from src.training.train import Train_model
from src.training.evaluation_metrics import evaluate_metrics
from src.utils.model_utils import load_best_model
from src.utils.seed import set_seed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--wandb-project", type=str, required=True, help="Tên dự án WandB của phiên bị ngắt (Ví dụ: Breast-Cancer-IDC_150_epochs)")
    parser.add_argument("--exp-dir", type=str, required=True, help="Đường dẫn tới thư mục kết quả bị ngắt (Ví dụ: results/attention_unet_bs32_lr0.0001_epochs150_2024_06_15)")
    parser.add_argument("--wandb-id", type=str, required=True, help="Mã ID của phiên WandB bị ngắt (Ví dụ: 1a2b3c4d)")
    parser.add_argument("--epochs", type=int, default=150, help="Tổng số epoch dự định chạy ban đầu")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size cũ")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate cũ")
    return parser.parse_args()

def main():
    args = parse_args()
    Config.update_from_args(args)
    set_seed(Config.SEED)

    # 1. Khôi phục các đường dẫn cũ
    LAST_MODEL_PATH = os.path.join(args.exp_dir, "checkpoints", "last.pth")
    BEST_MODEL_PATH = os.path.join(args.exp_dir, "checkpoints", "best.pth")
    LOG_PATH = os.path.join(args.exp_dir, "logs", "train_log.csv")
    METRICS_PATH = os.path.join(args.exp_dir, "metrics.json")

    if not os.path.exists(LAST_MODEL_PATH):
        print(f"[ERROR] Không tìm thấy file checkpoint tại: {LAST_MODEL_PATH}")
        return

    # 2. Tải Checkpoint "hiện trường"
    print(f"[INFO] Đang tải hiện trường từ: {LAST_MODEL_PATH}")
    checkpoint = torch.load(LAST_MODEL_PATH, map_location=Config.DEVICE)
    model_name = checkpoint.get("model", "attention_unet")
    start_epoch = checkpoint["epoch"] # Ví dụ: Sẽ lấy ra số 130
    print(f"[INFO] Khôi phục mô hình {model_name} | Tiếp tục chạy từ Epoch {start_epoch + 1}")
    #Đọc file best.pth để lấy kỷ lục F1 cũ (Tránh ghi đè nhầm)
    
    previous_best_f1 = 0.0
    if os.path.exists(BEST_MODEL_PATH):
        best_ckpt = torch.load(BEST_MODEL_PATH, map_location=Config.DEVICE)
        previous_best_f1 = best_ckpt.get("val_f1", 0.0)
        print(f"[INFO] Đã tìm thấy kỷ lục cũ. Best F1 hiện tại: {previous_best_f1:.4f}")
        
    # 3. Nối tiếp WandB
    if not Config.Turn_WandB_Off:
        wandb.init(
            project=f"{args.wandb_project}",
            id=args.wandb_id,
            resume="must"
        )

    # 4. Tải DataLoader
    train_loader = create_dataloader(Config.TRAIN_CSV, "train", Config.BATCH_SIZE, Config.TRAIN_SHUFFLE, Config.TRAIN_DROP_LAST)
    val_loader = create_dataloader(Config.VAL_CSV, "val", Config.BATCH_SIZE, Config.VAL_SHUFFLE, Config.VAL_DROP_LAST)
    test_loader = create_dataloader(Config.TEST_CSV, "test", Config.BATCH_SIZE, Config.TEST_SHUFFLE, Config.TEST_DROP_LAST)

    # 5. Khôi phục Mô hình và Thuật toán tối ưu (Cực kỳ quan trọng)
    model = build_model(model_name, in_channels=3).to(Config.DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    pos_weight, pos_weight_value = get_pos_weight(Config.TRAIN_CSV, Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = Adam(model.parameters(), lr=Config.BASE_LR)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    # 6. KÍCH HOẠT TRAIN NỐI TIẾP
    trainer = Train_model(model, optimizer, criterion, Config.DEVICE, scheduler=None)
    best_epoch_num = trainer.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        patience=5,
        log_path=LOG_PATH,
        best_model_path=BEST_MODEL_PATH,
        last_model_path=LAST_MODEL_PATH,
        model_name=model_name,
        start_epoch=start_epoch, 
        best_val_f1=previous_best_f1 # Truyền kỷ lục F1 cũ vào đây để tránh ghi đè nhầm
        )

    # ==========================================
    # ĐÁNH GIÁ ĐA MỐC VÀ LƯU JSON GIỐNG RUN.PY
    # ==========================================
    print("[INFO] Bắt đầu quá trình Đánh giá trên tập Test...")
    all_metrics_to_save = {
        "model": model_name,
        "batch_size": Config.BATCH_SIZE,
        "learning_rate": Config.BASE_LR,
        "pos_weight": float(pos_weight_value)
    }

    # ===== Load best model and evaluate =====
    print("[INFO] Loading best model for evaluation...")
    if Config.EPOCHS >= 150:
        print("[INFO] Kích hoạt chế độ đánh giá đa mốc (Multi-milestone Evaluation)")
        milestones = [50, 100, 150] 
        for milestone in milestones:
            # 1. Xác định đường dẫn file Checkpoint
            if milestone == 150:
                ckpt_path = BEST_MODEL_PATH # Mốc cuối cùng chính là best.pth
            else:
                ckpt_path = BEST_MODEL_PATH.replace(".pth", f"_at_epoch_{milestone}.pth")
                
            if not os.path.exists(ckpt_path):
                print(f"[WARNING] Không tìm thấy file checkpoint cho mốc {milestone} tại: {ckpt_path}")
                continue
            print(f"\n--- Đang đánh giá mốc {milestone} Epochs ---")
            checkpoint = torch.load(ckpt_path, map_location=Config.DEVICE)
            actual_best_epoch = checkpoint.get("epoch", "Unknown")
            model = load_best_model(model, ckpt_path, Config.DEVICE)
            metrics = evaluate_metrics(
                    model=model,
                    dataloader=test_loader,
                    device=Config.DEVICE
                )
            for k, v in metrics.items():
                if k != "confusion_matrix":
                    print(f"  {k}: {v:.4f}")
                else:
                    print(f"  {k}: {v}")
            # Ghi log kết quả Test lên WandB
            
            if not Config.Turn_WandB_Off and milestone == 150: # Chỉ log mốc 150 epochs lên WandB để tránh quá nhiều log
                wandb.log({
                    "Test Accuracy": metrics["accuracy"],
                    "Test Recall": metrics["recall"],
                    "Test Precision": metrics["precision"],
                    "Test F1": metrics["f1"],
                    "Test AUC": metrics["auc"],
                })
            #Ghi vào Dictionary tổng
            all_metrics_to_save[f"milestone_{milestone}"] = {
                    "best_epoch": actual_best_epoch,
                    "accuracy": metrics["accuracy"],
                    "recall": metrics["recall"],
                    "precision": metrics["precision"],
                    "f1": metrics["f1"],
                    "auc": metrics["auc"],
                    "confusion_matrix": metrics["confusion_matrix"]
                }        
    else:
        model = load_best_model(model, BEST_MODEL_PATH, Config.DEVICE)
        metrics = evaluate_metrics(
            model=model,
            dataloader=test_loader,
            device=Config.DEVICE
        )

        print("[INFO] Evaluation results:")
        for k, v in metrics.items():
            if k != "confusion_matrix":
                print(f"{k}: {v:.4f}")
            else:
                print(f"{k}: {v}")
            
        # Ghi log kết quả Test lên WandB
        if not Config.Turn_WandB_Off:
            wandb.log({
                "Test Accuracy": metrics["accuracy"],
                "Test Recall": metrics["recall"],
                "Test Precision": metrics["precision"],
                "Test F1": metrics["f1"],
                "Test AUC": metrics["auc"],
            })
        metrics_to_save = {
            "model": model_name,
            "batch_size": Config.BATCH_SIZE,
            "learning_rate": Config.BASE_LR,
            "pos_weight": float(pos_weight_value),
            "best_epoch": best_epoch_num,
            "accuracy": metrics["accuracy"],
            "recall": metrics["recall"],
            "precision": metrics["precision"],
            "f1": metrics["f1"],
            "auc": metrics["auc"],
            "confusion_matrix": metrics["confusion_matrix"]
            }  
    wandb.finish() # Kết thúc phiên làm việc với WandB
    with open(METRICS_PATH, "w") as f:
        if Config.EPOCHS >= 150:
            json.dump(all_metrics_to_save, f, indent=4) 
        else:
            json.dump(metrics_to_save, f, indent=4)

    print(f"[INFO] Metrics saved to {METRICS_PATH}")

if __name__ == "__main__":
    main()