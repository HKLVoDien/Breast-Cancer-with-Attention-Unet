# Mission: Training loop cho RGB Soft Voting Model.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import time
import csv
import wandb
import shutil, os, numpy as np
from tqdm import tqdm
from src.utils.callbacks import EarlyStopping
from src.training.evaluation_metrics import evaluate_metrics
from configs.default_configs import Config


class Train_model_RGB:
    """
    Training loop cho RGBSoftVotingModel.
    Khác với Train_model: lấy batch["image_r"], batch["image_g"], batch["image_b"]
    thay vì batch["image"] và truyền cả 3 vào model.forward(image_r, image_g, image_b).
    Toàn bộ logic WandB logging, checkpoint, anomaly detection, milestone giữ nguyên.
    """

    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def _calculate_metrics_from_arrays(self, y_true, y_pred):
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-8)
        precision = TP / (TP + FP + 1e-8)
        recall = TP / (TP + FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        return {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        for batch in tqdm(loader, leave=False):
            image_r = batch["image_r"].to(self.device)
            image_g = batch["image_g"].to(self.device)
            image_b = batch["image_b"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(image_r, image_g, image_b)  # (B,)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            if self.scheduler is not None:
                self.scheduler.step()

            running_loss += loss.item()
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.6).long()
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())

        epoch_loss = running_loss / len(loader)
        y_pred = torch.cat(all_preds).numpy().flatten()
        y_true = torch.cat(all_labels).numpy().flatten()
        metrics = self._calculate_metrics_from_arrays(y_true, y_pred)
        metrics["loss"] = epoch_loss
        return metrics

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                image_r = batch["image_r"].to(self.device)
                image_g = batch["image_g"].to(self.device)
                image_b = batch["image_b"].to(self.device)
                labels = batch["label"].float().to(self.device)

                outputs = self.model(image_r, image_g, image_b)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                probs = torch.sigmoid(outputs)
                preds = (probs >= 0.6).long()
                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        epoch_loss = running_loss / len(loader)
        y_pred = torch.cat(all_preds).numpy().flatten()
        y_true = torch.cat(all_labels).numpy().flatten()
        metrics = self._calculate_metrics_from_arrays(y_true, y_pred)
        metrics["loss"] = epoch_loss
        return metrics

    def fit(
        self,
        train_loader,
        val_loader,
        epochs,
        patience,
        log_path,
        best_model_path,
        last_model_path,
        model_name,
        start_epoch=0,
        best_val_f1=0.0,
    ):
        best_epoch_num = 1
        training_start_time = time.time()

        milestone_interval = 50
        milestones = [
            i for i in range(milestone_interval, epochs + 1, milestone_interval)
        ]
        ckpt_dir = os.path.dirname(last_model_path)
        all_epochs_dir = os.path.join(ckpt_dir, "all_epochs")
        os.makedirs(all_epochs_dir, exist_ok=True)
        if milestones:
            print(f"[INFO] Các mốc lưu checkpoint: {milestones}")

        prev_val_loss = float("inf")
        SPIKE_THRESHOLD_PERCENT = 50.0

        for epoch in range(start_epoch, epochs):
            train_results = self.train_one_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            train_loss, train_f1 = train_results["loss"], train_results["f1"]
            train_acc, train_prec, train_rec = (
                train_results["accuracy"],
                train_results["precision"],
                train_results["recall"],
            )

            val_loss, val_f1 = val_results["loss"], val_results["f1"]
            val_acc, val_prec, val_rec = (
                val_results["accuracy"],
                val_results["precision"],
                val_results["recall"],
            )

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Train Recall: {train_rec:.4f} | Train Precision: {train_prec:.4f} | Train F1: {train_f1:.4f} \n Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} |Val Recall: {val_rec:.4f} |Val Precision: {val_prec:.4f} | Val F1: {val_f1:.4f}"
            )

            # ========================================================
            # ANOMALY DETECTION
            # ========================================================
            if epoch > 0:
                delta_loss = val_loss - prev_val_loss
                spike_ratio = (delta_loss / (prev_val_loss + 1e-8)) * 100

                if spike_ratio > SPIKE_THRESHOLD_PERCENT:
                    anomaly_dir = os.path.dirname(best_model_path)
                    anomaly_folder = os.path.join(anomaly_dir, "anomaly_epochs")
                    os.makedirs(anomaly_folder, exist_ok=True)
                    anomaly_path = os.path.join(
                        anomaly_folder, f"anomaly_spike_epoch_{epoch+1}.pth"
                    )

                    print(
                        f"\n[CẢNH BÁO ĐỎ] Phát hiện Val Loss nhảy vọt tại Epoch {epoch+1}!"
                    )
                    print(
                        f"   - Val Loss cũ: {prev_val_loss:.4f} -> Val Loss mới: {val_loss:.4f}"
                    )
                    print(
                        f"   - Tỷ lệ tăng: +{spike_ratio:.2f}% (Ngưỡng: {SPIKE_THRESHOLD_PERCENT}%)"
                    )

                    torch.save(
                        {
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "spike_info": {
                                "previous_val_loss": float(prev_val_loss),
                                "current_val_loss": float(val_loss),
                                "delta_loss": float(delta_loss),
                                "spike_ratio_percent": float(spike_ratio),
                            },
                            "train_loss": float(train_loss),
                            "val_f1": float(val_f1),
                            "train_accuracy": float(train_acc),
                            "train_precision": float(train_prec),
                            "train_recall": float(train_rec),
                            "train_f1": float(train_f1),
                            "learning_rate": self.optimizer.param_groups[0]["lr"],
                        },
                        anomaly_path,
                    )
                    print(f"   -> Đã lưu bất thường tại: {anomaly_path}\n")

                    if not Config.Turn_WandB_Off:
                        wandb.log(
                            {"Anomaly Spike Ratio (%)": spike_ratio, "epoch": epoch + 1}
                        )

            prev_val_loss = val_loss

            # === WANDB LOGGING ===
            if not Config.Turn_WandB_Off:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "Train Loss": train_loss,
                        "Train Accuracy": train_acc,
                        "Train Precision": train_prec,
                        "Train Recall": train_rec,
                        "Train F1": train_f1,
                        "Val Loss": val_loss,
                        "Val F1": val_f1,
                        "Val Precision": val_prec,
                        "Val Recall": val_rec,
                        "Val Accuracy": val_acc,
                        "Learning Rate": self.optimizer.param_groups[0]["lr"],
                    }
                )

            # Ghi log CSV
            with open(log_path, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        epoch + 1,
                        train_loss,
                        train_acc,
                        train_prec,
                        train_rec,
                        train_f1,
                        val_loss,
                        val_f1,
                        val_acc,
                        val_rec,
                        val_prec,
                    ]
                )

            # Lưu Best Checkpoint
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_epoch_num = epoch + 1
                torch.save(
                    {
                        "epoch": best_epoch_num,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": float(val_loss),
                        "val_f1": float(val_f1),
                        "val_precision": float(val_prec),
                        "val_recall": float(val_rec),
                        "val_accuracy": float(val_acc),
                    },
                    best_model_path,
                )
                print(
                    f"[INFO] New best model saved with F1: {val_f1:.4f} at epoch {best_epoch_num}"
                )

            # Lưu Last Checkpoint
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": model_name,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                },
                last_model_path,
            )

            # Lưu checkpoint từng epoch
            epoch_ckpt_path = os.path.join(all_epochs_dir, f"epoch_{epoch+1}.pth")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "val_f1": float(val_f1),
                    "val_loss": float(val_loss),
                },
                epoch_ckpt_path,
            )

            # Lưu model tại các mốc
            current_epoch = epoch + 1
            if current_epoch in milestones:
                ckpt_dir = os.path.dirname(last_model_path)
                milestone_path = os.path.join(
                    ckpt_dir, f"milestone_epoch_{current_epoch}.pth"
                )
                torch.save(
                    {
                        "epoch": current_epoch,
                        "model": model_name,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "val_f1": float(val_f1),
                        "val_loss": float(val_loss),
                        "val_precision": float(val_prec),
                        "val_recall": float(val_rec),
                        "val_accuracy": float(val_acc),
                        "train_accuracy": float(train_acc),
                        "train_precision": float(train_prec),
                        "train_recall": float(train_rec),
                        "train_f1": float(train_f1),
                    },
                    milestone_path,
                )
                print(
                    f"[INFO] Milestone checkpoint saved at epoch {current_epoch} with F1: {val_f1:.4f}"
                )

        total_time = time.time() - training_start_time
        print(f"[INFO] Training completed in {total_time/60:.2f} minutes.")

        return best_epoch_num
