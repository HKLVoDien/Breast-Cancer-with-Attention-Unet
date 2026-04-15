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
from src.utils.soft_voting import get_soft_voting_preds
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
        running_loss_r = 0.0
        running_loss_g = 0.0
        running_loss_b = 0.0
        running_total_loss = 0.0
        all_preds = []
        all_labels = []
        for batch in tqdm(loader, leave=False):
            image_r = batch["image_r"].to(self.device)
            image_g = batch["image_g"].to(self.device)
            image_b = batch["image_b"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            # 1. Nhận 3 logit riêng biệt
            logit_r, logit_g, logit_b = self.model(image_r, image_g, image_b)
            # 2. Tính 3 hàm Loss riêng biệt cho 3 nhánh
            loss_r = self.criterion(logit_r, labels)
            loss_g = self.criterion(logit_g, labels)
            loss_b = self.criterion(logit_b, labels)

            # 3. Tổng hợp Loss và truyền ngược (cộng lại hoặc chia 3 đều được)
            total_loss = (loss_r + loss_g + loss_b) / 3.0
            total_loss.backward()
            self.optimizer.step()

            # Cập nhật running loss cho từng nhánh và tổng thể
            running_total_loss += total_loss.item()
            running_loss_r += loss_r.item()
            running_loss_g += loss_g.item()
            running_loss_b += loss_b.item()

            # Gọi hàm Soft Voting từ file utils
            preds, _ = get_soft_voting_preds(logit_r, logit_g, logit_b, threshold=0.6)
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
        if self.scheduler is not None:
            self.scheduler.step()
        y_pred = torch.cat(all_preds).numpy().flatten()
        y_true = torch.cat(all_labels).numpy().flatten()
        # Tính trung bình cho cả epoch
        num_batches = len(loader)
        metrics = self._calculate_metrics_from_arrays(y_true, y_pred)

        # Lưu cả 4 loss vào dictionary trả về
        metrics["loss_total"] = running_total_loss / num_batches
        metrics["loss_r"] = running_loss_r / num_batches
        metrics["loss_g"] = running_loss_g / num_batches
        metrics["loss_b"] = running_loss_b / num_batches
        return metrics

    def evaluate(self, loader):
        self.model.eval()
        running_loss_r = 0.0
        running_loss_g = 0.0
        running_loss_b = 0.0
        running_total_loss = 0.0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                image_r = batch["image_r"].to(self.device)
                image_g = batch["image_g"].to(self.device)
                image_b = batch["image_b"].to(self.device)
                labels = batch["label"].float().to(self.device)

                logit_r, logit_g, logit_b = self.model(image_r, image_g, image_b)
                # 2. Tính 3 hàm Loss riêng biệt cho 3 nhánh
                loss_r = self.criterion(logit_r, labels)
                loss_g = self.criterion(logit_g, labels)
                loss_b = self.criterion(logit_b, labels)
                # 3. Tổng hợp Loss và truyền ngược (cộng lại hoặc chia 3 đều được)
                total_loss = (loss_r + loss_g + loss_b) / 3.0
                # Cập nhật running loss cho từng nhánh và tổng thể
                running_total_loss += total_loss.item()
                running_loss_r += loss_r.item()
                running_loss_g += loss_g.item()
                running_loss_b += loss_b.item()

                # Gọi hàm Soft Voting từ file utils
                preds, _ = get_soft_voting_preds(
                    logit_r, logit_g, logit_b, threshold=0.6
                )
                all_preds.append(preds.detach().cpu())
                all_labels.append(labels.detach().cpu())
        # Scheduler step sau mỗi epoch (nếu có)
        if self.scheduler is not None:
            self.scheduler.step()
        y_pred = torch.cat(all_preds).numpy().flatten()
        y_true = torch.cat(all_labels).numpy().flatten()
        # Tính trung bình cho cả epoch
        num_batches = len(loader)
        metrics = self._calculate_metrics_from_arrays(y_true, y_pred)

        # Lưu cả 4 loss vào dictionary trả về
        metrics["loss_total"] = running_total_loss / num_batches
        metrics["loss_r"] = running_loss_r / num_batches
        metrics["loss_g"] = running_loss_g / num_batches
        metrics["loss_b"] = running_loss_b / num_batches
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

        for epoch in range(start_epoch, epochs):
            train_results = self.train_one_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            # Lấy ĐÚNG TÊN các loss đã lưu trong dictionary
            train_loss = train_results["loss_total"]
            train_loss_r = train_results["loss_r"]
            train_loss_g = train_results["loss_g"]
            train_loss_b = train_results["loss_b"]
            train_f1 = train_results["f1"]
            train_acc, train_prec, train_rec = (
                train_results["accuracy"],
                train_results["precision"],
                train_results["recall"],
            )

            # Lấy Validation Loss
            val_loss = val_results["loss_total"]
            val_f1 = val_results["f1"]
            val_acc, val_prec, val_rec = (
                val_results["accuracy"],
                val_results["precision"],
                val_results["recall"],
            )

            print(
                f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Train Accuracy: {train_acc:.4f} | Train Recall: {train_rec:.4f} | Train Precision: {train_prec:.4f} | Train F1: {train_f1:.4f} \n   Val Loss: {val_loss:.4f} | Val Accuracy: {val_acc:.4f} |Val Recall: {val_rec:.4f} |Val Precision: {val_prec:.4f} | Val F1: {val_f1:.4f}"
            )

            # === WANDB LOGGING ===
            if not Config.Turn_WandB_Off:
                wandb.log(
                    {
                        "epoch": epoch + 1,
                        "Train Loss Total": train_loss,
                        "Train Loss R": train_loss_r,
                        "Train Loss G": train_loss_g,
                        "Train Loss B": train_loss_b,
                        "Train Accuracy": train_acc,
                        "Train Precision": train_prec,
                        "Train Recall": train_rec,
                        "Train F1": train_f1,
                        "Val Loss Total": val_loss,
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
                        train_loss_r,
                        train_loss_g,
                        train_loss_b,
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
                        "train_loss": float(train_loss),
                        "train_loss_r": float(train_loss_r),
                        "train_loss_g": float(train_loss_g),
                        "train_loss_b": float(train_loss_b),
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
