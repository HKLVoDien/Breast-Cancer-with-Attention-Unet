# Mission: Huấn luyện mô hình trên dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 2.0
import torch
import time
import csv
import wandb
import shutil, os, numpy as np
from tqdm import tqdm
from src.utils.callbacks import EarlyStopping
from src.training.evaluation_metrics import evaluate_metrics
from configs.default_configs import Config


class Train_model:
    def __init__(self, model, optimizer, criterion, device, scheduler=None):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

    def _calculate_metrics_from_arrays(self, y_true, y_pred):
        """
        Hàm hỗ trợ tính toán metrics từ mảng numpy để tránh chạy lại model.
        Dựa trên logic threshold 0.6 của bạn.
        """
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
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # ------------------------------
            running_loss += loss.item()
            # Thu thập dữ liệu để tính metrics ngay tại đây
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.6).long()  # Khớp với threshold của bạn
            all_preds.append(preds.detach().cpu())
            all_labels.append(labels.detach().cpu())
        # --- DÒNG NÀY CHO CYCLIC LR ---
        if self.scheduler is not None:
            self.scheduler.step()
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
                images = batch["image"].to(self.device)
                labels = batch["label"].float().to(self.device)

                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()
                # Thu thập dữ liệu để tính metrics ngay tại đây
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
        """
        Đóng gói toàn bộ quá trình huấn luyện: lặp epoch, tính loss, tính metrics,
        ghi log, lưu model và early stopping.
        """
        best_epoch_num = 1
        # early_stopping = EarlyStopping(patience=patience, mode='min') # Dừng khi val_loss không giảm
        training_start_time = time.time()

        # Định nghĩa các mốc khi training để lưu model
        milestone_interval = 50  # Cứ 50 epoch sẽ tạo 1 mốc. Bạn có thể đổi tùy ý.
        milestones = [
            i for i in range(milestone_interval, epochs + 1, milestone_interval)
        ]
        # Lấy đường dẫn thư mục 'checkpoints'
        ckpt_dir = os.path.dirname(last_model_path)
        all_epochs_dir = os.path.join(ckpt_dir, "all_epochs")
        os.makedirs(all_epochs_dir, exist_ok=True)
        if milestones:
            print(f"[INFO] Các mốc lưu checkpoint: {milestones}")
        for epoch in range(start_epoch, epochs):
            train_results = self.train_one_epoch(train_loader)
            val_results = self.evaluate(val_loader)

            # Giải nén kết quả
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
            # === WANDB LOGGING MLOPS ===
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
                        val_acc,
                        val_rec,
                        val_prec,
                        val_f1,
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
            # Đường dẫn cho từng file epoch cụ thể
            epoch_ckpt_path = os.path.join(all_epochs_dir, f"epoch_{epoch+1}.pth")
            # Lưu checkpoint cho từng epoch
            # Lưu file (Đã bỏ qua optimizer để tối ưu dung lượng và tốc độ đẩy lên Drive)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": self.model.state_dict(),
                    "val_f1": float(val_f1),
                    "val_loss": float(val_loss),
                },
                epoch_ckpt_path,
            )

            # Kiểm tra Early Stopping
            # early_stopping(val_loss)
            # if early_stopping.early_stop:
            #     print(f"[INFO] Early stopping triggered at epoch {epoch+1}.")
            #     break

            # Lưu model tại các mốc đã định nếu có
            current_epoch = epoch + 1
            if current_epoch in milestones:
                # Tạo đường dẫn mới. Ví dụ: results/.../checkpoints/best_at_epoch_50.pth
                ckpt_dir = os.path.dirname(last_model_path)
                milestone_path = os.path.join(
                    ckpt_dir, f"milestone_epoch_{current_epoch}.pth"
                )
                # LƯU ĐẦY ĐỦ BỘ ĐỂ CÓ THỂ RESUME TỪ MỐC NÀY
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
