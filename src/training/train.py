# Mission: Huấn luyện mô hình trên dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 2.0
import torch
import time
import csv 
import wandb
import shutil, os
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
    def train_one_epoch(self, loader):
        self.model.train()
        running_loss = 0.0

        for batch in tqdm(loader, leave=False):
            images = batch["image"].to(self.device)
            labels = batch["label"].float().to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(images).squeeze(1)
            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()
            
        # --- DÒNG NÀY CHO CYCLIC LR ---
            if self.scheduler is not None:
                self.scheduler.step()
        # ------------------------------
            running_loss += loss.item()

        return running_loss / len(loader)

    def evaluate(self, loader):
        self.model.eval()
        running_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                images = batch["image"].to(self.device)
                labels = batch["label"].float().to(self.device)

                outputs = self.model(images).squeeze(1)
                loss = self.criterion(outputs, labels)
                running_loss += loss.item()

        return running_loss / len(loader)
    def fit(self, train_loader, val_loader, epochs, patience, log_path, best_model_path, last_model_path, model_name):
            """
            Đóng gói toàn bộ quá trình huấn luyện: lặp epoch, tính loss, tính metrics, 
            ghi log, lưu model và early stopping.
            """
            best_val_f1 = 0.0 
            best_epoch_num = 1
            #early_stopping = EarlyStopping(patience=patience, mode='min') # Dừng khi val_loss không giảm
            training_start_time = time.time()
            
            # Định nghĩa các mốc khi training để lưu model
            milestones = []
            if epochs >= 150:
                milestones = [50, 100]
            #Anomaly Detection
            prev_val_loss = float('inf')
            SPIKE_THRESHOLD_PERCENT = 100.0
            for epoch in range(epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_loss = self.evaluate(val_loader)
                
                # Tính toán chỉ số trên tập Val
                val_metrics = evaluate_metrics(self.model, val_loader, self.device)
                val_f1 = val_metrics["f1"]
                val_auc = val_metrics["auc"]
                val_precision = val_metrics["precision"]
                val_recall = val_metrics["recall"]
                val_accuracy = val_metrics["accuracy"]
                
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} |Val Recall: {val_recall:.4f} |Val Precision: {val_precision:.4f} | Val F1: {val_f1:.4f} | Val AUC: {val_auc:.4f}")
                # ========================================================
                # THỰC THI BẪY BẤT THƯỜNG (ANOMALY DETECTION)
                # ========================================================
                if epoch > 0: # Bỏ qua epoch 1 vì chưa có dữ liệu so sánh
                    delta_loss = val_loss - prev_val_loss
                    spike_ratio = (delta_loss / (prev_val_loss + 1e-8)) * 100
                    
                    if spike_ratio > SPIKE_THRESHOLD_PERCENT:
                        anomaly_dir = os.path.dirname(best_model_path)
                        anomaly_path = os.path.join(anomaly_dir, f"anomaly_spike_epoch_{epoch+1}.pth")
                        
                        print(f"\n[CẢNH BÁO ĐỎ] Phát hiện Val Loss nhảy vọt tại Epoch {epoch+1}!")
                        print(f"   - Val Loss cũ: {prev_val_loss:.4f} -> Val Loss mới: {val_loss:.4f}")
                        print(f"   - Tỷ lệ tăng: +{spike_ratio:.2f}% (Ngưỡng: {SPIKE_THRESHOLD_PERCENT}%)")
                        
                        torch.save({
                            "epoch": epoch + 1,
                            "model_state_dict": self.model.state_dict(),
                            "optimizer_state_dict": self.optimizer.state_dict(),
                            "spike_info": {
                                "previous_val_loss": float(prev_val_loss),
                                "current_val_loss": float(val_loss),
                                "delta_loss": float(delta_loss),
                                "spike_ratio_percent": float(spike_ratio)
                            },
                            "train_loss": float(train_loss),
                            "val_f1": float(val_f1),
                            "val_auc": float(val_auc),
                            "learning_rate": self.optimizer.param_groups[0]['lr']
                        }, anomaly_path)
                        print(f"   -> Đã lưu bất thường tại: {anomaly_path}\n")
                        
                        # Tùy chọn: Đẩy tỷ lệ nhảy vọt này lên WandB để dễ nhìn trên đồ thị
                        if not Config.Turn_WandB_Off:
                            wandb.log({"Anomaly Spike Ratio (%)": spike_ratio, "epoch": epoch + 1})
                
                # Cập nhật mốc Loss cho vòng lặp tiếp theo
                prev_val_loss = val_loss
                # === WANDB LOGGING MLOPS ===
                if not Config.Turn_WandB_Off:
                    wandb.log({
                        "epoch": epoch + 1,
                        "Train Loss": train_loss,
                        "Val Loss": val_loss,
                        "Val F1": val_f1,
                        "Val Precision": val_precision,
                        "Val Recall": val_recall,
                        "Val Accuracy": val_accuracy,
                        "Val AUC": val_auc,
                        "Learning Rate": self.optimizer.param_groups[0]['lr']
                    })
                # Ghi log CSV
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, train_loss, val_loss, val_f1, val_auc, val_precision, val_recall, val_accuracy])
                
                # Lưu Best Checkpoint
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch_num = epoch + 1
                    torch.save({
                        "epoch":  best_epoch_num,
                        "model_state_dict": self.model.state_dict(),
                        "val_loss": float(val_loss),
                        "val_f1": float(val_f1),
                        "val_auc": float(val_auc),
                        "val_precision": float(val_precision),
                        "val_recall": float(val_recall),
                        "val_accuracy": float(val_accuracy)
                    }, best_model_path)
                    print(f"[INFO] New best model saved with F1: {val_f1:.4f} at epoch {best_epoch_num}")
                
                # Lưu Last Checkpoint
                torch.save({
                    "epoch": epoch + 1,
                    "model": model_name,
                    "model_state_dict": self.model.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                }, last_model_path)
                
                # Kiểm tra Early Stopping
                # early_stopping(val_loss)
                # if early_stopping.early_stop:
                #     print(f"[INFO] Early stopping triggered at epoch {epoch+1}.")
                #     break
                
                # Lưu model tại các mốc đã định nếu có
                current_epoch = epoch + 1
                if current_epoch in milestones:
                    # Tạo đường dẫn mới. Ví dụ: results/.../checkpoints/best_at_epoch_50.pth
                    milestone_path = best_model_path.replace(".pth", f"_at_epoch_{current_epoch}.pth")
                    
                    # Kiểm tra xem file best gốc đã tồn tại chưa rồi mới copy
                    if os.path.exists(best_model_path):
                        shutil.copy(best_model_path, milestone_path)
                        print(f"\n[MILESTONE] Đã sao chép kỷ lục Best Model tính đến hiện tại ở mốc {current_epoch} thành: {milestone_path}\n")
                    else:
                        print(f"\n[MILESTONE] Cảnh báo: Chưa có best model nào được lưu để copy tại mốc {current_epoch}!\n")
                    
            total_time = time.time() - training_start_time
            print(f"[INFO] Training completed in {total_time/60:.2f} minutes.")
            
            return best_epoch_num