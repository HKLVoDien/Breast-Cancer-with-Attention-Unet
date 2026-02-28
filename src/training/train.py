# Mission: Huấn luyện mô hình trên dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 2.0
import torch
import time
import csv
from tqdm import tqdm
from src.utils.callbacks import EarlyStopping
from src.training.evaluation_metrics import evaluate_metrics

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
            early_stopping = EarlyStopping(patience=patience, mode='max')
            training_start_time = time.time()

            for epoch in range(epochs):
                train_loss = self.train_one_epoch(train_loader)
                val_loss = self.evaluate(val_loader)
                
                # Tính toán chỉ số trên tập Val
                val_metrics = evaluate_metrics(self.model, val_loader, self.device)
                val_f1 = val_metrics["f1"]
                val_recall = val_metrics["recall"]
                
                print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f} | Val Recall: {val_recall:.4f}")

                # Ghi log CSV
                with open(log_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch + 1, train_loss, val_loss, val_f1])
                
                # Lưu Best Checkpoint
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    best_epoch_num = epoch + 1
                    torch.save({
                        "epoch":  best_epoch_num,
                        "model_state_dict": self.model.state_dict(),
                        "val_f1": val_f1,
                        "val_recall": val_recall
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
                early_stopping(val_loss)
                if early_stopping.early_stop:
                    print(f"[INFO] Early stopping triggered at epoch {epoch+1}.")
                    break
                    
            total_time = time.time() - training_start_time
            print(f"[INFO] Training completed in {total_time/60:.2f} minutes.")
            
            return best_epoch_num