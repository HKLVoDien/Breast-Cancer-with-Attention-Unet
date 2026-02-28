# Mission: Tìm kiếm khoảng Learning Rate tối ưu (LR Range Test).
import os
import torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt

# Import thêm TrainDataLoaderIter từ thư viện
from torch_lr_finder import LRFinder, TrainDataLoaderIter

from configs.default_configs import Config
from src.models.build_models import build_model
from src.datasets.breast_cancer_dataloader import create_dataloader
from src.datasets.breast_cancer_dataframe import get_pos_weight

# ==========================================================
# 1. WRAPPER CHO DATALOADER (Khắc phục lỗi class 'dict')
# ==========================================================
class CustomTrainIter(TrainDataLoaderIter):
    def inputs_labels_from_batch(self, batch_data):
        # Trích xuất "image" và "label" từ dictionary
        # Đồng thời ép label sang float() giống trong file train.py
        return batch_data["image"], batch_data["label"].float()

# ==========================================================
# 2. WRAPPER CHO MODEL (Giúp output khớp shape với label)
# ==========================================================
class ModelWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # Hàm squeeze(1) giống hệt dòng 27 trong file train.py của bạn
        return self.model(x).squeeze(1)
# ==========================================================

def find_lr(model_name="attention_unet"):
    print("[INFO] Đang chuẩn bị Dataloader...")
    train_loader = create_dataloader(
        dataframe_csv_path=Config.TRAIN_CSV,
        split="train",
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print("[INFO] Đang khởi tạo Model và Optimizer...")
    # Khởi tạo model gốc
    base_model = build_model(name=model_name, in_channels=3).to(Config.DEVICE)
    
    # Bọc model gốc bằng ModelWrapper
    model = ModelWrapper(base_model)
    
    pos_weight, _ = get_pos_weight(Config.TRAIN_CSV, Config.DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    # Khởi tạo Adam với LR cực nhỏ làm điểm bắt đầu
    optimizer = Adam(model.parameters(), lr=1e-7, weight_decay=1e-4)

    print("[INFO] Bắt đầu tìm kiếm Learning Rate...")
    lr_finder = LRFinder(model, optimizer, criterion, device=Config.DEVICE)
    
    # Bọc train_loader bằng CustomTrainIter
    custom_train_loader = CustomTrainIter(train_loader)
    
    # Truyền custom_train_loader vào range_test
    lr_finder.range_test(custom_train_loader, end_lr=1.0, num_iter=100, step_mode="exp")

    # Lưu biểu đồ ra file ảnh
    os.makedirs(Config.RESULTS_DIR, exist_ok=True)
    plot_path = os.path.join(Config.RESULTS_DIR, f"lr_finder_plot_{model_name}.png")
    
    fig, ax = plt.subplots()
    lr_finder.plot(ax=ax) 
    fig.savefig(plot_path)
    print(f"[HOÀN THÀNH] Đã lưu biểu đồ tại: {plot_path}")
    
    print("[INFO] Đang hiển thị biểu đồ. Hãy đóng cửa sổ biểu đồ để kết thúc chương trình.")
    plt.show() 

    # Reset model về trạng thái ban đầu để tránh ảnh hưởng bộ nhớ
    lr_finder.reset()

if __name__ == "__main__":
    find_lr()