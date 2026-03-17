# Task: Tích hợp Attention UNET từ thư viện MONAI cho phân loại ảnh y tế.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import torch.nn as nn
from monai.networks.nets import AttentionUnet

class MonaiAttentionUNet(nn.Module):
    def __init__(self, in_channels=3):
        super(MonaiAttentionUNet, self).__init__()
        
        # 1. Khởi tạo thân mô hình Attention U-Net từ thư viện MONAI
        self.unet = AttentionUnet(
            spatial_dims=2,          # Xử lý ảnh 2D
            in_channels=in_channels,
            out_channels=64,         # Số lượng kênh đặc trưng đầu ra trước khi pooling
            channels=(64, 128, 256), # Cấu trúc channels tương đồng với mô hình tự build
            strides=(2, 2)           # Tương ứng với quá trình downsample
        )
        
        # 2. Thêm bộ phân loại (Classifier) cho bài toán Binary Classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(128),
            nn.Dropout(0.5),

            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(64),
            nn.Dropout(0.5),

            nn.Linear(64, 1) # Đầu ra 1 giá trị đồng bộ với toàn bộ project
        )

    def forward(self, x):
        # Đi qua phần thân trích xuất đặc trưng có cơ chế Attention
        x = self.unet(x)
        
        # Đi qua bộ gom cụm và phân loại
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

if __name__ == "__main__":
    # Test nhanh mô hình
    model = MonaiAttentionUNet(in_channels=3)
    x = torch.randn((2, 3, 224, 224))
    y = model(x)
    print("MONAI Attention U-Net output shape:", y.shape) # Kì vọng: [2, 1]