# Task: Thiết kế model ResNet Từ notebooks cho phân loại ảnh y tế sử dụng PyTorch.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch
import torch.nn as nn
import torchvision.models as models
class ResNet(nn.Module):
    def __init__(self, pretrained=True):
        super(ResNet, self).__init__()
        if pretrained:
            weights = models.ResNet18_Weights.DEFAULT
        else:
            weights = None
        # 1. Tải kiến trúc ResNet-18
        self.model = models.resnet18(weights=weights)
        
        # 2. Lấy số lượng đặc trưng đầu vào của lớp Fully Connected cuối cùng (thường là 512)
        num_features = self.model.fc.in_features
        
        # 3. Ghi đè lớp fc bằng cấu trúc chuỗi tùy chỉnh
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(0.5),
            
            # CHÚ Ý: Đầu ra là 1 để đồng bộ với Attention U-Net
            nn.Linear(256, 1) 
        )
        # 2. Áp dụng khởi tạo trọng số riêng cho phần FC này
        self._initialize_weights()

    def _initialize_weights(self):
        # Hàm nội bộ để khởi tạo trọng số
        for m in self.model.fc.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.fill_(0.01)
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    # Test nhanh mô hình
    model = ResNet()
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print("Output shape:", y.shape) # Kì vọng: [2, 1]