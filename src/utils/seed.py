#Task: Khởi tạo hàm set_seed để đảm bảo tính tái lập của mô hình trong quá trình huấn luyện.
#Author: Lê Văn Hoàn
#Version: 1.0   
import os
import random
import numpy as np  
import torch

def set_seed(seed=42):
    """Cố định hạt giống ngẫu nhiên cho kết quả chuẩn khoa học"""
    # 1. Cố định Hash Seed của Python
    os.environ["PYTHONHASHSEED"] = str(seed)
    
    # 2. Cấu hình workspace cho cuBLAS (Bắt buộc nếu dùng thuật toán tất định trên GPU)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    # 3. Cố định Random của Python và Numpy
    random.seed(seed)
    np.random.seed(seed)
    
    # 4. Cố định Random của PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # Nếu dùng nhiều GPU
    
    # 5. Cấu hình CUDNN
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False
    
    # 6. Ép buộc PyTorch sử dụng thuật toán tất định (có thể làm chậm tốc độ train một chút)
    #torch.use_deterministic_algorithms(True, warn_only=True)
