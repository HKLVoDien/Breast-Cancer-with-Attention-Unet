#Task: Khởi tạo hàm set_seed để đảm bảo tính tái lập của mô hình trong quá trình huấn luyện.
#Author: Lê Văn Hoàn
#Version: 1.0   
import random
import numpy as np  
import torch

def set_seed(seed=42):
    """Cố định hạt giống ngẫu nhiên cho kết quả chuẩn khoa học"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True 
    torch.backends.cudnn.benchmark = False