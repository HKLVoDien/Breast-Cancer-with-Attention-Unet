# Mission: Cấu hình mặc định cho dự án Breast Cancer.
# Author: Lê Văn Hoàn
# Version: 1.0
# configs/default_config.py
import torch
import os
import datetime
class Config:
    # System
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SEED = 42
    # Data paths
    DATA_ROOT = "data/IDC_regular_ps50_idx5"
    TRAIN_CSV = "data/metadata/train.csv"
    VAL_CSV   = "data/metadata/val.csv"
    TEST_CSV  = "data/metadata/test.csv"
    METADATA_CSV = "data/metadata/idc_metadata.csv"
    # DataLoader parameters
    TRAIN_SHUFFLE = True
    VAL_SHUFFLE = False
    TEST_SHUFFLE = False
    TRAIN_DROP_LAST = True
    VAL_DROP_LAST = False
    TEST_DROP_LAST = False
    # Training Hyperparameters
    EPOCHS = 30
    BATCH_SIZE = 32
    BASE_LR = 1e-4 # Cập nhật sau khi dùng LR Finder
    MAX_LR = 1e-3 # Cập nhật sau khi dùng LR Finder

    # Results paths
    RESULTS_DIR = "results"
    # WandB
    Turn_WandB_Off = True  # Đặt True để tắt hoàn toàn WandB (dành cho debug hoặc khi không muốn log)
    @classmethod
    def update_from_args(cls, args):
        """
        Hàm này giúp cập nhật các cấu hình dựa trên tham số 
        người dùng truyền vào từ Command Line (thông qua argparse).
        """
        if hasattr(args, 'batch_size') and args.batch_size is not None:
            cls.BATCH_SIZE = args.batch_size
        if hasattr(args, 'lr') and args.lr is not None:
            cls.BASE_LR = args.lr
        if hasattr(args, 'max_lr') and args.max_lr is not None:
            cls.MAX_LR = args.max_lr
        if hasattr(args, 'epochs') and args.epochs is not None:
            cls.EPOCHS = args.epochs
    # Tạo timestamp theo chuẩn: NămThángNgày (Ví dụ: 2024_06_15)
    timestamp = datetime.datetime.now().strftime("%Y_%m_%d")
    @classmethod
    def get_exp_dir(cls, model_name):
        """Tạo đường dẫn thư mục lưu kết quả riêng cho từng lần chạy"""
        exp_name = f"{model_name}_bs{cls.BATCH_SIZE}_lr{cls.BASE_LR}_epochs{cls.EPOCHS}_{cls.timestamp}"
        return os.path.join(cls.RESULTS_DIR, exp_name)
config = Config()
