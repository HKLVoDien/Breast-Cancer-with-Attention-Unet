# Mission: Chạy luồng xử lý chính: xây dựng metadata và huấn luyện mô hình UNet.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch
import argparse
from src.datasets.breast_cancer_dataframe import build_metadata_csv
from src.datasets.split_dataset import main as split_main
from src.training.train_unet import main as train_unet


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--build-data",
        action="store_true",
        help="Rebuild metadata CSV before training"
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    #Nếu cần xây dựng lại metadata thì thực hiện 
    if args.build_data:
        print("[INFO] Building metadata CSV...")
        build_metadata_csv()
        split_main()
        print("[INFO] Metadata CSV built successfully.")
    
    print("[INFO] Starting training...")
    train_unet()
