# Mission: xây dựng dataframe chứa thông tin về dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 1.0
import os
import pandas as pd
import numpy as np
import torch
import re
from src.datasets.check_img_size import check_and_export_invalid_images
def count_images(base_path):
    folders = os.listdir(base_path)
    total_images = 0
    for patient_id in folders:
        patient_path = os.path.join(base_path, patient_id)
        if not os.path.isdir(patient_path):
            continue
        for c in ["0", "1"]:
            class_path = os.path.join(patient_path, c)
            if os.path.isdir(class_path):
                total_images += len(os.listdir(class_path))

    return total_images
def extract_xy_from_filename(filename):
    """
    Ví dụ filename:
    10253_idx5_x1001_y1001_class0.png
    """
    match = re.search(r"_x(\d+)_y(\d+)", filename)
    if match:
        x = int(match.group(1))
        y = int(match.group(2))
        return x, y
    else:
        raise ValueError(f"Cannot extract x,y from filename: {filename}")

def breast_cancer_dataframe(base_path):
    records = []

    for patient_id in os.listdir(base_path):
        patient_path = os.path.join(base_path, patient_id)
        if not os.path.isdir(patient_path):
            continue

        for c in ["0", "1"]:
            class_path = os.path.join(patient_path, c)
            if not os.path.isdir(class_path):
                continue

            for image_name in os.listdir(class_path):
                x, y = extract_xy_from_filename(image_name)
                records.append({
                    "patient_id": patient_id,
                    "path": os.path.join(class_path, image_name),
                    "target": int(c),
                    "x": x,
                    "y": y
                })

    return pd.DataFrame(records)

def build_metadata_csv(
    data_root="data/IDC_regular_ps50_idx5",
    output_csv="data/metadata/idc_metadata.csv"
):
    print("[INFO] Bắt đầu quét thư mục và tạo DataFrame ban đầu...")
    df = breast_cancer_dataframe(data_root)
    
    print("[INFO] Bắt đầu kiểm tra và lọc ảnh lỗi kích thước...")
    invalid_data = check_and_export_invalid_images(df) 
    
    # Nếu phát hiện có ảnh lỗi (invalid_data không rỗng)
    if invalid_data:
        # Trích xuất danh sách các đường dẫn ảnh bị lỗi
        invalid_paths = [record["path"] for record in invalid_data]
        
        # Lọc bỏ các đường dẫn này khỏi dataframe gốc
        df_clean = df[~df["path"].isin(invalid_paths)]
        
        print(f"[INFO] Đã loại bỏ {len(invalid_paths)} ảnh lỗi.")
        print(f"[INFO] Số lượng ảnh sạch còn lại: {len(df_clean)}")
    else:
        df_clean = df
        print("[INFO] Dữ liệu hoàn hảo, không phát hiện ảnh lỗi!")
    
    # Lưu dataframe đã được làm sạch
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    df_clean.to_csv(output_csv, index=False)
    print(f"[INFO] Đã lưu metadata tại: {output_csv}")

    return df_clean

def get_pos_weight(csv_path, device):
    """
    Hàm đọc file CSV và tính toán pos_weight cho hàm loss BCEWithLogitsLoss
    nhằm giải quyết bài toán mất cân bằng dữ liệu.
    """
    df = pd.read_csv(csv_path)
    
    # Đếm số lượng mẫu positive (1) và negative (0)
    num_pos = (df["target"] == 1).sum()
    num_neg = (df["target"] == 0).sum()
    
    # Tính tỷ lệ
    pos_weight_value = num_neg / num_pos
    pos_weight = torch.tensor([pos_weight_value]).to(device)
    
    return pos_weight, float(pos_weight_value)


if __name__ == "__main__":
    base_path = "data/IDC_regular_ps50_idx5"
    total_images = count_images(base_path)
    print(f"Tổng số ảnh trong dataset: {total_images}")
    df = breast_cancer_dataframe(base_path)
    print(df.head())
    
    # Tạo thư mục lưu metadata 
    output_dir = "data/metadata"
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(
        os.path.join(output_dir, "idc_metadata.csv"),
        index=False
    )

    print("Saved metadata dataframe to idc_metadata.csv")