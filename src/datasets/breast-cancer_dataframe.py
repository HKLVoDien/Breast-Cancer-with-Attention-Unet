# Mission: xây dựng dataframe chứa thông tin về dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 1.0
import os
import pandas as pd
import numpy as np
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
                records.append({
                    "patient_id": patient_id,
                    "path": os.path.join(class_path, image_name),
                    "target": int(c)
                })

    return pd.DataFrame(records)


if __name__ == "__main__":
    base_path = "../../data/IDC_regular_ps50_idx5"
    total_images = count_images(base_path)
    print(f"Tổng số ảnh trong dataset: {total_images}")
    df = breast_cancer_dataframe(base_path)
    print(df.head())
    
    # Tạo thư mục lưu metadata 
    output_dir = "../../data/metadata"
    os.makedirs(output_dir, exist_ok=True)

    df.to_csv(
        os.path.join(output_dir, "idc_metadata.csv"),
        index=False
    )

    print("Saved metadata dataframe to idc_metadata.csv")