import pandas as pd
from PIL import Image
from tqdm import tqdm
import os

def check_and_export_invalid_images(
    dataframe,
    expected_size=(50, 50),
    output_csv='data/metadata/invalid_images.csv'
):
    """
    Kiểm tra kích thước ảnh và xuất danh sách các ảnh lỗi (path, w, h) ra file CSV.
    """
    df = dataframe.copy()
    invalid_records = []

    print(f"[INFO] Đang kiểm tra {len(df)} ảnh...")
    print(f"[INFO] Kích thước mong đợi: {expected_size}")

    # Tạo thư mục chứa file output nếu chưa có
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["path"]
        
        try:
            with Image.open(img_path) as img:
                w, h = img.size  # PIL: (W, H)
            
            # Kiểm tra nếu kích thước không khớp
            if (h, w) != expected_size:
                invalid_records.append({
                    "path": img_path,
                    "w": w,
                    "h": h
                })
                
        except Exception as e:
            # Trường hợp file ảnh bị hỏng không mở được
            print(f"\n[EXCEPTION] Không thể đọc ảnh: {img_path}")
            invalid_records.append({
                "path": img_path,
                "w": -1,  # Đánh dấu -1 cho ảnh hỏng hoàn toàn
                "h": -1
            })

    # Xuất danh sách ra file CSV bằng Pandas
    if invalid_records:
        invalid_df = pd.DataFrame(invalid_records)
        invalid_df.to_csv(output_csv, index=False)
        print(f"\n[INFO] Đã tìm thấy {len(invalid_records)} ảnh lỗi.")
        print(f"[INFO] Danh sách chi tiết đã được lưu tại: {output_csv}")
    else:
        print("\n[INFO] Không có ảnh nào bị lỗi kích thước.")
        
    return invalid_records

if __name__ == "__main__":
    check_and_export_invalid_images(
        dataframe=pd.read_csv('data/metadata/idc_metadata.csv'),
        expected_size=(50, 50),
        output_csv='data/metadata/invalid_images.csv'
    )