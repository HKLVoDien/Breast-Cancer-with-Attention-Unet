import pandas as pd
from PIL import Image
from tqdm import tqdm


def check_image_sizes(
    csv_path,
    expected_size=(50, 50),
    max_errors=20
):
    """
    Kiểm tra kích thước ảnh từ metadata CSV

    Args:
        csv_path (str): đường dẫn file CSV
        expected_size (tuple): (H, W) mong đợi
        max_errors (int): số lỗi tối đa in ra

    Returns:
        List[dict]: danh sách ảnh lỗi
    """

    df = pd.read_csv(csv_path)

    errors = []

    print(f"[INFO] Checking {len(df)} images...")
    print(f"[INFO] Expected size: {expected_size}")

    for idx, row in tqdm(df.iterrows(), total=len(df)):
        img_path = row["path"]
        label = row["target"]
        patient_id = row.get("patient_id", "N/A")

        try:
            with Image.open(img_path) as img:
                w, h = img.size  # PIL: (W, H)

            if (h, w) != expected_size:
                error = {
                    "patient_id": patient_id,
                    "label": label,
                    "image_path": img_path,
                    "height": h,
                    "width": w
                }
                errors.append(error)

                print(
                    f"[ERROR] patient_id={patient_id}, "
                    f"label={label}, "
                    f"size=({h},{w})\n"
                    f"        path={img_path}"
                )

                if len(errors) >= max_errors:
                    print("[STOP] Reached max_errors limit")
                    break

        except Exception as e:
            print(f"[EXCEPTION] Cannot read image: {img_path}")
            print(e)

    print(f"\n[SUMMARY] Found {len(errors)} invalid images")
    return errors
if __name__ == "__main__":
    check_image_sizes(
        csv_path='data/metadata/idc_metadata.csv',
        expected_size=(50, 50),
        max_errors=50
    )
