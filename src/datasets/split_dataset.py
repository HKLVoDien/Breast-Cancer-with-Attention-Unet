# Mission: Chia tập dữ liệu thành các phần train, val, test.
# Author: Lê Văn Hoàn
# Version: 1.0
import os
import pandas as pd
from sklearn.model_selection import train_test_split

CSV_PATH = "data/metadata/idc_metadata.csv"
OUTPUT_DIR = "data/metadata"
RANDOM_STATE = 0


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df = pd.read_csv(CSV_PATH)

    # Thêm hàm sorted() để chốt cứng thứ tự ID bệnh nhân (ví dụ: từ bé đến lớn)
    patients = sorted(df.patient_id.unique())
    train_ids, sub_ids = train_test_split(
        patients, test_size=0.3, random_state=RANDOM_STATE
    )

    val_ids, test_ids = train_test_split(
        sub_ids, test_size=0.5, random_state=RANDOM_STATE
    )

    train_df = df[df.patient_id.isin(train_ids)]
    val_df = df[df.patient_id.isin(val_ids)]
    test_df = df[df.patient_id.isin(test_ids)]

    train_df.to_csv(os.path.join(OUTPUT_DIR, "train.csv"), index=False)
    val_df.to_csv(os.path.join(OUTPUT_DIR, "val.csv"), index=False)
    test_df.to_csv(os.path.join(OUTPUT_DIR, "test.csv"), index=False)

    print("Split done:")
    print(f"Train patients: {len(train_ids)}")
    print(f"Val patients:   {len(val_ids)}")
    print(f"Test patients:  {len(test_ids)}")


if __name__ == "__main__":
    main()
