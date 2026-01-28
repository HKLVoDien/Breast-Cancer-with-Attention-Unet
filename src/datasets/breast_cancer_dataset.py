# Mission: Lấy dataset breast cancer từ thư mục và chuẩn bị cho việc huấn luyện model.
# Author: Lê Văn Hoàn
# Version: 1.0

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BreastCancerDataset(Dataset):

    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        patient_id = self.df.patient_id.values[idx]
        x_coord = self.df.x.values[idx]
        y_coord = self.df.y.values[idx]
        image_path = self.df.path.values[idx]

        image = Image.open(image_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if "target" in self.df.columns:
            target = int(self.df.target.values[idx])
        else:
            target = None

        return {
            "image": image,
            "label": target,
            "patient_id": patient_id,
            "x": x_coord,
            "y": y_coord
        }


if __name__ == "__main__":
    import torchvision.transforms as transforms

    CSV_PATH = "data/metadata/idc_metadata.csv"

    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    dataset = BreastCancerDataset(
        csv_path=CSV_PATH,
        transform=test_transform,
    )

    print("Dataset length:", len(dataset))

    sample = dataset[0]

    print("\nSample keys:", sample.keys())
    print("Patient ID:", sample["patient_id"])
    print("x, y:", sample["x"], sample["y"])
    print("Label:", sample["label"])
    print("Image tensor shape:", sample["image"].shape)
    print("Image dtype:", sample["image"].dtype)
