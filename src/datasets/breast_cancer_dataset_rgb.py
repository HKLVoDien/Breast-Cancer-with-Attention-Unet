# Mission: Dataset tách kênh màu RGB cho phương án RGB Soft Voting.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset


class BreastCancerDatasetRGB(Dataset):

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
            image = self.transform(image)  # (3, H, W)

        # Tách từng kênh màu
        r = image[0:1, :, :]  # (1, H, W)
        g = image[1:2, :, :]
        b = image[2:3, :, :]

        if "target" in self.df.columns:
            target = int(self.df.target.values[idx])
        else:
            target = None

        return {
            "image_r": r,
            "image_g": g,
            "image_b": b,
            "label": target,
            "patient_id": patient_id,
            "x": x_coord,
            "y": y_coord,
        }


if __name__ == "__main__":
    import torchvision.transforms as transforms

    CSV_PATH = "data/metadata/idc_metadata.csv"

    test_transform = transforms.Compose(
        [
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
        ]
    )

    dataset = BreastCancerDatasetRGB(csv_path=CSV_PATH, transform=test_transform)
    print("Dataset length:", len(dataset))

    sample = dataset[0]
    print("Sample keys:", list(sample.keys()))
    print("image_r shape:", sample["image_r"].shape)
    print("image_g shape:", sample["image_g"].shape)
    print("image_b shape:", sample["image_b"].shape)
    print("Label:", sample["label"])
