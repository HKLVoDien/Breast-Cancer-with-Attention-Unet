# Mission: DataLoader tách kênh màu RGB cho phương án RGB Soft Voting.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from .breast_cancer_dataset_rgb import BreastCancerDatasetRGB
from configs.default_configs import Config


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_transforms_rgb(split):
    """
    Giống get_transforms trong breast_cancer_dataloader.py nhưng tách riêng để
    không ảnh hưởng pipeline cũ.
    """
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(degrees=90),
                transforms.Resize((48, 48)),
                transforms.ColorJitter(
                    brightness=0.2,  # Độ sáng: Thay đổi ±20%
                    contrast=0.2,  # Độ tương phản: Thay đổi ±20%
                    saturation=0.2,  # Độ bão hòa màu (đậm/nhạt của thuốc nhuộm): ±20%
                    hue=0.05,  # Sắc độ màu:(±5%)
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:  # val, test
        return transforms.Compose(
            [
                transforms.Resize((48, 48)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )


def create_dataloader_rgb(
    dataframe_csv_path,
    split,
    batch_size,
    shuffle,
    drop_last,
    num_workers=2,
    pin_memory=True,
):
    """
    DataLoader RGB — trả về batch có keys: image_r, image_g, image_b, label.
    Dùng cho RGBSoftVotingModel và Train_model_RGB.
    """
    dataset = BreastCancerDatasetRGB(
        csv_path=dataframe_csv_path,
        transform=get_transforms_rgb(split),
    )

    g = torch.Generator()
    g.manual_seed(Config.SEED)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory,
        worker_init_fn=seed_worker,
        generator=g,
    )

    return loader
