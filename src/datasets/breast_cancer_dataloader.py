# Mission: xây dựng data loader chứa thông tin về dataset breast cancer.
# Author: Lê Văn Hoàn
# Version: 1.0
from torch.utils.data import DataLoader
from torchvision import transforms

from breast_cancer_dataset import BreastCancerDataset

def get_transforms(split):
    if split == "train":
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    else:  # val, test
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])


def create_dataloader(
    dataframe_csv_path,
    split,
    batch_size,
    shuffle,
    drop_last,
    num_workers=2,
    pin_memory=True
):
    """
    DataLoader patch-level IDC
    - Tham số truyền từ ngoài
    - Không hard-code batch_size, shuffle
    """

    dataset = BreastCancerDataset(
        csv_path=dataframe_csv_path,
        transform=get_transforms(split)
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return loader
