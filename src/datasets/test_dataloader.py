import torch
from breast_cancer_dataloader import create_dataloader


CSV_PATH = "../../data/metadata/idc_metadata.csv"


def test_single_batch(loader):
    """
    Test 1 batch duy nhất
    """
    batch = next(iter(loader))

    print("=== Test single batch ===")
    print("Batch keys:", batch.keys())
    print("Image shape:", batch["image"].shape)
    print("Label shape:", batch["label"].shape)
    print("Image dtype:", batch["image"].dtype)
    print("Patient IDs (first 3):", batch["patient_id"][:3])
    print("x coords (first 3):", batch["x"][:3])
    print("y coords (first 3):", batch["y"][:3])

    return batch


def test_label_distribution(batch):
    """
    Kiểm tra label có đúng 0/1 không
    """
    labels = batch["label"]
    print("\n=== Test label distribution ===")
    print("Unique labels in batch:", torch.unique(labels))


def test_image_normalization(batch):
    """
    Kiểm tra normalize có hoạt động không
    """
    img = batch["image"][0]

    print("\n=== Test image normalization ===")
    print("Min value:", img.min().item())
    print("Max value:", img.max().item())


def test_multiple_batches(loader, num_batches=3):
    """
    Test nhiều batch liên tiếp
    """
    print("\n=== Test multiple batches ===")
    for i, batch in enumerate(loader):
        print(f"Batch {i}: image shape {batch['image'].shape}")
        if i + 1 >= num_batches:
            break


def main():
    print("Creating DataLoader...")

    loader = create_dataloader(
        dataframe_csv_path=CSV_PATH,
        split="train",
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=2,   # debug dễ
        pin_memory=False
    )

    batch = test_single_batch(loader)
    test_label_distribution(batch)
    test_image_normalization(batch)
    test_multiple_batches(loader)


if __name__ == "__main__":
    main()
