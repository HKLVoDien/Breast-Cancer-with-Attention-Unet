import torch
from breast_cancer_dataloader import create_dataloader


CSV_PATH = "data/metadata/idc_metadata.csv"


def test_single_batch(loader):
    batch = next(iter(loader))

    image = batch["image"]
    label = batch["label"]
    patient_id = batch["patient_id"]
    x = batch["x"]
    y = batch["y"]

    print("=== Test single batch ===")
    print("Image shape:", image.shape)
    print("Label shape:", label.shape)
    print("Image dtype:", image.dtype)
    print("Patient IDs (first 3):", patient_id[:3])
    print("x coords (first 3):", x[:3])
    print("y coords (first 3):", y[:3])

    return image, label




def test_label_distribution(label):
    print("Unique labels in batch:", torch.unique(label))



def test_image_normalization(image):
    img = image[0]
    print("Min value:", img.min().item())
    print("Max value:", img.max().item())



def main():
    print("Creating DataLoader...")

    loader = create_dataloader(
        dataframe_csv_path=CSV_PATH,
        split="train",
        batch_size=8,
        shuffle=True,
        drop_last=False,
        num_workers=0,   # debug dễ
        pin_memory=False
    )

    image, label = test_single_batch(loader)
    test_label_distribution(label)
    test_image_normalization(image)



if __name__ == "__main__":
    main()
