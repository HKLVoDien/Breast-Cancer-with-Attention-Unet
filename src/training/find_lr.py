import torch
import torch.nn as nn
from torch.optim import Adam
import pandas as pd
from tqdm import tqdm

# Import từ project hiện tại
from src.datasets.breast_cancer_dataloader import create_dataloader
from src.models.build_models import build_model

def get_lr_search_scheduler(optimizer, min_lr, max_lr, max_iterations):
    return torch.optim.lr_scheduler.CyclicLR(
        optimizer=optimizer, base_lr=min_lr, max_lr=max_lr, 
        step_size_up=max_iterations, step_size_down=max_iterations, 
        mode="triangular", cycle_momentum=False
    )

def find_learning_rate(model_name="attention_unet", batch_size=32, start_lr=1e-7, end_lr=0.1):
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    TRAIN_CSV = "data/metadata/train.csv"

    print(f"[INFO] Tìm LR cho {model_name}...")
    train_loader = create_dataloader(
        dataframe_csv_path=TRAIN_CSV, split="train", 
        batch_size=batch_size, shuffle=True, drop_last=True
    )

    model = build_model(name=model_name, in_channels=3).to(DEVICE)

    # Tính pos_weight
    train_df = pd.read_csv(TRAIN_CSV)
    num_pos = (train_df["target"] == 1).sum()
    num_neg = (train_df["target"] == 0).sum()
    pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)
    
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = Adam(model.parameters(), lr=start_lr)
    
    max_iterations = len(train_loader)
    scheduler = get_lr_search_scheduler(optimizer, start_lr, end_lr, max_iterations)

    lrs = []
    losses = []

    model.train()
    for batch in tqdm(train_loader, leave=False):
        images = batch["image"].to(DEVICE)
        labels = batch["label"].float().to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images).squeeze(1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()

        lrs.append(scheduler.get_last_lr()[0])
        losses.append(loss.item())
        scheduler.step()

    # Chỉ lưu dữ liệu thô ra file CSV
    csv_path = "lr_finder_results.csv"
    pd.DataFrame({"lr": lrs, "loss": losses}).to_csv(csv_path, index=False)
    print(f"[INFO] Đã lưu kết quả tại {csv_path}")

if __name__ == "__main__":
    find_learning_rate(model_name="attention_unet", batch_size=32)