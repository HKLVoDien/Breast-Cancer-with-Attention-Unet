# Mission: Điều khiển luồng mô hình dùng để ứng dụng.
# Author: Lê Văn Hoàn
# Version: 1.0
# src/models/build_model.py

from src.models.unet import UNET
from src.models.attention_unet import AttentionUNET
import torch
def build_model(name: str, **kwargs):
    name = name.lower()

    if name == "unet":
        return UNET(**kwargs)

    elif name == "attention_unet":
        return AttentionUNET(**kwargs)

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Available: unet | attention_unet | resnet"
        )

if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    #Test xây dựng mô hình
    model_names = ["unet", "attention_unet"]
    for name in model_names:
        model = build_model(name, in_channels=3).to(DEVICE)
        x = torch.randn(2, 3, 224, 224).to(DEVICE)

        with torch.no_grad():
            y = model(x)

        print(name, y.shape)