# Mission: Điều khiển luồng mô hình dùng để ứng dụng.
# Author: Lê Văn Hoàn
# Version: 1.0
# src/models/build_model.py

from src.models.unet import UNET
from src.models.attention_unet import AttentionUNET
from src.models.resnet import ResNet
from src.models.monai_attention_unet import MonaiAttentionUNet
from src.models.rgb_soft_voting import RGBSoftVotingModel
from src.models.CA_attention_unet import AttentionUNET as CA_AttentionUNET
import torch


def build_model(name: str, **kwargs):
    name = name.lower()

    if name == "unet":
        return UNET(**kwargs)

    elif name == "attention_unet":
        return AttentionUNET(**kwargs)
    elif name == "attention_unet_v2":
        # Thêm dòng này: Khởi tạo bản V2 với giảm
        return AttentionUNET(features=[16, 32, 64, 128], **kwargs)
    elif name == "ca_attention_unet":
        return CA_AttentionUNET(features=[16, 32, 64, 128], **kwargs)
    elif name == "resnet":
        return ResNet(pretrained=True)
    elif name == "monai_attention_unet":
        return MonaiAttentionUNet(**kwargs)

    elif name == "rgb_soft_voting":
        return RGBSoftVotingModel()

    else:
        raise ValueError(
            f"Unknown model '{name}'. "
            "Available: unet | attention_unet | attention_unet_v2 | ca_attention_unet | resnet | monai_attention_unet | rgb_soft_voting"
        )


if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Test xây dựng mô hình
    model_names = [
        "unet",
        "attention_unet",
        "attention_unet_v2",
        "ca_attention_unet",
        "resnet",
    ]
    for name in model_names:
        model = build_model(name, in_channels=3).to(DEVICE)
        x = torch.randn(2, 3, 224, 224).to(DEVICE)

        with torch.no_grad():
            y = model(x)

        print(name, y.shape)
