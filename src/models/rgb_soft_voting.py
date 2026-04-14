# Mission: RGB Soft Voting Model — 3 nhánh AttentionUNET độc lập, mỗi nhánh nhận 1 kênh màu.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import torch.nn as nn
from src.models.attention_unet import AttentionUNET


class RGBSoftVotingModel(nn.Module):
    """
    Mô hình RGB Soft Voting:
      - Nhánh R: AttentionUNET nhận image_r (kênh đỏ repeat 3 lần)
      - Nhánh G: AttentionUNET nhận image_g (kênh xanh lá repeat 3 lần)
      - Nhánh B: AttentionUNET nhận image_b (kênh xanh dương repeat 3 lần)
    Kết quả: trung bình cộng 3 logit trước khi áp dụng sigmoid/loss.
    Loss vẫn dùng BCEWithLogitsLoss như các mô hình trước đó.

    Tham số:
        features: danh sách số kênh của AttentionUNET (mặc định [64, 128, 256])
    """

    def __init__(self, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256]
        self.branch_r = AttentionUNET(in_channels=3, features=features)
        self.branch_g = AttentionUNET(in_channels=3, features=features)
        self.branch_b = AttentionUNET(in_channels=3, features=features)

    def forward(self, image_r, image_g, image_b):
        """
        Args:
            image_r: Tensor (B, 3, H, W) — kênh R được repeat 3 lần
            image_g: Tensor (B, 3, H, W) — kênh G được repeat 3 lần
            image_b: Tensor (B, 3, H, W) — kênh B được repeat 3 lần
        Returns:
            logit_avg: Tensor (B,) — logit trung bình của 3 nhánh
        """
        logit_r = self.branch_r(image_r).squeeze(1)  # (B,)
        logit_g = self.branch_g(image_g).squeeze(1)
        logit_b = self.branch_b(image_b).squeeze(1)
        logit_avg = (logit_r + logit_g + logit_b) / 3.0
        return logit_avg


if __name__ == "__main__":
    model = RGBSoftVotingModel()
    x = torch.randn(2, 3, 48, 48)
    out = model(x, x, x)
    print("Output shape:", out.shape)  # (2,)
    print("Test passed!")
