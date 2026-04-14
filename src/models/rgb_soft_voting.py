# Mission: RGB Soft Voting Model — 3 nhánh AttentionUNET độc lập, mỗi nhánh nhận 1 kênh màu.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch
import torch.nn as nn
from src.models.soft_attention_unet import AttentionUNET


class RGBSoftVotingModel(nn.Module):
    """
    Mô hình RGB Soft Voting:
      - Nhánh R: AttentionUNET nhận image_r
      - Nhánh G: AttentionUNET nhận image_g
      - Nhánh B: AttentionUNET nhận image_b
    Kết quả: trung bình cộng 3 logit trước khi áp dụng sigmoid/loss.
    Loss vẫn dùng BCEWithLogitsLoss như các mô hình trước đó.
    """

    def __init__(self, features=None):
        super().__init__()
        if features is None:
            features = [16, 32, 64, 128]
        self.branch_r = AttentionUNET(in_channels=1, features=features)
        self.branch_g = AttentionUNET(in_channels=1, features=features)
        self.branch_b = AttentionUNET(in_channels=1, features=features)

    def forward(self, image_r, image_g, image_b):
        logit_r = self.branch_r(image_r).squeeze(1)  # (B,)
        logit_g = self.branch_g(image_g).squeeze(1)
        logit_b = self.branch_b(image_b).squeeze(1)

        return logit_r, logit_g, logit_b


if __name__ == "__main__":
    model = RGBSoftVotingModel()
    x = torch.randn(2, 3, 48, 48)
    out = model(x, x, x)
    print("Output shape:", out.shape)  # (2,)
    print("Test passed!")
