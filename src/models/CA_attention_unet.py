# Task: Thiết kế model Attention UNET kết hợp Channel Attention cho phân loại ảnh y tế sử dụng PyTorch.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# ==========================================
# 1. KHỐI CHANNEL ATTENTION (Lấy từ tư tưởng của CBAM)
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # Shared MLP (Mạng nơ-ron dùng chung)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        # Cộng hai nhánh lại và qua Sigmoid để ra trọng số cho từng kênh
        out = avg_out + max_out
        return self.sigmoid(out)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # TÍCH HỢP CHANNEL ATTENTION VÀO ĐUÔI DOUBLE CONV
        ratio = 16 if out_channels >= 16 else out_channels
        self.ca = ChannelAttention(out_channels, ratio=ratio)

    def forward(self, x):
        # Đi qua 2 lớp chập và ReLU
        x = self.conv(x)
        # Đi qua khối Channel Attention để nhân trọng số (Lọc "CÁI GÌ")
        x = x * self.ca(x)
        return x


class AttentionGate(nn.Module):
    def __init__(self, F_g, F_x, F_int):
        super().__init__()

        # Decoder feature
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )

        # Encoder feature (DOWN-SAMPLE)
        self.W_x = nn.Sequential(
            nn.Conv2d(F_x, F_int, kernel_size=1, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(F_int),
        )

        # Attention coefficient
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid(),
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        """
        g: decoder feature (low-res)
        x: encoder feature (high-res)
        """

        g1 = self.W_g(g)
        x1 = self.W_x(x)  # (B, F_int, H/2, W/2)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # (B, 1, H/2, W/2)

        # === PAPER STEP: UPSAMPLE ATTENTION MAP ===
        psi = torch.nn.functional.interpolate(
            psi, size=x.shape[2:], mode="bilinear", align_corners=False
        )

        return x * psi


class AttentionUNET(nn.Module):
    def __init__(
        self,
        in_channels=3,
        features=[64, 128, 256],
    ):
        super(AttentionUNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.attentions = nn.ModuleList()

        # Attention Gates
        for feature in reversed(features):
            self.attentions.append(
                AttentionGate(F_g=feature * 2, F_x=feature, F_int=feature // 2)
            )

        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature * 2,
                    feature,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(DoubleConv(feature * 2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        # Dropout cho bottleneck
        self.dropout = nn.Dropout(0.5)
        # cho binary classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),  # Dropout nhẹ trước khi phân loại
            nn.Linear(features[0], 1),
        )

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        x = self.dropout(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            # attention gate
            g = x  # decoder
            x_skip = skip_connections[idx // 2]  # encoder
            attn_skip = self.attentions[idx // 2](g, x_skip)
            # upsample
            x = self.ups[idx](x)
            # Xử lý kích thước nếu lẻ (như 161x161)
            if x.shape[2:] != attn_skip.shape[2:]:
                x = TF.resize(x, size=attn_skip.shape[2:])
            # concat
            concat = torch.cat((attn_skip, x), dim=1)
            x = self.ups[idx + 1](concat)

        # return self.final_conv(x)
        # === classification ===
        x = self.global_pool(x)  # (B, C, 1, 1)
        x = x.view(x.size(0), -1)  # (B, C)
        x = self.classifier(x)  # (B, 1)
        return x


def test():
    x = torch.randn((3, 1, 48, 48))
    model = AttentionUNET(in_channels=1)
    # Dành cho segmentation

    # preds = model(x)
    # assert preds.shape == x.shape

    # Dành cho binary classification
    y = model(x)
    print(y.shape)
    print("Test passed!")


if __name__ == "__main__":
    test()
