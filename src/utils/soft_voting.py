# Mission: Cung cấp hàm tiện ích cho cơ chế Soft Voting.
# Author: Lê Văn Hoàn
# Version: 1.0

import torch


def get_soft_voting_preds(logit_r, logit_g, logit_b, threshold=0.6):
    """
    Thực hiện Soft Voting trên cấp độ Logit.
    Gộp 3 logit từ 3 nhánh độc lập, tính trung bình, đưa qua hàm Sigmoid
    và trả về nhãn dự đoán cùng xác suất tương ứng.
    """
    # 1. Tính trung bình Logit (Giữ nguyên phổ giá trị -inf đến +inf)
    logit_avg = (logit_r + logit_g + logit_b) / 3.0

    # 2. Đưa qua Sigmoid để lấy xác suất
    probs_avg = torch.sigmoid(logit_avg)

    # 3. Áp dụng ngưỡng (Threshold)
    preds = (probs_avg >= threshold).long()

    return preds, probs_avg
