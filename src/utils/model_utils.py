# Mission: Các hàm tiện ích liên quan đến mô hình, như load checkpoint, v.v.
# Author: Lê Văn Hoàn
# Version: 1.0
import torch

def load_best_model(model, ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model