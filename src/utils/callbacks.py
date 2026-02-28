# Mission: Định nghĩa callback EarlyStopping để dừng huấn luyện sớm nếu mô hình không cải thiện trên tập Val.
# Author: Lê Văn Hoàn
# Version: 1.0
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == 'max' and score < self.best_score + self.min_delta) or \
             (self.mode == 'min' and score > self.best_score - self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0