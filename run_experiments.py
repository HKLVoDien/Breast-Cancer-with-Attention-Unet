import os

# Định nghĩa các kịch bản
batch_sizes = [16, 32]
learning_rates = [1e-3, 1e-4]

print("=== BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG ===")
for bs in batch_sizes:
    for lr in learning_rates:
        print(f"\n---> Đang chạy kịch bản: Batch Size = {bs}, LR = {lr}")
        # Lệnh gọi command line
        os.system(f"python run.py --model attention_unet --batch-size {bs} --lr {lr}")
        
print("\n=== ĐÃ HOÀN THÀNH TOÀN BỘ THỬ NGHIỆM! ===")