import os

# Định nghĩa các kịch bản
batch_sizes = [32, 64, 128]
epochs = 30
base_lr = 1e-4
max_lr = 1e-3
print("=== BẮT ĐẦU CHẠY THỬ NGHIỆM TỰ ĐỘNG ===")
for bs in batch_sizes:
        print(f"\n---> Đang chạy kịch bản: Batch Size = {bs} ")
        # Lệnh gọi command line
        os.system(f"python run.py --model attention_unet --batch-size {bs} --lr {base_lr} --max-lr {max_lr} --epochs {epochs}")
        
print("\n=== ĐÃ HOÀN THÀNH TOÀN BỘ THỬ NGHIỆM! ===")