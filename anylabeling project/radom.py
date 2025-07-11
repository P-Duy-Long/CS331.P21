import os
import shutil
import random

source_dir = "UTKFace" 
target_dir = "UTK500"  

if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# Lấy danh sách tất cả file ảnh trong thư mục gốc
image_files = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

# Chọn ngẫu nhiên 500 ảnh
selected_images = random.sample(image_files, 500)

for img in selected_images:
    source_path = os.path.join(source_dir, img)
    target_path = os.path.join(target_dir, img)
    shutil.copy(source_path, target_path)

print("Đã sao chép 500 ảnh vào thư mục:", target_dir)