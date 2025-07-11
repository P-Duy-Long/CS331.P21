import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split 

IMAGES_DIR = 'UTK500_labels/images'        
LABELS_DIR = 'UTK500_labels/labels'         
OUTPUT_DIR = 'UTK500_labels/processed_data' 
TARGET_SIZE = (128, 128)   

def load_and_preprocess_data(images_dir, labels_dir, target_size):
    """
    Tải ảnh và nhãn, tiền xử lý (resize, chuyển màu), và trả về dưới dạng NumPy arrays.
    """
    all_images = []
    all_labels = [] 

    # Lọc chỉ lấy các file ảnh (jpg, png, jpeg)
    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
    image_files.sort() 
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        # Lấy tên file ảnh không có đuôi để tìm file nhãn tương ứng
        base_name = os.path.splitext(img_file)[0]
        label_path = os.path.join(labels_dir, base_name + '.txt')

        if not os.path.exists(label_path):
            print(f"Warning: Label file for {img_file} ({label_path}) not found. Skipping.")
            continue

        # Đọc ảnh
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image {img_file}. Skipping.")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 


        with open(label_path, 'r') as f:
            lines = f.readlines()
            if not lines: # Ảnh không có nhãn
                print(f"Warning: No labels found in {label_path}. Skipping.")
                continue
            
            try:
                parts = list(map(float, lines[0].strip().split(' ')))
                bbox_norm = parts[1:]
                if len(bbox_norm) != 4:
                    raise ValueError("Label format incorrect (expected 4 values after class_id).")
            except (ValueError, IndexError) as e:
                print(f"Error parsing label for {img_file}: {e}. Skipping.")
                continue

        img_resized = cv2.resize(img, target_size)

        all_images.append(img_resized)
        all_labels.append(bbox_norm)

    return np.array(all_images), np.array(all_labels)

if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading and preprocessing data...")
    images, labels = load_and_preprocess_data(IMAGES_DIR, LABELS_DIR, TARGET_SIZE)
    print(f"Loaded {len(images)} images and {len(labels)} labels.")

    if len(images) == 0:
        print("No images found or processed. Exiting.")
        exit()


    images = images.astype('float32') / 255.0

    # --- PHẦN CHIA TẬP DATASET ---
    print("Splitting data into training and validation sets...")
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels, test_size=0.2, random_state=42
    )

    print(f"Training images: {len(X_train)} (80%)")
    print(f"Validation images: {len(X_val)} (20%)")


    print(f"Saving processed data to {OUTPUT_DIR}/...")
    np.save(os.path.join(OUTPUT_DIR, 'X_train.npy'), X_train)
    np.save(os.path.join(OUTPUT_DIR, 'y_train.npy'), y_train)
    np.save(os.path.join(OUTPUT_DIR, 'X_val.npy'), X_val)
    np.save(os.path.join(OUTPUT_DIR, 'y_val.npy'), y_val)
    print("Data saved successfully.")