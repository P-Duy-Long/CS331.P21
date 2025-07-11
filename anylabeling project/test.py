import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# --- Cấu hình đường dẫn ---
MODEL_PATH = 'UTK500_labels/face_localization_model.h5'
IMG_HEIGHT, IMG_WIDTH = 128, 128 

def resize_and_pad(image, target_size):
    """
    Resizes an image to fit within target_size while maintaining aspect ratio,
    and pads the remaining space with black.
    Returns the processed image and the scaling factors and padding applied.
    """
    h_orig, w_orig = image.shape[:2]
    target_h, target_w = target_size
    scale = min(target_w / w_orig, target_h / h_orig)
    new_w = int(w_orig * scale)
    new_h = int(h_orig * scale)

    # Resize ảnh
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    pad_w = target_w - new_w
    pad_h = target_h - new_h
    
    # Pad evenly on both sides
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    padded_image = cv2.copyMakeBorder(resized_image, 
                                      pad_top, pad_bottom, 
                                      pad_left, pad_right, 
                                      cv2.BORDER_CONSTANT, value=[0, 0, 0]) # Black padding

    return padded_image, scale, pad_left, pad_top

# --- Hàm Denormalize Bounding Box (ĐÃ ĐIỀU CHỈNH ĐỂ XỬ LÝ PADDING) ---
def denormalize_bbox_with_padding(bbox_norm, original_width, original_height, 
                                  img_width_model, img_height_model, 
                                  scale_factor, pad_left, pad_top):
    x_center_norm, y_center_norm, w_norm, h_norm = bbox_norm

    x_center_padded = x_center_norm * img_width_model
    y_center_padded = y_center_norm * img_height_model
    w_padded = w_norm * img_width_model
    h_padded = h_norm * img_height_model

    x_min_padded = x_center_padded - w_padded / 2
    y_min_padded = y_center_padded - h_padded / 2
    x_max_padded = x_center_padded + w_padded / 2
    y_max_padded = y_center_padded + h_padded / 2

    x_min_resized = x_min_padded - pad_left
    y_min_resized = y_min_padded - pad_top
    x_max_resized = x_max_padded - pad_left
    y_max_resized = y_max_padded - pad_top

    x_min_orig = int(x_min_resized / scale_factor)
    y_min_orig = int(y_min_resized / scale_factor)
    x_max_orig = int(x_max_resized / scale_factor)
    y_max_orig = int(y_max_resized / scale_factor)

    x_min_orig = max(0, x_min_orig)
    y_min_orig = max(0, y_min_orig)
    x_max_orig = min(original_width - 1, x_max_orig)
    y_max_orig = min(original_height - 1, y_max_orig)
    
    return [x_min_orig, y_min_orig, x_max_orig, y_max_orig]

if __name__ == "__main__":
    # --- Tải mô hình đã huấn luyện ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        # Khi tải mô hình, chỉ định hàm loss và metrics tùy chỉnh
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError(), 
                          'MeanSquaredError': tf.keras.metrics.MeanSquaredError()} 
        
        model = load_model(MODEL_PATH, custom_objects=custom_objects) 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure {MODEL_PATH} exists and is a valid Keras model file.")
        exit()

    # --- Khởi tạo camera ---
    cap = cv2.VideoCapture(0) # 0 thường là webcam mặc định

    if not cap.isOpened():
        print("Error: Could not open camera. Please check if camera is connected or in use.")
        exit()

    print("\n--- Realtime Face Localization ---")
    print("This model predicts a SINGLE bounding box per frame.")
    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break

        original_height, original_width, _ = frame.shape

        processed_frame, scale_factor, pad_left, pad_top = resize_and_pad(frame, (IMG_WIDTH, IMG_HEIGHT))
        
        processed_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB) 
        processed_frame = processed_frame / 255.0 
        processed_frame = np.expand_dims(processed_frame, axis=0)

        # --- Dự đoán ---
        predictions_norm = model.predict(processed_frame, verbose=0)[0] 

        bbox_denorm = denormalize_bbox_with_padding(predictions_norm, 
                                                   original_width, original_height, 
                                                   IMG_WIDTH, IMG_HEIGHT,
                                                   scale_factor, pad_left, pad_top)
        x_min, y_min, x_max, y_max = bbox_denorm

        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2) 
        cv2.putText(frame, "Face", (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        cv2.imshow('Realtime Face Localization', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows() 
    print("Realtime localization stopped.")