import os
import numpy as np
import tensorflow as tf
import cv2 
import matplotlib.pyplot as plt
import random 
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError 
from sklearn.metrics import average_precision_score # Thêm để tính AP/mAP

# --- Cấu hình đường dẫn ---
MODEL_PATH = 'UTK500_labels/face_localization_model.h5'
PROCESSED_DATA_DIR = 'UTK500_labels/processed_data'
ORIGINAL_IMAGES_DIR = 'UTK500_labels/images' 

IMG_HEIGHT, IMG_WIDTH = 128, 128

def denormalize_bbox(bbox_norm, original_width, original_height):
    """
    Chuyển đổi bounding box từ tọa độ chuẩn hóa [0, 1] về tọa độ pixel thực tế.
    bbox_norm: [x_center_norm, y_center_norm, w_norm, h_norm]
    Trả về: [x_min, y_min, x_max, y_max]
    """
    x_center, y_center, width, height = bbox_norm

    x_min = int((x_center - width / 2) * original_width)
    y_min = int((y_center - height / 2) * original_height)
    x_max = int((x_center + width / 2) * original_width)
    y_max = int((y_center + height / 2) * original_height)

    # Đảm bảo tọa độ không vượt quá biên ảnh
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(original_width - 1, x_max)
    y_max = min(original_height - 1, y_max)
    
    return [x_min, y_min, x_max, y_max]

def calculate_iou(boxA, boxB):
    """
    Tính Intersection over Union (IoU) giữa hai bounding box.
    boxA, boxB: [x_min, y_min, x_max, y_max]
    """
    # Xác định tọa độ của hình chữ nhật giao nhau
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxB[3], boxB[3]) # Sửa lỗi chính tả: boxB[3] thay vì boxB[2]
    
    # Tính diện tích phần giao nhau
    inter_width = max(0, xB - xA + 1)
    inter_height = max(0, yB - yA + 1)
    inter_area = inter_width * inter_height

    # Tính diện tích của hai hình chữ nhật
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # Tính diện tích phần hợp
    iou = inter_area / float(boxA_area + boxB_area - inter_area)

    return iou

if __name__ == "__main__":
    # --- Tải mô hình đã huấn luyện ---
    print(f"Loading model from {MODEL_PATH}...")
    try:
        custom_objects = {'mse': tf.keras.losses.MeanSquaredError(), 
                          'MeanSquaredError': tf.keras.metrics.MeanSquaredError()} 
        
        model = load_model(MODEL_PATH, custom_objects=custom_objects) 
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
        print(f"Please ensure {MODEL_PATH} exists and is a valid Keras model file.")
        exit()

    # --- Tải dữ liệu validation ---
    print(f"Loading validation data from {PROCESSED_DATA_DIR}/...")
    try:
        X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
        print("Validation data loaded successfully.")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    except FileNotFoundError:
        print(f"Error: Validation data files not found in {PROCESSED_DATA_DIR}.")
        print("Please ensure prepare_data.py has been run successfully.")
        exit()

    # --- Đánh giá mô hình trên tập validation (Loss & MSE) ---
    print("\n--- Evaluating Model Metrics (Loss, MSE) ---")
    loss, mse = model.evaluate(X_val, y_val, verbose=1)
    print(f"Validation Loss (MSE): {loss:.6f}")
    print(f"Validation Mean Squared Error: {mse:.6f}")

    # --- Thực hiện dự đoán trên tập validation ---
    print("\n--- Making Predictions for Advanced Metrics ---")
    predictions_norm = model.predict(X_val) # Dự đoán bounding box chuẩn hóa

    # --- Plotting MSE distribution for each coordinate ---
    print("\n--- Plotting MSE distribution for bounding box coordinates ---")
    squared_errors = (predictions_norm - y_val)**2

    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    coord_names = ['X-Center', 'Y-Center', 'Width', 'Height']

    for i in range(4):
        axes[i].hist(squared_errors[:, i], bins=50, color='skyblue', edgecolor='black')
        axes[i].set_title(f'Squared Error Distribution for {coord_names[i]}')
        axes[i].set_xlabel('Squared Error')
        axes[i].set_ylabel('Frequency')
        axes[i].grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()


    confidence_scores = np.ones(len(X_val)) # Giả định tất cả dự đoán đều có độ tin cậy 1.0

    all_detections = [] # Format: [image_id, class_id, confidence_score, x_min, y_min, x_max, y_max]
    all_ground_truths = [] # Format: [image_id, class_id, x_min, y_min, x_max, y_max]


    for i in range(len(X_val)):
        gt_bbox_norm = y_val[i] 
        pred_bbox_norm = predictions_norm[i]
        
        # Denormalize bbox về kích thước 128x128 để tính IoU.
        gt_bbox_denorm = denormalize_bbox(gt_bbox_norm, IMG_WIDTH, IMG_HEIGHT)
        pred_bbox_denorm = denormalize_bbox(pred_bbox_norm, IMG_WIDTH, IMG_HEIGHT)
        
        # Class_id = 0 cho lớp 'face' duy nhất
        all_ground_truths.append([i, 0, gt_bbox_denorm[0], gt_bbox_denorm[1], gt_bbox_denorm[2], gt_bbox_denorm[3]])
        all_detections.append([i, 0, confidence_scores[i], pred_bbox_denorm[0], pred_bbox_denorm[1], pred_bbox_denorm[2], pred_bbox_denorm[3]])

    all_detections = np.array(all_detections)
    all_ground_truths = np.array(all_ground_truths)

    print("\n--- Calculating AP and mAP ---")


    iou_thresholds = np.arange(0.5, 1.0, 0.05) 
    
    average_precisions_at_iou = []

    for iou_thresh in iou_thresholds:
        # Sắp xếp các dự đoán theo điểm tin cậy giảm dần
        sorted_detections = all_detections[np.argsort(-all_detections[:, 2])]

        true_positives = np.zeros(len(sorted_detections))
        false_positives = np.zeros(len(sorted_detections))
        
        matched_gt = set() # Theo dõi các ground truth đã được phát hiện (để tránh đếm trùng)

        for det_idx, det in enumerate(sorted_detections):
            det_image_id = int(det[0])
            det_bbox = det[3:].astype(int) 

            best_iou = 0
            best_gt_idx_in_all = -1 # Chỉ mục trong all_ground_truths
            
            # Tìm ground truth phù hợp nhất cho dự đoán này trong cùng ảnh
            # Filter ground truths for the current image
            current_image_gts_indices = np.where(all_ground_truths[:, 0] == det_image_id)[0]
            
            for gt_idx_in_all in current_image_gts_indices:
                gt = all_ground_truths[gt_idx_in_all]
                gt_bbox = gt[2:].astype(int)
                iou = calculate_iou(det_bbox, gt_bbox)

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx_in_all = gt_idx_in_all # Lưu chỉ mục toàn cục

            # Nếu có sự chồng lấn tốt và ground truth chưa được phát hiện
            # Sử dụng best_gt_idx_in_all để theo dõi duy nhất
            if best_iou >= iou_thresh and best_gt_idx_in_all != -1 and best_gt_idx_in_all not in matched_gt:
                true_positives[det_idx] = 1
                matched_gt.add(best_gt_idx_in_all)
            else:
                false_positives[det_idx] = 1

        cum_tp = np.cumsum(true_positives)
        cum_fp = np.cumsum(false_positives)

        precision = cum_tp / (cum_tp + cum_fp + 1e-6) # Thêm epsilon để tránh chia cho 0
        recall = cum_tp / len(all_ground_truths) 
        
        # Xử lý trường hợp không có dự đoán nào hoặc không có ground truth
        if len(precision) == 0 or len(all_ground_truths) == 0:
            ap = 0.0
        else:
            # Chuẩn bị để nội suy đường cong PR
            precision = np.concatenate(([0.], precision, [0.]))
            recall = np.concatenate(([0.], recall, [1.]))

            # Nội suy: lấy giá trị precision lớn nhất ở phía bên phải
            for i in range(len(precision) - 1, 0, -1):
                precision[i - 1] = np.maximum(precision[i - 1], precision[i])

            # Tính diện tích dưới đường cong PR (interpolated)
            i = np.where(recall[1:] != recall[:-1])[0]
            ap = np.sum((recall[i + 1] - recall[i]) * precision[i + 1])

        average_precisions_at_iou.append(ap)
        print(f"   AP at IoU={iou_thresh:.2f}: {ap:.4f}")

    mAP = np.mean(average_precisions_at_iou)
    print(f"\nMean Average Precision (mAP) @ IoU=[0.50:0.05:0.95]: {mAP:.4f}")

    # --- Trực quan hóa đường cong Precision-Recall cho IoU=0.5 ---
    print("\n--- Plotting Precision-Recall Curve (IoU=0.5) ---")
    iou_thresh_for_plot = 0.5
    # Thực hiện lại tính toán TP/FP/Precision/Recall cho ngưỡng này để vẽ
    
    true_positives_plot = np.zeros(len(sorted_detections))
    false_positives_plot = np.zeros(len(sorted_detections))
    matched_gt_plot = set()

    for det_idx, det in enumerate(sorted_detections):
        det_image_id = int(det[0])
        det_bbox = det[3:].astype(int)

        best_iou = 0
        best_gt_idx_in_all = -1
        current_image_gts_indices = np.where(all_ground_truths[:, 0] == det_image_id)[0]
        
        for gt_idx_in_all in current_image_gts_indices:
            gt = all_ground_truths[gt_idx_in_all]
            gt_bbox = gt[2:].astype(int)
            iou = calculate_iou(det_bbox, gt_bbox)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx_in_all = gt_idx_in_all
        
        if best_iou >= iou_thresh_for_plot and best_gt_idx_in_all != -1 and best_gt_idx_in_all not in matched_gt_plot:
            true_positives_plot[det_idx] = 1
            matched_gt_plot.add(best_gt_idx_in_all)
        else:
            false_positives_plot[det_idx] = 1

    cum_tp_plot = np.cumsum(true_positives_plot)
    cum_fp_plot = np.cumsum(false_positives_plot)
    precision_plot = cum_tp_plot / (cum_tp_plot + cum_fp_plot + 1e-6) 
    recall_plot = cum_tp_plot / len(all_ground_truths)

    plt.figure(figsize=(8, 6))
    plt.plot(recall_plot, precision_plot, marker='.')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Precision-Recall Curve (IoU Threshold = {iou_thresh_for_plot})')
    plt.grid(True)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()


    # --- Trực quan hóa kết quả (10 ảnh ngẫu nhiên) ---
    print("\n--- Visualizing some random predictions ---")
    
    original_image_files = sorted([f for f in os.listdir(ORIGINAL_IMAGES_DIR) 
                                   if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    
    num_samples_to_display = 10 
    
    val_indices = range(len(X_val)) 
    
    selected_indices = random.sample(val_indices, min(num_samples_to_display, len(X_val)))

    plt.figure(figsize=(15, 10))
    for i, idx_in_val in enumerate(selected_indices):
        
        # CHÚ Ý: Đây là giải pháp TẠM THỜI để có ảnh gốc để vẽ.
        # Ảnh gốc được chọn ngẫu nhiên, KHÔNG PHẢI ảnh tương ứng với idx_in_val.
        # Để chính xác, prepare_data.py cần lưu đường dẫn hoặc ID ảnh gốc cho mỗi mẫu.
        random_original_img_file = random.choice(original_image_files)
        original_img_path = os.path.join(ORIGINAL_IMAGES_DIR, random_original_img_file)
        original_img = cv2.imread(original_img_path)
        if original_img is None:
            print(f"Could not load original image: {random_original_img_file}. Skipping visualization.")
            continue
        
        h_orig, w_orig, _ = original_img.shape
        
        gt_bbox_norm = y_val[idx_in_val]
        pred_bbox_norm = predictions_norm[idx_in_val]
        
        gt_bbox = denormalize_bbox(gt_bbox_norm, w_orig, h_orig)
        pred_bbox = denormalize_bbox(pred_bbox_norm, w_orig, h_orig)
        
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)

        cv2.rectangle(original_img_rgb, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
        cv2.putText(original_img_rgb, "GT", (gt_bbox[0], gt_bbox[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.rectangle(original_img_rgb, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (255, 0, 0), 2)
        cv2.putText(original_img_rgb, "Pred", (pred_bbox[0], pred_bbox[1] - 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        plt.subplot(2, 5, i + 1) 
        plt.imshow(original_img_rgb)
        plt.title(f"Sample {idx_in_val + 1}")
        plt.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nEvaluation completed.")
