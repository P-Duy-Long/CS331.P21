import os

id_mapping_old_to_new = {
    1: 0,  # bicycle
    2: 1,  # bus
    3: 2,  # car
    4: 3,  # motorcycle
    5: 4,  # parking meter
    6: 5,  # person
    7: 6,  # traffic light
    8: 7   # truck
}

# --- CÁC ĐƯỜNG DẪN CẦN THAY ĐỔI ---
# Đảm bảo đường dẫn đến thư mục gốc của dataset YOLO của bạn là chính xác.
yolo_dataset_root_folder = 'C:/Users/ACER/OneDrive/Máy tính/RCNN&Yolo/yolo_dataset' # <-- THAY ĐỔI ĐƯỜNG DẪN NÀY!

# --- CÁC ĐƯỜNG DẪN CON (ĐƯỢC XÂY DỰNG TỰ ĐỘNG) ---
yolo_labels_input_dir = os.path.join(yolo_dataset_root_folder, 'labels')
yolo_labels_output_dir = os.path.join(yolo_dataset_root_folder, 'labels_processed') # Thư mục mới để lưu kết quả
yolo_classes_txt_input_path = os.path.join(yolo_dataset_root_folder, 'classes.txt') # File classes.txt gốc (để tham chiếu nếu cần)
yolo_classes_txt_output_path = os.path.join(yolo_dataset_root_folder, 'classes_final.txt') # File classes.txt đã chỉnh sửa

print("--- BẮT ĐẦU QUÁ TRÌNH CHỈNH SỬA FILE .TXT NHÃN YOLO ---")
print(f"Thư mục labels gốc: {yolo_labels_input_dir}")
print(f"Thư mục đầu ra cho labels đã xử lý: {yolo_labels_output_dir}")

# --- 1. Xử lý các file nhãn YOLO (.txt) ---
print(f"\nĐang xử lý các file nhãn YOLO trong thư mục: {yolo_labels_input_dir}")
try:
    if not os.path.exists(yolo_labels_input_dir):
        print(f"Lỗi: Thư mục nhãn YOLO không tồn tại tại {yolo_labels_input_dir}. Vui lòng kiểm tra đường dẫn.")
    else:
        os.makedirs(yolo_labels_output_dir, exist_ok=True) # Tạo thư mục đầu ra nếu chưa có

        processed_files_count = 0
        for filename in os.listdir(yolo_labels_input_dir):
            if filename.endswith('.txt'):
                input_file_path = os.path.join(yolo_labels_input_dir, filename)
                output_file_path = os.path.join(yolo_labels_output_dir, filename)

                updated_lines = []
                with open(input_file_path, 'r', encoding='utf-8') as f: # Đảm bảo encoding
                    for line in f:
                        parts = line.strip().split()
                        if parts:
                            try:
                                old_class_id = int(parts[0])
                                
                                if old_class_id == 0: 
                                    continue 

                                # Ánh xạ ID cũ sang ID mới
                                if old_class_id in id_mapping_old_to_new:
                                    parts[0] = str(id_mapping_old_to_new[old_class_id])
                                    updated_lines.append(' '.join(parts))
                                else:
                                    print(f"Cảnh báo: ID lớp {old_class_id} trong file '{filename}' không có trong bảng ánh xạ hoặc đã bị loại bỏ ('auto'). Bỏ qua dòng này.")
                            except ValueError:
                                print(f"Cảnh báo: Không thể chuyển đổi class ID thành số trong file '{filename}', dòng: '{line.strip()}'. Bỏ qua dòng này.")
                        else:
                            updated_lines.append(line.strip()) # Giữ nguyên dòng trống

                with open(output_file_path, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(updated_lines))
                processed_files_count += 1
        
        print(f"Đã xử lý và cập nhật ID trong {processed_files_count} file YOLO .txt. Kết quả lưu vào: {yolo_labels_output_dir}")

except Exception as e:
    print(f"Có lỗi xảy ra khi xử lý file .txt của YOLO: {e}")


# --- 2. Xử lý file classes.txt (quan trọng để đồng bộ) ---
print(f"\nĐang tạo file classes.txt cuối cùng: {yolo_classes_txt_output_path}")
try:
    # Thứ tự các lớp sau khi loại bỏ 'auto' và ánh xạ lại ID từ 0 đến 7
    final_class_names_order = [
        "bicycle",        # ID 0
        "bus",            # ID 1
        "car",            # ID 2
        "motorcycle",     # ID 3
        "parking meter",  # ID 4
        "person",         # ID 5
        "traffic light",  # ID 6
        "truck"           # ID 7
    ]

    with open(yolo_classes_txt_output_path, 'w', encoding='utf-8') as f:
        for name in final_class_names_order:
            f.write(name + '\n')
    print(f"Đã tạo file classes.txt cuối cùng và lưu vào: {yolo_classes_txt_output_path}")

except Exception as e:
    print(f"Có lỗi xảy ra khi tạo classes.txt cuối cùng: {e}")

print("\n--- HOÀN TẤT QUÁ TRÌNH CHỈNH SỬA FILE .TXT NHÃN YOLO ---")
print("Vui lòng kiểm tra các file đã xử lý trong thư mục:")
print(f"- File .txt đã cập nhật: {yolo_labels_output_dir}")
print(f"- File classes.txt cuối cùng: {yolo_classes_txt_output_path}")