import json
import os

print("--- BẮT ĐẦU SCRIPT CHUYỂN ĐỔI ANNOTATIONS.JSON CHO FASTER R-CNN ---")

# --- ĐƯỜNG DẪN CẦN THAY ĐỔI ---
# Đường dẫn đến thư mục R-CNN dataset của bạn trong Colab
# Đảm bảo đây là đường dẫn gốc của thư mục RCNN_dataset sau khi giải nén
RCNN_CUSTOM_DATASET_ROOT = 'C:/Users/ACER/OneDrive/Máy tính/RCNN&Yolo/RCNN_dataset' # THAY THẾ NẾU KHÁC!

# Đường dẫn file JSON ĐẦU VÀO (thường là annotations.json gốc)
input_json_path = os.path.join(RCNN_CUSTOM_DATASET_ROOT, 'annotations.json')

# Đường dẫn file JSON ĐẦU RA (sau khi xử lý)
output_json_path = os.path.join(RCNN_CUSTOM_DATASET_ROOT, 'annotations_processed_final.json')

# --- Bảng ánh xạ ID cũ (từ 1-8) sang ID mới (0-7), loại bỏ ID 0 ('auto') ---
# Đây là các lớp mà bạn QUAN TÂM và muốn sử dụng.
# ID 0 (background) sẽ được xử lý ngầm bởi mô hình, không cần có trong labels.
target_class_mapping = {
    # Tên lớp : ID MỚI (liên tục từ 0)
    "bicycle": 0,
    "bus": 1,
    "car": 2,
    "motorcycle": 3,
    "parking meter": 4,
    "person": 5,
    "traffic light": 6,
    "truck": 7
}

# Tạo ánh xạ ngược từ ID cũ của bạn sang ID mới
# Để dễ dàng ánh xạ category_id trong annotations từ ID cũ sang ID mới
# Giả sử file JSON gốc của bạn có các ID cho các lớp này là 1-8
# Nếu ID cũ của bạn khác, bạn cần cập nhật ánh xạ này
# Ví dụ: nếu bicycle là ID 1, bus là ID 2, ..., truck là ID 8.
# (Nếu bạn đã chạy script trước và nó là 0-7, thì ánh xạ này sẽ là 0->0, 1->1, v.v.)
# Nhưng để chắc chắn, chúng ta sẽ xây dựng ánh xạ dựa trên tên.
# Chúng ta cần một ánh xạ từ ID COCO gốc sang ID mới của mình.
# Cách an toàn nhất là đọc categories từ JSON gốc.

try:
    print(f"Đang đọc file JSON gốc từ: {input_json_path}")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # --- Xử lý Categories ---
    # Tạo ánh xạ từ ID cũ trong JSON gốc sang tên lớp
    old_id_to_name = {cat['id']: cat['name'] for cat in coco_data.get('categories', [])}
    
    # Tạo danh sách categories MỚI với ID đã được ánh xạ
    updated_categories = []
    category_id_map = {} # Ánh xạ từ OLD_ID -> NEW_ID
    
    # Sắp xếp các lớp mục tiêu theo ID mới để đảm bảo tính nhất quán
    sorted_target_classes = sorted(target_class_mapping.items(), key=lambda item: item[1])

    for class_name, new_id in sorted_target_classes:
        # Tìm ID cũ của lớp này trong file JSON gốc
        old_id_found = None
        for cat in coco_data.get('categories', []):
            if cat['name'] == class_name:
                old_id_found = cat['id']
                break
        
        if old_id_found is not None:
            updated_categories.append({
                "id": new_id,
                "name": class_name,
                "supercategory": "vehicle" # Hoặc giữ nguyên supercategory từ cũ nếu có
            })
            category_id_map[old_id_found] = new_id
        else:
            print(f"Cảnh báo: Không tìm thấy lớp '{class_name}' trong categories gốc của JSON. Bỏ qua.")

    coco_data['categories'] = updated_categories
    print(f"Đã cập nhật {len(updated_categories)} danh mục mới.")
    print(f"Bảng ánh xạ ID cũ sang ID mới: {category_id_map}")

    # --- Xử lý Annotations ---
    updated_annotations = []
    num_removed_annotations = 0
    for ann in coco_data.get('annotations', []):
        old_category_id = ann['category_id']
        
        if old_category_id in category_id_map:
            ann['category_id'] = category_id_map[old_category_id]
            updated_annotations.append(ann)
        else:
            # Nếu ID cũ không có trong bảng ánh xạ (ví dụ: là 'auto' ID 0, hoặc lớp không mong muốn khác)
            # thì bỏ qua annotation này
            num_removed_annotations += 1
            # print(f"  Loại bỏ annotation với category_id cũ: {old_category_id} (lớp: {old_id_to_name.get(old_category_id, 'Không xác định')})")

    coco_data['annotations'] = updated_annotations
    print(f"Đã cập nhật {len(updated_annotations)} annotations. Đã loại bỏ {num_removed_annotations} annotations.")

    # --- Lưu file JSON mới ---
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=4)

    print(f"\nChuyển đổi hoàn tất! File JSON mới đã được lưu tại: {output_json_path}")
    print("Vui lòng kiểm tra file này. Nếu mọi thứ đúng, bạn có thể đổi tên nó thành 'annotations.json'")
    print(f"và sử dụng nó trong code đánh giá/huấn luyện Faster R-CNN của bạn.")

except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file JSON đầu vào tại '{input_json_path}'. Vui lòng kiểm tra đường dẫn.")
except json.JSONDecodeError:
    print(f"Lỗi: Không thể đọc file JSON '{input_json_path}'. Kiểm tra định dạng file JSON.")
except Exception as e:
    print(f"Đã xảy ra lỗi không mong muốn: {e}")

print("\n--- KẾT THÚC SCRIPT CHUYỂN ĐỔI ---")