import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # Cần import Sequential để xây lại model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten # Import các lớp cần thiết
from tensorflow.keras.optimizers import Adam # Import Adam optimizer
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- Cấu hình đường dẫn ---
# Không cần MODEL_JSON_PATH nữa vì chúng ta sẽ xây lại model từ code
MODEL_H5_PATH = 'model/emotion_model.h5'    # Path tới file H5 chứa trọng số
TEST_DATA_DIR = 'data/test'

# --- Định nghĩa từ điển cảm xúc ---
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}
class_labels = list(emotion_dict.values())

# --- Xây dựng lại cấu trúc mô hình ---
# Đây là bước quan trọng nhất để khắc phục lỗi 'Could not locate class Sequential'
# Phải đảm bảo kiến trúc này giống HỆT kiến trúc bạn đã dùng để train
print("Đang xây dựng lại cấu trúc mô hình...")
emotion_model = Sequential()

emotion_model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
emotion_model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
emotion_model.add(MaxPooling2D(pool_size=(2, 2)))
emotion_model.add(Dropout(0.25))

emotion_model.add(Flatten())
emotion_model.add(Dense(1024, activation='relu'))
emotion_model.add(Dropout(0.5))
emotion_model.add(Dense(7, activation='softmax')) # 7 là số lượng lớp cảm xúc
print("Đã xây dựng lại cấu trúc mô hình thành công.")


# --- Tải trọng số vào mô hình đã xây lại ---
print(f"Đang tải trọng số mô hình từ {MODEL_H5_PATH}...")
try:
    if not os.path.exists(MODEL_H5_PATH):
        raise FileNotFoundError(f"File mô hình '{MODEL_H5_PATH}' không tìm thấy.")
    
    emotion_model.load_weights(MODEL_H5_PATH)
    print("Đã tải trọng số mô hình thành công.")
except Exception as e:
    print(f"Lỗi khi tải trọng số: {e}")
    print("Vui lòng đảm bảo file trọng số .h5 tồn tại và hợp lệ.")
    exit()

# --- Biên dịch lại mô hình ---
# Đây là bước bắt buộc để mô hình có thể được evaluate() và predict()
# Phải khớp với optimizer và loss đã dùng khi train (Adam(lr=0.0001, decay=1e-6), categorical_crossentropy)
emotion_model.compile(loss='categorical_crossentropy', 
                      optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001, decay=1e-6), 
                      metrics=['accuracy'])
print("Đã biên dịch lại mô hình để đánh giá.")

# --- Khởi tạo Data Generator cho tập Test ---
IMG_HEIGHT, IMG_WIDTH = 48, 48 

test_data_gen = ImageDataGenerator(rescale=1./255)

print(f"Đang chuẩn bị dữ liệu test từ thư mục: {TEST_DATA_DIR}")
test_generator = test_data_gen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=64,
    color_mode="grayscale",
    class_mode='categorical',
    shuffle=False 
)
print("Đã tiền xử lý dữ liệu test.")

# --- Đánh giá tổng quát ---
print("\n--- Đang đánh giá tổng quát mô hình trên tập test ---")
loss, accuracy = emotion_model.evaluate(test_generator)
print(f"Test Loss: {loss:.4f}")
print(f"Test Accuracy: {accuracy:.4f}")

# --- Thực hiện dự đoán trên tập test ---
print("\n--- Đang thực hiện dự đoán trên tập test để tạo báo cáo chi tiết ---")
test_generator.reset() 

steps_for_prediction = test_generator.samples // test_generator.batch_size
if test_generator.samples % test_generator.batch_size != 0:
    steps_for_prediction += 1

predictions = emotion_model.predict(test_generator, steps=steps_for_prediction, verbose=1)
predicted_classes = np.argmax(predictions, axis=1)

true_classes = test_generator.classes

# --- Tạo và hiển thị Confusion Matrix ---
print("\n--- Confusion Matrix ---")
cm = confusion_matrix(true_classes, predicted_classes)
print(cm)

plt.figure(figsize=(9, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Nhãn dự đoán')
plt.ylabel('Nhãn thực tế')
plt.title('Ma trận nhầm lẫn (Confusion Matrix)')
plt.show()

# --- Tạo và hiển thị Classification Report ---
print("\n--- Báo cáo phân loại (Classification Report) ---")
print(classification_report(true_classes, predicted_classes, target_names=class_labels))

print("\n--- Đánh giá hoàn tất ---")