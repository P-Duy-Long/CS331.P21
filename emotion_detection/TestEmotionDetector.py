import cv2
import numpy as np
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
import os

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 
                4: "Neutral", 5: "Sad", 6: "Surprised"}

try:
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(1024, activation='relu'),
        Dropout(0.5),
        Dense(7, activation='softmax')
    ])

    model.load_weights('model/emotion_model.h5')
    print("Đã tải model thành công")

    face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    if face_cascade.empty():
        raise Exception("Lỗi khi tải classifier khuôn mặt")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Không thể mở webcam")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Lỗi khi đọc frame")
            break

        frame = cv2.resize(frame, (1280, 720))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )

        for (x, y, w, h) in faces:
            # Vẽ hình chữ nhật
            cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 2)
            
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = roi_gray.astype('float32') / 255.0

            prediction = model.predict(roi_gray, verbose=0)
            emotion_label = emotion_dict[np.argmax(prediction)]

            cv2.putText(frame, emotion_label, (x+5, y-20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Nhận diện cảm xúc', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Lỗi: {str(e)}")

finally:
    # Dọn dẹp
    if 'cap' in locals():
        cap.release()
    cv2.destroyAllWindows()
