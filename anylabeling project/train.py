import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

PROCESSED_DATA_DIR = 'UTK500_labels/processed_data' 
MODEL_SAVE_PATH = 'UTK500_labels/face_localization_model.h5' 


IMG_HEIGHT, IMG_WIDTH = 128, 128
CHANNELS = 3 # RGB images

# --- Tham số huấn luyện ---
BATCH_SIZE = 32
EPOCHS = 50 
LEARNING_RATE = 0.001

if __name__ == "__main__":
    print(f"Loading processed data from {PROCESSED_DATA_DIR}/...")
    try:
        X_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_train.npy'))
        y_train = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_train.npy'))
        X_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'X_val.npy'))
        y_val = np.load(os.path.join(PROCESSED_DATA_DIR, 'y_val.npy'))
        print("Data loaded successfully.")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}, y_val shape: {y_val.shape}")
    except FileNotFoundError:
        print(f"Error: Processed data files not found in {PROCESSED_DATA_DIR}.")
        print("Please ensure prepare_data.py has been run successfully.")
        exit()

    # --- Xây dựng mô hình CNN ---
    print("Building CNN model...")
    model = Sequential([

        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, CHANNELS)),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),

        Flatten(),

        Dense(128, activation='relu'),
        Dropout(0.3), 

        Dense(4, activation='sigmoid') 
    ])


    print("Compiling model...")
    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE), loss='mse', metrics=['mse'])

    model.summary() 

    checkpoint_callback = ModelCheckpoint(
        filepath=MODEL_SAVE_PATH,
        monitor='val_loss',
        save_best_only=True,
        mode='min', 
        verbose=1
    )

    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=10,
        mode='min',
        restore_best_weights=True,
        verbose=1
    )

    # --- Huấn luyện mô hình ---
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val),
        callbacks=[checkpoint_callback, early_stopping_callback], 
        verbose=1
    )
    print("Model training completed.")

    print("Plotting training history...")
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mse'], label='Training MSE')
    plt.plot(history.history['val_mse'], label='Validation MSE')
    plt.title('Model Mean Squared Error')
    plt.xlabel('Epoch')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show() 
    print(f"Best model saved to {MODEL_SAVE_PATH}")