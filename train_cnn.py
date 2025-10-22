"""
train_cnn.py (quick test)
Smaller, faster configuration for a quick smoke-test.
- TARGET_SIZE  : 128x128 (faster)
- BATCH_SIZE   : 16
- EPOCHS       : 3
- steps_per_epoch / validation_steps are limited for speed
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Settings (modified for a fast test)
DATA_DIR = "Dataset"
TARGET_SIZE = (128, 128)   # smaller for speed
BATCH_SIZE = 16
EPOCHS = 3
MODEL_DIR = "models"
PLOTS_DIR = "plots"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Data generators (rescale to [0,1], augmentation for training)
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

validation_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Classes:", train_generator.class_indices)
print(f"Train samples: {train_generator.samples}, Val samples: {validation_generator.samples}")

# Build a small CNN
def build_cnn(input_shape=(128,128,3), num_classes=2):
    model = Sequential()
    model.add(Conv2D(32, (3,3), activation='relu', padding='same', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(64, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Conv2D(128, (3,3), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2,2)))

    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

model = build_cnn(input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=num_classes)
model.compile(optimizer=Adam(learning_rate=1e-4),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

# Callbacks
checkpoint_path = os.path.join(MODEL_DIR, "cnn_best.h5")
checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

# Limit steps for a quick test (safe defaults)
steps_per_epoch = min(100, max(1, train_generator.samples // BATCH_SIZE))
validation_steps = min(50, max(1, validation_generator.samples // BATCH_SIZE))

print(f"Using TARGET_SIZE={TARGET_SIZE}, BATCH_SIZE={BATCH_SIZE}, EPOCHS={EPOCHS}")
print(f"steps_per_epoch={steps_per_epoch}, validation_steps={validation_steps}")

# Train
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    callbacks=[checkpoint, early]
)

# Save final model (checkpoint already saved best)
final_model_path = os.path.join(MODEL_DIR, "cnn_final.h5")
model.save(final_model_path)

# Plot history (if matplotlib is installed)
try:
    acc = history.history.get('accuracy', [])
    val_acc = history.history.get('val_accuracy', [])
    loss = history.history.get('loss', [])
    val_loss = history.history.get('val_loss', [])

    plt.figure(figsize=(8,4))
    plt.plot(acc, label='train_acc')
    plt.plot(val_acc, label='val_acc')
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_accuracy_test.png"))
    plt.close()

    plt.figure(figsize=(8,4))
    plt.plot(loss, label='train_loss')
    plt.plot(val_loss, label='val_loss')
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(PLOTS_DIR, "cnn_loss_test.png"))
    plt.close()
except Exception as e:
    print("Could not save plots (matplotlib missing or error):", e)

print(f"Quick test complete. Best model saved to: {checkpoint_path}")
