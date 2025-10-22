"""
train_transfer_learning.py
Train and fine-tune multiple pretrained models (VGG16, ResNet50, MobileNetV2, InceptionV3, EfficientNetB0).
Saves the best model for each base model under models/<model_name>_best.h5
Also writes class_indices.json (mapping folder name -> index) for the Streamlit app.
"""

import os
import json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

# Settings
DATA_DIR = "Dataset"
TARGET_SIZE = (224, 224)   # standard size for most pretrained nets
BATCH_SIZE = 16            # reduce if you run out of memory
EPOCHS = 8                 # moderate; increase later for final training
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Create / update class_indices.json from Dataset/train folder
train_dir = os.path.join(DATA_DIR, "train")
folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_indices = {folders[i]: i for i in range(len(folders))}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)
print("Saved class indices to class_indices.json")
print(class_indices)

# Data generators
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

# We'll use train and validation directories from Dataset/train & Dataset/val
train_generator = train_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "train"),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

num_classes = len(train_generator.class_indices)
print("Detected classes:", train_generator.class_indices)
print(f"Train samples: {train_generator.samples}, Val samples: {val_generator.samples}")

# Base models mapping
BASE_MODELS = {
    "VGG16": (tf.keras.applications.VGG16, tf.keras.applications.vgg16.preprocess_input),
    "ResNet50": (tf.keras.applications.ResNet50, tf.keras.applications.resnet.preprocess_input),
    "MobileNetV2": (tf.keras.applications.MobileNetV2, tf.keras.applications.mobilenet_v2.preprocess_input),
    "InceptionV3": (tf.keras.applications.InceptionV3, tf.keras.applications.inception_v3.preprocess_input),
    "EfficientNetB0": (tf.keras.applications.EfficientNetB0, tf.keras.applications.efficientnet.preprocess_input)
}

def build_finetune_model(base_constructor, input_shape=(224,224,3), num_classes=2):
    base = base_constructor(weights='imagenet', include_top=False, input_shape=input_shape)
    base.trainable = False  # freeze initially

    x = base.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base.input, outputs=outputs)
    return model, base

# Loop through each base model, train and fine-tune
for name, (constructor, preprocess_fn) in BASE_MODELS.items():
    print(f"\n\n=== TRAINING {name} ===")
    try:
        model, base = build_finetune_model(constructor, input_shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3), num_classes=num_classes)
    except Exception as e:
        print(f"Could not instantiate {name}: {e}")
        continue

    model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    
    checkpoint_path = os.path.join(MODEL_DIR, f"{name}_best.h5")
    checkpoint = ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True, verbose=1)

    # Train (feature extraction)
    steps_per_epoch = max(1, train_generator.samples // BATCH_SIZE)
    validation_steps = max(1, val_generator.samples // BATCH_SIZE)

    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early]
    )

    # Unfreeze last blocks for fine-tuning
    base.trainable = True
    # Unfreeze top ~20% of layers
    fine_tune_at = int(len(base.layers) * 0.8)
    for layer in base.layers[:fine_tune_at]:
        layer.trainable = False
    for layer in base.layers[fine_tune_at:]:
        layer.trainable = True

    model.compile(optimizer=Adam(learning_rate=1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    print("Starting fine-tuning (low learning rate)...")
    history_fine = model.fit(
        train_generator,
        epochs=3,  # short fine-tune; increase if you want
        validation_data=val_generator,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[checkpoint, early]
    )

    print(f"{name} training complete. Best model saved to {checkpoint_path}")
