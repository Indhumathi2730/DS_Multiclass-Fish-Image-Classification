# quick_mobilenet.py — fast transfer-learning test (MobileNetV2)
import os, json
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

DATA_DIR = "Dataset"
TARGET_SIZE = (128,128)
BATCH_SIZE = 8
EPOCHS = 3
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# write class_indices if not present
train_dir = os.path.join(DATA_DIR, "train")
folders = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
class_indices = {folders[i]: i for i in range(len(folders))}
with open("class_indices.json", "w") as f:
    json.dump(class_indices, f, indent=2)

train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=20, zoom_range=0.1, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(os.path.join(DATA_DIR,"train"),
                                              target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=True)
val_gen = val_datagen.flow_from_directory(os.path.join(DATA_DIR,"val"),
                                          target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False)

num_classes = len(train_gen.class_indices)
print("Classes:", train_gen.class_indices)

base = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(TARGET_SIZE[0],TARGET_SIZE[1],3))
base.trainable = False
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
outputs = Dense(num_classes, activation='softmax')(x)
model = Model(inputs=base.input, outputs=outputs)

model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

checkpoint = ModelCheckpoint(os.path.join(MODEL_DIR, "MobileNetV2_quick.h5"), monitor='val_accuracy', save_best_only=True, verbose=1)
early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1)

steps = max(1, train_gen.samples // BATCH_SIZE)
val_steps = max(1, val_gen.samples // BATCH_SIZE)
print(f"steps={steps}, val_steps={val_steps}")

model.fit(train_gen, epochs=EPOCHS, validation_data=val_gen, steps_per_epoch=steps, validation_steps=val_steps, callbacks=[checkpoint, early])

print("Done. Model saved to models/MobileNetV2_quick.h5")
