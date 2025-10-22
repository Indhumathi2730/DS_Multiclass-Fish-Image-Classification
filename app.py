# app.py
"""
Streamlit app for Multiclass Fish Image Classification

Requirements:
  - streamlit
  - tensorflow
  - pillow
  - numpy

Run:
  (venv active)
  streamlit run app.py
"""

import os
import json
import numpy as np
from PIL import Image
import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Fish Classifier", layout="centered")

MODEL_DIR = "models"
DEFAULT_TARGET_SIZE = (224, 224)  # default; models trained here use 224x224 unless quick models used 128

@st.cache_resource
def load_class_indices():
    if os.path.exists("class_indices.json"):
        with open("class_indices.json", "r") as f:
            return json.load(f)
    # fallback: infer from folders under Dataset/train
    train_folder = os.path.join("Dataset", "train")
    if os.path.isdir(train_folder):
        folders = sorted([d for d in os.listdir(train_folder) if os.path.isdir(os.path.join(train_folder, d))])
        return {folders[i]: i for i in range(len(folders))}
    return {}

@st.cache_resource
def load_models():
    models = {}
    model_files = sorted([f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")])
    for m in model_files:
        path = os.path.join(MODEL_DIR, m)
        try:
            models[m] = tf.keras.models.load_model(path)
        except Exception as e:
            st.warning(f"Could not load {m}: {e}")
    return models

def preprocess_image_pil(img: Image.Image, target_size):
    img = img.convert("RGB")
    img = img.resize(target_size)
    arr = np.array(img).astype("float32") / 255.0
    arr = np.expand_dims(arr, axis=0)
    return arr

def get_model_input_size(model):
    try:
        shape = model.input_shape
        if shape and len(shape) >= 3:
            h, w = int(shape[1]), int(shape[2])
            return (h, w)
    except Exception:
        pass
    return DEFAULT_TARGET_SIZE

st.title("🐠 Multiclass Fish Image Classification")
st.write("Upload a fish image and choose a model. Models are loaded from the `models/` folder.")

# Load class indices and models
class_indices = load_class_indices()
if class_indices:
    st.info(f"Loaded {len(class_indices)} classes.")
else:
    st.warning("No class_indices.json found and could not infer classes. Please create class_indices.json.")

models = load_models()
if not models:
    st.error("No models found in the models/ directory. Run the training scripts first (train_cnn.py, train_transfer_learning.py, or quick_mobilenet.py).")
    st.stop()

model_name = st.selectbox("Choose model", list(models.keys()))
model = models[model_name]

# Determine model input size
target_size = get_model_input_size(model)
st.write(f"Model input size: {target_size[0]} x {target_size[1]}")

# Allow webp uploads too
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
if uploaded_file is not None:
    try:
        img = Image.open(uploaded_file)
    except Exception as e:
        st.error(f"Could not open the image: {e}")
        st.stop()

    # ✅ Modern Streamlit syntax fix
    st.image(img, caption="Uploaded image", use_container_width=True)

    # Preprocess and predict
    X = preprocess_image_pil(img, target_size)
    preds = model.predict(X)[0]
    top_idx = int(np.argmax(preds))
    confidence = float(preds[top_idx]) * 100.0

    # Invert mapping class->index to index->class
    idx_to_class = {v: k for k, v in class_indices.items()} if class_indices else {}
    predicted_label = idx_to_class.get(top_idx, str(top_idx))

    st.markdown("### 🎯 Prediction")
    st.write(f"**Predicted class:** {predicted_label}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # Show top-3 predictions
    top_k = 3
    top_indices = preds.argsort()[-top_k:][::-1]
    st.markdown("### 🔝 Top predictions")
    for i in top_indices:
        label = idx_to_class.get(int(i), str(int(i)))
        prob = float(preds[int(i)]) * 100.0
        st.write(f"- {label}: {prob:.2f}%")
