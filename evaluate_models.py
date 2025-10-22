"""
evaluate_models.py
Load saved models and evaluate on validation folder. Produce classification report and confusion matrix images.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import pandas as pd
import json

DATA_DIR = "Dataset"
TARGET_SIZE = (224,224)
BATCH_SIZE = 16
MODEL_DIR = "models"
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Load class mapping
if os.path.exists("class_indices.json"):
    with open("class_indices.json","r") as f:
        class_indices = json.load(f)
else:
    folders = sorted([d for d in os.listdir(os.path.join(DATA_DIR,"train")) if os.path.isdir(os.path.join(DATA_DIR,"train",d))])
    class_indices = {folders[i]: i for i in range(len(folders))}

idx_to_class = {v:k for k,v in class_indices.items()}

# Validation generator
val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_directory(
    os.path.join(DATA_DIR, "val"),
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

# Find models
model_files = [f for f in os.listdir(MODEL_DIR) if f.endswith(".h5")]
print("Models found:", model_files)

reports = {}
for mfile in model_files:
    print("Evaluating:", mfile)
    model = load_model(os.path.join(MODEL_DIR, mfile))
    val_generator.reset()
    preds = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_generator.classes

    target_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
    reports[mfile] = report

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=target_names, yticklabels=target_names)
    plt.title(f"Confusion Matrix - {mfile}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"confusion_{mfile.replace('.','_')}.png"))
    plt.close()

# Save CSV summary
rows = []
for mfile, rep in reports.items():
    for cls, vals in rep.items():
        if cls in ["accuracy", "macro avg", "weighted avg"]:
            continue
        rows.append({
            "model": mfile,
            "class": cls,
            "precision": vals["precision"],
            "recall": vals["recall"],
            "f1-score": vals["f1-score"],
            "support": vals["support"]
        })
df = pd.DataFrame(rows)
df.to_csv("model_classwise_metrics.csv", index=False)
print("Saved model_classwise_metrics.csv")
