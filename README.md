# 🐠 Multiclass Fish Image Classification

### 🎯 Project Objective
To build and deploy a **deep learning-based image classification system** that identifies multiple species of fish from images.  
The project demonstrates model training (CNN + Transfer Learning), evaluation, and deployment via a Streamlit web application.

---

### 📂 Folder Structure


---

### 🧠 Overview
This project classifies fish species into multiple categories using **Convolutional Neural Networks (CNN)** and **Transfer Learning** architectures such as **VGG16**, **ResNet50**, and **MobileNetV2**.

It includes:
- Data preprocessing with TensorFlow’s `ImageDataGenerator`
- Custom CNN model training  
- Transfer Learning with pretrained architectures  
- Model evaluation using precision, recall, F1-score  
- Real-time prediction web app using Streamlit  

---

### 🧰 Tech Stack
| Tool / Library | Purpose |
|----------------|----------|
| **Python 3.10** | Programming language |
| **TensorFlow / Keras** | Model building and training |
| **NumPy, Pandas** | Data manipulation |
| **Matplotlib / Seaborn** | Visualization |
| **Streamlit** | Web app deployment |
| **Pillow (PIL)** | Image loading and preprocessing |

---

### ⚙️ Installation & Setup

#### 1️⃣ Create and activate virtual environment
```bash
py -3.10 -m venv venv
.\venv\Scripts\activate

pip install --upgrade pip
pip install tensorflow==2.12.0 streamlit pillow numpy matplotlib seaborn pandas scikit-learn

python -c "import tensorflow as tf; print('TF Version:', tf.__version__)"

The Streamlit interface allows users to:

Upload new fish images

Select a model dynamically

View predictions instantly


---

That’s it 💫  
Once you paste and save this file, your project is **fully ready for submission or GitHub upload**.  

Would you like me to make a **short “README summary section”** (like a paragraph you can paste into your report or presentation slide)?
