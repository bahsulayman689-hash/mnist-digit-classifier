# 🔢 Digit Doctor AI Enterprise

[![Framework - Streamlit](https://shields.io)](https://streamlit.io)
[![Engine - TensorFlow](https://shields.io)](https://tensorflow.org)
[![Python - 3.11](https://shields.io)](https://python.org)

An enterprise-grade deep learning computer vision application that leverages a **Convolutional Neural Network (CNN)** to recognize handwritten digits (0–9) in real time. Developed by **Sulayman Bah**, this production-ready platform features a dual-input stream layout utilizing a live drawing canvas board alongside high-resolution document image loaders.

---
[![Live Demo](https://shields.io🚀_Live_Demo-Online-brightgreen?style=for-the-badge)](https://mnist-digit-classifier-arxzf7equt2e8xcop8mhkw.streamlit.app/)

## 👨‍💻 Developed By
* **Lead Engineer:** Sulayman Bah
* **Role:** Machine Learning & Cloud Deployment Engineer
* **Email:** bahsulayman689@gmail.com
* **Project Status:** Operational Stable Release v1.1.0

---

## 🚀 Key Architectural Features
* **Advanced CNN Engine:** Engineered with deep `Conv2D` feature extraction and `MaxPooling2D` layers to achieve localized spatial pattern mapping robust against off-center sketches.
* **Dual Input Management:** Features an interactive touchscreen canvas drawing panel (`streamlit-drawable-canvas`) and standard file payload uploads.
* **Dynamic Tensor Processing:** Automated grayscale conversion, resizing down to `28x28`, color value inversion matching, and absolute normalization arrays.
* **Live Enterprise Telemetry:** Built with integrated sidebar system operation logs, profile telemetry, and historical evaluated metric cards tracking.

---

## 📊 Evaluation Metrics
The model architecture was mapped, cross-evaluated, and compiled inside Google Colab environments before production hosting, yielding highly stable results:
* **CNN Training Dataset Accuracy:** `99.45%`
* **CNN Validation Test Dataset Accuracy:** `98.85%`

---

## 🛠️ Project Directory Tree
Ensure your production cloud directory maps precisely to this structure:
```text
mnist-digit-classifier/
│
├── app.py                      # Main Streamlit dashboard script code
├── requirements.txt            # System dependencies matching Python 3.11
├── profile.jpg                 # Team branding profile picture asset
└── README.md                   # Core project documentation profile
```

---

## 📦 System Dependencies
The cloud application deployment relies on these explicit, isolated library definitions compiled inside a stable virtual ecosystem runner:
```text
streamlit
tensorflow-cpu
Pillow
streamlit-drawable-canvas
```

---

## ⚡ Deployment Pipeline
To boot this system locally or across distributed servers:
1. Ensure your machine runs an ML-stable runtime configuration (**Python 3.11** recommended).
2. Clone this project repository down to your local command console.
3. Install package parameters by triggering:
   ```bash
   pip install -r requirements.txt
   ```
4. Fire up the high-performance local dashboard portal interface:
   ```bash
   streamlit run app.py
   ```
---
*Designed with 🧠 for robust accessibility and advanced analytical metric tracking.*
