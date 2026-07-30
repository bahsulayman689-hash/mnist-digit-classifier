import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import os

# Configure a clean mobile-first workspace presentation layout
st.set_page_config(page_title="Digit Classifier", page_icon="🔢", layout="centered")

st.title("🔢 MNIST Handwritten Digit Classifier")
st.markdown("Upload or draw a handwritten digit (0-9) to see the neural network analyze it in real time.")

# 1. FIXED ARCHITECTURE PIPELINE (Cached to protect memory pools)
@st.cache_resource
def build_and_train_weights():
    # Load native MNIST data sets inside cloud memory autonomously
    (X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    
    # Scale matrix pixel configurations to match training distribution arrays
    X_train = X_train / 255.0
    
    # Corrected setup parameters matching explicit softmax probability distribution math
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(50, activation='relu'),
        keras.layers.Dense(10, activation='softmax') # Fixed from sigmoid to fix loss alignment bugs
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    
    # Lightweight fast training iteration pass directly inside the deployment workspace container
    with st.spinner("Initializing neural network node configurations..."):
        model.fit(X_train, y_train, epochs=5, verbose=0)
    return model

model = build_and_train_weights()

# 2. FILE INTERACTIVE LOADER FIELD WIDGET
uploaded_file = st.file_uploader("📸 Upload a Handwritten Digit Image (PNG, JPG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read incoming structural payload values
    raw_image = Image.open(uploaded_file)
    
    # Render interactive input layout display card elements
    st.image(raw_image, caption="Uploaded Document Artifact Source", width=200)
    
    with st.spinner("Processing image matrix structures..."):
        # Convert image to pure grayscale (1 channel matrix mapping)
        gray_image = ImageOps.grayscale(raw_image)
        
        # Resize image dimensions to match your target matrix configurations: 28 x 28
        resized_image = gray_image.resize((28, 28))
        
        # MNIST images use white text on a black background.
        # If the user uploads black text on a white page, invert the pixels dynamically.
        img_array = np.array(resized_image)
        if np.mean(img_array) > 127:
            img_array = 255 - img_array
            
        # Standardize matrix configurations (Value scaling division layer matched to 1./255 math)
        normalized_array = img_array / 255.0
        
        # Expand dimensions to include single batch input node structures: [1, 28, 28]
        input_reshaped = np.reshape(normalized_array, [1, 28, 28])
        
        # 3. EXECUTE FORWARD PROPAGATION PREDICTION PASS
        prediction_probabilities = model.predict(input_reshaped)
        predicted_class_label = np.argmax(prediction_probabilities[0])
        confidence_score = float(prediction_probabilities[0][predicted_class_label])
        
        # Render clean metric visual result structures for users
        st.success(f"## Recognized Digit: {predicted_class_label}")
        st.metric(label="Model Identification Confidence", value=f"{confidence_score * 100:.2f}%")
        
        # Render a bar chart showing the probability scores across digits 0-9
        st.markdown("### Probability Distribution Map Across Nodes:")
        st.bar_chart(prediction_probabilities[0])
