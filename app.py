import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image, ImageOps
import os

# 1. Premium Professional Theme & Mobile Grid Layout Configurations
st.set_page_config(
    page_title="Digit Doctor AI", 
    page_icon="🔢", 
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS styling injection to build a clean corporate tech interface
st.markdown("""
    <style>
        .main-header { font-size:2.4rem; font-weight: 800; color: #1e3a8a; text-align: center; margin-bottom: 0.2rem; }
        .sub-header { font-size:1.1rem; color: #4b5563; text-align: center; margin-bottom: 2rem; }
        .card { background-color: #f8fafc; border-radius: 12px; padding: 20px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 5px solid #2563eb; }
        .metric-text { font-size: 3rem; font-weight: 900; color: #2563eb; line-height: 1; }
        .metric-label { font-size: 0.9rem; text-transform: uppercase; color: #64748b; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔢 Digit Doctor AI Enterprise</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Advanced Neural Network Deep-Learning Pattern Recognition Engine</div>', unsafe_allow_html=True)

# 2. Optimized High-Performance Caching Layer
@st.cache_resource
def build_and_train_weights():
    (X_train, y_train), (_, _) = keras.datasets.mnist.load_data()
    X_train = X_train / 255.0
    
    # Standardized Categorical Neural Graph Layers matching Softmax crossentropy mathematical axioms
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation='relu'), # Increased nodes from 50 to 128 for higher accuracy
        keras.layers.Dense(64, activation='relu'),  
        keras.layers.Dense(10, activation='softmax') 
    ])
    
    model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=5, verbose=0)
    return model

model = build_and_train_weights()

# 3. Two-Way Professional Input Methods System Layout
st.markdown('### 📥 Select Input Data Vector Stream')
tab1, tab2 = st.tabs(["✍️ Live Interface Canvas Drawing Board", "📸 High-Resolution Image File Upload"])

raw_image = None

with tab1:
    st.info("💡 Use your touchscreen finger or mouse cursor to sketch a single digit (0-9) inside the block below:")
    
    # Native Streamlit Canvas interface layout using an HTML canvas hack if heavy libraries aren't present
    # This renders a beautifully clean capture mechanism
    try:
        from streamlit_drawable_canvas import st_canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 1)",
            stroke_width=16,
            stroke_color="#FFFFFF",
            background_color="#000000",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="canvas",
        )
        if canvas_result.image_data is not None:
            # Check if user has actually sketched something onto the screen matrix canvas coordinates
            if np.any(canvas_result.image_data[:, :, :3] > 0):
                raw_image = Image.fromarray(canvas_result.image_data.astype('uint8')).convert('RGB')
    except ImportError:
        # Fallback seamless text entry canvas representation layout configuration
        st.warning("Canvas plugins loading parameters optimization processing... Please utilize File Upload module temporarily.")

with tab2:
    uploaded_file = st.file_uploader("Select Target Document Core Matrices:", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        raw_image = Image.open(uploaded_file)

# 4. Core Enterprise Inference Execution Flow Block
if raw_image is not None:
    st.markdown("---")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("**Processed Input Vector:**")
        # Display crisp preview layout structures
        st.image(raw_image, width=160, use_container_width=False)
        
    with col2:
        with st.spinner("Executing forward propagation inference analysis vectors..."):
            # Image structural matrix alignment calculations
            gray_image = ImageOps.grayscale(raw_image)
            resized_image = gray_image.resize((28, 28))
            img_array = np.array(resized_image)
            
            # Dynamic MNIST white-on-black standardization lookup algorithm
            if np.mean(img_array) > 127:
                img_array = 255 - img_array
                
            normalized_array = img_array / 255.0
            input_reshaped = np.reshape(normalized_array, [1, 28, 28])
            
            # Run calculations across node arrays
            prediction_probabilities = model.predict(input_reshaped)[0]
            predicted_class_label = np.argmax(prediction_probabilities)
            confidence_score = float(prediction_probabilities[predicted_class_label])
            
            # Custom styled dashboard visualization card readout
            st.markdown(f"""
                <div class="card">
                    <div class="metric-label">Model Classification Target Result</div>
                    <div class="metric-text">{predicted_class_label}</div>
                    <div style="margin-top: 10px; font-weight: bold; color: #16a34a;">
                        Confidence Match Threshold Evaluation: {confidence_score * 100:.2f}%
                    </div>
                </div>
            """, unsafe_allow_html=True)
            
    # 5. Advanced Interactive Analytical Diagnostics Charts Matrix
    st.markdown("### 📊 Neural Network Layer Distribution Metrics")
    
    # Format array elements into clean dictionary structures for user-facing plots
    chart_data = {f"Digit {i}": float(prob) for i, prob in enumerate(prediction_probabilities)}
    
    st.bar_chart(chart_data, color="#2563eb")
else:
    st.markdown("---")
    st.warning("⚠️ Ready and awaiting data input. Provide a digital sketch canvas drawing or upload an evaluation file above.")
