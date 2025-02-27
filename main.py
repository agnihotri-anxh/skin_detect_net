import streamlit as st
import numpy as np
import os
import requests
from tensorflow.keras.models import load_model
from PIL import Image

# GitHub release link to model file
GITHUB_RELEASE_LINK = "https://github.com/agnihotri-anxh/skin_detect_net/releases/download/model/skin_cancer_cnn1.h5"
MODEL_PATH = "skin_cancer_cnn1.h5"

def download_model():
    """Downloads the model from GitHub Releases if not present."""
    if not os.path.exists(MODEL_PATH):
        st.info("⏳ Downloading model... Please wait!")
        try:
            response = requests.get(GITHUB_RELEASE_LINK, stream=True, allow_redirects=True)
            with open(MODEL_PATH, "wb") as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            st.success("✅ Model downloaded successfully!")
        except Exception as e:
            st.error(f"⚠️ Error downloading model: {e}")
            return None
    return MODEL_PATH

# Download model if not available
model_path = download_model()

# Load the model
model = None
if model_path and os.path.exists(model_path):
    try:
        model = load_model(model_path)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"⚠️ Failed to load model: {e}")

# Prediction function
def predict_skin_cancer(img, model):
    if model is None:
        return "❌ Error: Model not loaded", 0.0  

    img = img.resize((224, 224))  
    img_array = np.array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_label = "🔴 Malignant" if prediction > 0.5 else "🟢 Benign"

    return class_label, prediction[0][0]  

# Streamlit UI
st.title("🩺 Skin Cancer Detection App")

st.markdown("""
    ### 🔍 Upload skin lesion images for **Malignant vs. Benign** classification.
""")

uploaded_images = st.file_uploader("📤 Upload Images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

if uploaded_images:
    for uploaded_image in uploaded_images:
        img = Image.open(uploaded_image)
        class_label, confidence = predict_skin_cancer(img, model)

        st.image(img, caption=f"🖼 Prediction: **{class_label}** \n Confidence: {confidence:.2%}", use_column_width=True)
        st.markdown("---")

st.markdown("""
    ### About the Model:
    This model uses CNN architecture for predicting whether a skin lesion is **Benign** or **Malignant** based on images of skin lesions.

    #### Features:
    - **Input**: Skin lesion images
    - **Output**: **Benign** or **Malignant** classification

    #### How to use:
    1. Upload an image of a skin lesion.
    2. The model will predict if it's **Benign** or **Malignant**.

    #### Understanding the Results:
    - **Benign**: These are non-cancerous lesions that do not spread to other parts of the body.
    - **Malignant**: These are cancerous lesions that have the potential to grow and spread to other parts of the body.
""")
