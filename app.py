import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import json
import pandas as pd
from io import BytesIO

# Page config
st.set_page_config(page_title="Skin Detect Net", page_icon="🏥", layout="wide")

# Simple CSS
st.markdown("""
<style>
    .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
              padding: 2rem; border-radius: 15px; text-align: center; color: white; 
              margin-bottom: 2rem; box-shadow: 0 8px 32px rgba(0,0,0,0.1); }
    .card { background: white; border-radius: 15px; padding: 2rem; margin: 1rem 0; 
            box-shadow: 0 4px 20px rgba(0,0,0,0.08); border: 1px solid #e0e0e0; }
    .diagnosis { border-radius: 15px; padding: 2rem; margin: 1rem 0; 
                 border-left: 6px solid; box-shadow: 0 6px 25px rgba(0,0,0,0.1); }
    .malignant { border-left-color: #dc3545; background: linear-gradient(135deg, #fff5f5 0%, #ffe6e6 100%); }
    .benign { border-left-color: #28a745; background: linear-gradient(135deg, #f0fff4 0%, #e6ffe6 100%); }
    .confidence { background: #6c757d; color: white; padding: 0.5rem 1rem; 
                  border-radius: 20px; font-weight: 600; display: inline-block; margin-left: 1rem; }
</style>
""", unsafe_allow_html=True)

# Model and data paths
MODEL_PATH = "skin_cancer_mobilenetv2.h5"
USERS_FILE = "users.json"

# Load users
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    return {"doctor1": "password1", "doctor2": "password2"}

if 'USERS' not in st.session_state:
    st.session_state['USERS'] = load_users()

# Load model
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    st.error("Model file not found!")
    return None

model = load_model()

# Prediction function
def predict_skin_cancer(img, model):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return ("🟥 Malignant" if prediction > 0.5 else "🟩 Benign", float(prediction[0][0]))

# Main app functions
def show_header(title, subtitle):
    st.markdown(f"""
    <div class="header">
        <h1 style="font-size: 3rem; margin-bottom: 0.5rem;">{title}</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">{subtitle}</p>
    </div>
    """, unsafe_allow_html=True)

def login_page():
    show_header("🏥 Skin Detect Net", "AI-Powered Skin Cancer Detection")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        username = st.text_input("👤 Username", placeholder="Enter username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Sign In", use_container_width=True):
                if username in st.session_state['USERS'] and st.session_state['USERS'][username] == password:
                    st.session_state["logged_in"] = True
                    st.session_state["username"] = username
                    st.success("🎉 Welcome!")
                    st.balloons()
                else:
                    st.error("❌ Invalid credentials")
        
        if st.button("📝 Create Account", use_container_width=True):
            st.session_state["show_register"] = True
        st.markdown('</div>', unsafe_allow_html=True)

def register_page():
    show_header("🏥 Skin Detect Net", "Create New Account")
    
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        new_username = st.text_input("👤 Username", placeholder="Choose username")
        new_password = st.text_input("🔒 Password", type="password", placeholder="Choose password")
        
        if st.button("🚀 Create Account", use_container_width=True):
            if not new_username or not new_password:
                st.error("❌ Fill all fields")
            elif new_username in st.session_state['USERS']:
                st.error("❌ Username exists")
            else:
                st.session_state['USERS'][new_username] = new_password
                with open(USERS_FILE, "w") as f:
                    json.dump(st.session_state['USERS'], f)
                st.success("✅ Account created!")
                st.session_state["show_register"] = False
        
        if st.button("🔙 Back to Login", use_container_width=True):
            st.session_state["show_register"] = False
        st.markdown('</div>', unsafe_allow_html=True)

def main_app():
    show_header("🏥 Skin Detect Net", f"Welcome, {st.session_state.get('username', 'User')}")
    
    if 'show_report' not in st.session_state:
        st.session_state['show_report'] = False

    if not st.session_state['show_report']:
        # Input form
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>👤 Patient Information</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            name = st.text_input("Full Name", placeholder="Patient name")
            age = st.number_input("Age", 0, 120, 25)
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        with col2:
            contact = st.text_input("Contact", placeholder="Phone number")
            notes = st.text_area("Notes", placeholder="Additional information")
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Image upload
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown('<h3>📸 Upload Image</h3>', unsafe_allow_html=True)
        uploaded_image = st.file_uploader("Choose image", type=["jpg", "jpeg", "png"])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        
        if st.button("🔍 Analyze", use_container_width=True, type="primary"):
            if uploaded_image and model and name:
                with st.spinner("🔬 Analyzing..."):
                    img = Image.open(uploaded_image)
                    class_label, confidence = predict_skin_cancer(img, model)
                    st.session_state['report_data'] = {
                        'patient_info': {"name": name, "age": age, "gender": gender, "contact": contact, "notes": notes},
                        'img_bytes': uploaded_image.getvalue(),
                        'class_label': class_label,
                        'confidence': confidence
                    }
                    st.session_state['show_report'] = True
                    st.rerun()
            else:
                st.error("❌ Please fill all required fields and upload an image")
        st.markdown('</div>', unsafe_allow_html=True)
    
    else:
        # Report display
        data = st.session_state['report_data']
        patient = data['patient_info']
        img = Image.open(BytesIO(data['img_bytes']))
        class_label, confidence = data['class_label'], data['confidence']
        
        # Report header
        st.markdown(f"""
        <div class="card">
            <h2 style="text-align: center;">📋 Medical Report</h2>
            <p style="text-align: center; color: #666;">{pd.Timestamp.now().strftime('%B %d, %Y at %I:%M %p')}</p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.5])
        
        with col1:
            # Patient info
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>👤 Patient Information</h4>', unsafe_allow_html=True)
            st.markdown(f"""
            <p><strong>Name:</strong> {patient['name']}</p>
            <p><strong>Age:</strong> {patient['age']} years</p>
            <p><strong>Gender:</strong> {patient['gender']}</p>
            <p><strong>Contact:</strong> {patient['contact']}</p>
            <p><strong>Notes:</strong> {patient['notes']}</p>
            """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Image
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>📸 Lesion Image</h4>', unsafe_allow_html=True)
            st.image(img, use_column_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            # Diagnosis
            diag_class = 'malignant' if "Malignant" in class_label else 'benign'
            st.markdown(f"""
            <div class="diagnosis {diag_class}">
                <h3>🔬 AI Diagnosis</h3>
                <div style="display: flex; align-items: center; margin: 1rem 0;">
                    <span style="font-size: 1.5rem;">{class_label}</span>
                    <span class="confidence">Confidence: {confidence:.1%}</span>
                </div>
                <p>Analysis based on deep learning model trained on medical imaging data.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown('<h4>📋 Recommendations</h4>', unsafe_allow_html=True)
            if "Malignant" in class_label:
                st.markdown("""
                <p><strong>⚠️ Important:</strong> Lesion shows characteristics associated with skin cancer.</p>
                <p><strong>🚨 Immediate Action:</strong></p>
                <ul>
                    <li>Consult dermatologist immediately</li>
                    <li>Schedule medical examination</li>
                    <li>Early detection improves outcomes</li>
                </ul>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <p><strong>✅ Good News:</strong> Lesion appears benign.</p>
                <p><strong>📋 Recommendations:</strong></p>
                <ul>
                    <li>Monitor for changes</li>
                    <li>Regular skin checks</li>
                    <li>Protect from UV radiation</li>
                </ul>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer and actions
        st.markdown("""
        <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 10px; margin-top: 2rem;">
            <p><strong>🏥 Skin Detect Net</strong> | AI-Powered Skin Cancer Detection</p>
            <p style="font-size: 0.8rem; color: #666;">⚠️ For educational purposes only. Consult healthcare professionals for diagnosis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 New Analysis", use_container_width=True):
            st.session_state['show_report'] = False
            st.rerun()

# Main app logic
if __name__ == "__main__":
    # Initialize session state
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "show_register" not in st.session_state:
        st.session_state["show_register"] = False
    
    # Route to appropriate page
    if st.session_state["show_register"]:
        register_page()
    elif not st.session_state["logged_in"]:
        login_page()
    else:
        main_app() 
