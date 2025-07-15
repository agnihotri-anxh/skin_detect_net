import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
from datetime import datetime
import os

# Page config
st.set_page_config(
    page_title="Skin Cancer Detection System",
    page_icon="🏥",
    layout="wide"
)

# Load model
@st.cache_resource
def load_model():
    model_path = "skin_cancer_mobilenetv2.h5"
    model_url = "https://github.com/agnihotri-anxh/skin_detect_net/releases/download/Model_files/skin_cancer_mobilenetv2.h5"
    
    if not os.path.exists(model_path):
        print("Downloading model from GitHub release...")
        r = requests.get(model_url, stream=True)
        with open(model_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
    
    return tf.keras.models.load_model(model_path)

# Load user data
def load_users():
    try:
        with open('users.json', 'r') as f:
            return json.load(f)
    except:
        return {}

def save_users(users):
    with open('users.json', 'w') as f:
        json.dump(users, f, indent=2)

# Main CSS
st.markdown("""
<style>
    .main-header {
        text-align: center;
        color: #2c3e50;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    .form-container {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .result-box {
        background: #e8f5e8;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    .warning-box {
        background: #fff3cd;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    .danger-box {
        background: #f8d7da;
        padding: 1.5rem;
        border-radius: 8px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
    .status-badge {
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        text-align: center;
        margin: 1rem 0;
    }
    .status-benign { background: #d4edda; color: #155724; }
    .status-malignant { background: #f8d7da; color: #721c24; }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">🏥 Skin Cancer Detection System</h1>', unsafe_allow_html=True)

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = 'input'

# Load model and users
model = load_model()
users = load_users()

# Navigation
if st.session_state.current_page == 'input':
    st.markdown("### Patient Information")
    
    with st.container():
        col1, col2 = st.columns(2)
        
        with col1:
            patient_name = st.text_input("Full Name *", placeholder="Enter patient's full name")
            age = st.number_input("Age *", min_value=0, max_value=120, value=30)
            gender = st.selectbox("Gender *", ["Male", "Female", "Other"])
            
        with col2:
            contact = st.text_input("Contact Number", placeholder="Phone number")
            email = st.text_input("Email Address", placeholder="Email address")
            symptoms = st.text_area("Symptoms (Optional)", placeholder="Describe any symptoms...")
    
    st.markdown("### Image Upload")
    uploaded_file = st.file_uploader("Upload Skin Lesion Image *", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
    
    if st.button("Analyze Image", type="primary", use_container_width=True):
        if patient_name and age and gender and uploaded_file:
            # Process image
            img = Image.open(uploaded_file)
            img = img.resize((224, 224))
            img_array = np.array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            
            # Predict
            prediction = model.predict(img_array)
            probability = prediction[0][0]
            result = "Malignant" if probability > 0.5 else "Benign"
            confidence = probability if result == "Malignant" else 1 - probability
            
            # Save data
            report_id = f"RPT_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            report_data = {
                "report_id": report_id,
                "patient_name": patient_name,
                "age": age,
                "gender": gender,
                "contact": contact,
                "email": email,
                "symptoms": symptoms,
                "result": result,
                "confidence": float(confidence),
                "timestamp": datetime.now().isoformat()
            }
            
            users[report_id] = report_data
            save_users(users)
            
            st.session_state.report_data = report_data
            st.session_state.current_page = 'report'
            st.rerun()
        else:
            st.error("Please fill all required fields and upload an image.")

elif st.session_state.current_page == 'report':
    report_data = st.session_state.report_data
    
    st.markdown(f"## Medical Report - {report_data['report_id']}")
    st.markdown(f"**Date:** {datetime.fromisoformat(report_data['timestamp']).strftime('%B %d, %Y at %I:%M %p')}")
    
    # Patient Info
    st.markdown("### Patient Information")
    col1, col2 = st.columns(2)
    with col1:
        st.write(f"**Name:** {report_data['patient_name']}")
        st.write(f"**Age:** {report_data['age']} years")
    with col2:
        st.write(f"**Gender:** {report_data['gender']}")
        if report_data['contact']:
            st.write(f"**Contact:** {report_data['contact']}")
    
    # Results
    st.markdown("### Analysis Results")
    
    if report_data['result'] == "Benign":
        st.markdown(f'<div class="result-box status-badge status-benign">DIAGNOSIS: BENIGN</div>', unsafe_allow_html=True)
        st.markdown(f"**Confidence Level:** {report_data['confidence']:.1%}")
        st.markdown("**Clinical Interpretation:** The analyzed skin lesion appears to be benign. However, regular monitoring is recommended.")
    else:
        st.markdown(f'<div class="danger-box status-badge status-malignant">DIAGNOSIS: MALIGNANT</div>', unsafe_allow_html=True)
        st.markdown(f"**Confidence Level:** {report_data['confidence']:.1%}")
        st.markdown("**Clinical Interpretation:** The analyzed skin lesion shows characteristics consistent with malignancy. Immediate medical consultation is strongly advised.")
    
    # Recommendations
    st.markdown("### Clinical Recommendations")
    if report_data['result'] == "Benign":
        st.markdown("""
        - Schedule follow-up examination in 3-6 months
        - Monitor for any changes in size, color, or texture
        - Protect from sun exposure
        - Report any new symptoms immediately
        """)
    else:
        st.markdown("""
        - **URGENT:** Schedule immediate consultation with a dermatologist
        - Avoid any manipulation of the lesion
        - Document any changes in appearance
        - Consider biopsy for definitive diagnosis
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("*This report is generated by an AI system and should be reviewed by qualified medical professionals.*")
    
    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("New Analysis", use_container_width=True):
            st.session_state.current_page = 'input'
            st.rerun()
    with col2:
        if st.button("Print Report", use_container_width=True):
            st.info("Print functionality would be implemented here") 
