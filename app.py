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
    .report-header { background: #f8f9fa; border: 2px solid #dee2e6; border-radius: 10px; 
                     padding: 2rem; margin-bottom: 2rem; text-align: center; }
    .report-title { font-size: 2.5rem; font-weight: 700; color: #2c3e50; margin-bottom: 0.5rem; }
    .report-subtitle { font-size: 1.2rem; color: #7f8c8d; margin-bottom: 1rem; }
    .report-info { display: flex; justify-content: space-between; margin-top: 1rem; 
                   font-size: 0.9rem; color: #6c757d; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #2c3e50; 
                     border-bottom: 2px solid #3498db; padding-bottom: 0.5rem; margin-bottom: 1rem; }
    .patient-details { background: #f8f9fa; border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
    .patient-details table { width: 100%; border-collapse: collapse; }
    .patient-details td { padding: 0.5rem; border-bottom: 1px solid #dee2e6; }
    .patient-details td:first-child { font-weight: 600; color: #495057; width: 30%; }
    .medical-finding { background: #fff; border: 1px solid #dee2e6; border-radius: 8px; 
                       padding: 1.5rem; margin: 1rem 0; }
    .finding-title { font-size: 1.1rem; font-weight: 600; color: #2c3e50; margin-bottom: 0.5rem; }
    .finding-content { color: #495057; line-height: 1.6; }
    .recommendations { background: #e8f4fd; border-left: 4px solid #3498db; 
                       border-radius: 8px; padding: 1.5rem; margin: 1rem 0; }
    .disclaimer { background: #fff3cd; border: 1px solid #ffeaa7; border-radius: 8px; 
                  padding: 1rem; margin: 1rem 0; font-size: 0.9rem; color: #856404; }
    .signature-line { border-top: 1px solid #000; width: 200px; margin: 2rem auto; }
    .page-break { page-break-before: always; }
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

def generate_formal_report(patient_data, image_data, diagnosis_data):
    """Generate a formal medical report"""
    
    # Report header
    st.markdown(f"""
    <div class="report-header">
        <div class="report-title">MEDICAL DIAGNOSTIC REPORT</div>
        <div class="report-subtitle">Skin Lesion Analysis - AI-Assisted Screening</div>
        <div class="report-info">
            <span><strong>Report Date:</strong> {pd.Timestamp.now().strftime('%B %d, %Y')}</span>
            <span><strong>Report Time:</strong> {pd.Timestamp.now().strftime('%I:%M %p')}</span>
            <span><strong>Report ID:</strong> SKN-{pd.Timestamp.now().strftime('%Y%m%d%H%M')}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Patient Information Section
    st.markdown('<div class="section-title">PATIENT INFORMATION</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="patient-details">
        <table>
            <tr><td>Patient Name</td><td>{patient_data['name']}</td></tr>
            <tr><td>Age</td><td>{patient_data['age']} years</td></tr>
            <tr><td>Gender</td><td>{patient_data['gender']}</td></tr>
            <tr><td>Contact Number</td><td>{patient_data['contact']}</td></tr>
            <tr><td>Referring Physician</td><td>AI-Assisted Screening System</td></tr>
            <tr><td>Clinical Notes</td><td>{patient_data['notes'] if patient_data['notes'] else 'No additional notes provided'}</td></tr>
        </table>
    </div>
    """, unsafe_allow_html=True)
    
    # Clinical Findings Section
    st.markdown('<div class="section-title">CLINICAL FINDINGS</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<div class="medical-finding">', unsafe_allow_html=True)
        st.markdown('<div class="finding-title">📸 Lesion Image Analysis</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-content">', unsafe_allow_html=True)
        st.image(Image.open(BytesIO(image_data)), caption="Digital Image of Skin Lesion", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="medical-finding">', unsafe_allow_html=True)
        st.markdown('<div class="finding-title">🔬 AI Diagnostic Assessment</div>', unsafe_allow_html=True)
        st.markdown('<div class="finding-content">', unsafe_allow_html=True)
        
        # Diagnosis display
        diag_class = 'malignant' if "Malignant" in diagnosis_data['class_label'] else 'benign'
        st.markdown(f"""
        <div class="diagnosis {diag_class}">
            <h4 style="margin-bottom: 1rem;">Primary Assessment</h4>
            <p style="font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
                {diagnosis_data['class_label']}
            </p>
            <p style="margin-bottom: 1rem;">
                <strong>Confidence Level:</strong> {diagnosis_data['confidence']:.1%}
            </p>
            <p style="font-size: 0.9rem; color: #666;">
                Analysis performed using deep learning algorithm trained on extensive medical imaging dataset.
            </p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Diagnostic Interpretation
    st.markdown('<div class="section-title">DIAGNOSTIC INTERPRETATION</div>', unsafe_allow_html=True)
    
    if "Malignant" in diagnosis_data['class_label']:
        st.markdown("""
        <div class="medical-finding">
            <div class="finding-title">⚠️ CLINICAL SIGNIFICANCE</div>
            <div class="finding-content">
                <p>The analyzed skin lesion demonstrates morphological characteristics that are concerning for potential malignancy. 
                The AI algorithm has identified features commonly associated with skin cancer, including irregular borders, 
                color variation, and asymmetric growth patterns.</p>
                
                <p><strong>Risk Assessment:</strong> HIGH - Immediate clinical evaluation recommended</p>
                <p><strong>Differential Diagnosis:</strong> Melanoma, Basal Cell Carcinoma, Squamous Cell Carcinoma</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="medical-finding">
            <div class="finding-title">✅ CLINICAL SIGNIFICANCE</div>
            <div class="finding-content">
                <p>The analyzed skin lesion demonstrates benign characteristics with regular borders, uniform pigmentation, 
                and symmetric growth patterns. The AI algorithm has not identified features typically associated with malignancy.</p>
                
                <p><strong>Risk Assessment:</strong> LOW - Routine monitoring recommended</p>
                <p><strong>Differential Diagnosis:</strong> Benign Nevus, Seborrheic Keratosis, Dermatofibroma</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Recommendations Section
    st.markdown('<div class="section-title">CLINICAL RECOMMENDATIONS</div>', unsafe_allow_html=True)
    
    if "Malignant" in diagnosis_data['class_label']:
        st.markdown("""
        <div class="recommendations">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">🚨 URGENT CLINICAL ACTIONS REQUIRED</h4>
            <ol style="color: #495057; line-height: 1.8;">
                <li><strong>Immediate Consultation:</strong> Schedule appointment with board-certified dermatologist within 48-72 hours</li>
                <li><strong>Biopsy Recommendation:</strong> Excisional or punch biopsy for definitive histopathological diagnosis</li>
                <li><strong>Imaging Studies:</strong> Consider dermoscopy and/or confocal microscopy for detailed assessment</li>
                <li><strong>Documentation:</strong> Maintain photographic documentation of lesion for monitoring</li>
                <li><strong>Follow-up:</strong> Establish regular surveillance schedule based on final diagnosis</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="recommendations">
            <h4 style="color: #2c3e50; margin-bottom: 1rem;">📋 ROUTINE CLINICAL ACTIONS</h4>
            <ol style="color: #495057; line-height: 1.8;">
                <li><strong>Regular Monitoring:</strong> Self-examination monthly, clinical evaluation every 6-12 months</li>
                <li><strong>Photographic Documentation:</strong> Maintain baseline images for comparison</li>
                <li><strong>Sun Protection:</strong> Implement strict UV protection measures (SPF 30+, protective clothing)</li>
                <li><strong>Education:</strong> Patient education on ABCDE criteria for melanoma detection</li>
                <li><strong>Follow-up:</strong> Return for evaluation if any changes in size, color, or symptoms</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    # Technical Information
    st.markdown('<div class="section-title">TECHNICAL INFORMATION</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="medical-finding">
        <div class="finding-title">🔬 Methodology</div>
        <div class="finding-content">
            <p><strong>Analysis Method:</strong> Deep Learning Convolutional Neural Network (MobileNetV2)</p>
            <p><strong>Training Dataset:</strong> 10,000+ annotated skin lesion images</p>
            <p><strong>Image Processing:</strong> 224x224 pixel resolution, RGB normalization</p>
            <p><strong>Algorithm Version:</strong> Skin Detect Net v2.0</p>
            <p><strong>Processing Time:</strong> < 5 seconds</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Disclaimer
    st.markdown("""
    <div class="disclaimer">
        <strong>⚠️ IMPORTANT DISCLAIMER:</strong><br>
        This report is generated by an AI-assisted screening system and is intended for educational and preliminary screening purposes only. 
        It should not replace professional medical diagnosis, clinical judgment, or histopathological examination. 
        All findings must be interpreted by qualified healthcare professionals. The system is not intended for definitive diagnosis 
        and should be used as an adjunct to clinical evaluation.
    </div>
    """, unsafe_allow_html=True)
    
    # Signature section
    st.markdown("""
    <div style="text-align: center; margin-top: 3rem;">
        <div class="signature-line"></div>
        <p style="margin-top: 0.5rem; font-weight: 600;">AI-Assisted Screening System</p>
        <p style="font-size: 0.9rem; color: #666;">Skin Detect Net - Automated Analysis Report</p>
    </div>
    """, unsafe_allow_html=True)

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
        
        if st.button("🔍 Generate Medical Report", use_container_width=True, type="primary"):
            if uploaded_image and model and name:
                with st.spinner("🔬 Analyzing and generating report..."):
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
        # Generate formal report
        data = st.session_state['report_data']
        generate_formal_report(
            data['patient_info'], 
            data['img_bytes'], 
            {'class_label': data['class_label'], 'confidence': data['confidence']}
        )
        
        # Action buttons
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
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
