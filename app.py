import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
import json
import pandas as pd
from io import BytesIO

# Path to the improved model
MODEL_PATH = "skin_cancer_mobilenetv2.h5"
USERS_FILE = "users.json"

# Persistent user storage functions
def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, "r") as f:
            return json.load(f)
    else:
        # Default demo users
        return {"doctor1": "password1", "doctor2": "password2"}

def save_users(users):
    with open(USERS_FILE, "w") as f:
        json.dump(users, f)

# Load users into session_state
if 'USERS' not in st.session_state:
    st.session_state['USERS'] = load_users()
USERS = st.session_state['USERS']

# Load the model
@st.cache_resource
def load_skin_cancer_model():
    if os.path.exists(MODEL_PATH):
        return load_model(MODEL_PATH)
    else:
        st.error("Model file not found. Please train and save the model as 'skin_cancer_mobilenetv2.h5'.")
        return None

model = load_skin_cancer_model()

def register():
    st.title("📝 Register New User")
    new_username = st.text_input("Choose a Username", key="register_username")
    new_password = st.text_input("Choose a Password", type="password", key="register_password")
    register_btn = st.button("Register")
    if register_btn:
        if not new_username or not new_password:
            st.error("Username and password cannot be empty.")
        elif new_username in USERS:
            st.error("Username already exists. Please choose another.")
        else:
            USERS[new_username] = new_password
            save_users(USERS)
            st.success(f"User '{new_username}' registered successfully! You can now log in.")
            st.session_state["show_register"] = False
            st.session_state["show_login_redirect"] = True
    if st.session_state.get("show_login_redirect", False):
        if st.button("Go to Login"):
            st.session_state["show_login_redirect"] = False
            st.session_state["show_register"] = False


def login():
    # Full-page background image and centered login form
    st.markdown("""
    <style>
    body, html, [data-testid="stAppViewContainer"] {
        height: 100vh;
        width: 100vw;
        margin: 0;
        padding: 0;
        overflow: hidden;
    }
    .login-bg {
        position: fixed;
        top: 0; left: 0; right: 0; bottom: 0;
        width: 100vw;
        height: 100vh;
        z-index: 0;
        background: url('docimage.png') no-repeat center center fixed;
        background-size: cover;
        filter: brightness(0.92);
    }
    .login-center-form {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        background: rgba(255,255,255,0.90);
        border-radius: 20px;
        box-shadow: 0 8px 32px rgba(44,62,80,0.18);
        border: 1.5px solid #e0e0e0;
        padding: 2.7rem 2.7rem 2.2rem 2.7rem;
        min-width: 340px;
        max-width: 390px;
        z-index: 2;
        display: flex;
        flex-direction: column;
        align-items: center;
    }
    .login-logo {
        width: 64px;
        height: 64px;
        margin-bottom: 0.7rem;
        border-radius: 50%;
        box-shadow: 0 2px 8px rgba(44,62,80,0.10);
        object-fit: cover;
        background: #e3f0fc;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .login-appname {
        font-size: 1.7rem;
        font-weight: 800;
        color: #2E86C1;
        margin-bottom: 0.2rem;
        text-align: center;
        letter-spacing: 1px;
    }
    .login-tagline {
        font-size: 1.08rem;
        color: #117A65;
        margin-bottom: 1.7rem;
        text-align: center;
        font-weight: 500;
    }
    .login-title {
        font-size: 1.25rem;
        font-weight: 700;
        color: #2E4053;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .login-subtitle {
        font-size: 1.01rem;
        color: #555;
        margin-bottom: 1.7rem;
        text-align: center;
    }
    .login-btn {
        width: 100%;
        background: #2E86C1;
        color: #fff;
        font-weight: 600;
        border-radius: 6px;
        padding: 0.7rem 0;
        margin-top: 0.5rem;
        margin-bottom: 0.7rem;
        border: none;
        font-size: 1.1rem;
        box-shadow: 0 2px 8px rgba(44,62,80,0.07);
        transition: background 0.2s;
    }
    .login-btn:hover {
        background: #145A8A;
    }
    .login-signup-link {
        color: #2E86C1;
        font-weight: 500;
        text-decoration: underline;
        cursor: pointer;
        text-align: center;
        margin-top: 0.7rem;
    }
    </style>
    <div class='login-bg'></div>
    <div class='login-center-form'>
        <img src='docimage.png' class='login-logo' alt='Doctor Logo'>
        <div class='login-appname'>Skin Detect Net</div>
        <div class='login-tagline'>AI-powered Skin Cancer Detection</div>
        <div class='login-title'>Sign In</div>
        <div class='login-subtitle'>Enter your username and password to sign in</div>
    </div>
    """, unsafe_allow_html=True)

    # Place the form fields inside the centered form using Streamlit widgets
    with st.container():
        st.markdown("""
        <style>
        .stTextInput > div > div > input {
            background: #f8fafc;
            border-radius: 6px;
            border: 1.5px solid #e0e0e0;
            font-size: 1.08rem;
            margin-bottom: 1.1rem;
        }
        </style>
        """, unsafe_allow_html=True)
        username = st.text_input("Username", key="login_username")
        password = st.text_input("Password", type="password", key="login_password")
        login_btn = st.button("Sign In", key="login_btn")
        st.markdown("<div class='login-signup-link'>Don't have an account? <a href='#' onclick=\"window.location.reload();\">Sign up</a></div>", unsafe_allow_html=True)
        if login_btn:
            if username in USERS and USERS[username] == password:
                st.session_state["logged_in"] = True
                st.session_state["username"] = username
                st.success(f"Welcome, {username}!")
            else:
                st.error("Invalid username or password.")

# Prediction function
def predict_skin_cancer(img, model):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_label = "🟥 Malignant" if prediction > 0.5 else "🟩 Benign"
    return class_label, float(prediction[0][0])

def patient_form():
    st.header("Patient Details")
    name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=0, max_value=120, step=1)
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    contact = st.text_input("Contact Number")
    notes = st.text_area("Additional Notes")
    return {"name": name, "age": age, "gender": gender, "contact": contact, "notes": notes}

def main_app():
    # Custom CSS for placement-ready, professional look
    st.markdown("""
    <style>
    body, .report-header-bar, .section-title, .info-table, .diagnosis-bar, .summary-box, .footer {
        font-family: 'Segoe UI', 'Arial', sans-serif;
    }
    .report-header-bar {
        display: flex;
        flex-direction: column;
        align-items: center;
        margin-bottom: 1.2rem;
        width: 100%;
    }
    .report-header-bar img {
        margin-bottom: 0.2rem;
    }
    .report-title {
        font-size: 2rem;
        font-weight: 700;
        color: #2E86C1;
        margin-bottom: 0.1rem;
        letter-spacing: 0.5px;
    }
    .report-subtitle {
        font-size: 1.1rem;
        color: #117A65;
        margin-bottom: 0.2rem;
    }
    .report-date {
        font-size: 0.98rem;
        color: #888;
        margin-bottom: 0.2rem;
    }
    .divider {
        border-top: 1px solid #e0e0e0;
        margin: 0.7rem 0 1.2rem 0;
        width: 100%;
    }
    .info-table {
        width: 100%;
        border-collapse: collapse;
        font-size: 1.05rem;
    }
    .info-table td {
        padding: 0.18rem 0.7rem 0.18rem 0.1rem;
        border: none;
    }
    .image-frame {
        border: 1.5px solid #e0e0e0;
        border-radius: 7px;
        padding: 0.3rem;
        background: #fafbfc;
        margin-bottom: 0.5rem;
        text-align: center;
        width: 100%;
    }
    .diagnosis-bar {
        display: flex;
        align-items: center;
        background: #f8f9fa;
        border-radius: 7px;
        padding: 0.7rem 1.2rem;
        margin: 1.2rem 0 0.7rem 0;
        font-size: 1.18rem;
        font-weight: 500;
        width: 100%;
        border-left: 5px solid #2E86C1;
    }
    .diagnosis-malignant {
        color: #C0392B;
        border-left: 5px solid #C0392B;
    }
    .diagnosis-benign {
        color: #117A65;
        border-left: 5px solid #117A65;
    }
    .diagnosis-label {
        font-size: 1.25rem;
        font-weight: 600;
        margin-right: 1.2rem;
    }
    .confidence-box {
        background: #F4F6F7;
        border-radius: 6px;
        padding: 0.4rem 0.9rem;
        font-size: 1.01rem;
        margin-left: 1.2rem;
        display: inline-block;
        color: #555;
    }
    .summary-box {
        background: #E8F8F5;
        border-radius: 7px;
        padding: 0.8rem 1.1rem;
        margin-top: 0.7rem;
        font-size: 1.08rem;
        font-weight: 500;
        display: flex;
        align-items: flex-start;
        width: 100%;
        color: #222;
    }
    .footer {
        text-align: right;
        font-size: 0.95rem;
        color: #aaa;
        margin-top: 1.5rem;
        margin-bottom: 0.2rem;
        width: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='report-header-bar'>
        <img src='https://img.icons8.com/ios-filled/100/2E86C1/hospital-room.png' width='54'>
        <div class='report-title'>Skin Detect Net</div>
        <div class='report-subtitle'>Skin Cancer Detection Report</div>
        <div class='report-date'><b>Date:</b> {date}</div>
    </div>
    <div class='divider'></div>
    """.format(date=pd.Timestamp.now().strftime('%Y-%m-%d')), unsafe_allow_html=True)

    if 'show_report' not in st.session_state:
        st.session_state['show_report'] = False
    if 'report_data' not in st.session_state:
        st.session_state['report_data'] = None

    if not st.session_state['show_report']:
        # Patient form and image upload
        patient_info = patient_form()
        uploaded_image = st.file_uploader("Upload Skin Lesion Image", type=["jpg", "jpeg", "png"])
        submit_btn = st.button("Submit & Generate Report")
        if submit_btn and uploaded_image and model and patient_info['name']:
            img = Image.open(uploaded_image)
            class_label, confidence = predict_skin_cancer(img, model)
            st.session_state['report_data'] = {
                'patient_info': patient_info,
                'img_bytes': uploaded_image.getvalue(),
                'class_label': class_label,
                'confidence': confidence
            }
            st.session_state['show_report'] = True
            st.rerun()
    else:
        # Dedicated report page, full width
        report_data = st.session_state['report_data']
        patient_info = report_data['patient_info']
        img = Image.open(BytesIO(report_data['img_bytes']))
        class_label = report_data['class_label']
        confidence = report_data['confidence']
        col1, col2 = st.columns([1.2, 1.8], gap="large")
        with col1:
            st.markdown("<div class='section-title'>Patient Information</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <table class='info-table'>
                <tr><td><b>Name:</b></td><td>{patient_info['name']}</td></tr>
                <tr><td><b>Age:</b></td><td>{patient_info['age']}</td></tr>
                <tr><td><b>Gender:</b></td><td>{patient_info['gender']}</td></tr>
                <tr><td><b>Contact:</b></td><td>{patient_info['contact']}</td></tr>
                <tr><td><b>Notes:</b></td><td>{patient_info['notes']}</td></tr>
            </table>
            """, unsafe_allow_html=True)
            st.markdown("<div class='section-title'>Lesion Image</div>", unsafe_allow_html=True)
            st.markdown("<div class='image-frame'>", unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("<div style='font-size:0.97rem; color:#888; margin-top:0.2rem;'>Skin lesions of this type are analyzed for cancer risk.</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            diag_class = 'diagnosis-malignant' if class_label == "🟥 Malignant" else 'diagnosis-benign'
            st.markdown(f"<div class='section-title'>Diagnosis</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='diagnosis-bar {diag_class}'><span class='diagnosis-label'>{class_label}</span> <span class='confidence-box'><b>Confidence:</b> {confidence:.2%}</span></div>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1.01rem; color:#555; margin-bottom:0.7rem;'>This result is based on the analysis of the uploaded skin lesion image using a deep learning model trained for skin cancer detection.</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='section-title'>Summary & Recommendation</div>", unsafe_allow_html=True)
            if class_label == "🟥 Malignant":
                st.markdown("<div class='summary-box'>The lesion is predicted to be malignant. This means that, based on the analysis, the skin lesion shows signs that may be associated with skin cancer. It is very important to consult a dermatologist or healthcare specialist as soon as possible for a thorough examination, diagnosis, and appropriate treatment. Early evaluation and intervention can make a significant difference in outcomes. Please do not ignore this result and seek professional medical advice promptly.</div>", unsafe_allow_html=True)
            else:
                st.markdown("<div class='summary-box'>The lesion is predicted to be benign. This means that, based on the analysis, the skin lesion does not show signs of cancer. However, it is always a good idea to monitor the area for any changes and consult a healthcare professional for routine skin checks or if you notice anything unusual.</div>", unsafe_allow_html=True)
        st.markdown("<div class='footer'>Generated by Skin Detect Net | Placement Demo</div>", unsafe_allow_html=True)
        if st.button("Back to Form"):
            st.session_state['show_report'] = False
            st.session_state['report_data'] = None
            st.rerun()

if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False
    if "show_register" not in st.session_state:
        st.session_state["show_register"] = False
    if "show_login_redirect" not in st.session_state:
        st.session_state["show_login_redirect"] = False
    if st.session_state["show_register"]:
        register()
    elif not st.session_state["logged_in"]:
        login()
    else:
        main_app() 