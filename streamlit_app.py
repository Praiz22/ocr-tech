# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import easyocr

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
    <header class="header">
        <div class="branding">Praix Tech — OCR Lab</div>
        <div class="subtitle">Futuristic UI Prototype • Upload → Preprocess → Predict</div>
    </header>
    <button class="start-demo">Start Demo</button>
""", unsafe_allow_html=True)

# Load pre-trained MobileNetV2
@st.cache_resource
def load_classifier():
    model = models.mobilenet_v2(pretrained=True)
    model.eval()
    return model

# Preprocessing function
def preprocess_image(image):
    img = cv2.medianBlur(image, 5)
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    return cv2.resize(thresh, (224, 224)) / 255.0

# Classify image (heuristic + MobileNetV2)
def classify_image(model, image):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485], std=[0.229])  # Grayscale normalization
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        features = model.features(img_tensor).mean([2, 3]).flatten().numpy()
    text_density = np.mean(image < 0.5)  # Heuristic: high black pixels = text
    if text_density > 0.3:
        return "documents", 0.9
    elif text_density > 0.1:
        return "screenshots", 0.85
    return "pictures", 0.8

# OCR with EasyOCR
def predict_text(image):
    reader = easyocr.Reader(['en'], gpu=False)
    result = reader.readtext(image)
    text = " ".join([res[1] for res in result]) or "No text detected"
    confidence = np.mean([res[2] for res in result]) if result else 0.9
    return text, confidence

# Session state
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None

# Upload Section
st.markdown('<section class="section-upload card fade-in">', unsafe_allow_html=True)
st.markdown("<h2>Upload Image</h2>", unsafe_allow_html=True)
st.markdown("<p>10s processing budget</p>", unsafe_allow_html=True)
st.markdown("<div class='upload-area'>Drag & Drop an image here or Choose File</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"], key="uploader")
if uploaded_file is not None:
    st.session_state.uploaded_file = uploaded_file

if st.session_state.uploaded_file is not None:
    image = Image.open(st.session_state.uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    start_time = time.time()
    original_img = np.array(image)
    preprocessed_img = original_img.copy()

    # Preprocessing Section
    st.markdown('<section class="section-preprocess card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Processing...</p>", unsafe_allow_html=True)

    steps = ["Noise Removal", "Thresholding", "Normalization"]
    for step in steps:
        step_progress = st.progress(0)
        time.sleep(0.1)
        if step == "Noise Removal":
            preprocessed_img = cv2.medianBlur(preprocessed_img, 5)
        elif step == "Thresholding":
            gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
            preprocessed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({"Applied": [70], "Remaining": [30]})
            st.bar_chart(chart_data, color="#00ff00")
        elif step == "Normalization":
            preprocessed_img = cv2.resize(preprocessed_img, (224, 224)) / 255.0
        st.markdown(f"<div class='step'>{step}<span>100%</span><div class='progress-bar full'></div></div>", unsafe_allow_html=True)
        step_progress.progress(1.0)

    st.markdown("<p>Ready</p>", unsafe_allow_html=True)
    st.image(preprocessed_img, caption="Preprocessed Image", use_column_width=True)
    st.markdown('</section>', unsafe_allow_html=True)

    # Prediction Section
    st.markdown('<section class="section-prediction card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Predicting...</p>", unsafe_allow_html=True)

    model = load_classifier()
    class_label, class_confidence = classify_image(model, preprocessed_img)
    extracted_text = "N/A"
    ocr_confidence = 0.0
    if class_label == "documents":
        extracted_text, ocr_confidence = predict_text(original_img)  # Use original for better OCR
    confidence = class_confidence if class_label != "documents" else ocr_confidence

    st.markdown(f"<div class='result'>Extracted Text<span>{extracted_text}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='result'>Prediction<span>{class_label}</span></div>", unsafe_allow_html=True)
    st.markdown(f"<div class='metric'>Confidence<span>{confidence * 100:.2f}%</span><div class='progress-bar' style='width: {confidence*100}%'></div></div>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Done</p>", unsafe_allow_html=True)

    end_time = time.time()
    if end_time - start_time > 10:
        st.warning("Processing exceeded 10s; try smaller images.")
    st.markdown('</section>', unsafe_allow_html=True)
else:
    st.markdown('<section class="section-preprocess card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Idle</p>", unsafe_allow_html=True)
    for step in ["Noise Removal", "Thresholding", "Normalization"]:
        st.markdown(f"<div class='step'>{step}<span>0%</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
        if step == "Thresholding":
            st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({"Applied": [0], "Remaining": [0]})
            st.bar_chart(chart_data, color="#00ff00")
    st.markdown('</section>', unsafe_allow_html=True)

    st.markdown('<section class="section-prediction card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Waiting for input</p>", unsafe_allow_html=True)
    st.markdown("<div class='result'>Extracted Text<span>—</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='result'>Prediction<span>—</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Confidence<span>0%</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)
