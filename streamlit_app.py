# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
import pandas as pd
import os
from utils.ocr_model import load_ocr_model, classify_image, predict_text
from utils.preprocessing import preprocess_image, deskew_image

# Load CSS (assumed copied to root from utils/)
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
    <header class="header">
        <div class="branding">Praix Tech — OCR Lab</div>
        <div class="subtitle">Futuristic UI Prototype • Upload → Preprocess → Train → Predict</div>
    </header>
    <button class="start-demo">Start Demo</button>
""", unsafe_allow_html=True)

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
    image = Image.open(st.session_state.uploaded_file)
    st.image(image, use_column_width=True)

    # Save uploaded image for optional training
    os.makedirs("dataset/uploaded", exist_ok=True)
    image.save(f"dataset/uploaded/{time.time()}.jpg")
st.markdown('</section>', unsafe_allow_html=True)

if st.session_state.uploaded_file is not None:
    start_time = time.time()
    original_img = np.array(image.convert('RGB'))
    preprocessed_img = original_img.copy()

    # Preprocessing Section
    st.markdown('<section class="section-preprocess card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Processing...</p>", unsafe_allow_html=True)

    steps = ["Noise Removal", "Thresholding", "Deskew", "Normalization"]
    for step in steps:
        step_progress = st.progress(0)
        time.sleep(0.15)  # UI feedback
        if step == "Noise Removal":
            preprocessed_img = cv2.medianBlur(preprocessed_img, 5)
        elif step == "Thresholding":
            gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
            preprocessed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({"Applied": [70], "Remaining": [30]})
            st.bar_chart(chart_data, color="#00ff00")
        elif step == "Deskew":
            preprocessed_img = deskew_image(preprocessed_img)
        elif step == "Normalization":
            preprocessed_img = cv2.resize(preprocessed_img, (224, 224)) / 255.0
        st.markdown(f"<div class='step'>{step}<span>100%</span><div class='progress-bar full'></div></div>", unsafe_allow_html=True)
        step_progress.progress(1.0)

    st.markdown("<p>Ready</p>", unsafe_allow_html=True)
    st.image(preprocessed_img, use_column_width=True)
    st.markdown('</section>', unsafe_allow_html=True)

    # Training Section (Simulated)
    st.markdown('<section class="section-training card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Training Model</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Training...</p>", unsafe_allow_html=True)

    epochs = 10
    for epoch in range(epochs):
        time.sleep(0.08)
        accuracy = min(1.0, (epoch + 1) / epochs * 0.95 + 0.05)
        loss = max(0.05, 1 - accuracy)
        st.markdown(f"<div class='metric'>Epoch<span>{epoch + 1} / {epochs}</span><div class='progress-bar' style='width: {((epoch + 1) / epochs)*100}%'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>Accuracy<span>{accuracy:.2%}</span><div class='progress-bar' style='width: {accuracy*100}%'></div></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>Loss<span>{loss:.3f}</span><div class='progress-bar' style='width: {(1 - loss)*100}%'></div></div>", unsafe_allow_html=True)

    st.markdown("<p class='status'>Completed</p>", unsafe_allow_html=True)
    if st.button("Queue Image for Training"):
        st.success("Image queued for offline training. Run `utils/train.py` to update model.")
    st.markdown('</section>', unsafe_allow_html=True)

    # Prediction Section
    st.markdown('<section class="section-prediction card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Predicting...</p>", unsafe_allow_html=True)

    try:
        model = load_ocr_model("model/custom_ocr_model.pt")
        class_label, class_confidence = classify_image(model, preprocessed_img, return_confidence=True)
        extracted_text = "N/A"
        ocr_confidence = 0.0
        if class_label == "documents":
            extracted_text, ocr_confidence = predict_text(model, preprocessed_img)
        confidence = class_confidence if class_label != "documents" else ocr_confidence

        st.markdown(f"<div class='result'>Extracted Text<span>{extracted_text}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'>Prediction<span>{class_label}</span></div>", unsafe_allow_html=True)
        st.markdown(f"<div class='metric'>Confidence<span>{confidence * 100:.2f}%</span><div class='progress-bar' style='width: {confidence*100}%'></div></div>", unsafe_allow_html=True)
        st.markdown("<p class='status'>Done</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")

    end_time = time.time()
    if end_time - start_time > 10:
        st.warning("Processing exceeded 10s; optimize model or hardware.")
    st.markdown('</section>', unsafe_allow_html=True)
else:
    st.markdown('<section class="section-preprocess card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Data Preprocessing</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Idle</p>", unsafe_allow_html=True)
    for step in ["Noise Removal", "Thresholding", "Deskew", "Normalization"]:
        st.markdown(f"<div class='step'>{step}<span>0%</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
        if step == "Thresholding":
            st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({"Applied": [0], "Remaining": [0]})
            st.bar_chart(chart_data, color="#00ff00")
    st.markdown('</section>', unsafe_allow_html=True)

    st.markdown('<section class="section-training card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Training Model</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Paused</p>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Epoch<span>0 / 10</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Accuracy<span>0%</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Loss<span>1.000</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)

    st.markdown('<section class="section-prediction card fade-in">', unsafe_allow_html=True)
    st.markdown("<h2>Prediction Results</h2>", unsafe_allow_html=True)
    st.markdown("<p class='status'>Waiting for input</p>", unsafe_allow_html=True)
    st.markdown("<div class='result'>Extracted Text<span>—</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='result'>Prediction<span>—</span></div>", unsafe_allow_html=True)
    st.markdown("<div class='metric'>Confidence<span>0%</span><div class='progress-bar empty'></div></div>", unsafe_allow_html=True)
    st.markdown('</section>', unsafe_allow_html=True)
