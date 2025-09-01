# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time  # For simulating progress
from ocr_model import load_ocr_model, predict_text
from utils import preprocess_image, deskew_image  # Import specific if needed

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown("""
    <header class="header">
        <div class="branding">Praix Tech — OCR Lab</div>
        <div class="title">Futuristic UI Prototype • Upload → Preprocess → Train → Predict</div>
    </header>
""", unsafe_allow_html=True)

# Main container
with st.container():

    # Upload Image Section
    st.markdown('<div class="card fade-in section-upload">', unsafe_allow_html=True)
    st.subheader("Upload Image")
    st.markdown("<p>10s processing budget</p>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drag & Drop an image here or Choose File", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image Preview", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file is not None:
        original_img = np.array(image.convert('RGB'))  # Convert to OpenCV format
        preprocessed_img = original_img.copy()

        # Data Preprocessing Section
        st.markdown('<div class="card fade-in section-preprocess">', unsafe_allow_html=True)
        st.subheader("Data Preprocessing")
        st.markdown("<p class='status'>Processing...</p>", unsafe_allow_html=True)

        steps = ["Noise Removal", "Thresholding", "Deskew", "Normalization"]
        progress_bars = {step: st.progress(0) for step in steps}

        for i, step in enumerate(steps):
            time.sleep(0.5)  # Simulate time
            progress_bars[step].progress(1.0)  # Complete each step

            if step == "Noise Removal":
                preprocessed_img = cv2.medianBlur(preprocessed_img, 5)
                st.markdown("<div class='step'><p>Noise Removal 100%</p></div>", unsafe_allow_html=True)
            elif step == "Thresholding":
                gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
                preprocessed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                st.markdown("<div class='step'><p>Thresholding 100%</p></div>", unsafe_allow_html=True)
                # Applied vs Remaining chart
                st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
                chart_data = {"Applied": [50], "Remaining": [50]}  # Dummy
                st.bar_chart(chart_data)
            elif step == "Deskew":
                preprocessed_img = deskew_image(preprocessed_img)
                st.markdown("<div class='step'><p>Deskew 100%</p></div>", unsafe_allow_html=True)
            elif step == "Normalization":
                preprocessed_img = cv2.resize(preprocessed_img, (224, 224))  # Example size
                preprocessed_img = preprocessed_img / 255.0  # Normalize
                st.markdown("<div class='step'><p>Normalization 100%</p></div>", unsafe_allow_html=True)

        st.markdown("<p class='status'>Ready</p>", unsafe_allow_html=True)
        st.image(preprocessed_img, caption="Preprocessed Image", use_column_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Training Model Section (Simulated)
        st.markdown('<div class="card fade-in section-training">', unsafe_allow_html=True)
        st.subheader("Training Model")
        st.markdown("<p class='status'>Training...</p>", unsafe_allow_html=True)

        epochs = 10
        epoch_progress = st.progress(0)
        accuracy_progress = st.progress(0)
        loss_progress = st.progress(0)

        for epoch in range(epochs):
            time.sleep(0.3)  # Simulate
            epoch_progress.progress((epoch + 1) / epochs)
            accuracy = (epoch + 1) / epochs  # Simulated
            loss = 1 - accuracy  # Simulated
            accuracy_progress.progress(accuracy)
            loss_progress.progress(loss)  # Perhaps inverse for visualization
            st.markdown(f"<p>Epoch {epoch + 1} / {epochs}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Accuracy {accuracy:.2f}</p>", unsafe_allow_html=True)
            st.markdown(f"<p>Loss {loss:.3f}</p>", unsafe_allow_html=True)

        st.markdown("<p class='status'>Completed</p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # Prediction Results Section
        st.markdown('<div class="card fade-in section-prediction">', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        st.markdown("<p class='status'>Predicting...</p>", unsafe_allow_html=True)

        try:
            model = load_ocr_model("model/custom_ocr_model.pt")
            extracted_text, label, confidence = predict_text(model, preprocessed_img)

            st.markdown(f"<div class='result'><p>Extracted Text</p><p>{extracted_text}</p></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='result'><p>Prediction</p><p>{label}</p></div>", unsafe_allow_html=True)
            st.progress(confidence)
            st.markdown(f"<p>Confidence {confidence * 100:.2f}%</p>", unsafe_allow_html=True)
            st.markdown("<p class='status'>Done</p>", unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

        st.markdown('</div>', unsafe_allow_html=True)
    else:
        # Show idle states if no upload
        st.markdown('<div class="card fade-in section-preprocess">', unsafe_allow_html=True)
        st.subheader("Data Preprocessing")
        st.markdown("<p class='status'>Idle</p>", unsafe_allow_html=True)
        st.markdown("<div class='step'><p>Noise Removal 0%</p><div class='progress-empty'></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='step'><p>Thresholding 0%</p><div class='progress-empty'></div></div>", unsafe_allow_html=True)
        st.markdown("<p>Applied vs Remaining</p>", unsafe_allow_html=True)
        st.bar_chart({"Applied": [0], "Remaining": [0]})
        st.markdown("<div class='step'><p>Deskew 0%</p><div class='progress-empty'></div></div>", unsafe_allow_html=True)
        st.markdown("<div class='step'><p>Normalization 0%</p><div class='progress-empty'></div></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card fade-in section-training">', unsafe_allow_html=True)
        st.subheader("Training Model")
        st.markdown("<p class='status'>Paused</p>", unsafe_allow_html=True)
        st.markdown("<p>Epoch 0 / 10</p><div class='progress-empty'></div>", unsafe_allow_html=True)
        st.markdown("<p>Accuracy 0%</p><div class='progress-empty'></div>", unsafe_allow_html=True)
        st.markdown("<p>Loss 1.000</p><div class='progress-empty'></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="card fade-in section-prediction">', unsafe_allow_html=True)
        st.subheader("Prediction Results")
        st.markdown("<p class='status'>Waiting for input</p>", unsafe_allow_html=True)
        st.markdown("<div class='result'><p>Extracted Text</p><p>—</p></div>", unsafe_allow_html=True)
        st.markdown("<div class='result'><p>Prediction</p><p>—</p><div class='progress-empty'></div><p>Confidence 0%</p></div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
