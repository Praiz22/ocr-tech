# app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time  # For simulating progress
from ocr_model import load_ocr_model, predict_text
from utils import preprocess_image

# Load CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Hero Section
st.markdown("""
    <div class="hero">
        <div class="branding">FuturOCR</div>
        <h1 class="title">Advanced OCR Application</h1>
        <p class="subtitle">Extract text from images with cutting-edge AI in a futuristic interface.</p>
    </div>
""", unsafe_allow_html=True)

# Image Upload Panel
st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
st.subheader("Upload Image")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image Preview", use_column_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# Preprocessing Panel
if uploaded_file is not None:
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("Image Preprocessing")
    
    # Simulate preprocessing steps with progress
    steps = ["Noise Removal", "Thresholding", "Deskew", "Normalization"]
    preprocessed_img = np.array(image.convert('RGB'))  # Convert to OpenCV format
    progress_bar = st.progress(0)
    
    for i, step in enumerate(steps):
        st.markdown(f"<p class='step'>{step}</p>", unsafe_allow_html=True)
        time.sleep(0.5)  # Simulate time
        progress_bar.progress((i + 1) / len(steps))
        
        # Apply actual preprocessing step-by-step
        if step == "Noise Removal":
            preprocessed_img = cv2.medianBlur(preprocessed_img, 5)
        elif step == "Thresholding":
            gray = cv2.cvtColor(preprocessed_img, cv2.COLOR_RGB2GRAY)
            preprocessed_img = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        elif step == "Deskew":
            # Placeholder for deskew (simplified, actual logic in utils)
            preprocessed_img = utils.deskew_image(preprocessed_img)
        elif step == "Normalization":
            preprocessed_img = cv2.resize(preprocessed_img, (224, 224))  # Example size
            preprocessed_img = preprocessed_img / 255.0  # Normalize
    
    st.image(preprocessed_img, caption="Preprocessed Image", use_column_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Training Panel (Placeholder: Simulate or show static training metrics)
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("Model Training Status")
    st.write("Displaying simulated training progress for demonstration. In production, load real history.")
    
    epochs = 10
    epoch_progress = st.progress(0)
    accuracy_progress = st.progress(0)
    loss_progress = st.progress(0)
    
    for epoch in range(epochs):
        time.sleep(0.3)  # Simulate
        epoch_progress.progress((epoch + 1) / epochs)
        accuracy = (epoch + 1) / epochs  # Simulated increasing accuracy
        loss = 1 - accuracy  # Simulated decreasing loss
        accuracy_progress.progress(accuracy)
        loss_progress.progress(loss)
        st.markdown(f"<p>Epoch: {epoch + 1}/{epochs} | Accuracy: {accuracy:.2f} | Loss: {loss:.2f}</p>", unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

    # Prediction Results Panel
    st.markdown('<div class="card fade-in">', unsafe_allow_html=True)
    st.subheader("OCR Prediction Results")
    
    try:
        model = load_ocr_model("model/custom_ocr_model.h5")  # Or .pt/.onnx with appropriate loader
        extracted_text, label, confidence = predict_text(model, preprocessed_img)
        
        st.markdown(f"<p class='result'>Extracted Text: {extracted_text}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result'>Predicted Label: {label}</p>", unsafe_allow_html=True)
        
        # Confidence bar
        st.progress(confidence)
        st.markdown(f"<p>Confidence: {confidence * 100:.2f}%</p>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
    
    st.markdown('</div>', unsafe_allow_html=True)
else:
    st.info("Please upload an image to begin.")
