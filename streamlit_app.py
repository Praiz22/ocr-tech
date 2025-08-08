import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text

st.title("OCR-Based Image Classification System")
st.write("Upload an image, extract text using OCR, and classify it.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert to OpenCV format
    img_cv = np.array(image)

    # Preprocess
    processed_img = preprocess_image(img_cv)

    # OCR
    extracted_text = pytesseract.image_to_string(processed_img)
    st.subheader("Extracted Text")
    st.text(extracted_text)

    # Classification
    category = classify_text(extracted_text)
    st.subheader("Predicted Category")
    st.success(category)
