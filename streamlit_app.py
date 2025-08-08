import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text

# --------------------------
# Page Config
# --------------------------
st.set_page_config(
    page_title="OCR-Based Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# --------------------------
# Custom CSS for Glassmorphism + Branding
# --------------------------
st.markdown("""
    <style>
    body {
        background: linear-gradient(135deg, #0a0a0a, #1a1a1a);
        color: white;
    }
    .glass-card {
        background: rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(0,0,0,0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.18);
    }
    .header-glass {
        background: linear-gradient(120deg, rgba(255, 0, 150, 0.2), rgba(0, 153, 255, 0.2));
        padding: 2rem;
        border-radius: 20px;
        text-align: center;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    .footer {
        margin-top: 2rem;
        padding: 1rem;
        font-size: 0.8rem;
        text-align: center;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# --------------------------
# Glassmorphism Header
# --------------------------
st.markdown("""
    <div class="header-glass">
        <h1 style='color:white; font-weight:bold;'>🖼️ OCR-Based Image Classification System</h1>
        <p style='color:white;'>By Praix Tech & Jahsmine — Inspired by removebg & erase.bg styles</p>
    </div>
""", unsafe_allow_html=True)

st.write("")

# --------------------------
# Layout
# --------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    if uploaded_file:
        img_cv = np.array(image)
        processed_img = preprocess_image(img_cv)
        extracted_text = pytesseract.image_to_string(processed_img)
        st.subheader("📝 Extracted Text")
        st.text(extracted_text)

        category = classify_text(extracted_text)
        st.subheader("📌 Predicted Category")
        st.success(category)
    else:
        st.info("Upload an image to see results.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------
# Footer with Credit & Disclaimer
# --------------------------
st.markdown("""
    <div class="footer">
        Developed by <b>Praix Tech</b> & <b>Jahsmine</b> for educational & research purposes 
        under the supervision of <b>Mrs. Oguniyi</b>.  
        <br>
        <b>Disclaimer:</b> This tool is intended solely for educational use.
    </div>
""", unsafe_allow_html=True)

