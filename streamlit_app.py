import streamlit as st
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text
import base64

# --- Page Config ---
st.set_page_config(
    page_title="OCR-Based Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# --- Custom CSS ---
st.markdown("""
<style>
body {
    background: #f5f7fa;
    color: #333;
    font-family: 'Segoe UI', sans-serif;
}
.hero {
    text-align: center;
    padding: 3rem 1rem;
}
.hero img {
    max-width: 300px;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.15);
}
.hero h1 {
    font-weight: bold;
    margin-top: 1rem;
}
.hero p {
    font-size: 1.1rem;
    color: #555;
}
.carousel {
    display: flex;
    overflow: hidden;
    position: relative;
    border-radius: 15px;
}
.carousel img {
    width: 100%;
    height: auto;
}
.card {
    background: white;
    padding: 2rem;
    border-radius: 15px;
    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}
.footer {
    margin-top: 2rem;
    padding: 1rem;
    font-size: 0.85rem;
    text-align: center;
    background: #f1f3f6;
    border-radius: 10px;
    color: #555;
}
.copy-btn {
    background: #0078ff;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    cursor: pointer;
}
.copy-btn:hover {
    background: #005fcc;
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <img src="https://raw.githubusercontent.com/yourusername/yourrepo/main/ocr.gif" alt="OCR Animation">
    <h1>🖼️ OCR-Based Image Classification System</h1>
    <p>By Praix Tech & Jahsmine — Professional UI inspired by praixtech.netlify.app & erase.bg</p>
</div>
""", unsafe_allow_html=True)

# --- Upload & Processing ---
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    if uploaded_file:
        img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        processed_img = preprocess_image(img_cv)
        extracted_text = pytesseract.image_to_string(processed_img)

        # --- Display OCR Output ---
        st.subheader("📝 Extracted Text")
        st.text_area("", extracted_text, height=200)

        # --- Copy to Clipboard ---
        b64_text = base64.b64encode(extracted_text.encode()).decode()
        st.markdown(f"""
            <a class="copy-btn" href="data:text/plain;base64,{b64_text}" download="extracted_text.txt">⬇ Download Text</a>
        """, unsafe_allow_html=True)

        # --- Classification ---
        category = classify_text(img_cv, processed_img)
        st.subheader("📌 Predicted Category")
        st.success(category)
    else:
        st.info("Upload an image to see results.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    Developed by <b>Praix Tech</b> & <b>Jahsmine</b> for educational & research purposes under the supervision of <b>Mrs. Oguniyi</b>.<br>
    <b>Disclaimer:</b> This tool is intended solely for educational use.
</div>
""", unsafe_allow_html=True)
