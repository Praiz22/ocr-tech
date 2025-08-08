import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text
import base64

# Point pytesseract to the system-installed tesseract binary
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# Page Config
st.set_page_config(
    page_title="OCR-Based Image Classification",
    page_icon="🖼️",
    layout="wide"
)

# Custom CSS and AOS animations
st.markdown("""
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            background: linear-gradient(180deg, #f5f7fa 0%, #e8ecf3 100%);
            color: #222;
        }
        .hero {
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 4rem 2rem;
            flex-wrap: wrap;
        }
        .hero-text {
            max-width: 500px;
        }
        .hero-text h1 {
            font-size: 2.5rem;
            font-weight: 700;
            margin-bottom: 1rem;
            color: #1e293b;
        }
        .hero-text p {
            font-size: 1.1rem;
            color: #475569;
            margin-bottom: 2rem;
        }
        .btn-primary {
            background-color: #2563eb;
            color: white;
            padding: 0.75rem 1.5rem;
            border-radius: 8px;
            text-decoration: none;
            font-weight: 500;
            display: inline-block;
        }
        .hero-img img {
            max-width: 350px;
            border-radius: 16px;
            box-shadow: 0 8px 20px rgba(0,0,0,0.1);
        }
        .carousel {
            position: relative;
            margin: 2rem auto;
            max-width: 700px;
            overflow: hidden;
            border-radius: 12px;
            box-shadow: 0 4px 16px rgba(0,0,0,0.1);
        }
        .carousel img {
            width: 100%;
            display: block;
        }
        .copy-btn {
            background-color: #0ea5e9;
            color: white;
            padding: 0.4rem 0.8rem;
            border-radius: 5px;
            font-size: 0.85rem;
            cursor: pointer;
            border: none;
        }
        .footer {
            margin-top: 3rem;
            padding: 1rem;
            font-size: 0.85rem;
            text-align: center;
            background: #f1f5f9;
            border-radius: 8px;
            color: #475569;
        }
    </style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero" data-aos="fade-up">
    <div class="hero-text">
        <h1>OCR-Based Image Classification Tool</h1>
        <p>Extract text and classify images into categories like text documents, pictures, screenshots, and more. Powered by Praix Tech & Jahsmine.</p>
        <a href="#upload-section" class="btn-primary">Upload an Image</a>
    </div>
    <div class="hero-img">
        <img src="assets/ocr.gif" alt="OCR Demo">
    </div>
</div>
""", unsafe_allow_html=True)

# --- Carousel Section ---
st.markdown("""
<div class="carousel" data-aos="zoom-in">
    <img src="https://i.ibb.co/pzsKn3M/sample1.jpg" alt="Sample OCR 1">
</div>
""", unsafe_allow_html=True)

# --- Upload & Processing Section ---
st.markdown("<h2 id='upload-section'>Upload & Process</h2>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("📤 Upload an Image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

with col2:
    if uploaded_file:
        img_cv = np.array(image)
        processed_img = preprocess_image(img_cv)
        extracted_text = pytesseract.image_to_string(processed_img)

        st.subheader("📝 Extracted Text")
        st.text_area("OCR Output", extracted_text, height=200)

        # --- Copy to Clipboard ---
        escaped_ocr = extracted_text.replace("`", "\\`")
        copy_html = f"""
        <button class="copy-btn" id="copyBtn">Copy to clipboard</button>
        <script>
        const btn = document.getElementById('copyBtn');
        btn.addEventListener('click', async () => {{
            await navigator.clipboard.writeText(`{escaped_ocr}`);
            btn.innerText = 'Copied ✓';
            setTimeout(()=> btn.innerText = 'Copy to clipboard', 1400);
        }});
        </script>
        """
        st.markdown(copy_html, unsafe_allow_html=True)

        # --- Download as TXT ---
        b64 = base64.b64encode(extracted_text.encode()).decode()
        st.download_button(
            label="💾 Download Extracted Text",
            data=extracted_text,
            file_name="extracted_text.txt",
            mime="text/plain"
        )

        # --- Classification ---
        category = classify_text(extracted_text)
        st.subheader("📌 Predicted Category")
        st.success(category)
    else:
        st.info("Upload an image to see results.")

# --- Footer ---
st.markdown("""
<div class="footer">
    Developed by <b>Praix Tech</b> & <b>Jahsmine</b> for educational purposes under the supervision of <b>Mrs. Oguniyi</b>.<br>
    <b>Disclaimer:</b> This tool is intended solely for educational use.
</div>
""", unsafe_allow_html=True)

# AOS.js script
st.markdown("""
<script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
<script>AOS.init();</script>
""", unsafe_allow_html=True)
