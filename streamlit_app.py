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
    background: #f7fafd;
    color: #222;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.card {
    background: #fff;
    padding: 2rem 1.5rem;
    border-radius: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.10);
    margin-bottom: 2rem;
}
.hero {
    text-align: center;
    padding: 2.5rem 1rem;
}
.hero h1 {
    font-weight: 900;
    color: #223;
    margin-bottom: 0.5rem;
}
.hero p {
    color: #456;
    font-size: 1.12rem;
}
.footer {
    margin-top:2rem;
    padding:1.3rem;
    text-align:center;
    background:#f1f3f6;
    border-radius:10px;
    font-size:0.98rem;
    color:#666;
}
a.gh-link { display:inline-block; margin-top:8px; text-decoration:none; }
a.gh-link img { width:24px; vertical-align:middle; }
.copy-btn {
    background: #0078ff;
    color: white;
    border: none;
    padding: 0.5rem 1rem;
    border-radius: 8px;
    cursor: pointer;
    text-decoration: none;
    font-size: 1rem;
    margin-top: 0.5rem;
    display:inline-block;
}
.copy-btn:hover { background: #005fcc; }
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <img src="https://raw.githubusercontent.com/Praiz22/ocr-tech/main/ocr.gif" alt="OCR Animation">
    <h1>🖼️ OCR-Based Image Classification System</h1>
    <p>By Praix Tech & Jahsmine — Professional UI inspired by praixtech.netlify.app & erase.bg</p>
</div>
""", unsafe_allow_html=True)

# --- Upload & Processing ---
col1, col2 = st.columns(2)

def metric_bar(label, value, max_value=1.0, color='#0078ff'):
    pct = int(100 * min(value / max_value, 1.0) * 100)
    bar = f"""
    <div style='margin-bottom:0.5rem'>
      <div style='font-size:1rem;margin-bottom:0.2rem'>{label}: <b>{value:.4f}</b></div>
      <div style='background:#e0e5ec;border-radius:8px;overflow:hidden;height:18px;'>
        <div style='width:{pct}%;background:{color};height:100%;transition:width 1s'></div>
      </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)

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

        st.subheader("📝 Extracted Text")
        st.text_area("", extracted_text, height=200)

        b64_text = base64.b64encode(extracted_text.encode()).decode()
        st.markdown(f"""
            <a class="copy-btn" href="data:text/plain;base64,{b64_text}" download="extracted_text.txt">⬇ Download Text</a>
        """, unsafe_allow_html=True)

        # Classification with metrics
        result = classify_text(img_cv, processed_img)

        st.subheader("📌 Predicted Category")
        st.success(f"{result['category']}  —  Confidence: {result['score']*100:.1f}%")

        st.subheader("📊 Classification Metrics")
        metric_bar("Text Ratio", result['text_ratio'], 0.05, '#36c')
        metric_bar("Edge Density", result['edge_density'], 0.05, '#2b8')
        metric_bar("Color Variance", result['color_variance'], 1.0, '#c63')
        metric_bar("Text Pixels Ratio", result['text_pixels_ratio'], 0.05, '#d39')
        # If using DL confidence, display it
        if 'dl_conf' in result:
            metric_bar("DL Confidence", result['dl_conf'], 1.0, '#e80')
        st.write(f"- **Aspect Ratio:** `{result['aspect_ratio']:.2f}`")
        st.write(f"- **Image Size:** `{result['width']} x {result['height']}`")
    else:
        st.info("Upload an image to see results.")
    st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    Developed by <b>Praix Tech</b> & <b>Jahsmine</b> for educational & research purposes under the supervision of <b>Mrs. Oguniyi</b>.<br>
    <b>Disclaimer:</b> This tool is intended solely for educational use.<br>
    <a class="gh-link" href="https://github.com/Praiz22/ocr-tech" target="_blank">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" /> View on GitHub
    </a>
</div>
""", unsafe_allow_html=True)
