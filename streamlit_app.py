import streamlit as st
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text
import base64

# --- Glassmorphism CSS ---
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #ece9f6 0%, #f7fafd 100%);
    font-family: 'Inter', 'Segoe UI', sans-serif;
    color: #181c2f;
}
.card {
    background: rgba(255,255,255,0.35);
    box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.15);
    backdrop-filter: blur(12px);
    -webkit-backdrop-filter: blur(12px);
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.18);
    padding: 2.3rem 1.7rem;
    margin-bottom: 2rem;
    transition: box-shadow .3s;
}
.card:hover {
    box-shadow: 0 16px 40px 0 rgba(31, 38, 135, 0.22);
}
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.2rem 1rem;
}
.hero h1 {
    font-weight: 900;
    color: #223;
    margin-bottom: 0.5rem;
    letter-spacing: -0.03em;
}
.hero p {
    color: #456;
    font-size: 1.15rem;
    margin-bottom: 0;
}
.stButton>button, .copy-btn {
    background: linear-gradient(90deg, #6f86d6 0%, #48c6ef 100%);
    color: white;
    border: none;
    padding: 0.65rem 1.6rem;
    border-radius: 10px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.07rem;
    margin-top: 0.6rem;
    box-shadow: 0 1px 3px 0 rgba(31,38,135,.07);
    transition: background .2s;
}
.stButton>button:hover, .copy-btn:hover {
    background: linear-gradient(90deg, #48c6ef 0%, #6f86d6 100%);
}
.footer {
    margin-top:2rem;
    padding:1.3rem;
    text-align:center;
    background: rgba(255,255,255,0.36);
    border-radius:18px;
    font-size:1rem;
    color:#444;
    box-shadow:0 3px 16px 0 rgba(31,38,135,.08)
}
a.gh-link { display:inline-block; margin-top:10px; text-decoration:none; }
a.gh-link img { width:26px; vertical-align:middle; }
.metric-bar {
    margin-bottom: 0.6rem;
    border-radius: 8px;
    background: rgba(200,225,255,0.19);
    box-shadow: 0 1px 5px rgba(31,38,135,0.09);
}
.metric-fill {
    border-radius: 8px;
    height: 21px;
    background: linear-gradient(90deg, #48e6d6, #6f86d6);
    transition: width 1s;
}
@media (max-width: 900px) {
    .card { padding: 1.2rem 0.7rem; }
    .hero { padding: 1.2rem 0.3rem 1rem 0.3rem; }
}
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

# --- Metric Bar Glassmorphism ---
def metric_bar(label, value, max_value=1.0):
    pct = int(100 * min(value / max_value, 1.0))
    bar = f"""
    <div class="metric-bar">
      <div style='font-size:1rem;margin-bottom:0.2rem'>{label}: <b>{value:.4f}</b></div>
      <div style='background:rgba(220,220,255,0.33);border-radius:8px;overflow:hidden;height:21px;'>
        <div class="metric-fill" style='width:{pct}%;'></div>
      </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)

# --- Upload & Processing ---
col1, col2 = st.columns([1,1.2])
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
        metric_bar("Text Ratio", result['text_ratio'], 0.05)
        metric_bar("Edge Density", result['edge_density'], 0.05)
        metric_bar("Color Variance", result['color_variance'], 1.0)
        metric_bar("Text Pixels Ratio", result['text_pixels_ratio'], 0.05)
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
