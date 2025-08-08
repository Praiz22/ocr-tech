import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text, set_rf_model
import base64
import joblib  # For loading scikit-learn model

# --- Clean, Professional Minimal CSS ---
st.markdown("""
<style>
body {
    background: #fff !important;
    color: #111 !important;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.card {
    background: rgba(42,42,42,0.10);
    box-shadow: 0 4px 32px 0 rgba(30,30,30,0.10);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    border-radius: 16px;
    padding: 2rem 1.5rem;
    margin-bottom: 2rem;
}
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
}
.hero h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 800;
    color: #111;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.hero p {
    color: #444;
    font-size: 1.08rem;
    margin-bottom: 0;
}
.stButton>button, .copy-btn {
    background: #222;
    color: #fff;
    border: none;
    padding: 0.48rem 1.3rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    box-shadow: 0 1px 3px 0 rgba(0,0,0,.09);
    transition: background .18s;
}
.stButton>button:hover, .copy-btn:hover {
    background: #444;
}
.footer {
    margin-top:2rem;
    padding:1.3rem;
    text-align:center;
    background: rgba(42,42,42,0.11);
    border-radius:14px;
    font-size:1rem;
    color:#1b1b1b;
    box-shadow:0 2px 10px 0 rgba(30,30,30,.10)
}
a.gh-link { display:inline-block; margin-top:10px; text-decoration:none; }
a.gh-link img { width:26px; vertical-align:middle; }
.metric-bar {
    margin-bottom: 0.6rem;
    border-radius: 7px;
    background: rgba(42,42,42,0.11);
}
.metric-fill {
    border-radius: 7px;
    height: 17px;
    background: #111;
    transition: width 1s;
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1>🖼️ OCR-Based Image Classification</h1>
    <p>By Praix Tech & Jahsmine — Minimal, clean, professional UI</p>
</div>
""", unsafe_allow_html=True)

# --- Metric Bar ---
def metric_bar(label, value, max_value=1.0):
    pct = int(100 * min(value / max_value, 1.0))
    bar = f"""
    <div class="metric-bar">
      <div style='font-size:1rem;margin-bottom:0.18rem;color:#111'>{label}: <b>{value:.4f}</b></div>
      <div style='background:rgba(40,40,40,0.10);border-radius:7px;overflow:hidden;height:17px;'>
        <div class="metric-fill" style='width:{pct}%;'></div>
      </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)

# --- Load ML Model (if available) ---
# Example: place 'rf_model.joblib' and 'rf_labels.joblib' in your repo root if you want ML
ml_model = None
ml_label_map = None
try:
    ml_model = joblib.load("rf_model.joblib")
    ml_label_map = joblib.load("rf_labels.joblib")
    set_rf_model(ml_model, ml_label_map)
except Exception:
    pass

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

        result = classify_text(img_cv, processed_img)

        st.subheader("📌 Predicted Category")
        st.success(f"{result['category']}  —  Confidence: {result['score']*100:.1f}%")
        if result.get('ml_label'):
            st.write(f"<span style='font-size:0.92rem;color:#222'><b>ML:</b> {result['ml_label']} ({result['ml_conf']*100:.1f}%)</span>", unsafe_allow_html=True)

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
