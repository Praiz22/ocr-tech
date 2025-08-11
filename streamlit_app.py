import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import pytesseract
import joblib

from utils.preprocessing import preprocess_image
from utils.classify import classify_text, set_rf_model
from utils.ocr_utils import ocr_ensemble

# ---- Styling ----
st.markdown("""
<style>
:root {
    --accent: #007bff;
    --accent2: #00c6ff;
    --bg: #f8f9fb;
    --card-bg: #fff;
    --text-dark: #111;
    --text-muted: #555;
}
.stApp { background-color: var(--bg); color: var(--text-dark); }
.card { background: var(--card-bg); padding: 1rem; border-radius: 10px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06); margin-bottom: 1rem; }
.info-box { background-color: #e7f1ff; border: 1px solid #b6daff;
            padding: 0.75rem; border-radius: 8px; }
.copy-btn { display: inline-block; background: var(--accent);
            color: #fff; padding: 0.5rem 1.2rem; border-radius: 6px;
            text-decoration: none; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("""
<div class="card" style="text-align:center">
    <h1>🖼️ Document & Image Classifier</h1>
    <p>Analyze images with OCR & AI — classify content in seconds.</p>
</div>
""", unsafe_allow_html=True)

# ---- Load model ----
try:
    ml_model = joblib.load("rf_model.joblib")
    ml_label_map = joblib.load("rf_labels.joblib")
    set_rf_model(ml_model, ml_label_map)
except:
    pass

# ---- Upload ----
st.markdown('<div class="card">', unsafe_allow_html=True)
uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file:
    image = Image.open(uploaded_file)
    img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # ---- OCR ----
    extracted_text, best_proc, ocr_details = ocr_ensemble(img_cv)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📝 Extracted Text")
    if extracted_text.strip():
        st.text_area("", extracted_text, height=200)
        b64 = base64.b64encode(extracted_text.encode()).decode()
        st.markdown(f'<a class="copy-btn" href="data:text/plain;base64,{b64}" download="text.txt">⬇ Download</a>', unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>No text detected in the image.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---- Classification ----
    processed_img = preprocess_image(img_cv)
    result = classify_text(img_cv, processed_img)
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Predicted Category")
    if result.get('category'):
        st.markdown(f"<div class='info-box'><b>{result['category']}</b> — Confidence: {result['score']*100:.1f}%</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='info-box'>Could not classify the image.</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.markdown('<div class="card"><div class="info-box">Upload an image to begin analysis.</div></div>', unsafe_allow_html=True)

# ---- Footer ----
st.markdown("""
<div class="card" style="text-align:center; font-size:0.9rem; color:var(--text-muted)">
    Developed by <b>Praix Tech</b>, <b>Jahsmine</b> & <b>John Olumide</b> — 2025
</div>
""", unsafe_allow_html=True)
