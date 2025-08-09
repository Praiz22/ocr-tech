import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
from utils.preprocessing import preprocess_image
from utils.classify import classify_text, set_rf_model
import base64
import joblib

# --- Praix Tech Light Mode Styling ---
st.markdown("""
<style>
:root {
    --accent: #007bff;
    --accent2: #00c6ff;
    --text-dark: #111;
    --text-muted: #555;
    --bg: #f8f9fb;
    --card-bg: #fff;
    --info-bg: #e6f0ff;
    --info-text: #084298;
}

/* Global background and font */
.stApp {
    background-color: var(--bg);
    font-family: 'Inter', sans-serif;
    color: var(--text-dark);
}

/* Page container */
.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 2rem;
    max-width: 1400px;
}

/* Card container */
.card {
    background: var(--card-bg);
    border-radius: 14px;
    padding: 1.6rem;
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.06);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 1.4rem;
}
.card:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 28px rgba(0, 0, 0, 0.08);
}

/* Hero section */
.hero {
    background: var(--card-bg);
    padding: 2.4rem 2rem;
    border-radius: 16px;
    box-shadow: 0 8px 24px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
    text-align: center;
}
.hero h1 {
    font-weight: 900;
    font-size: 2.4rem;
    margin-bottom: 0.4rem;
    letter-spacing: -0.02em;
}
.hero p {
    color: var(--text-muted);
    font-size: 1.05rem;
    margin: 0;
}

/* Info box */
.info-box {
    background: var(--info-bg);
    color: var(--info-text);
    padding: 0.8rem 1rem;
    border-radius: 8px;
    font-weight: 500;
    font-size: 0.96rem;
    margin-top: 0.5rem;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, var(--accent), var(--accent2));
    color: #fff;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    font-weight: 600;
    font-size: 1rem;
    cursor: pointer;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 14px rgba(0, 123, 255, 0.25);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 18px rgba(0, 123, 255, 0.35);
}

/* Download button link */
.copy-btn {
    display: inline-block;
    background: var(--accent);
    color: #fff !important;
    padding: 0.55rem 1.3rem;
    border-radius: 8px;
    font-weight: 600;
    text-decoration: none;
    font-size: 0.95rem;
}
.copy-btn:hover {
    background: #0056b3;
}

/* Metric bars */
.metric-bar {
    margin-bottom: 0.9rem;
}
.metric-bar-label {
    font-weight: 600;
    font-size: 0.95rem;
    margin-bottom: 0.25rem;
}
.metric-bar-value {
    color: var(--text-muted);
    font-weight: normal;
}
.metric-fill-container {
    height: 12px;
    background: #e9ecef;
    border-radius: 6px;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    background: linear-gradient(90deg, #6dd5ed, #2193b0);
    transition: width 1.4s ease-in-out;
}

/* Footer */
.footer {
    margin-top: 2rem;
    padding: 1.3rem;
    background: var(--card-bg);
    border-radius: 14px;
    font-size: 0.9rem;
    color: var(--text-muted);
    text-align: center;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
}
.footer b {
    color: var(--text-dark);
}
.gh-link img {
    width: 22px;
    vertical-align: middle;
    margin-right: 6px;
}
</style>
""", unsafe_allow_html=True)

# --- Hero ---
st.markdown("""
<div class="hero">
    <h1>🖼️ Document & Image Classifier</h1>
    <p>Analyze images with OCR & AI — classify content accurately in seconds.</p>
</div>
""", unsafe_allow_html=True)

# --- Metric bar ---
def metric_bar(label, value, max_value=1.0):
    pct = int(100 * min(value / max_value, 1.0))
    st.markdown(f"""
    <div class="metric-bar">
      <div class="metric-bar-label">{label}: <span class="metric-bar-value">{value:.4f}</span></div>
      <div class="metric-fill-container">
        <div class="metric-fill" style='width:{pct}%;'></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# --- Load model ---
try:
    ml_model = joblib.load("rf_model.joblib")
    ml_label_map = joblib.load("rf_labels.joblib")
    set_rf_model(ml_model, ml_label_map)
except Exception:
    ml_model = None

# --- Layout ---
col1, col2 = st.columns([1, 1.5], gap="large")

# --- Upload ---
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        try:
            image = Image.open(uploaded_file)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            uploaded_file = None

# --- Processing ---
with col2:
    if uploaded_file:
        try:
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            processed_img = preprocess_image(img_cv)
            extracted_text = pytesseract.image_to_string(processed_img)

            # Text extraction card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📝 Extracted Text")
            if extracted_text.strip():
                st.text_area("", extracted_text, height=200)
                b64_text = base64.b64encode(extracted_text.encode()).decode()
                st.markdown(f"""
                    <a class="copy-btn" href="data:text/plain;base64,{b64_text}" download="extracted_text.txt">⬇ Download Text</a>
                """, unsafe_allow_html=True)
            else:
                st.markdown("<div class='info-box'>No text detected in the image.</div>", unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

            # Classification
            result = classify_text(img_cv, processed_img)

            # Category card (now using info-box)
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📌 Predicted Category")
            if result.get('category'):
                st.markdown(
                    f"<div class='info-box'><b>{result['category']}</b> — Confidence: {result['score']*100:.1f}%</div>",
                    unsafe_allow_html=True
                )
                if result.get('ml_label'):
                    st.markdown(f"**ML Model Prediction:** {result['ml_label']} ({result['ml_conf']*100:.1f}%)")
            else:
                st.warning("Could not classify the image.")
            st.markdown('</div>', unsafe_allow_html=True)

            # Metrics card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("📊 Classification Metrics")
            metric_bar("Text Ratio", result['text_ratio'], 0.05)
            metric_bar("Edge Density", result['edge_density'], 0.05)
            metric_bar("Color Variance", result['color_variance'], 1.0)
            metric_bar("Text Pixels Ratio", result['text_pixels_ratio'], 0.05)
            st.write(f"- **Aspect Ratio:** `{result['aspect_ratio']:.2f}`")
            st.write(f"- **Image Size:** `{result['width']} x {result['height']}`")
            st.markdown('</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Error: {e}")
            st.info("Try another image.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("<div class='info-box'>Upload an image to begin analysis.</div>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    Developed by <b>Praix Tech</b>, <b>Jahsmine</b> & <b>John Olumide</b> — 2025<br>
    <b>Disclaimer:</b> Educational use only.<br>
    <a class="gh-link" href="https://github.com/Praiz22/ocr-tech" target="_blank">
        <img src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png" alt="GitHub" /> View on GitHub
    </a>
</div>
""", unsafe_allow_html=True)
