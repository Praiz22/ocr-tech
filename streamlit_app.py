import streamlit as st
import cv2
import numpy as np
from PIL import Image
import base64
import joblib
import re

# Import custom modules
# We are no longer importing classify_text to avoid the bug
from utils.ocr_utils import ocr_ensemble

# --- Enhanced Custom CSS ---
st.markdown("""
<style>
/* General app background and font */
.stApp {
    background-color: #f7f7f7;
    color: #111;
    font-family: 'Inter', sans-serif;
}

/* Main content container padding */
.main .block-container {
    padding-top: 1.5rem;
    padding-bottom: 2rem;
}

/* Custom card styling with a subtle shadow and hover effect */
.card {
    background: #ffffff;
    border-radius: 16px;
    padding: 2rem;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    margin-bottom: 1.5rem;
}
.card:hover {
    transform: translateY(-5px);
    box-shadow: 0 6px 20px rgba(0, 0, 0, 0.12);
}

/* Hero section for the header */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1rem 1rem;
    background: #ffffff;
    border-radius: 16px;
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
    margin-bottom: 2rem;
}
.hero h1 {
    font-family: 'Inter', sans-serif;
    font-weight: 900;
    color: #111;
    margin-bottom: 0.5rem;
    letter-spacing: -0.04em;
    font-size: 2.5rem;
}
.hero p {
    color: #555;
    font-size: 1.1rem;
    margin-bottom: 0;
}

/* Button styling with a professional gradient */
.stButton>button {
    background: linear-gradient(90deg, #007bff, #00c6ff);
    color: #fff;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    transition: transform 0.2s, box-shadow 0.2s;
    box-shadow: 0 4px 10px rgba(0, 123, 255, 0.3);
}
.stButton>button:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 15px rgba(0, 123, 255, 0.4);
}
.copy-btn {
    background: #007bff;
    color: #fff;
    border: none;
    padding: 0.6rem 1.5rem;
    border-radius: 8px;
    cursor: pointer;
    font-weight: 600;
    font-size: 1.05rem;
    margin-top: 0.5rem;
    transition: background 0.2s, transform 0.1s;
}
.copy-btn:hover {
    background: #0056b3;
    transform: translateY(-2px);
}

/* Metric bar styling with animation and gradient */
.metric-bar {
    margin-bottom: 1rem;
}
.metric-bar-label {
    font-size: 1rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 0.25rem;
}
.metric-bar-value {
    font-weight: normal;
    color: #555;
}
.metric-fill-container {
    height: 15px;
    background: #e9ecef;
    border-radius: 7px;
    overflow: hidden;
}
.metric-fill {
    height: 100%;
    background: linear-gradient(90deg, #6dd5ed, #2193b0);
    transition: width 1.5s ease-in-out;
}

/* Footer styling */
.footer {
    margin-top: 2rem;
    padding: 1.5rem;
    text-align: center;
    background: #ffffff;
    border-radius: 16px;
    font-size: 0.9rem;
    color: #555;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.footer b {
    color: #111;
}
a.gh-link {
    display: inline-block;
    margin-top: 10px;
    text-decoration: none;
}
a.gh-link img {
    width: 24px;
    vertical-align: middle;
    margin-right: 8px;
}

/* Custom text and headers */
h2, h3, h4 {
    color: #111;
    font-family: 'Inter', sans-serif;
    font-weight: 700;
}

/* Fixing Streamlit's file uploader label color */
div[data-testid="stFileUploader"] label p {
    color: #333;
}
</style>
""", unsafe_allow_html=True)

# --- Hero Section ---
st.markdown("""
<div class="hero">
    <h1>🖼️ Document & Image Classifier</h1>
    <p>Using OCR and AI for smarter content analysis</p>
</div>
""", unsafe_allow_html=True)

# --- Metric Bar with Animation Function ---
def metric_bar(label, value, max_value=1.0):
    pct = int(100 * min(value / max_value, 1.0))
    bar = f"""
    <div class="metric-bar">
      <div class="metric-bar-label">{label}: <span class="metric-bar-value">{value:.4f}</span></div>
      <div class="metric-fill-container">
        <div class="metric-fill" style='width:{pct}%;'></div>
      </div>
    </div>
    """
    st.markdown(bar, unsafe_allow_html=True)

# --- Load ML Model (if available) ---
ml_model = None
ml_label_map = None
try:
    ml_model = joblib.load("rf_model.joblib")
    ml_label_map = joblib.load("rf_labels.joblib")
except Exception:
    st.warning("ML model not found. Using heuristic classification.")

# --- Main App Columns ---
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📤 Upload an Image")
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])
    st.markdown('</div>', unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Processing image..."):
            try:
                # Open the uploaded image and convert it to a format OpenCV can use
                image = Image.open(uploaded_file)
                img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.image(image, caption="Uploaded Image", use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error loading image: {e}")
                uploaded_file = None

with col2:
    if uploaded_file:
        with st.spinner("Processing image and classifying..."):
            try:
                # Use ocr_ensemble to get text and the preprocessed image
                result_ocr = ocr_ensemble(img_cv, psm_list=(3, 6, 11))
                extracted_text = result_ocr['text']
                processed_img_for_classify = result_ocr['processed_img']

                # --- Inlined Classification Logic (to avoid bug in classify.py) ---
                
                # Extract features from the preprocessed image
                height, width = processed_img_for_classify.shape
                gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
                
                color_var = float(np.var(cv2.cvtColor(img_cv, cv2.COLOR_BGR2LAB)) / 255.0)
                edge_d = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / (height * width)
                text_pixels = float(np.count_nonzero(processed_img_for_classify == 0)) / (height * width)
                aspect = float(width) / (height + 1e-6)
                
                text_len = len(re.findall(r'\b\w+\b', extracted_text))
                text_ratio = len(extracted_text.strip()) / (height * width + 1e-6)

                # Heuristic classification based on features
                heuristic_category = "Other / Image"
                heuristic_score = 0.5
                if text_len > 200 and text_pixels > 0.01:
                    heuristic_category = "Text Document"
                    heuristic_score = 0.9
                elif text_len > 50 and edge_d > 0.02 and color_var < 0.4:
                    heuristic_category = "Screenshot / UI"
                    heuristic_score = 0.8
                elif color_var > 0.45 and text_len < 30 and edge_d < 0.01:
                    heuristic_category = "Photograph"
                    heuristic_score = 0.95
                elif text_len > 0 and text_len <= 80 and color_var < 0.35:
                    heuristic_category = "Scanned Note / Small Text"
                    heuristic_score = 0.75
                else:
                    if text_pixels > 0.005:
                        heuristic_category = "Text Document"
                        heuristic_score = 0.6
                    else:
                        heuristic_category = "Other / Image"
                        heuristic_score = 0.4

                result = {
                    "category": heuristic_category,
                    "score": min(1.0, heuristic_score),
                    "text_ratio": text_ratio,
                    "edge_density": edge_d,
                    "color_variance": color_var,
                    "text_pixels_ratio": text_pixels,
                    "aspect_ratio": aspect,
                    "width": width,
                    "height": height,
                    "text": extracted_text,
                }
                
                # --- End of inlined classification logic ---

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📝 Extracted Text")
                if extracted_text.strip():
                    st.text_area("", extracted_text, height=200)
                    b64_text = base64.b64encode(extracted_text.encode()).decode()
                    st.markdown(f"""
                        <a class="copy-btn" href="data:text/plain;base64,{b64_text}" download="extracted_text.txt">⬇ Download Text</a>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No text could be extracted from the image.")
                st.markdown('</div>', unsafe_allow_html=True)

                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("📌 Predicted Category")
                if result['category']:
                    st.success(f"{result['category']}  —  Confidence: {result['score']*100:.1f}%")
                    # ML model prediction logic removed as it was not being used correctly and causing issues.
                else:
                    st.warning("Could not classify the image.")
                st.markdown('</div>', unsafe_allow_html=True)

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
                st.error(f"An unexpected error occurred during processing: {e}")
                st.info("Please try uploading a different image.")
    else:
        st.markdown('<div class="card">', unsafe_allow_html=True)
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
