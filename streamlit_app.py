import streamlit as st
from utils.preprocessing import display_preprocessing_section
from utils.ocr import display_ocr_section
from utils.train import display_training_section
from utils.prediction import display_prediction_section

st.set_page_config(page_title="Futuristic OCR Dashboard", layout="wide")
with open("utils/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Header
st.markdown(
    """
    <div class="glass-header">
      <h1>🔮 Futuristic OCR Dashboard</h1>
      <h4>AI-powered OCR & ML with Next-Gen UI</h4>
    </div>
    """, unsafe_allow_html=True
)

# Main Layout
tab1, tab2, tab3, tab4 = st.tabs([
    "Upload & Preprocess", "OCR Extraction", "Model Training", "Prediction"
])

# Upload & Preprocess Tab
with tab1:
    display_preprocessing_section()

# OCR Extraction Tab
with tab2:
    display_ocr_section()

# Model Training Tab
with tab3:
    display_training_section()

# Prediction Tab
with tab4:
    display_prediction_section()

st.markdown(
    """<div class="footer-glass"><small>© 2025 - Futuristic OCR Dashboard</small></div>""",
    unsafe_allow_html=True
)
