import streamlit as st
from PIL import Image

def display_preprocessing_section():
    st.markdown('<div class="glass-card"><h3>📷 Upload & Preprocess</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        # Placeholder for preprocessing metrics
        st.markdown('<div class="metric-glass">Preprocessing metrics will appear here.</div>', unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)
