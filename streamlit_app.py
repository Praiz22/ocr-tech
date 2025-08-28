import streamlit as st
from PIL import Image
import time
import numpy as np

# === Inject your custom CSS (adapted from your HTML) ===
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

st.markdown(
    """
    <div class="glow"></div>
    <div class="shell">
      <div class="hero fade-in">
        <div class="brand">
          <div class="logo"></div>
          <div>
            <div class="title">Praix Tech — OCR Lab</div>
            <div class="subtitle">Futuristic UI Prototype • Upload → Preprocess → Train → Predict</div>
          </div>
        </div>
        <button class="btn" id="startDemo">Start Demo</button>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# === Upload Section ===
with st.container():
    st.markdown("""
    <section class="card col-12 fade-in">
      <div class="head">
        <h3>Upload Image</h3>
        <span class="soft-accent">10s processing budget</span>
      </div>
      """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="fileInput")
    st.markdown("</section>", unsafe_allow_html=True)
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

# === Preprocessing Section ===
with st.container():
    st.markdown("""
    <section class="card col-8 fade-in delay">
      <div class="head">
        <h3>Data Preprocessing</h3>
        <span class="soft-accent" id="preStatus">Idle</span>
      </div>
      <div class="steps" id="preSteps">
        <span class="chip active">Noise Removal</span>
        <span class="chip">Thresholding</span>
        <span class="chip">Deskew</span>
        <span class="chip">Normalization</span>
        <span class="chip">Ready</span>
      </div>
      <div class="metrics" style="margin-top:14px">
        <div class="metric">
          <div class="row"><span class="label">Noise Removal</span><span class="value" id="m_noise">0%</span></div>
          <div class="bar"><i id="b_noise"></i></div>
        </div>
        <div class="metric">
          <div class="row"><span class="label">Thresholding</span><span class="value" id="m_thresh">0%</span></div>
          <div class="bar"><i id="b_thresh"></i></div>
        </div>
        <div class="metric">
          <div class="row"><span class="label">Deskew</span><span class="value" id="m_skew">0%</span></div>
          <div class="bar"><i id="b_skew"></i></div>
        </div>
        <div class="metric">
          <div class="row"><span class="label">Normalization</span><span class="value" id="m_norm">0%</span></div>
          <div class="bar"><i id="b_norm"></i></div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    # Optionally, add live metric updates here using st.progress or st.empty()

# === Training Panel ===
with st.container():
    st.markdown("""
    <section class="card col-4 fade-in delay">
      <div class="head">
        <h3>Training Model</h3>
        <span class="soft-accent" id="trainStatus">Paused</span>
      </div>
      <div class="metric" style="margin-bottom:12px">
        <div class="row"><span class="label">Epoch</span><span class="value" id="m_epoch">0 / 10</span></div>
        <div class="bar"><i id="b_epoch"></i></div>
      </div>
      <div class="spark" style="margin-bottom:10px">
        <div style="flex:1">
          <div class="row" style="margin-bottom:8px">
            <span class="label">Accuracy</span><span class="value" id="m_acc">0%</span>
          </div>
          <div class="bar"><i id="b_acc"></i></div>
        </div>
      </div>
      <div class="spark">
        <div style="flex:1">
          <div class="row" style="margin-bottom:8px">
            <span class="label">Loss</span><span class="value" id="m_loss">1.000</span>
          </div>
          <div class="bar"><i id="b_loss"></i></div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    # Add st.button for "Train" and show live metrics with st.empty()

# === Prediction Results Section ===
with st.container():
    st.markdown("""
    <section class="card col-12 fade-in delay2">
      <div class="head">
        <h3>Prediction Results</h3>
        <span class="soft-accent" id="predStatus">Waiting for input</span>
      </div>
      <div class="grid" style="gap:16px">
        <div class="col-8">
          <div class="result">
            <span class="tag soft-accent">Extracted Text</span>
            <div class="mono" id="ocrText">—</div>
          </div>
        </div>
        <div class="col-4">
          <div class="result">
            <span class="tag soft-accent">Prediction</span>
            <div class="mono"><strong id="predLabel">—</strong></div>
            <div class="confidence">
              <div class="row"><span class="label">Confidence</span><span class="value" id="confVal">0%</span></div>
              <div class="bar"><i id="confBar"></i></div>
            </div>
          </div>
        </div>
      </div>
    </section>
    """, unsafe_allow_html=True)
    # Add OCR & prediction logic here, updating the content

# === JS for Glow Animation (optional in Streamlit, handled by CSS above) ===

# Footer or other elements can be added as needed.
