# streamlit_app.py
import streamlit as st
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"
import numpy as np
from PIL import Image
from io import BytesIO
from utils.preprocessing import preprocess_image, prepare_for_display
from utils.classify import smart_classify
import base64
import streamlit.components.v1 as components
import time

st.set_page_config(
    page_title="PraixTech — OCR Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ----------------------
# Styles: light professional tone + subtle glass
# ----------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#f6f8fa;
      --card:#ffffff;
      --muted:#6b7280;
      --accent:#0b66ff;
      --glass: rgba(255,255,255,0.8);
    }
    body { background: var(--bg); color: #0b1724; }
    .top-hero {
        background: linear-gradient(90deg, rgba(11,102,255,0.06), rgba(3,201,183,0.03));
        border-radius: 16px;
        padding: 28px;
        margin-bottom: 18px;
        box-shadow: 0 6px 20px rgba(15,23,42,0.05);
    }
    .app-card {
        background: var(--card);
        border-radius: 12px;
        padding: 18px;
        box-shadow: 0 6px 20px rgba(15,23,42,0.04);
    }
    h1 { margin: 0; font-size: 30px; }
    .muted { color: var(--muted); font-size:14px; }
    .accent { color: var(--accent); font-weight:700; }
    .footer {
        margin-top: 36px;
        padding: 16px;
        text-align:center;
        color: #475569;
        font-size: 13px;
    }
    .signature {
        display:inline-block;
        margin-top:10px;
        padding:8px 14px;
        border-radius:999px;
        background: linear-gradient(90deg, rgba(11,102,255,0.12), rgba(3,201,183,0.06));
        box-shadow: 0 8px 30px rgba(11,102,255,0.06);
        font-weight:600;
        color: var(--accent);
    }
    /* small animation on cards */
    .fade-in { opacity:0; transform: translateY(8px); transition: all 450ms ease; }
    .fade-in.visible { opacity:1; transform: translateY(0px); }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Small JS to animate visible elements (works in Streamlit)
# ----------------------
components.html(
    """
    <script>
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) e.target.classList.add('visible');
      });
    }, {threshold: 0.15});
    document.querySelectorAll('.fade-in').forEach(n => observer.observe(n));
    </script>
    """,
    height=0,
    scrolling=False,
)

# ----------------------
# Header / Hero
# ----------------------
st.markdown(
    """
    <div class="top-hero">
      <div style="display:flex; align-items:center; justify-content:space-between; gap:12px;">
        <div>
          <h1>OCR Image Classification — Praix Tech</h1>
          <div class="muted">Robust OCR + smart image heuristics to automatically categorize images (documents, screenshots, photos, designs) — responsive and built for demos.</div>
        </div>
        <div style="text-align:right">
          <div class="signature">Praix Tech • Jahsmine</div>
          <div style="font-size:12px; color:#64748b; margin-top:6px;">Supervised by Mrs. Oguniyi</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------
# Sidebar controls
# ----------------------
st.sidebar.markdown("### Settings")
use_advanced = st.sidebar.checkbox("Enable advanced heuristics", value=True)
show_debug = st.sidebar.checkbox("Show debug values", value=False)
download_txt = st.sidebar.checkbox("Enable text download", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("Tip: Upload screenshots, receipts, photos or UI pictures to test categorization.")

# ----------------------
# Main layout: left = upload & preview, right = results & details
# ----------------------
left, right = st.columns([1.1, 1])

with left:
    st.markdown('<div class="app-card fade-in">', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload an image", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    st.write("")  # spacing
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        # small display helper
        st.image(prepare_for_display(image), caption="Uploaded image", use_container_width=True)
        st.write("") 
    else:
        st.info("Upload a file to start. Try a screenshot, receipt, or a photo.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="app-card fade-in">', unsafe_allow_html=True)
    if uploaded_file:
        # Spinner while processing
        with st.spinner("Processing image and extracting text..."):
            # convert to CV2
            img_cv = np.array(image)[:, :, ::-1].copy()  # RGB->BGR
            processed = preprocess_image(img_cv)  # returns cleaned grayscale/binary for OCR
            # classify
            result = smart_classify(img_cv, processed, advanced=use_advanced)
            time.sleep(0.4)  # small UX delay so spinner is visible

        # Show category clear
        st.markdown(f"### 📌 Predicted Category: **{result['category']}**")
        st.markdown(f"**Confidence (heuristic):** {result['score']:.2f}")
        st.write("")

        # Show OCR text
        st.subheader("📝 Extracted Text")
        ocr_text = result.get("text", "")
        if ocr_text.strip() == "":
            st.write("*No readable text detected.*")
        else:
            st.text_area("Detected text (editable)", value=ocr_text, height=200, key="ocr_text_area", help="You can edit and download this text.")

        # Download button
        if download_txt and ocr_text.strip():
            bts = BytesIO()
            bts.write(ocr_text.encode("utf-8"))
            bts.seek(0)
            st.download_button("⬇️ Download extracted text (.txt)", data=bts, file_name="extracted_text.txt", mime="text/plain")

        # Extras: image stats & debug
        st.subheader("Image Details & Checks")
        col_a, col_b = st.columns(2)
        col_a.metric("Width", result["width"])
        col_a.metric("Height", result["height"])
        col_b.metric("Text chars", len(ocr_text))
        col_b.metric("Edge density", f"{result['edge_density']:.3f}")

        if show_debug:
            st.write("### Debug info")
            st.json({
                "text_len": len(ocr_text),
                "text_ratio": result["text_ratio"],
                "edge_density": result["edge_density"],
                "color_variance": result["color_variance"],
                "aspect_ratio": result["aspect_ratio"]
            })

    else:
        st.subheader("Results")
        st.write("Upload an image to run OCR and classification.")
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Footer (credits + disclaimer)
# ----------------------
st.markdown(
    f"""
    <div class="footer">
      <div>Developed by <strong>Praix Tech</strong> & <strong>Jahsmine</strong> for educational & research purposes under the supervision of <strong>Mrs. Oguniyi</strong>.</div>
      <div style="margin-top:8px; color:#94a3b8; font-size:12px;">
        Disclaimer: This tool is for demonstration & educational use only. Results are heuristic-based and not guaranteed for production use.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)
