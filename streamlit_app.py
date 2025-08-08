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
import streamlit.components.v1 as components
import time
import base64

st.set_page_config(
    page_title="Praix Tech — OCR Image Classifier",
    page_icon="🖼️",
    layout="wide",
    initial_sidebar_state="auto",
)

# ----------------------
# Styles: light professional tone
# ----------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#f8fafc;
      --card:#ffffff;
      --muted:#64748b;
      --accent:#0b66ff;
      --teal:#06b6d4;
      --glass: rgba(255,255,255,0.85);
    }
    html, body, [class*="css"]  {
      background: var(--bg) !important;
      color: #06202a !important;
    }
    .hero {
        background: linear-gradient(90deg, rgba(11,102,255,0.06), rgba(6,182,212,0.04));
        border-radius: 14px;
        padding: 26px;
        margin-bottom: 18px;
        box-shadow: 0 8px 30px rgba(2,6,23,0.04);
    }
    .card {
        background: var(--card);
        border-radius: 10px;
        padding: 14px;
        box-shadow: 0 6px 18px rgba(2,6,23,0.04);
    }
    h1 { margin: 0; font-size: 28px; }
    .muted { color: var(--muted); font-size:14px; }
    .accent { color: var(--accent); font-weight:700; }
    .note { font-size:13px; color:#475569; }
    .pill {
        display:inline-block;
        padding:8px 12px;
        border-radius:999px;
        background: linear-gradient(90deg, rgba(11,102,255,0.12), rgba(6,182,212,0.06));
        color:var(--accent);
        font-weight:600;
    }
    .section-title { font-size:18px; margin-bottom:6px; }
    .grid { display:grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap:12px; }
    .fade { opacity:0; transform:translateY(8px); transition: all 420ms ease; }
    .fade.visible { opacity:1; transform:translateY(0px); }
    .small-muted { font-size:12px; color:#64748b; }
    .footer {
        margin-top:30px;
        padding:18px;
        text-align:center;
        color:#475569;
        font-size:13px;
    }
    </style>
    """, unsafe_allow_html=True
)

# ----------------------
# Intersection Observer for fade-in
# ----------------------
components.html(
    """
    <script>
    const observer = new IntersectionObserver((entries) => {
      entries.forEach(e => {
        if (e.isIntersecting) e.target.classList.add('visible');
      });
    }, {threshold: 0.12});
    document.addEventListener("DOMContentLoaded", function() {
      document.querySelectorAll('.fade').forEach(n => observer.observe(n));
    });
    </script>
    """,
    height=0,
)

# ----------------------
# Header / Hero
# ----------------------
st.markdown(
    """
    <div class="hero fade">
      <div style="display:flex; gap:12px; align-items:center; justify-content:space-between;">
        <div>
          <h1>Praix Tech — OCR Image Classification</h1>
          <div class="muted">Hybrid OCR + image heuristics to automatically classify images (Documents, Screenshots, Photos, Designs). Built for demonstration & evaluation.</div>
        </div>
        <div style="text-align:right">
          <div class="pill">Praix Tech • Jahsmine</div>
          <div class="small-muted" style="margin-top:6px;">Supervisor: Mrs. Oguniyi</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True
)

# ----------------------
# Sidebar: quick controls
# ----------------------
st.sidebar.markdown("### Controls & Settings")
use_advanced = st.sidebar.checkbox("Enable advanced heuristics", value=True)
show_debug = st.sidebar.checkbox("Show debug details", value=False)
download_txt = st.sidebar.checkbox("Enable download button", value=True)
st.sidebar.markdown("---")
st.sidebar.markdown("**Demo tips**: Try a screenshot, printed invoice, receipt or photo. Use the debug toggle to inspect metrics.")

# ----------------------
# Main layout - uploader & results
# ----------------------
col_left, col_right = st.columns([1.1, 1])

with col_left:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)
    st.markdown("### Upload & Preview")
    uploaded_file = st.file_uploader("Upload an image (PNG / JPG / JPEG)", type=["png", "jpg", "jpeg"], accept_multiple_files=False)
    st.caption("Max 200MB — screenshots, photos, scanned docs.")
    st.write("")
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        display = prepare_for_display(image)
        st.image(display, caption="Preview — uploaded image", use_container_width=True)
        st.write("")
    else:
        st.info("Upload an image to begin. Example images below help you test categories.")
    st.markdown("</div>", unsafe_allow_html=True)

    # Example gallery and documentation
    st.markdown('<div class="card fade" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("### Examples & Quick Guide")
    st.markdown("""
    <div class="grid">
      <div><strong>Text Document</strong><div class="small-muted">Scanned books, printed pages, pdf screenshots.</div></div>
      <div><strong>Screenshot / UI</strong><div class="small-muted">App UI screenshots, digital receipts, chat captures.</div></div>
      <div><strong>Photograph</strong><div class="small-muted">Camera photos with high color variance and little text.</div></div>
      <div><strong>Designs / Mockups</strong><div class="small-muted">UI designs, social posts, image-heavy layouts.</div></div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("### How it works (short)")
    st.markdown("""
    1. Image preprocessing (denoise, resize, binarize).  
    2. OCR extraction (Tesseract) to find textual content.  
    3. Image heuristics (edge density, color variance, text-pixel ratio).  
    4. Hybrid scoring to predict category and confidence.  
    """)
    st.markdown("</div>", unsafe_allow_html=True)

    # Large explanation section for supervisor
    st.markdown('<div class="card fade" style="margin-top:12px;">', unsafe_allow_html=True)
    st.markdown("### Project Overview (for evaluators)")
    st.markdown("""
    **Objective:** Build an OCR-augmented classification system that automatically categorizes images into document types and image categories using a fast hybrid approach.  
    **Architecture:** Client (Streamlit UI) → Server (Streamlit Cloud runs OCR + heuristics) → Firebase (optional for storage)  
    **Key strengths:** Fast inference, no heavy training required, clear explainability via heuristics and readable OCR output.  
    **Limitations:** Heuristic-based; accuracy depends on image quality and language support available to Tesseract; not intended as a production-grade OCR pipeline without further training and enterprise-grade models.
    """)
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)
    st.markdown("### Results & Actions")
    if uploaded_file:
        with st.spinner("Processing image..."):
            img_cv = np.array(image)[:, :, ::-1].copy()  # RGB -> BGR for cv2
            processed_bin = preprocess_image(img_cv)
            result = smart_classify(img_cv, processed_bin, advanced=use_advanced)
            time.sleep(0.35)

        st.markdown(f"#### Predicted Category: **{result['category']}**")
        st.markdown(f"**Confidence (heuristic):** {result['score']:.2f}")
        st.write("")

        st.subheader("Extracted Text")
        ocr_text = result.get("text", "")
        if ocr_text.strip() == "":
            st.write("*No readable text detected.*")
        else:
            # editable box for supervisor to tweak
            st.text_area("Detected text (editable)", value=ocr_text, height=220, key="ocr_text")

            # Copy to clipboard: JS + components
            copy_js = f"""
            <button id="copy-btn" style="padding:8px 12px; border-radius:8px; border:0; background:#0b66ff; color:white;">Copy to Clipboard</button>
            <script>
            const btn = document.getElementById('copy-btn');
            btn.addEventListener('click', async () => {{
              const txt = document.querySelector('textarea').value;
              await navigator.clipboard.writeText(txt);
              btn.innerText = 'Copied ✓';
              setTimeout(()=>{{ btn.innerText = 'Copy to Clipboard'; }}, 1600);
            }});
            </script>
            """
            components.html(copy_js, height=50)

            # Download
            if download_txt:
                bts = BytesIO()
                bts.write(ocr_text.encode("utf-8"))
                bts.seek(0)
                st.download_button("⬇️ Download extracted text (.txt)", data=bts, file_name="extracted_text.txt", mime="text/plain")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Image Metrics")
        cols = st.columns(3)
        cols[0].metric("Width", result["width"])
        cols[0].metric("Height", result["height"])
        cols[1].metric("Detected text chars", len(ocr_text))
        cols[1].metric("Edge density", f"{result['edge_density']:.3f}")
        cols[2].metric("Color variance", f"{result['color_variance']:.3f}")
        st.write("")

        if show_debug:
            st.subheader("Debug / Heuristics")
            st.json({
                "text_ratio": result["text_ratio"],
                "text_pixels_ratio": result.get("text_pixels_ratio", 0),
                "edge_density": result["edge_density"],
                "color_variance": result["color_variance"],
                "aspect_ratio": result["aspect_ratio"]
            })
    else:
        st.info("Waiting for an image upload...")

    st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Large comprehensive documentation card
# ----------------------
st.markdown('<div class="card fade" style="margin-top:14px;">', unsafe_allow_html=True)
st.markdown("## Deep Dive — Notes for the Supervisor")
st.markdown("""
**Dataset & testing**  
- This demo uses heuristic rules and Tesseract OCR. For production, we recommend collecting labeled examples and training a small classifier (SVM/LightGBM/NN) on features (text-length, color variance, edge density, OCR token counts, aspect ratio).  

**Extensions possible**  
- Replace heuristics with a small trained model (we can scaffold that).  
- Integrate Google Vision API or other modern OCR for better accuracy on complex layouts and languages.  
- Add Firebase integration for storage and audit logs.

**Evaluation checklist**  
- Test with printed documents, low-res scans, screenshots of apps, photographed receipts, and designed posters.  
- Note OCR quality: some fonts and handwritings will need training or custom models.
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

# ----------------------
# Footer
# ----------------------
st.markdown(
    """
    <div class="footer">
      <div><strong>Developed by Praix Tech & Jahsmine</strong> — for educational & research purposes under the supervision of <strong>Mrs. Oguniyi</strong>.</div>
      <div style="margin-top:6px; color:#94a3b8; font-size:12px;">
        Disclaimer: Demo / research tool — heuristic-based. Not for production use without further validation.
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# small final note
st.caption("UI updated for supervisor presentation — responsive, comprehensive, and ready for demo.")
