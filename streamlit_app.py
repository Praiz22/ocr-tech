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
    page_title="Praix Tech — OCR Classifier",
    page_icon="🖼️",
    layout="wide",
)

# ------------------------------
# Styles: light, modern, professional
# ------------------------------
st.markdown(
    """
    <style>
    :root{
      --bg:#f7fbfc;
      --card:#ffffff;
      --muted:#6b7280;
      --accent:#0b66ff;
      --teal:#06b6d4;
      --soft:#eef7ff;
    }
    html, body, [class*="css"]  {
      background: var(--bg) !important;
      color: #052225 !important;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* Hero */
    .hero {
      background: linear-gradient(90deg, rgba(11,102,255,0.06), rgba(6,182,212,0.04));
      border-radius: 14px;
      padding: 28px;
      margin-bottom: 18px;
      box-shadow: 0 8px 30px rgba(2,6,23,0.04);
    }
    .hero h1 { margin:0; font-size:32px; letter-spacing: -0.3px; }
    .hero p { margin:6px 0 0 0; color:var(--muted); }

    /* main card */
    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 16px;
      box-shadow: 0 6px 18px rgba(2,6,23,0.04);
      margin-bottom: 14px;
    }

    .muted { color: var(--muted); }
    .accent { color: var(--accent); font-weight:600; }
    .pill {
      display:inline-block;
      padding:8px 12px;
      border-radius:999px;
      background: linear-gradient(90deg, rgba(11,102,255,0.12), rgba(6,182,212,0.06));
      color:var(--accent);
      font-weight:600;
    }

    /* Carousel styling container within Streamlit component */
    .carousel-wrapper { width:100%; max-width:940px; margin:auto; }
    .carousel {
      position:relative;
      overflow:hidden;
      border-radius:12px;
      background: linear-gradient(180deg, rgba(255,255,255,0.9), rgba(255,255,255,0.95));
      border:1px solid rgba(11,102,255,0.06);
      padding:10px;
    }
    .slides { display:flex; transition: transform 450ms ease; gap:12px; align-items:center; }
    .slide { min-width:100%; flex-shrink:0; display:flex; justify-content:center; align-items:center; }
    .slide img { width:100%; height:auto; border-radius:10px; object-fit:cover; max-height:360px; }

    .carousel .nav {
      position:absolute; top:50%; transform:translateY(-50%);
      background:rgba(0,0,0,0.06); border-radius:999px; padding:8px; cursor:pointer;
    }
    .nav.left { left:12px; } .nav.right { right:12px; }

    /* responsive */
    @media (max-width:768px) {
      .hero h1 { font-size:22px; }
    }

    /* small fade-in animation for cards */
    .fade { opacity:0; transform:translateY(8px); transition: all 420ms ease; }
    .fade.visible { opacity:1; transform:translateY(0px); }

    /* buttons */
    .btn {
      background: linear-gradient(90deg, var(--accent), #06b6d4);
      color:white; padding:10px 14px; border-radius:10px; border:0; cursor:pointer; font-weight:600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------------------
# Carousel HTML + JS (autoplay + arrows)
# uses embedded SVG placeholders (data URIs) so no external images required
# ------------------------------
carousel_html = """
<div class="carousel-wrapper">
  <div class="carousel" id="carousel">
    <div class="slides" id="slides">
      <!-- Slide 1 -->
      <div class="slide"><img src='data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720"><rect width="100%" height="100%" fill="%23e6f2ff"/><text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="36" fill="%230b66ff">Sample — Text Document</text><text x="50%" y="60%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="20" fill="%236b7280">Printed pages, invoices, scans</text></svg>'/></div>

      <!-- Slide 2 -->
      <div class="slide"><img src='data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720"><rect width="100%" height="100%" fill="%23fff7ed"/><text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="36" fill="%23fb923c">Sample — Screenshot</text><text x="50%" y="60%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="20" fill="%236b7280">App UIs, chat captures, receipts</text></svg>'/></div>

      <!-- Slide 3 -->
      <div class="slide"><img src='data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="720"><rect width="100%" height="100%" fill="%23eefaf6"/><text x="50%" y="45%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="36" fill="%2300b894">Sample — Photograph</text><text x="50%" y="60%" dominant-baseline="middle" text-anchor="middle" font-family="Arial" font-size="20" fill="%236b7280">Camera photos & designs</text></svg>'/></div>
    </div>

    <div class="nav left" id="prev">&#10094;</div>
    <div class="nav right" id="next">&#10095;</div>
  </div>
</div>

<script>
(function() {
  const slides = document.getElementById('slides');
  const total = slides.children.length;
  let idx = 0;
  let width = slides.children[0].clientWidth + 12; // gap
  function update() { slides.style.transform = `translateX(${-idx * width}px)`; }
  window.addEventListener('resize', () => { width = slides.children[0].clientWidth + 12; update(); });

  document.getElementById('prev').addEventListener('click', () => {
    idx = (idx - 1 + total) % total;
    update();
  });
  document.getElementById('next').addEventListener('click', () => {
    idx = (idx + 1) % total;
    update();
  });

  // autoplay
  let autoplay = setInterval(() => {
    idx = (idx + 1) % total;
    update();
  }, 3000);

  // pause on hover
  const carousel = document.getElementById('carousel');
  carousel.addEventListener('mouseenter', () => clearInterval(autoplay));
  carousel.addEventListener('mouseleave', () => {
    autoplay = setInterval(() => {
      idx = (idx + 1) % total; update();
    }, 3000);
  });
})();
</script>
"""

# render hero + carousel
st.markdown(
    f"""
    <div class="hero fade" id="hero">
      <div style="display:flex; justify-content:space-between; align-items:center; gap:12px;">
        <div>
          <h1>Praix Tech — OCR-Based Image Classifier</h1>
          <p class="muted">Upload an image — quickly extract text and auto-categorize into Document, Screenshot, Photo, or Other.</p>
        </div>
        <div style="text-align:right;">
          <div class="pill">Praix Tech • Jahsmine</div>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

components.html(carousel_html, height=420)

# ------------------------------
# Main upload + result cards
# ------------------------------
left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)
    st.markdown("### Upload image")
    upload = st.file_uploader("Drag & drop or click to upload (PNG / JPG / JPEG)", type=["png","jpg","jpeg"])
    st.caption("Tip: images with clear text work best for OCR.")
    if upload:
        pil = Image.open(upload).convert("RGB")
        preview = prepare_for_display(pil)
        st.image(preview, caption="Preview", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card fade">', unsafe_allow_html=True)
    st.markdown("### Actions")
    if upload:
        # We'll show copy/download actions in right column under OCR text too
        st.write("You can copy or download extracted text after processing.")
    else:
        st.write("Upload an image to see actions.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card fade">', unsafe_allow_html=True)
    st.markdown("### Results")
    if upload:
        with st.spinner("Processing — preprocessing & OCR..."):
            img_cv = np.array(pil)[:, :, ::-1].copy()  # RGB -> BGR
            processed_bin = preprocess_image(img_cv)
            result = smart_classify(img_cv, processed_bin, advanced=True)
            time.sleep(0.3)  # small UX delay
        st.markdown(f"**Category:** <span class='accent'>{result['category']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {result['score']:.2f}")
        st.markdown("")

        st.subheader("Extracted text")
        ocr_text = result.get("text", "")
        if not ocr_text.strip():
            st.write("_No text detected._")
        else:
            # editable text area
            txt_area = st.text_area("Detected text (editable)", value=ocr_text, height=220, key="ocr_text_area")

            # Copy to clipboard button (JS)
            copy_button = """
            <button id="copyBtn" class="btn">Copy to clipboard</button>
            <script>
            const btn = document.getElementById('copyBtn');
            btn.addEventListener('click', async () => {
              const ta = document.querySelector('textarea');
              const text = ta.value;
              await navigator.clipboard.writeText(text);
              btn.innerText = 'Copied ✓';
              setTimeout(()=>{ btn.innerText = 'Copy to clipboard'; }, 1400);
            });
            </script>
            """
            components.html(copy_button, height=50)

            # Download
            bts = BytesIO()
            bts.write(txt_area.encode('utf-8'))
            bts.seek(0)
            st.download_button("⬇️ Download extracted text (.txt)", data=bts, file_name="extracted_text.txt", mime="text/plain")

        st.markdown("<hr/>", unsafe_allow_html=True)
        st.subheader("Image metrics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Width", result["width"])
        c1.metric("Height", result["height"])
        c2.metric("Detected chars", len(ocr_text))
        c2.metric("Edge density", f"{result['edge_density']:.3f}")
        c3.metric("Color variance", f"{result['color_variance']:.3f}")
    else:
        st.info("No image uploaded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

# ------------------------------
# Minimal footer (signature + disclaimer)
# ------------------------------
st.markdown(
    """
    <div style="margin-top:18px; text-align:center;">
      <div style="display:inline-block; padding:10px 14px; border-radius:999px; background:linear-gradient(90deg, rgba(11,102,255,0.12), rgba(6,182,212,0.06)); font-weight:600; color:#0b66ff;">Praix Tech • Jahsmine</div>
      <div style="margin-top:8px; color:#64748b; font-size:13px;">Developed for educational & research purposes. Disclaimer: results are heuristic-based and for demo only.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# small script to add .visible class to fade elements after load
components.html(
    """
    <script>
    document.querySelectorAll('.fade').forEach(el => {
      setTimeout(()=> el.classList.add('visible'), 120);
    });
    </script>
    """,
    height=0,
)
