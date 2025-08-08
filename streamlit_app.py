# streamlit_app.py
import streamlit as st
import cv2
import pytesseract
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"  # keep for Streamlit Cloud
import numpy as np
from PIL import Image
from io import BytesIO
from utils.preprocessing import preprocess_image, prepare_for_display
from utils.classify import smart_classify
import streamlit.components.v1 as components
import time

# Page config
st.set_page_config(page_title="Praix Tech — OCR Classifier", page_icon="🖼️", layout="wide")

# -------------------------
# CSS + AOS (animations) + fonts
# -------------------------
st.markdown(
    """
    <!-- Google Fonts -->
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@600;700&family=Inter:wght@300;400;600&display=swap" rel="stylesheet">

    <!-- AOS for simple reveal animations -->
    <link href="https://unpkg.com/aos@2.3.1/dist/aos.css" rel="stylesheet">

    <style>
    :root{
      --bg:#f6fbfc;
      --card:#ffffff;
      --muted:#6b7280;
      --accent:#0b66ff;
      --teal:#06b6d4;
      --glass: rgba(255,255,255,0.85);
    }
    html, body, [class*="css"] {
      background: var(--bg) !important;
      color: #042023 !important;
      font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial;
    }

    /* HERO */
    .hero {
      background: linear-gradient(90deg, rgba(11,102,255,0.06), rgba(6,182,212,0.03));
      border-radius: 14px;
      padding: 30px;
      margin-bottom: 20px;
      box-shadow: 0 10px 30px rgba(2,6,23,0.04);
      display:flex;
      gap:24px;
      align-items:center;
      justify-content:space-between;
    }
    .hero-left { flex:1; min-width: 260px; }
    .hero-right { width:420px; max-width:42%; display:flex; justify-content:center; align-items:center; }

    .title { font-size:28px; font-weight:700; margin-bottom:6px; font-family: Poppins, sans-serif; }
    .subtitle { color:var(--muted); margin:0; font-size:14px; }

    .btn-primary {
      background: linear-gradient(90deg,var(--accent), #06b6d4);
      color:white; padding:10px 14px; border-radius:10px; border:0; font-weight:600; cursor:pointer;
    }

    .card {
      background: var(--card);
      border-radius: 12px;
      padding: 18px;
      box-shadow: 0 6px 18px rgba(2,6,23,0.04);
      margin-bottom: 14px;
    }

    /* carousel */
    .carousel {
      position: relative;
      overflow: hidden;
      border-radius: 12px;
      border:1px solid rgba(11,102,255,0.06);
    }
    .slides { display:flex; transition: transform 480ms ease; gap:12px; }
    .slide { min-width:100%; flex-shrink:0; display:flex; justify-content:center; align-items:center; }
    .slide img { width:100%; height:auto; border-radius:8px; max-height:360px; object-fit:cover; }

    .nav {
      position:absolute; top:50%; transform:translateY(-50%);
      background: rgba(2,6,23,0.06); padding:8px; border-radius:999px; cursor:pointer;
      user-select:none;
    }
    .nav.left { left:12px; }
    .nav.right { right:12px; }

    /* responsive */
    @media (max-width:880px){
      .hero { flex-direction:column; align-items:flex-start; }
      .hero-right { width:100%; max-width:100%; }
    }

    .muted { color:var(--muted); font-size:13px; }
    .small { font-size:13px; color:#475569; }

    /* fade-in helper */
    .fade { opacity:0; transform: translateY(8px); transition: all 420ms ease; }
    .fade.aos-animate { opacity:1; transform: translateY(0); }

    /* copy button styling in components */
    .copy-btn { background: linear-gradient(90deg,var(--accent),#06b6d4); color:white; padding:8px 12px; border-radius:8px; border:0; cursor:pointer; font-weight:600; }
    </style>
    <script src="https://unpkg.com/aos@2.3.1/dist/aos.js"></script>
    """,
    unsafe_allow_html=True,
)

# Initialize AOS
components.html("<script>AOS.init({duration:650, once:true});</script>", height=0)

# -------------------------
# Helper: carousel HTML (autoplay + arrows)
# We'll reference images in /assets/ (repo /assets folder)
# -------------------------
carousel_html = """
<div style="padding:12px;">
  <div class="carousel" id="carousel">
    <div class="slides" id="slides">
      <div class="slide"><img src="assets/carousel1.jpg" alt="sample1"/></div>
      <div class="slide"><img src="assets/carousel2.jpg" alt="sample2"/></div>
      <div class="slide"><img src="assets/carousel3.jpg" alt="sample3"/></div>
    </div>
    <div class="nav left" id="prev">&#10094;</div>
    <div class="nav right" id="next">&#10095;</div>
  </div>
</div>

<script>
(function(){
  const slides = document.getElementById('slides');
  const total = slides.children.length;
  let idx = 0;
  function width(){ return slides.children[0].clientWidth + 12; }
  function update(){ slides.style.transform = `translateX(${-idx * width()}px)`; }
  window.addEventListener('resize', update);

  document.getElementById('prev').addEventListener('click', () => { idx = (idx - 1 + total) % total; update(); });
  document.getElementById('next').addEventListener('click', () => { idx = (idx + 1) % total; update(); });

  // autoplay
  let auto = setInterval(()=> { idx = (idx + 1) % total; update(); }, 3000);

  const carousel = document.getElementById('carousel');
  carousel.addEventListener('mouseenter', ()=> clearInterval(auto));
  carousel.addEventListener('mouseleave', ()=> { auto = setInterval(()=> { idx = (idx + 1) % total; update(); }, 3000); });

  // initial sizing
  setTimeout(update, 120);
})();
</script>
"""

# -------------------------
# HERO (left: text + CTAs, right: ocr.gif)
# -------------------------
st.markdown(
    f"""
    <div class="hero fade" data-aos="fade-up">
      <div class="hero-left">
        <div class="title">OCR-Based Image Classification Tool</div>
        <div class="subtitle">Fast OCR + smart image heuristics — classify screenshots, scanned docs, photos & more.</div>
        <div style="height:12px;"></div>
        <div style="display:flex; gap:10px;">
          <button class="btn-primary" onclick="window.scrollTo({{top: document.body.scrollHeight*0.05, behavior:'smooth'}})">Upload Image</button>
          <button class="btn-primary" onclick="document.getElementById('carousel').scrollIntoView({behavior:'smooth'})" style="background:transparent; color:var(--accent); border:1px solid rgba(11,102,255,0.12)">See samples</button>
        </div>
        <div style="height:8px;"></div>
        <div class="small">Developed by <strong>Praix Tech</strong> & <strong>Jahsmine</strong></div>
      </div>
      <div class="hero-right" style="text-align:center;">
        <!-- show your ocr.gif from assets -->
        <div style="width:100%; max-width:420px; border-radius:12px; overflow:hidden; box-shadow: 0 8px 24px rgba(2,6,23,0.06)">
          <img src="assets/ocr.gif" style="width:100%; display:block;" alt="ocr gif" />
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# carousel
components.html(carousel_html, height=420)

# -------------------------
# Upload + Results cards
# -------------------------
left, right = st.columns([1.05, 1])

with left:
    st.markdown('<div class="card fade" data-aos="fade-up">', unsafe_allow_html=True)
    st.markdown("### Upload image")
    upload = st.file_uploader("Drag and drop or click to upload (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"])
    st.caption("Tip: Use clear images for best OCR results.")
    if upload:
        pil = Image.open(upload).convert("RGB")
        display = prepare_for_display(pil)
        st.image(display, caption="Preview", use_container_width=True)
    else:
        st.info("Upload an image to begin. Sample images above show expected categories.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="card fade" data-aos="fade-up">', unsafe_allow_html=True)
    st.markdown("### Actions")
    st.write("After processing you can copy or download the extracted text.")
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown('<div class="card fade" data-aos="fade-up">', unsafe_allow_html=True)
    st.markdown("### Results")
    if upload:
        with st.spinner("Processing image (preprocessing → OCR → heuristics)..."):
            img_cv = np.array(pil)[:, :, ::-1].copy()  # RGB->BGR
            processed_bin = preprocess_image(img_cv)
            result = smart_classify(img_cv, processed_bin, advanced=True)
            time.sleep(0.35)

        # Category + confidence
        st.markdown(f"**Category:** <span style='color:var(--accent); font-weight:700'>{result['category']}</span>", unsafe_allow_html=True)
        st.markdown(f"**Confidence:** {result['score']:.2f}")
        st.markdown("")

        # OCR text
        st.subheader("Extracted text")
        ocr_text = result.get("text", "")
        if not ocr_text.strip():
            st.write("_No readable text detected._")
        else:
            # editable text area
            txt_val = st.text_area("Detected text (editable)", value=ocr_text, height=220, key="ocr_text_area")

            # copy to clipboard button (JS)
            copy_html = f"""
            <div>
              <button class="copy-btn" id="copyBtn">Copy to clipboard</button>
            </div>
            <script>
            const btn = document.getElementById('copyBtn');
            btn.addEventListener('click', async () => {{
              const ta = document.querySelector('textarea#ocr_text_area');
              // Streamlit renders textareas multiple times; find one whose value equals `{ocr_text.replace("`","\\`")}`:
              const t = document.querySelector('textarea');
              const text = t ? t.value : `{ocr_text.replace("`","\\`")}`;
              await navigator.clipboard.writeText(text);
              btn.innerText = 'Copied ✓';
              setTimeout(()=> btn.innerText = 'Copy to clipboard', 1400);
            }});
            </script>
            """
            components.html(copy_html, height=50)

            # download button
            bts = BytesIO()
            bts.write(txt_val.encode("utf-8"))
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

# -------------------------
# Footer (small)
# -------------------------
st.markdown(
    """
    <div style="margin-top:18px; text-align:center;">
      <div style="display:inline-block; padding:10px 14px; border-radius:999px; background:linear-gradient(90deg, rgba(11,102,255,0.12), rgba(6,182,212,0.06)); font-weight:600; color:#0b66ff;">Praix Tech • Jahsmine</div>
      <div style="margin-top:10px; color:#64748b; font-size:13px;">Developed for educational & research purposes under the supervision of Mrs. Oguniyi. Disclaimer: demo tool, heuristic results.</div>
    </div>
    """,
    unsafe_allow_html=True,
)

# final small script to trigger simple fade
components.html("<script>document.querySelectorAll('.fade').forEach((el,i)=> setTimeout(()=> el.classList.add('aos-animate'), 120 + i*60));</script>", height=0)
