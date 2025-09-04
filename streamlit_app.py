################################################################################
# OCR-TECH: Advanced OCR with Glassmorphism UI and Bounding Box Overlay
#
# - EasyOCR extraction with confidence filtering & bounding box overlays
# - Tesseract fallback
# - Compact, glassy, portable UI with animation during processing
# - Toggle to display bounding boxes on original image
# - All previous features: metrics, classification, download/copy, extensibility
#
# AUTHOR: Praiz22 & Copilot, 2025
################################################################################

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageDraw, ImageFont, ImageFilter
import easyocr
import pytesseract
import re
import time
from io import BytesIO
from urllib.parse import quote

################################################################################
# Streamlit Setup and Constants
################################################################################

# Set up Streamlit page
st.set_page_config(
    page_title="OCR-TECH",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# -- UI CSS: glassmorphism, compact containers, animation, overlays --
st.markdown("""
<style>
:root {
  --bg-1: #fff;
  --card-bg: rgba(255,255,255,0.18);
  --card-border: rgba(255,255,255,0.15);
  --brand: #ff7a18;
  --brand-2: #ff4d00;
  --radius-xl: 16px;
  --radius-lg: 13px;
  --radius-md: 10px;
  --radius-sm: 6px;
}
.stApp {background: linear-gradient(135deg,#fff5eb 0%,#ffe7cc 100%);}
.ocr-container {max-width: 870px; margin:0 auto;}
.ocr-card {
  background: var(--card-bg);
  border: 2px solid var(--card-border);
  border-radius: var(--radius-xl);
  box-shadow: 0 9px 24px rgba(0,0,0,0.09);
  padding: 2.2rem 1.4rem;
  margin-bottom: 1.6rem;
}
.header {text-align: center; margin-bottom: 1.5rem;}
.header h1 {font-size: 2.1rem; color:var(--brand);}
.header p {color: #444;}
.file-upload-section {text-align:center; padding-bottom:1.3rem;}
.image-row {display: flex; flex-direction: row; gap: 1.35rem; justify-content: center; align-items: flex-start;}
.image-container {
  width:100%; max-width:260px; max-height:290px; overflow:hidden;
  border-radius: 10px; background:rgba(255,255,255,0.13);
  box-shadow:0 2px 10px rgba(0,0,0,0.04); display:flex;align-items:center;justify-content:center;position:relative;
}
.image-preview-container.processing::before {
  content: '';
  position: absolute;
  left: 0; width: 100%; height: 100%;
  top: 0;
  background: repeating-linear-gradient(
    120deg,
    rgba(255,255,255,0) 0%,
    rgba(255,255,255,0.35) 30%,
    rgba(255,255,255,0.8) 47%,
    rgba(255,255,255,0.35) 70%,
    rgba(255,255,255,0) 100%
  );
  background-size: 120% 120%;
  animation: streakDown 1.1s cubic-bezier(.77,0,.18,1) infinite;
  z-index: 2;
  pointer-events: none;
}
@keyframes streakDown {
    0% { top: -80%; }
    100% { top: 120%; }
}
.results-grid {display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 1rem;}
.metric-card {
  padding: 0.9rem;
  border-radius: var(--radius-lg);
  background: rgba(255,255,255,0.26);
  border: 1px solid rgba(255,255,255,0.12);
  color: #111 !important;
}
.metric-value { font-size: 1.1rem; font-weight: 700; color: var(--brand-2); }
.progress-bar-container {height: 7px; background: #e0e0e0; border-radius: 5px; overflow: hidden; margin-top: 0.4rem;}
.progress-bar {height: 100%; background: var(--brand); transition: width 0.2s;}
.progress-bar.success {background: #0aa574;}
.text-output-card {
  background: rgba(255,255,255,0.29);
  padding: 1rem;
  border-radius: var(--radius-xl);
  border: 1px solid rgba(255,255,255,0.07);
  margin-top: 1rem;
  color: #191919 !important;
}
.text-output-card pre {
  white-space: pre-wrap; word-wrap: break-word; font-family: 'Poppins', sans-serif;
  color: #191919 !important; background: transparent !important; font-size: 1.03rem; line-height: 1.35;
}
.button-row {display: flex; justify-content: center; gap: 0.9rem; margin-top: 1.2rem;}
.ocr-button, .ocr-button:visited, .ocr-button:hover {
  background-color: var(--brand); color: #fff !important; padding: 0.8rem 1.7rem;
  border-radius: var(--radius-lg); font-size: 1rem; font-weight: 600; border: none;
  cursor: pointer; transition: all 0.2s; text-decoration: none !important; display: inline-block;
}
.ocr-button:hover {background-color: var(--brand-2);}
.st-emotion-cache-1c7y31u {border: 2px dashed #ececec; border-radius: var(--radius-lg); padding: 1.2rem;}
</style>
""", unsafe_allow_html=True)

################################################################################
# Utility: EasyOCR Reader as Resource
################################################################################

@st.cache_resource(show_spinner="Loading EasyOCR model...")
def get_easyocr_reader():
    """
    Loads EasyOCR English reader as a singleton resource.
    """
    return easyocr.Reader(['en'])

################################################################################
# Extraction Logic: EasyOCR with Confidence Filtering & Bounding Boxes
################################################################################

def recognize_text_easyocr(image, min_conf=0.2):
    """
    Uses EasyOCR to extract text with bounding boxes and confidence filtering.
    Returns:
        result (list): List of (bbox, text, prob)
    """
    reader = get_easyocr_reader()
    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
    result = reader.readtext(img_np)
    return [
        (bbox, text.strip(), prob)
        for bbox, text, prob in result
        if prob >= min_conf and text.strip()
    ]

def get_cleaned_text_from_easyocr_results(results):
    """
    Concatenates filtered OCR results into a single cleaned string.
    """
    return "\n".join([text for bbox, text, prob in results])

def overlay_ocr_boxes_on_image(image, ocr_results, font_size=18):
    """
    Overlays bounding boxes and recognized text onto the image.
    Args:
        image (PIL.Image): The original image.
        ocr_results (list): List of (bbox, text, prob)
        font_size (int): Font size for overlayed text.
    Returns:
        PIL.Image: Image with bounding boxes and text overlay.
    """
    # Convert for drawing
    img_draw = image.convert("RGBA")
    draw = ImageDraw.Draw(img_draw)
    try:
        font = ImageFont.truetype("arial.ttf", font_size)
    except:
        font = ImageFont.load_default()

    for bbox, text, prob in ocr_results:
        pts = [tuple(map(int, pt)) for pt in bbox]
        draw.line(pts + [pts[0]], fill=(255, 32, 32, 170), width=3)
        # Draw a filled background for text
        text_w, text_h = draw.textsize(text, font=font)
        tx, ty = pts[0]
        draw.rectangle([(tx, ty - text_h - 4), (tx + text_w + 8, ty)], fill=(255,255,255,196))
        draw.text((tx + 4, ty - text_h - 2), text, fill=(32,32,32,255), font=font)
    return img_draw

################################################################################
# Extraction Logic: Tesseract Fallback
################################################################################

def extract_text_tesseract(image):
    """
    Attempts OCR extraction with multiple Tesseract PSM settings.
    """
    for psm in [6, 11, 3, 7]:
        try:
            text = pytesseract.image_to_string(image, config=f"--psm {psm}")
        except Exception:
            continue
        filtered = clean_text(text)
        if len(filtered) > 8 and any(c.isalpha() for c in filtered):
            return filtered
    return ""

################################################################################
# Text Cleaning, Metrics, Classification, and Frequency
################################################################################

def clean_text(text: str) -> str:
    """
    Removes non-ASCII, junk lines, and keeps only readable text.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\r]+', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'[\n]{3,}', '\n\n', text)
    # Remove lines that are mostly non-alphabetic (noisy OCR)
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if len(line.strip()) > 1 and (
            sum(c.isalpha() or c.isdigit() for c in line) / max(1, len(line.strip()))
        ) > 0.4:
            filtered.append(line)
    return "\n".join(filtered).strip()

def classify_document(text, img, processed_img):
    """
    Heuristic document classification.
    """
    words = text.split()
    real_words = [w for w in words if len(w) > 2 and re.match(r"[a-zA-Z]", w)]
    arr = np.array(img.convert("L"))
    avg_brightness = arr.mean()
    if len(words) < 4 or len(real_words) < 2 or avg_brightness > 240 or avg_brightness < 15:
        return "Picture", 99
    arr_proc = np.array(processed_img.convert("L"))
    std = arr_proc.std()
    if std > 55 and std < 95:
        return "Handwritten Note", 94
    return "Document", 92

def word_frequency(text):
    """
    Returns word frequency dictionary.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word in freq: freq[word] += 1
        else: freq[word] = 1
    return freq

def top_n_words(freq, n=5):
    """
    Returns top n most frequent words.
    """
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

################################################################################
# Preprocessing (Contrast, Binarize, Denoise)
################################################################################

def preprocess_image(img, processed_placeholder, status_placeholder):
    """
    Preprocesses image for OCR: grayscale, contrast enhance, binarize, denoise.
    """
    status_placeholder.markdown('**Normalizing...**')
    gray = ImageOps.grayscale(img)
    processed_placeholder.image(gray, caption="Processed", use_container_width=True)
    status_placeholder.markdown('**Enhancing Contrast...**')
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    status_placeholder.markdown('**Binarizing...**')
    arr = np.array(enhanced)
    mean = arr.mean()
    binarized = (arr > mean - 10).astype(np.uint8) * 255
    bin_img = Image.fromarray(binarized)
    status_placeholder.markdown('**Denoising...**')
    denoised = bin_img.filter(ImageFilter.MedianFilter(size=3))
    return denoised

################################################################################
# Streamlit Session State
################################################################################

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None
if "show_boxes" not in st.session_state:
    st.session_state.show_boxes = True

################################################################################
# Main UI Layout and Logic
################################################################################

st.markdown("""
<div class="ocr-container">
  <div class="header">
    <h1>OCR-TECH</h1>
    <p>Advanced Text Extraction & Bounding Box Visualization</p>
  </div>
  <div class="ocr-card">
    <div class="file-upload-section">
      <h4>Upload an Image</h4>
      <p>Drag and drop or click below to choose a file.</p>
      <div style="margin-top:1.2rem;width:100%;">
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")
st.markdown("""</div></div></div>""", unsafe_allow_html=True)

# Toggle for bounding box overlays
st.session_state.show_boxes = st.checkbox(
    "Show bounding boxes on original image (EasyOCR results)", value=True
)

# -- File upload: load and reset state if new image --
if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.uploaded_image = None
        st.session_state.processing = False
        try:
            image_data = uploaded_file.getvalue()
            _ = Image.open(BytesIO(image_data))
            st.session_state.uploaded_image = image_data
            st.rerun()
        except Exception:
            st.error("The file you uploaded could not be identified as a valid image. Please try a different file.")
            st.session_state.uploaded_image = None

################################################################################
# Main OCR Processing Block
################################################################################

if st.session_state.uploaded_image and not st.session_state.processing:
    st.session_state.processing = True
    image = Image.open(BytesIO(st.session_state.uploaded_image)).convert("RGB")
    status_text = st.empty()
    metric_grid_placeholder = st.empty()
    text_output_placeholder = st.empty()

    # ---------- IMAGE PREVIEWS IN FIXED CONTAINERS (with animation on left) ----------
    st.markdown('<div class="image-row">', unsafe_allow_html=True)
    with st.container():
        st.markdown(
            '<div class="image-container">'
            f'<div class="image-preview-container {"processing" if st.session_state.processing else ""}">',
            unsafe_allow_html=True)
        # If show_boxes, overlay the bounding boxes (after OCR)
        preview_img = image
        ocr_results = []
        try:
            ocr_results = recognize_text_easyocr(image)
            if st.session_state.show_boxes and len(ocr_results):
                preview_img = overlay_ocr_boxes_on_image(image, ocr_results)
        except Exception:
            ocr_results = []
            preview_img = image
        st.image(preview_img, caption="Original (with boxes)" if st.session_state.show_boxes else "Original", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        processed_image_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # ---------- PREPROCESS AND OCR ---------------------
    processed_image = preprocess_image(image, processed_image_placeholder, status_text)
    status_text.markdown('**Extracting Text...**')
    time.sleep(0.18)

    # Primary extraction: EasyOCR with confidence filtering (from your repo)
    try:
        if not ocr_results:
            ocr_results = recognize_text_easyocr(processed_image)
        extracted_text = get_cleaned_text_from_easyocr_results(ocr_results)
        filtered = clean_text(extracted_text)
        if not (len(filtered) > 8 and any(c.isalpha() for c in filtered)):
            # Fallback to Tesseract
            filtered = extract_text_tesseract(processed_image)
    except Exception as ex:
        filtered = extract_text_tesseract(processed_image)

    # ---------- CLASSIFICATION & METRICS --------------------------
    label, confidence = classify_document(filtered, image, processed_image)
    word_count = len(filtered.split())
    char_count = len(filtered.replace(" ", "").replace("\n", ""))
    avg_word_length = char_count / word_count if word_count > 0 else 0
    freq = word_frequency(filtered)
    top_words = top_n_words(freq, 5)

    # ---------- METRICS GRID --------------------------------------
    metric_grid_placeholder.markdown(f"""
    <div class="results-grid">
      <div class="metric-card">
        <h4>Prediction</h4>
        <div class="metric-value" style="color:var(--brand);">{label}</div>
        <p style="color:#444; font-size:0.83rem;">(Classified)</p>
      </div>
      <div class="metric-card">
        <h4>Confidence</h4>
        <div class="metric-value">{confidence}%</div>
        <div class="progress-bar-container">
          <div class="progress-bar success" style="width:{confidence}%;"></div>
        </div>
      </div>
      <div class="metric-card">
        <h4>Text Count</h4>
        <div class="metric-value">{word_count}</div>
        <p style="color:#444; font-size:0.83rem;">(Words)</p>
      </div>
      <div class="metric-card">
        <h4>Char/Word</h4>
        <div class="metric-value">{avg_word_length:.2f}</div>
        <p style="color:#444; font-size:0.83rem;">(Avg len)</p>
      </div>
      <div class="metric-card">
        <h4>Top Words</h4>
        <div class="metric-value">{', '.join([f"{w}({c})" for w, c in top_words]) or '-'}</div>
        <p style="color:#444; font-size:0.83rem;">(Freq.)</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ---------- TEXT + COPY/DOWNLOAD --------------------------
    quoted_text = quote(filtered)
    text_output_placeholder.markdown(f"""
    <div class="text-output-card">
      <h4>Extracted Text</h4>
      <pre id="ocrText">{filtered or "[No visible text]"}</pre>
      <div class="button-row">
        <button class="ocr-button" onclick="copyToClipboard()">Copy Text</button>
        <a href="data:text/plain;charset=utf-8,{quoted_text}" download="extracted_text.txt" class="ocr-button">Download .txt</a>
      </div>
    </div>
    <script>
      function copyToClipboard() {{
        const textToCopy = document.getElementById('ocrText').innerText;
        navigator.clipboard.writeText(textToCopy).then(() => {{
          var toastMessage = "Text copied!";
          var streamlitContainer = window.parent.document.querySelector('.st-toast');
          if (streamlitContainer) {{
            var toastDiv = document.createElement('div');
            toastDiv.className = 'st-toast';
            toastDiv.innerHTML = '<div style="background-color: #f0f2f6; padding: 10px; border-radius: 5px;">' + toastMessage + '</div>';
            streamlitContainer.appendChild(toastDiv);
            setTimeout(() => {{ toastDiv.remove(); }}, 3000);
          }} else {{
            console.log(toastMessage);
          }}
        }}).catch(err => {{
          console.error('Could not copy text: ', err);
        }});
      }}
    </script>
    """, unsafe_allow_html=True)
    status_text.markdown('**Done!**')
    st.session_state.processing = False

st.markdown("""
<div style="text-align: center; margin-top: 1.5rem;">
  <p style="color:#444; font-size:0.8rem;">OCR-TECH - ADELEKE, OLADOKUN, OLALEYE (with Copilot)</p>
</div>
""", unsafe_allow_html=True)
