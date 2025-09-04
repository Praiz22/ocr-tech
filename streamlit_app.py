import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from scipy.ndimage import sobel
from io import BytesIO
import numpy as np
import re
import time
from urllib.parse import quote

# -------------------------------------------------------------------------------------------------
# Page configuration
# -------------------------------------------------------------------------------------------------
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="collapsed")

# -------------------------------------------------------------------------------------------------
# Glassmorphism & UI CSS
# -------------------------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
:root {
  --bg-1: #ffffff;
  --bg-2: #fff5eb;
  --bg-3: #ffe7cc;
  --card-bg: rgba(255,255,255,0.19);
  --card-border: rgba(255,255,255,0.18);
  --card-shadow: 0 18px 44px rgba(0,0,0,0.18);
  --text-1: #191919 !important;
  --text-2: #3a3a3a !important;
  --brand: #ff7a18;
  --brand-2: #ff4d00;
  --muted: #e9e9e9;
  --success: #0aa574;
  --warning: #d97a00;
  --radius-xl: 22px;
  --radius-lg: 18px;
  --radius-md: 14px;
  --radius-sm: 8px;
}
body { font-family: 'Poppins', sans-serif; color: var(--text-1);}
.stApp { background: linear-gradient(135deg, var(--bg-2) 0%, var(--bg-3) 100%); min-height: 100vh; padding: 2rem;}
.ocr-container { max-width: 900px; width: 100%; margin: 0 auto; display: flex; flex-direction: column; gap: 2rem;}
.ocr-card {
  background: var(--card-bg);
  backdrop-filter: blur(25px);
  border: 2px solid var(--card-border);
  border-radius: var(--radius-xl);
  padding: 2.5rem;
  box-shadow: var(--card-shadow);
  color: var(--text-1) !important;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  width: 100%;
}
.header { text-align: center; margin-bottom: 2rem;}
.header h1 { font-size: 2.5rem; font-weight: 700; color: var(--brand); margin: 0;}
.header p { color: var(--text-2); margin: 0.5rem 0 0; font-weight: 500;}
.file-upload-section {
  display: flex; flex-direction: column; align-items: center; justify-content: center;
  text-align: center; padding-bottom: 2rem; border-bottom: 1px dashed var(--muted); width: 100%;
}
.results-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;}
.metric-card {
  padding: 1.5rem;
  border-radius: var(--radius-lg);
  background: rgba(255,255,255,0.30);
  border: 1px solid rgba(255,255,255,0.16);
  box-shadow: 0 4px 12px rgba(0,0,0,0.12);
  color: #111 !important;
}
.metric-card h4 { margin: 0 0 0.5rem; font-size: 1rem; color: var(--text-2);}
.metric-value { font-size: 1.8rem; font-weight: 700; color: var(--brand-2);}
.progress-bar-container { height: 8px; background: var(--muted); border-radius: var(--radius-sm); overflow: hidden; margin-top: 0.5rem;}
.progress-bar { height: 100%; background: var(--brand); transition: width 0.3s ease-in-out;}
.progress-bar.success { background: var(--success);}
.progress-bar.warning { background: var(--warning);}
.text-output-card {
  grid-column: 1 / -1;
  background: rgba(255,255,255,0.32);
  padding: 2rem;
  border-radius: var(--radius-xl);
  border: 1px solid rgba(255,255,255,0.14);
  box-shadow: 0 4px 12px rgba(0,0,0,0.10);
  margin-top: 1.5rem;
  color: #191919 !important;
}
.text-output-card pre {
  white-space: pre-wrap;
  word-wrap: break-word;
  font-family: 'Poppins', sans-serif;
  color: #191919 !important;
  background: transparent !important;
  font-size: 1.08rem;
  line-height: 1.48;
}
.button-row { display: flex; justify-content: center; gap: 1rem; margin-top: 2rem;}
.ocr-button, .ocr-button:visited, .ocr-button:hover {
  background-color: var(--brand);
  color: var(--bg-1) !important;
  padding: 1rem 2.5rem;
  border-radius: var(--radius-lg);
  font-size: 1.1rem;
  font-weight: 600;
  border: none;
  cursor: pointer;
  transition: all 0.2s ease;
  text-decoration: none !important;
  display: inline-block;
}
.ocr-button:hover {
  background-color: var(--brand-2);
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0,0,0,0.2);
  text-decoration: none !important;
}
.st-emotion-cache-1c7y31u {
    border: 2px dashed var(--muted);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    background: rgba(255,255,255,0.18);
    transition: background 0.3s ease;
    color: #191919 !important;
}
.st-emotion-cache-1c7y31u:hover {
    background: rgba(255,255,255,0.27);
}
.st-emotion-cache-1c7y31u div p { color: var(--text-1); font-weight: 500;}
.image-preview-container { position: relative; overflow: hidden; border-radius: var(--radius-md);}
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
  animation: streakDown 1.6s cubic-bezier(.77,0,.18,1) infinite;
  z-index: 2;
  pointer-events: none;
}
@keyframes streakDown {
    0% { top: -80%; }
    100% { top: 120%; }
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------------------------------------------
# Helper functions for OCR, text cleaning, and classification
# -------------------------------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Remove non-ASCII, junk lines, and keep only readable text."""
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

def multi_psm_extract(image: Image.Image) -> str:
    """Try multiple Tesseract PSMs for best legit text extraction."""
    for psm in [6, 11, 3, 7]:
        text = pytesseract.image_to_string(image, config=f"--psm {psm}")
        filtered = clean_text(text)
        # Only return if legit text present
        if len(filtered) > 8 and any(c.isalpha() for c in filtered):
            return filtered
    return ""

def preprocess_image(image: Image.Image, processed_placeholder, status_placeholder) -> Image.Image:
    status_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Normalizing Image...</h4>', unsafe_allow_html=True)
    time.sleep(0.3)
    gray = ImageOps.grayscale(image)
    processed_placeholder.image(gray, caption="Processed Image", use_container_width=True)
    status_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Enhancing Contrast...</h4>', unsafe_allow_html=True)
    time.sleep(0.3)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.3)
    status_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Binarizing...</h4>', unsafe_allow_html=True)
    time.sleep(0.3)
    arr = np.array(enhanced)
    mean = arr.mean()
    binarized = (arr > mean - 12).astype(np.uint8) * 255
    bin_img = Image.fromarray(binarized)
    status_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Removing Noise...</h4>', unsafe_allow_html=True)
    time.sleep(0.3)
    denoised = bin_img.filter(ImageFilter.MedianFilter(size=3))
    return denoised

def is_blank_or_picture(text: str, img: Image.Image) -> bool:
    words = text.split()
    if len(words) < 4: return True
    real_words = [w for w in words if len(w) > 2 and re.match(r"[a-zA-Z]", w)]
    if len(real_words) < 2: return True
    arr = np.array(img.convert("L"))
    avg_brightness = arr.mean()
    if avg_brightness > 240 or avg_brightness < 15: return True
    return False

def is_handwritten(image: Image.Image) -> bool:
    arr = np.array(image.convert("L"))
    std = arr.std()
    return std > 55 and std < 95

def contains_form_lines(image: Image.Image) -> bool:
    arr = np.array(image)
    edges_h = sobel(arr, axis=0)
    edges_v = sobel(arr, axis=1)
    return (np.abs(edges_h) > 70).sum() > arr.shape[0] * 6 and (np.abs(edges_v) > 70).sum() > arr.shape[1] * 6

def classify_document(text, img, processed_img):
    if is_blank_or_picture(text, img): return "Picture", 99
    if is_handwritten(processed_img): return "Handwritten Note", 94
    if contains_form_lines(processed_img): return "Scanned Form", 92
    LOW_TEXT = text.lower()
    doc_types = {
        "Invoice": ["invoice", "bill to", "invoice number", "tax", "payment due", "amount due", "billed to", "vat"],
        "Receipt": ["receipt", "thank you for your purchase", "subtotal", "cashier", "transaction id", "change due"],
        "Report": ["report", "summary", "analysis", "findings", "conclusion", "abstract", "introduction", "methodology"],
        "Contract": ["contract", "agreement", "terms and conditions", "effective date", "parties", "whereas", "this agreement"],
        "Memo": ["memorandum", "memo", "interoffice", "to:", "from:", "date:", "subject:"],
        "Agenda": ["agenda", "meeting", "minutes", "discussion points", "location:", "time:"],
        "Medical": ["prescription", "rx", "refill", "patient", "diagnosis", "doctor", "hospital", "medication"],
        "Resume": ["resume", "curriculum vitae", "experience", "education", "skills", "objective", "work history", "email:"],
        "Legal": ["affidavit", "will", "deed", "court", "judgment", "plaintiff", "defendant", "statute", "legal"],
        "Financial": ["balance sheet", "income statement", "cash flow", "assets", "liabilities", "revenue", "expenses"],
        "Letter": ["dear", "sincerely", "regards", "yours truly", "addressed to"],
    }
    best_match, highest = "Miscellaneous", 0
    for doc, keys in doc_types.items():
        score = sum(1 for k in keys if k in LOW_TEXT)
        if score > highest:
            best_match, highest = doc, score
    word_count = len(text.split())
    confidence = min(99, 70 + highest * 7 + (5 if word_count > 100 else 0))
    return best_match, confidence

def word_frequency(text: str) -> dict:
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word in freq: freq[word] += 1
        else: freq[word] = 1
    return freq

def top_n_words(freq: dict, n: int = 6) -> list:
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

# -------------------------------------------------------------------------------------------------
# Session state
# -------------------------------------------------------------------------------------------------
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# -------------------------------------------------------------------------------------------------
# Main UI
# -------------------------------------------------------------------------------------------------
st.markdown("""
<div class="ocr-container">
  <div class="header">
    <h1>OCR-TECH</h1>
    <p>Optical Character Recognition & Image Classification</p>
  </div>
  <div class="ocr-card" style="margin-bottom:1.5rem;">
    <div class="file-upload-section">
      <h4 style="color:var(--text-1);">Upload an Image</h4>
      <p style="color:var(--text-2);">Drag and drop a file here or click below to choose a file.</p>
      <div style="margin-top:1.5rem;width:100%;">
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")
st.markdown("""</div></div></div>""", unsafe_allow_html=True)

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

if st.session_state.uploaded_image and not st.session_state.processing:
    st.session_state.processing = True
    image = Image.open(BytesIO(st.session_state.uploaded_image)).convert("RGB")
    status_text = st.empty()
    metric_grid_placeholder = st.empty()
    text_output_placeholder = st.empty()
    original_col, processed_col = st.columns(2)
    with original_col:
        st.markdown(f'<div class="image-preview-container {"processing" if st.session_state.processing else ""}">', unsafe_allow_html=True)
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with processed_col:
        processed_image_placeholder = st.empty()
        processed_image = preprocess_image(image, processed_image_placeholder, status_text)
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Extracting Text...</h4>', unsafe_allow_html=True)
    time.sleep(0.15)
    extracted_text = multi_psm_extract(processed_image)
    label, confidence = classify_document(extracted_text, image, processed_image)
    word_count = len(extracted_text.split())
    char_count = len(extracted_text.replace(" ", "").replace("\n", ""))
    avg_word_length = char_count / word_count if word_count > 0 else 0
    freq = word_frequency(extracted_text)
    top_words = top_n_words(freq, 6)
    metric_grid_placeholder.markdown(f"""
    <div class="results-grid">
      <div class="metric-card">
        <h4>Prediction</h4>
        <div class="metric-value" style="color:var(--brand);">{label}</div>
        <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Classified document type)</p>
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
        <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Words extracted)</p>
      </div>
      <div class="metric-card">
        <h4>Characters/Word</h4>
        <div class="metric-value">{avg_word_length:.2f}</div>
        <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Average length)</p>
      </div>
      <div class="metric-card">
        <h4>Top Words</h4>
        <div class="metric-value">{', '.join([f"{w}({c})" for w, c in top_words]) or '-'}</div>
        <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Most frequent)</p>
      </div>
    </div>
    """, unsafe_allow_html=True)
    quoted_text = quote(extracted_text)
    text_output_placeholder.markdown(f"""
    <div class="text-output-card">
      <h4 style="color:var(--text-1);">Extracted Text</h4>
      <pre id="ocrText" style="color:#191919 !important;">{extracted_text or "[No visible text]"}</pre>
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
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Done!</h4>', unsafe_allow_html=True)
    st.session_state.processing = False

st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
  <p style="color:var(--text-2); font-size:0.8rem;">OCR-TECH - Designed by ADELEKE, OLADOKUN, and OLALEYE</p>
</div>
""", unsafe_allow_html=True)

# (You can now add even more features, batch, ML, advanced NLP, pagination, etc.)
