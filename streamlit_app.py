import streamlit as st
import pytesseract
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
from io import BytesIO
import re
import time
import base64
import numpy as np
from urllib.parse import quote
from scipy.ndimage import sobel
from typing import Tuple, Dict, List, Any

# --- Streamlit Page Settings ---
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="collapsed")

# --- Custom CSS for Consistent UI Style ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  :root {
    --bg-1: #ffffff;
    --bg-2: #fff5eb;
    --bg-3: #ffe7cc;
    --card-bg: rgba(255, 255, 255, 0.5);
    --card-border: rgba(255, 255, 255, 0.6);
    --card-shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
    --text-1: #1f1f1f;
    --text-2: #5a5a5a;
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
  body {
    font-family: 'Poppins', sans-serif;
    color: var(--text-1);
  }
  .stApp {
    background: linear-gradient(135deg, var(--bg-2) 0%, var(--bg-3) 100%);
    min-height: 100vh;
    padding: 2rem;
  }
  .ocr-container {
    max-width: 900px;
    width: 100%;
    margin: 0 auto;
    display: flex;
    flex-direction: column;
    gap: 2rem;
  }
  .ocr-card {
    background: var(--card-bg);
    backdrop-filter: blur(16px);
    border: 1px solid var(--card-border);
    border-radius: var(--radius-xl);
    padding: 2.5rem;
    box-shadow: var(--card-shadow);
    color: var(--text-1);
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    width: 100%;
  }
  .header {
    text-align: center;
    margin-bottom: 2rem;
  }
  .header h1 {
    font-size: 2.5rem;
    font-weight: 700;
    color: var(--brand);
    margin: 0;
  }
  .header p {
    color: var(--text-2);
    margin: 0.5rem 0 0;
    font-weight: 500;
  }
  .file-upload-section {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding-bottom: 2rem;
    border-bottom: 1px dashed var(--muted);
    width: 100%;
  }
  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
    gap: 1.5rem;
  }
  .metric-card {
    padding: 1.5rem;
    border-radius: var(--radius-lg);
    background: rgba(255, 255, 255, 0.5);
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
  }
  .metric-card h4 {
    margin: 0 0 0.5rem;
    font-size: 1rem;
    color: var(--text-2);
  }
  .metric-value {
    font-size: 1.8rem;
    font-weight: 700;
    color: var(--brand-2);
  }
  .progress-bar-container {
    height: 8px;
    background: var(--muted);
    border-radius: var(--radius-sm);
    overflow: hidden;
    margin-top: 0.5rem;
  }
  .progress-bar {
    height: 100%;
    background: var(--brand);
    transition: width 0.3s ease-in-out;
  }
  .progress-bar.success {
    background: var(--success);
  }
  .progress-bar.warning {
    background: var(--warning);
  }
  .text-output-card {
    grid-column: 1 / -1;
    background: rgba(255, 255, 255, 0.5);
    padding: 2rem;
    border-radius: var(--radius-xl);
    border: 1px solid rgba(255, 255, 255, 0.4);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    margin-top: 1.5rem;
  }
  .text-output-card pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Poppins', sans-serif;
    color: #1f1f1f !important;
    background: transparent !important;
  }
  .button-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 2rem;
  }
  .ocr-button, .ocr-button:visited, .ocr-button:hover {
    background-color: var(--brand);
    color: var(--bg-1);
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
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
    text-decoration: none !important;
  }
  .st-emotion-cache-1c7y31u {
      border: 2px dashed var(--muted);
      border-radius: var(--radius-lg);
      padding: 2rem;
      text-align: center;
      cursor: pointer;
      background: rgba(255, 255, 255, 0.2);
      transition: background 0.3s ease;
  }
  .st-emotion-cache-1c7y31u:hover {
      background: rgba(255, 255, 255, 0.3);
  }
  .st-emotion-cache-1c7y31u div p {
      color: var(--text-1);
      font-weight: 500;
  }
  .image-preview-container {
    position: relative;
    overflow: hidden;
    border-radius: var(--radius-md);
  }
  .image-preview-container.processing::before {
    content: '';
    position: absolute;
    top: -40%;
    left: 0;
    width: 100%;
    height: 35%;
    background: linear-gradient(
      to bottom,
      rgba(255,255,255,0) 0%,
      rgba(255,255,255,0.7) 40%,
      var(--brand) 60%,
      rgba(255,255,255,0.7) 80%,
      rgba(255,255,255,0) 100%
    );
    animation: streakDown 1.2s linear infinite;
    z-index: 2;
  }
  @keyframes streakDown {
      0% { top: -40%; }
      100% { top: 110%; }
  }
</style>
""", unsafe_allow_html=True)

# --- Helper Functions for OCR & Classification ---

def clean_text(text: str) -> str:
    """
    Cleans OCR text output by removing non-ASCII, redundant whitespace, and excessive newlines.
    """
    cleaned = re.sub(r'[^\x00-\x7F]+', ' ', text)
    cleaned = re.sub(r'[\r]+', '', cleaned)
    cleaned = re.sub(r'[ ]{2,}', ' ', cleaned)
    cleaned = re.sub(r'[\n]{3,}', '\n\n', cleaned)
    return cleaned.strip()

def preprocess_image(image: Image.Image, processed_image_placeholder, status_text_placeholder) -> Image.Image:
    """
    Runs image normalization, contrast enhancement, binarization, and denoising. Animates UI with status messages.
    """
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Normalizing Image...</h4>', unsafe_allow_html=True)
    time.sleep(0.25)
    gray = ImageOps.grayscale(image)
    processed_image_placeholder.image(gray, caption="Processed Image", use_container_width=True)
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Enhancing Contrast...</h4>', unsafe_allow_html=True)
    time.sleep(0.25)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Binarizing...</h4>', unsafe_allow_html=True)
    time.sleep(0.25)
    arr = np.array(enhanced)
    mean = arr.mean()
    binarized = (arr > mean - 15).astype(np.uint8) * 255
    bin_img = Image.fromarray(binarized)
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Denoising...</h4>', unsafe_allow_html=True)
    time.sleep(0.25)
    denoised = bin_img.filter(ImageFilter.MedianFilter(size=3))
    return denoised

def count_real_words(text: str) -> int:
    """
    Counts the number of real words (length > 2, alphabetic) in text.
    """
    words = text.split()
    return len([w for w in words if len(w) > 2 and re.match(r"[a-zA-Z]", w)])

def is_blank_or_picture(text: str, img: Image.Image) -> bool:
    """
    Determines if image is a photo/picture or has no meaningful text.
    """
    words = text.split()
    if len(words) < 5:
        return True
    if count_real_words(text) < 2:
        return True
    arr = np.array(img.convert("L"))
    avg_brightness = arr.mean()
    if avg_brightness > 240 or avg_brightness < 15:
        return True
    return False

def is_handwritten(image: Image.Image) -> bool:
    """
    Simple heuristic: high pixel stddev = handwriting.
    """
    arr = np.array(image.convert("L"))
    std = arr.std()
    return std > 55 and std < 95

def contains_form_lines(image: Image.Image) -> bool:
    """
    Looks for dominant horizontal/vertical lines (likely a form).
    """
    arr = np.array(image)
    edges_h = sobel(arr, axis=0)
    edges_v = sobel(arr, axis=1)
    return (np.abs(edges_h) > 70).sum() > arr.shape[0] * 6 and (np.abs(edges_v) > 70).sum() > arr.shape[1] * 6

def classify_document(text: str, img: Image.Image, processed_img: Image.Image) -> Tuple[str, int]:
    """
    Multi-stage classifier: picture/photo, handwritten, form, then text patterns.
    """
    if is_blank_or_picture(text, img):
        return "Picture", 99
    if is_handwritten(processed_img):
        return "Handwritten Note", 92
    if contains_form_lines(processed_img):
        return "Scanned Form", 93
    LOW_TEXT = text.lower()
    doc_types: Dict[str, List[str]] = {
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
    best_match = "Miscellaneous"
    highest = 0
    for doc, keys in doc_types.items():
        score = sum(1 for k in keys if k in LOW_TEXT)
        if score > highest:
            best_match, highest = doc, score
    word_count = len(text.split())
    confidence = min(99, 70 + highest * 7 + (5 if word_count > 100 else 0))
    return best_match, confidence

def extract_text(image: Image.Image) -> str:
    """
    Applies multiple OCR strategies for best extraction.
    """
    text = pytesseract.image_to_string(image, config="--psm 6")
    if len(clean_text(text)) < 5:
        text = pytesseract.image_to_string(image, config="--psm 11")
    if len(clean_text(text)) < 5:
        text = pytesseract.image_to_string(image, config="--psm 7")
    return clean_text(text)

def get_image_metadata(image: Image.Image) -> Dict[str, Any]:
    """
    Returns metadata such as dimensions, mode, average brightness, stddev.
    """
    arr = np.array(image.convert("L"))
    return {
        "width": image.width,
        "height": image.height,
        "mode": image.mode,
        "avg_brightness": arr.mean(),
        "stddev_brightness": arr.std(),
    }

def word_frequency(text: str) -> Dict[str, int]:
    """
    Returns word frequency dictionary for the text.
    """
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def top_n_words(freq: Dict[str, int], n: int = 5) -> List[Tuple[str, int]]:
    """
    Returns the top N most common words.
    """
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

# --- Streamlit UI State Management ---
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# --- Main App Layout ---
st.markdown("""
<div class="ocr-container">
    <div class="header">
        <h1>OCR-TECH</h1>
        <p>Optical Character Recognition & Image Classification</p>
    </div>
    <div class="ocr-card">
        <div class="file-upload-section">
            <h4 style="color:var(--text-1);">Upload an Image</h4>
            <p style="color:var(--text-2);">Drag and drop a file here or click below to choose a file.</p>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")

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
    time.sleep(0.4)
    extracted_text = extract_text(processed_image)
    label, confidence = classify_document(extracted_text, image, processed_image)
    word_count = len(extracted_text.split())
    char_count = len(extracted_text.replace(" ", "").replace("\n", ""))
    avg_word_length = char_count / word_count if word_count > 0 else 0

    meta = get_image_metadata(image)
    freq = word_frequency(extracted_text)
    top_words = top_n_words(freq, 7)

    # --- Results Grid with More Metrics ---
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
            <h4>Image Size</h4>
            <div class="metric-value">{meta['width']}×{meta['height']}</div>
            <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">({meta['mode']})</p>
        </div>
        <div class="metric-card">
            <h4>Brightness/StdDev</h4>
            <div class="metric-value">{meta['avg_brightness']:.1f}/{meta['stddev_brightness']:.1f}</div>
            <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Luminance stats)</p>
        </div>
        <div class="metric-card">
            <h4>Top Words</h4>
            <div class="metric-value">{', '.join([f"{w}({c})" for w, c in top_words]) or '-'}</div>
            <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Most frequent)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # --- Output text & download/copy ---
    quoted_text = quote(extracted_text)
    text_output_placeholder.markdown(f"""
    <div class="text-output-card">
        <h4 style="color:var(--text-1);">Extracted Text</h4>
        <pre id="ocrText" style="color:var(--text-1);">{extracted_text or "[No visible text]"}</pre>
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

# --- Footer ---
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="color:var(--text-2); font-size:0.8rem;">OCR-TECH - Designed by ADELEKE, OLADOKUN, and OLALEYE</p>
</div>
""", unsafe_allow_html=True)

# --- Reserved for future: advanced NLP, ML, and batch processing features ---
# ... You can extend here with custom models, batch uploads, advanced charting, etc. ...
