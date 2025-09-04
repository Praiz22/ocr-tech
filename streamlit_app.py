# streamlit_app.py

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageOps, ImageFilter, ImageEnhance
import easyocr
import pytesseract
import re
import time
import base64
from urllib.parse import quote
from io import BytesIO

# ----------------------------------------
# PAGE & GLOBALS
# ----------------------------------------
# Set Streamlit to use a wide layout and a custom title.
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="expanded")

# ----------------------------------------
# CSS for glassmorphism and compact containers
# ----------------------------------------
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  
  :root {
    --bg-1: #ffffff;
    --bg-2: #fff5eb;
    --bg-3: #ffe7cc;
    --card-bg: rgba(255, 255, 255, 0.5); /* More transparent for glass effect */
    --card-border: rgba(255, 255, 255, 0.6);
    --card-shadow: 0 18px 44px rgba(0, 0, 0, 0.28);
    --text-1: #1f1f1f; /* Dark text for readability on light background */
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
    text-align: center;
    padding-bottom: 2rem;
    border-bottom: 1px dashed var(--muted);
  }
  
  .image-row {
    display: flex;
    flex-direction: row;
    gap: 1rem; /* Reduced gap */
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap; /* Allow wrapping on small screens */
  }
  
  .image-container {
    width: 100%;
    flex: 1 1 250px; /* New fluid sizing for three images side-by-side */
    max-width: 300px; /* Optional, but helps maintain aspect ratio on very large screens */
    height: 100%; /* Height is fluid with width */
    max-height: 250px;
    overflow: hidden;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.13);
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }

  /* Make st.image inside the container fluid */
  .image-container img {
      width: 100%;
      height: 100%;
      object-fit: contain; /* Scale image to fit container */
  }
  
  .image-preview-container.processing::before {
    content: '';
    position: absolute;
    left: 0;
    width: 100%;
    height: 100%;
    top: 0;
    background: repeating-linear-gradient(
      120deg,
      rgba(255, 255, 255, 0) 0%,
      rgba(255, 255, 255, 0.35) 30%,
      rgba(255, 255, 255, 0.8) 47%,
      rgba(255, 255, 255, 0.35) 70%,
      rgba(255, 255, 255, 0) 100%
    );
    background-size: 120% 120%;
    animation: streakDown 1.1s cubic-bezier(.77, 0, .18, 1) infinite;
    z-index: 2;
    pointer-events: none;
  }
  
  @keyframes streakDown {
    0% {
      top: -80%;
    }
    100% {
      top: 120%;
    }
  }

  .results-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
    gap: 1rem;
  }
  
  .metric-card {
    padding: 0.9rem;
    border-radius: var(--radius-lg);
    background: rgba(255, 255, 255, 0.26);
    border: 1px solid rgba(255, 255, 255, 0.12);
    color: #111 !important;
  }
  
  .metric-value {
    font-size: 1.1rem;
    font-weight: 700;
    color: var(--brand-2);
  }
  
  .progress-bar-container {
    height: 7px;
    background: #e0e0e0;
    border-radius: 5px;
    overflow: hidden;
    margin-top: 0.4rem;
  }
  
  .progress-bar {
    height: 100%;
    background: var(--brand);
    transition: width 0.2s;
  }
  
  .progress-bar.success {
    background: var(--success);
  }
  
  .text-output-card {
    background: rgba(255, 255, 255, 0.29);
    padding: 1rem;
    border-radius: var(--radius-xl);
    border: 1px solid rgba(255, 255, 255, 0.07);
    margin-top: 1rem;
    color: #191919 !important;
  }
  
  .text-output-card pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Poppins', sans-serif;
    color: #191919 !important;
    background: transparent !important;
    font-size: 1.03rem;
    line-height: 1.35;
  }
  
  .button-row {
    display: flex;
    justify-content: center;
    gap: 0.9rem;
    margin-top: 1.2rem;
  }
  
  .ocr-button,
  .ocr-button:visited,
  .ocr-button:hover {
    background-color: var(--brand);
    color: #fff !important;
    padding: 0.8rem 1.7rem;
    border-radius: var(--radius-lg);
    font-size: 1rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: all 0.2s;
    text-decoration: none !important;
    display: inline-block;
  }
  
  .ocr-button:hover {
    background-color: var(--brand-2);
  }
  
  /* Style for the Streamlit file uploader to match custom design */
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

  /* Style for the file uploader's text to ensure visibility */
  .st-emotion-cache-1c7y31u div p {
      color: var(--text-1) !important;
      font-weight: 500;
  }

  /* Style for the uploaded file name text */
  .st-emotion-cache-1869e5d div p {
      color: black !important;
  }

  .github-icon-svg {
    width: 20px;
    height: 20px;
    fill: #444;
    transition: transform 0.3s ease-in-out;
  }
  
  .github-link:hover .github-icon-svg {
    transform: rotate(360deg);
  }
</style>
""", unsafe_allow_html=True)

# Set the path to the Tesseract executable.
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# ----------------------------------------
# OCR Extraction Logic (using EasyOCR & Tesseract)
# ----------------------------------------
@st.cache_resource(show_spinner=False)
def get_easyocr_reader(lang_list):
    """Caches the EasyOCR reader with a specific language list."""
    # This prevents reloading the model every time the page refreshes
    return easyocr.Reader(lang_list)

def recognize_text_easyocr(image, langs, min_conf=0.2):
    """Uses EasyOCR to recognize text with a minimum confidence threshold."""
    reader = get_easyocr_reader(langs)
    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
    result = reader.readtext(img_np)
    lines = [text.strip() for bbox, text, prob in result if prob >= min_conf and text.strip()]
    return "\n".join(lines), result

def clean_text(text: str) -> str:
    """
    Cleans extracted text by removing non-ASCII characters,
    excessive newlines, and non-textual lines.
    """
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    text = re.sub(r'[\r]+', '', text)
    text = re.sub(r'[ ]{2,}', ' ', text)
    text = re.sub(r'[\n]{3,}', '\n\n', text)
    lines = text.splitlines()
    filtered = []
    for line in lines:
        if len(line.strip()) > 1 and (
            sum(c.isalpha() or c.isdigit() for c in line) / max(1, len(line.strip()))
        ) > 0.4:
            filtered.append(line)
    return "\n".join(filtered).strip()

def draw_text_on_image(image, results):
    """
    Draws bounding boxes and text from EasyOCR results onto the image.
    """
    img_np = np.array(image.convert("RGB"))
    for (bbox, text, prob) in results:
        if prob >= 0.2:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            cv2.rectangle(img=img_np, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=2)
            cv2.putText(img=img_np, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
    return Image.fromarray(img_np)

def extract_text(image, langs):
    """
    Extracts text, first attempting with EasyOCR, then falling back to
    Tesseract if EasyOCR fails or produces poor results.
    Returns the extracted text and the raw EasyOCR results.
    """
    easyocr_results = None
    try:
        text, easyocr_results = recognize_text_easyocr(image, langs)
        filtered = clean_text(text)
        if len(filtered) > 8 and any(c.isalpha() for c in filtered):
            return filtered, easyocr_results
    except Exception as e:
        st.warning(f"EasyOCR failed: {e}. Falling back to Tesseract.")

    # Fallback to Tesseract with different PSM modes
    for psm in [6, 11, 3, 7]:
        text = pytesseract.image_to_string(image, config=f"--psm {psm}")
        filtered = clean_text(text)
        if len(filtered) > 8 and any(c.isalpha() for c in filtered):
            return filtered, [] # Return empty list for no EasyOCR results
    return "", []

# ----------------------------------------
# Image Preprocessing Pipeline (Enhanced)
# ----------------------------------------
def preprocess_image(img, processed_placeholder, status_placeholder):
    """
    Applies a series of image processing techniques to enhance OCR accuracy.
    Includes dynamic updates for a more interactive feel.
    """
    status_placeholder.markdown('**Normalizing Image...**')
    time.sleep(0.1)
    gray = ImageOps.grayscale(img)
    processed_placeholder.image(gray, caption="Grayscale", use_container_width=True)
    
    status_placeholder.markdown('**Enhancing Contrast...**')
    time.sleep(0.1)
    enhancer = ImageEnhance.Contrast(gray)
    enhanced = enhancer.enhance(2.0)
    processed_placeholder.image(enhanced, caption="Contrast Enhanced", use_container_width=True)
    
    status_placeholder.markdown('**Binarizing (Otsu)...**')
    time.sleep(0.1)
    arr = np.array(enhanced)
    _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin_img = Image.fromarray(binarized)
    processed_placeholder.image(bin_img, caption="Binarized (Otsu)", use_container_width=True)
    
    status_placeholder.markdown('**Denoising (Morphological)...**')
    time.sleep(0.1)
    kernel = np.ones((1, 1), np.uint8)
    denoised_np = cv2.morphologyEx(binarized, cv2.MORPH_OPEN, kernel)
    denoised = Image.fromarray(denoised_np)
    processed_placeholder.image(denoised, caption="Denoised", use_container_width=True)
    
    return denoised

# ----------------------------------------
# Utilities (metrics, classification)
# ----------------------------------------
def classify_document(text, img, processed_img, ocr_results):
    """
    Classifies the image based on extracted text and image properties.
    """
    words = text.split()
    real_words = [w for w in words if len(w) > 2 and re.match(r"[a-zA-Z]", w)]
    
    word_count = len(words)
    line_count = len(text.splitlines())
    char_count = len(text.replace(" ", "").replace("\n", ""))
    
    img_width, img_height = img.size
    aspect_ratio = max(img_width, img_height) / min(img_width, img_height) if min(img_width, img_height) > 0 else 1
    
    arr_proc = np.array(processed_img.convert("L"))
    std = arr_proc.std()
    
    document_score = 0
    handwritten_score = 0
    
    if word_count > 50 and line_count > 10:
        document_score += 40
    if char_count / (img_width * img_height) > 0.005:
        document_score += 20
    if 1.4 < aspect_ratio < 1.6:
        document_score += 20
    if std > 55 and std < 95:
        handwritten_score += 40
    
    keywords = {'invoice', 'receipt', 'report', 'statement', 'bill', 'form'}
    for word in real_words:
        if word.lower() in keywords:
            document_score += 15
            break
            
    if word_count < 10 or len(real_words) < 5:
        return "Picture", 99
        
    if document_score > handwritten_score:
        return "Document", min(100, 70 + document_score // 2)
    elif handwritten_score > document_score:
        return "Handwritten Note", min(100, 70 + handwritten_score // 2)
    else:
        return "Document", min(100, 70 + document_score // 2)

def word_frequency(text):
    """Calculates the frequency of each word in the text."""
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq

def top_n_words(freq, n=5):
    """Returns the top N most frequent words."""
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

# ----------------------------------------
# Streamlit Session State Management
# ----------------------------------------
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# ----------------------------------------
# Sidebar for Settings
# ----------------------------------------
st.sidebar.header("Settings")
st.sidebar.markdown("Configure your OCR extraction preferences.")
# Added 'yo' for Yoruba language support
available_langs = ['en', 'ar', 'ru', 'ch_sim', 'ja', 'yo']
selected_langs = st.sidebar.multiselect(
    'Select language(s) for OCR',
    options=available_langs,
    default=['en'],
    help="Select the languages present in the image. Multiple languages are supported."
)

if not selected_langs:
    st.sidebar.warning("Please select at least one language.")
    st.stop()

# ----------------------------------------
# Main UI
# ----------------------------------------
st.markdown("""
<div class="ocr-container">
    <div class="header">
        <h1>OCR-TECH</h1>
        <p>Better Text Extraction from Images (Powered by EasyOCR)</p>
    </div>
    <div class="ocr-card">
        <div class="file-upload-section">
            <h4>Upload an Image</h4>
            <p>Drag and drop or click below to choose a file.</p>
            <div style="margin-top:1.2rem;width:100%;">
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
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(
            '<div class="image-container">'
            f'<div class="image-preview-container {"processing" if st.session_state.processing else ""}">',
            unsafe_allow_html=True)
        st.image(image, caption="Original", use_container_width=True)
        st.markdown('</div></div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        processed_image_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        overlayed_image_placeholder = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    status_text = st.empty()
    metric_grid_placeholder = st.empty()
    text_output_placeholder = st.empty()
    
    start_time = time.time()
    
    # Dynamic Status Updates
    status_text.markdown('**✨ Starting image analysis...**')
    time.sleep(0.3)
    
    status_text.markdown('**🧠 Preprocessing image for better accuracy...**')
    preprocess_start = time.time()
    processed_image = preprocess_image(image, processed_image_placeholder, status_text)
    preprocess_time = time.time() - preprocess_start
    status_text.markdown('**Preprocessing complete. 👌**')
    
    status_text.markdown('**📝 Extracting text with EasyOCR...**')
    extract_start = time.time()
    extracted_text, ocr_results = extract_text(processed_image, selected_langs)
    extract_time = time.time() - extract_start
    
    status_text.markdown('**🎨 Drawing text overlay on image...**')
    overlay_start = time.time()
    overlayed_image = None
    if ocr_results:
        overlayed_image = draw_text_on_image(image, ocr_results)
        overlayed_image_placeholder.image(overlayed_image, caption="Text Overlay", use_container_width=True)
    else:
        overlayed_image_placeholder.image(image, caption="No OCR Results", use_container_width=True)
    overlay_time = time.time() - overlay_start
    
    total_time = time.time() - start_time
    
    status_text.markdown('**🗂️ Classifying and analyzing results...**')
    time.sleep(0.3)
    label, confidence = classify_document(extracted_text, image, processed_image, ocr_results)
    word_count = len(extracted_text.split())
    char_count = len(extracted_text.replace(" ", "").replace("\n", ""))
    avg_word_length = char_count / word_count if word_count > 0 else 0
    freq = word_frequency(extracted_text)
    top_words = top_n_words(freq, 5)

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
            <h4>Processing</h4>
            <div class="metric-value">{total_time:.2f}s</div>
            <p style="color:#444; font-size:0.83rem;">(Total Time)</p>
        </div>
        <div class="metric-card">
            <h4>Top Words</h4>
            <div class="metric-value">{', '.join([f"{w}({c})" for w, c in top_words]) or '-'}</div>
            <p style="color:#444; font-size:0.83rem;">(Freq.)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    quoted_text = quote(extracted_text)
    text_output_placeholder.markdown(f"""
    <div class="text-output-card">
        <h4>Extracted Text</h4>
        <pre id="ocrText">{extracted_text or "[No visible text]"}</pre>
        <div class="button-row">
            <button class="ocr-button" onclick="copyToClipboard()">Copy Text</button>
            <a href="data:text/plain;charset=utf-8,{quoted_text}" download="extracted_text.txt" class="ocr-button">Download .txt</a>
        </div>
    </div>
    <script>
        function copyToClipboard() {{
            const textToCopy = document.getElementById('ocrText').innerText;
            navigator.clipboard.writeText(textToCopy).then(() => {{
                window.parent.postMessage({{
                    streamlit: {{
                        type: 'streamlit:setComponentValue',
                        value: {{
                            key: 'copied_success',
                            value: true
                        }}
                    }}
                }}, '*')
            }}).catch(err => {{
                console.error('Could not copy text: ', err);
            }});
        }}
    </script>
    """, unsafe_allow_html=True)
    status_text.markdown('**✅ Processing complete!**')
    st.session_state.processing = False
    
    if st.session_state.get('copied_success'):
        st.toast("Text copied to clipboard!")
        st.session_state.copied_success = False

st.markdown("""
<div style="text-align: center; margin-top: 1.5rem;">
    <p style="color:#444; font-size:0.8rem;">OCR-TECH - ADELEKE, OLADOKUN, OLALEYE</p>
    <a href="https://github.com/Praiz22/ocr-tech" target="_blank" class="github-link">
        <span style="display:inline-flex; align-items:center; gap:5px; color:#444; font-size:0.8rem; font-weight: 500;">
            Github Repo- Praiztech
            <svg class="github-icon-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 496 512"><!--!Font Awesome Free 6.5.1 by @fontawesome - https://fontawesome.com License - https://fontawesome.com/license/free Copyright 2024 Fonticons, Inc.--><path d="M165.9 397.4c0 2-2.3 4-4.9 4-2.7 0-4.9-2-4.9-4 0-2 2.3-4 4.9-4 2.7 0 4.9 2 4.9 4zm-14-100.2c.4 1.3.1 2.5-.6 3.5-.9 1.4-1.9 2.7-3.2 3.8-1.5 1.3-3.2 2.6-5 3.6-2.6 1.3-5.5 2.2-8.5 2.5-3.3 .4-6.6-.7-9.3-2.6-2.5-1.7-4.4-4.1-5.6-7-.9-2.2-1.3-4.5-1.4-6.8-2.1-4.9-1.9-9.8.5-14.7 1.5-2.8 3.5-5.5 5.9-7.8 1.9-1.8 4-3.5 6.2-5.1 2.3-1.6 4.7-3 7.2-4.1 2.3-1.2 4.9-2.2 7.6-2.7 2.3-.5 4.6-1.1 7-.9 2.5 .3 5 .8 7.3 1.9 2.1 .9 4.1 2.2 5.9 3.8 2.3 2.1 4.2 4.5 5.8 7.2 1.3 2 2.2 4.2 2.7 6.6 .5 2.4 .7 4.9 .5 7.4-.2 2.6-.8 5.1-1.7 7.5zm-51.5-7.4c.5 1.2.3 2.6-.5 3.8-.9 1.4-2.1 2.7-3.5 3.9-1.6 1.4-3.5 2.6-5.5 3.7-2.6 1.4-5.5 2.2-8.6 2.5-3.4 .4-6.8-.7-9.6-2.7-2.7-1.7-4.7-4.2-6-7.2-.9-2.3-1.3-4.8-1.4-7.2-2.3-5.2-2.2-10.4-.1-15.5 1.6-3 3.7-5.8 6.3-8.2 2.1-1.9 4.3-3.7 6.6-5.4 2.4-1.7 4.9-3.2 7.6-4.3 2.4-1.2 5.1-2.2 7.9-2.7 2.4-.5 4.9-1.1 7.4-.9 2.6 .3 5.2 .8 7.5 2 2.2 1 4.2 2.4 6 4 2.3 2.1 4.2 4.6 5.9 7.4 1.4 2.2 2.2 4.6 2.7 7.1 .5 2.5 .7 5 .5 7.6-.2 2.6-.8 5.1-1.8 7.5zm-5.1-47.5c.3 1.2 .1 2.4-.6 3.5-.9 1.4-2.1 2.7-3.4 3.9-1.5 1.4-3.3 2.6-5.2 3.6-2.4 1.2-5 1.9-7.8 2.1-3.1 .3-6.2-.7-8.8-2.4-2.4-1.6-4.2-3.9-5.4-6.6-.8-2-1.2-4.1-1.3-6.3-2-4.7-1.9-9.5 .4-14.2 1.4-2.7 3.3-5.2 5.6-7.4 1.8-1.7 3.8-3.3 5.9-4.9 2.2-1.6 4.6-3 7.1-4.1 2.2-1.1 4.6-2 7.1-2.5 2.1-.4 4.3-.9 6.5-.7 2.3 .2 4.6 .6 6.7 1.6 2 .9 3.9 2.2 5.6 3.7 2.2 2.1 4 4.5 5.5 7.1 1.2 2 2 4.1 2.5 6.3 .4 2.2 .6 4.5 .4 6.8-.2 2.2-.6 4.5-1.5 6.6zm-11.4 102c.4 2.1-.5 4.3-2.6 5.5-2.2 1.2-4.6 1.9-7.1 2.1-3.2 .3-6.4-.8-9.1-2.9-2.7-2.1-4.7-4.8-6.1-8-1.2-2.7-1.8-5.6-1.9-8.5-.8-5.3-.2-10.7 2.2-15.8 1.8-3.8 4.2-7.3 7-10.4 2.5-2.8 5.3-5.3 8.3-7.5 2.7-2 5.6-3.7 8.6-5.1 3-1.4 6.2-2.4 9.4-2.8 3.3-.4 6.7-.8 10-1.1 3.5-.3 7.1-.6 10.6-.2 3.7 .4 7.3 1.2 10.8 2.6 3.3 1.4 6.5 3.1 9.6 5.2 3.2 2.2 6.1 4.7 8.8 7.6 2.5 2.6 4.6 5.5 6.3 8.7 1.5 3.2 2.5 6.6 3.2 10.1 .7 3.4 .9 6.9 .6 10.4-.3 3.3-.8 6.7-1.7 9.9zm135-26.1c.5 1.2 .3 2.6-.5 3.8-.9 1.4-2.1 2.7-3.5 3.9-1.6 1.4-3.5 2.6-5.5 3.7-2.6 1.4-5.5 2.2-8.6 2.5-3.4 .4-6.8-.7-9.6-2.7-2.7-1.7-4.7-4.2-6-7.2-.9-2.3-1.3-4.8-1.4-7.2-2.3-5.2-2.2-10.4-.1-15.5 1.6-3 3.7-5.8 6.3-8.2 2.1-1.9 4.3-3.7 6.6-5.4 2.4-1.7 4.9-3.2 7.6-4.3 2.4-1.2 5.1-2.2 7.9-2.7 2.4-.5 4.9-1.1 7.4-.9 2.6 .3 5.2 .8 7.5 2 2.2 1 4.2 2.4 6 4 2.3 2.1 4.2 4.6 5.9 7.4 1.4 2.2 2.2 4.6 2.7 7.1 .5 2.5 .7 5 .5 7.6-.2 2.6-.8 5.1-1.8 7.5zm-5.1-47.5c.3 1.2 .1 2.4-.6 3.5-.9 1.4-2.1 2.7-3.4 3.9-1.5 1.4-3.3 2.6-5.2 3.6-2.4 1.2-5 1.9-7.8 2.1-3.1 .3-6.2-.7-8.8-2.4-2.4-1.6-4.2-3.9-5.4-6.6-.8-2-1.2-4.1-1.3-6.3-2-4.7-1.9-9.5 .4-14.2 1.4-2.7 3.3-5.2 5.6-7.4 1.8-1.7 3.8-3.3 5.9-4.9 2.2-1.6 4.6-3 7.1-4.1 2.2-1.1 4.6-2 7.1-2.5 2.1-.4 4.3-.9 6.5-.7 2.3 .2 4.6 .6 6.7 1.6 2 .9 3.9 2.2 5.6 3.7 2.2 2.1 4 4.5 5.5 7.1 1.2 2 2 4.1 2.5 6.3 .4 2.2 .6 4.5 .4 6.8-.2 2.2-.6 4.5-1.5 6.6zm114.2 60.1c.3 1.2 .1 2.4-.6 3.5-.9 1.4-2.1 2.7-3.4 3.9-1.5 1.4-3.3 2.6-5.2 3.6-2.4 1.2-5 1.9-7.8 2.1-3.1 .3-6.2-.7-8.8-2.4-2.4-1.6-4.2-3.9-5.4-6.6-.8-2-1.2-4.1-1.3-6.3-2-4.7-1.9-9.5 .4-14.2 1.4-2.7 3.3-5.2 5.6-7.4 1.8-1.7 3.8-3.3 5.9-4.9 2.2-1.6 4.6-3 7.1-4.1 2.2-1.1 4.6-2 7.1-2.5 2.1-.4 4.3-.9 6.5-.7 2.3 .2 4.6 .6 6.7 1.6 2 .9 3.9 2.2 5.6 3.7 2.2 2.1 4 4.5 5.5 7.1 1.2 2 2 4.1 2.5 6.3 .4 2.2 .6 4.5 .4 6.8-.2 2.2-.6 4.5-1.5 6.6zm-29.3 103.1c-1.4 1.2-3.2 2.2-5.1 3.1-2.2 1-4.6 1.5-7.1 1.5-2.7 0-5.3-.4-7.8-1.2-2.5-.8-4.9-2-7.1-3.6-2.2-1.5-4.2-3.3-5.9-5.3-1.6-2.1-2.8-4.5-3.7-7.1-.8-2.5-1.2-5.2-1.2-7.9 0-3.1 .5-6.2 1.5-9.1 1-2.9 2.3-5.7 3.9-8.2 1.7-2.6 3.6-5 5.7-7.2 2-2.1 4.2-3.9 6.6-5.4 2.4-1.5 4.9-2.7 7.6-3.6 2.5-.8 5.1-1.2 7.8-1.2 2.6 0 5.2 .4 7.7 1.2 2.5 .8 4.9 2 7.2 3.6 2.3 1.5 4.4 3.4 6.2 5.5 1.7 2.1 3 4.5 3.9 7.1 .8 2.6 1.2 5.2 1.2 8-.1 3.2-.5 6.3-1.6 9.3-1 2.9-2.3 5.7-4 8.2zm-28-144.1c-1.4 1.2-3.2 2.2-5.1 3.1-2.2 1-4.6 1.5-7.1 1.5-2.7 0-5.3-.4-7.8-1.2-2.5-.8-4.9-2-7.1-3.6-2.2-1.5-4.2-3.3-5.9-5.3-1.6-2.1-2.8-4.5-3.7-7.1-.8-2.5-1.2-5.2-1.2-7.9 0-3.1 .5-6.2 1.5-9.1 1-2.9 2.3-5.7 3.9-8.2 1.7-2.6 3.6-5 5.7-7.2 2-2.1 4.2-3.9 6.6-5.4 2.4-1.5 4.9-2.7 7.6-3.6 2.5-.8 5.1-1.2 7.8-1.2 2.6 0 5.2 .4 7.7 1.2 2.5 .8 4.9 2 7.2 3.6 2.3 1.5 4.4 3.4 6.2 5.5 1.7 2.1 3 4.5 3.9 7.1 .8 2.6 1.2 5.2 1.2 8-.1 3.2-.5 6.3-1.6 9.3-1 2.9-2.3 5.7-4 8.2zm23.4 216c-2.3 2.1-4.2 4.6-5.9 7.4-1.4 2.2-2.2 4.6-2.7 7.1-.5 2.5-.7 5-.5 7.6 .2 2.6 .8 5.1 1.8 7.5 1.3 3.1 3 5.7 5.1 8 2.1 2.2 4.6 4.1 7.4 5.9 2.8 1.8 5.7 3 8.8 3.9 3.1 .9 6.3 1.3 9.4 1.3 3.3 0 6.6-.4 9.8-1.3 3.2-.8 6.2-2.2 9.1-4 2.8-1.7 5.5-3.7 7.8-5.9 2.4-2.3 4.3-4.9 5.8-7.8 1.4-2.9 2.3-6.1 2.7-9.3 .4-3.2 .5-6.5 .1-9.7-.5-3.1-1.3-6.1-2.5-9-.9-2.1-2.2-4.1-3.7-6-1.4-1.8-3-3.4-4.8-4.9-1.9-1.5-3.9-2.8-6.1-3.9-2.2-1.1-4.6-2-7.1-2.5-2.5-.5-5.1-.7-7.6-.6-2.5 .1-5 .6-7.3 1.6-2.2 1-4.3 2.3-6.3 3.8zm11.3-88.7c.3 1.2 .1 2.4-.6 3.5-.9 1.4-2.1 2.7-3.4 3.9-1.5 1.4-3.3 2.6-5.2 3.6-2.4 1.2-5 1.9-7.8 2.1-3.1 .3-6.2-.7-8.8-2.4-2.4-1.6-4.2-3.9-5.4-6.6-.8-2-1.2-4.1-1.3-6.3-2-4.7-1.9-9.5 .4-14.2 1.4-2.7 3.3-5.2 5.6-7.4 1.8-1.7 3.8-3.3 5.9-4.9 2.2-1.6 4.6-3 7.1-4.1 2.2-1.1 4.6-2 7.1-2.5 2.1-.4 4.3-.9 6.5-.7 2.3 .2 4.6 .6 6.7 1.6 2 .9 3.9 2.2 5.6 3.7 2.2 2.1 4 4.5 5.5 7.1 1.2 2 2 4.1 2.5 6.3 .4 2.2 .6 4.5 .4 6.8-.2 2.2-.6 4.5-1.5 6.6zm-113.8 62.7c.4 1.3.1 2.5-.6 3.5-.9 1.4-1.9 2.7-3.2 3.8-1.5 1.3-3.2 2.6-5 3.6-2.6 1.3-5.5 2.2-8.5 2.5-3.3 .4-6.6-.7-9.3-2.6-2.5-1.7-4.4-4.1-5.6-7-.9-2.2-1.3-4.5-1.4-6.8-2.1-4.9-1.9-9.8.5-14.7 1.5-2.8 3.5-5.5 5.9-7.8 1.9-1.8 4-3.5 6.2-5.1 2.3-1.6 4.7-3 7.2-4.1 2.3-1.2 4.9-2.2 7.6-2.7 2.3-.5 4.6-1.1 7-.9 2.5 .3 5 .8 7.3 1.9 2.1 .9 4.1 2.2 5.9 3.8 2.3 2.1 4.2 4.5 5.8 7.2 1.3 2 2.2 4.2 2.7 6.6 .5 2.4 .7 4.9 .5 7.4-.2 2.6-.8 5.1-1.7 7.5zM248 8C111 8 0 119 0 256s111 248 248 248 248-111 248-248S385 8 248 8zm44.2 222.1c1.3 1.3 2.5 2.7 3.7 4.2 1.3 1.5 2.3 3.1 3.3 4.7 1.2 2 2 4.1 2.6 6.3 .7 2.2 1 4.5 1 6.8 0 3-.5 6-1.4 8.9-.9 2.8-2.2 5.5-3.8 8-1.6 2.6-3.4 5-5.6 7.2-2.2 2.1-4.6 3.9-7.1 5.4-2.5 1.5-5.1 2.7-7.8 3.6-2.7 .9-5.4 1.2-8.1 1.2-2.9 0-5.8-.4-8.7-1.4-3-.9-5.9-2.2-8.6-3.8-2.7-1.6-5.3-3.6-7.8-5.8-2.5-2.3-4.8-5-6.6-7.9-1.7-3-3-6.2-3.8-9.6-.8-3.4-1.2-6.9-1.1-10.4 .1-3.3 .7-6.6 1.7-9.9 1-3.2 2.4-6.2 4.1-9 1.5-2.5 3.4-4.8 5.6-6.9 2.1-2.1 4.5-3.9 7-5.4 2.5-1.5 5.1-2.6 7.8-3.5 2.7-.9 5.5-1.2 8.2-1.2 2.9 0 5.8 .4 8.7 1.4 3 .9 5.8 2.2 8.5 3.8 2.7 1.6 5.3 3.6 7.8 5.9 2.5 2.3 4.8 5.1 6.6 8.1 1.7 3 3 6.3 3.8 9.7 .8 3.4 1.2 6.9 1.1 10.4-.1 3.3-.7 6.6-1.8 9.9z"/></svg>
    </a>
</div>
""", unsafe_allow_html=True)
