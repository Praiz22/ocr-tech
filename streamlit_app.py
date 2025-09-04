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
from skimage.transform import rotate, probabilistic_hough_line, hough_line
from skimage.feature import canny
from math import degrees

# ----------------------------------------
# PAGE & GLOBALS
# ----------------------------------------
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="expanded")

# IMPORTANT: Tesseract must be installed and in your system's PATH.
# For Debian/Ubuntu: sudo apt-get install tesseract-ocr
# For macOS: brew install tesseract
# On Windows, download from: https://tesseract-ocr.github.io/tessdoc/Downloads.html
# If not in PATH, uncomment the line below and set your specific path.
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

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
    --card-bg: rgba(255, 255, 255, 0.1); /* VERY transparent */
    --card-border: rgba(255, 255, 255, 0.2); /* VERY transparent */
    --card-shadow: 0 5px 15px rgba(0, 0, 0, 0.08);
    --text-1: #1f1f1f; /* Darkest text */
    --text-2: #5a5a5a; /* Slightly lighter dark text */
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
    color: var(--text-1); /* Default body text color */
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
    color: var(--text-1); /* Ensure card text is dark */
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
  
  /* Styling for the larger file uploader container */
  .st-emotion-cache-1h50xby {
      background: var(--card-bg);
      backdrop-filter: blur(16px);
      border: 1px solid var(--card-border);
      border-radius: var(--radius-xl);
      padding: 3rem; /* Increased padding */
      box-shadow: var(--card-shadow);
      display: flex;
      flex-direction: column;
      align-items: center;
      justify-content: center;
      min-height: 250px; /* Make the box taller */
  }
  
  .st-emotion-cache-1h50xby:hover {
      border-color: var(--brand);
  }

  /* Text inside the file uploader */
  .st-emotion-cache-1h50xby p, 
  .st-emotion-cache-1h50xby h5 {
      color: var(--text-1) !important;
  }

  .image-row {
    display: flex;
    flex-direction: row;
    gap: 1rem;
    justify-content: center;
    align-items: flex-start;
    flex-wrap: wrap;
  }
  
  .image-container {
    width: 100%;
    flex: 1 1 250px;
    max-width: 300px;
    height: 100%;
    max-height: 250px;
    overflow: hidden;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.08); /* More transparent */
    box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
  }
  
  .image-container img {
      width: 100%;
      height: 100%;
      object-fit: contain;
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
    background: rgba(255, 255, 255, 0.1); /* More transparent */
    border: 1px solid rgba(255, 255, 255, 0.08); /* More transparent */
    color: var(--text-1) !important; /* Ensure metric text is dark */
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
    background: rgba(255, 255, 255, 0.15); /* More transparent */
    padding: 1rem;
    border-radius: var(--radius-xl);
    border: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 1rem;
    color: var(--text-1) !important; /* Ensure text output is dark */
  }
  
  .text-output-card pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Poppins', sans-serif;
    color: var(--text-1) !important;
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
  
  /* --- Global Text Color Overrides for Streamlit Components --- */
  
  /* All h tags */
  h1, h2, h3, h4, h5, h6 {
      color: var(--text-1) !important;
  }

  /* All p tags */
  p {
      color: var(--text-2) !important;
  }

  /* Specific targeting for file uploader text (filename and size) */
  .st-emotion-cache-1c7y31u p {
      color: black !important;
  }

  /* Target the "Ready to process?" text (if still white) */
  .st-emotion-cache-1869e5d div p {
      color: black !important;
  }

  /* Sidebar text labels */
  .st-emotion-cache-16idsys p {
      color: var(--text-1) !important;
  }
  .st-emotion-cache-10mrprx p {
      color: var(--text-1) !important;
  }

  /* Text inside status messages */
  .stStatus .st-emotion-cache-13xt0m9 p {
      color: var(--text-1) !important;
  }
  .stStatus .st-emotion-cache-1jm7cjm p {
      color: var(--text-1) !important;
  }

  /* Caption text under images */
  .st-emotion-cache-fofk9f p {
      color: var(--text-2) !important;
  }

  /* Help text for widgets */
  .st-emotion-cache-p4dp75 p {
    color: var(--text-2) !important;
  }

  /* --- Other elements for consistency --- */
  .github-icon-svg {
    width: 20px;
    height: 20px;
    fill: #444; /* Darker fill for icon */
    transition: transform 0.3s ease-in-out;
  }
  
  .github-link:hover .github-icon-svg {
    transform: rotate(360deg);
  }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------
# IMAGE PROCESSING & DESKEWING
# ----------------------------------------
def deskew_image(image):
    """
    Corrects the skew of an image using the Hough Transform.
    Returns the deskewed image.
    """
    with st.status("Deskewing image...", expanded=True) as status:
        img_np = np.array(image.convert('L'))
        edges = canny(img_np, sigma=1)
        h, theta, d = hough_line(edges)
        
        # Find the angle corresponding to the most prominent line
        _, angles, _ = hough_line(edges)
        
        # Get the most frequent angle
        max_angle = 0
        if len(angles) > 0:
            # Convert radians to degrees
            angles_deg = np.rad2deg(angles)
            angles_deg[angles_deg > 45] -= 90
            angles_deg[angles_deg < -45] += 90
            
            # Use histogram to find the most prominent angle
            hist, bins = np.histogram(angles_deg, bins=90, range=(-45, 45))
            max_angle = bins[np.argmax(hist)]
        
        status.update(label=f"Skew angle detected: {max_angle:.2f} degrees. Rotating...", state="running")
        
        # Rotate the original image
        rotated_img = Image.fromarray((rotate(np.array(image), -max_angle, resize=True, mode='edge') * 255).astype(np.uint8))
        status.update(label="Deskewing complete!", state="complete")
        return rotated_img

def preprocess_image(img, contrast_factor, deskew):
    """
    Applies a series of image processing techniques to enhance OCR accuracy.
    """
    with st.status("Preprocessing image...", expanded=True) as status:
        if deskew:
            img = deskew_image(img)

        status.update(label="Converting to Grayscale...", state="running")
        gray = ImageOps.grayscale(img)
        
        status.update(label="Enhancing Contrast...", state="running")
        enhancer = ImageEnhance.Contrast(gray)
        enhanced = enhancer.enhance(contrast_factor)
        
        status.update(label="Binarizing (Otsu)...", state="running")
        arr = np.array(enhanced)
        _, binarized = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        bin_img = Image.fromarray(binarized)
        
        status.update(label="Preprocessing complete!", state="complete")
        return bin_img

# ----------------------------------------
# OCR EXTRACTION
# ----------------------------------------
@st.cache_data(show_spinner=False)
def get_easyocr_reader(lang_list):
    return easyocr.Reader(lang_list)

def recognize_text_easyocr(image, langs, min_conf=0.2):
    reader = get_easyocr_reader(langs)
    img_np = np.array(image) if not isinstance(image, np.ndarray) else image
    result = reader.readtext(img_np)
    lines = [text.strip() for bbox, text, prob in result if prob >= min_conf and text.strip()]
    return "\n".join(lines), result

def clean_text(text: str) -> str:
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
    img_np = np.array(image.convert("RGB"))
    for (bbox, text, prob) in results:
        if prob >= 0.2:
            (top_left, top_right, bottom_right, bottom_left) = bbox
            top_left = (int(top_left[0]), int(top_left[1]))
            bottom_right = (int(bottom_right[0]), int(bottom_right[1]))
            
            cv2.rectangle(img=img_np, pt1=top_left, pt2=bottom_right, color=(255, 0, 0), thickness=2)
            cv2.putText(img=img_np, text=text, org=(top_left[0], top_left[1] - 10), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(255, 0, 0), thickness=2)
    return Image.fromarray(img_np)

def extract_text(image, langs, tesseract_psm_modes):
    """
    Extracts text, first attempting with EasyOCR, then falling back to
    Tesseract with different PSM modes if EasyOCR fails.
    """
    easyocr_results = None
    try:
        text, easyocr_results = recognize_text_easyocr(image, langs)
        filtered = clean_text(text)
        if len(filtered) > 8 and any(c.isalpha() for c in filtered):
            return filtered, easyocr_results
    except Exception as e:
        st.warning(f"EasyOCR failed: {e}. Falling back to Tesseract.")

    for psm in tesseract_psm_modes:
        try:
            text = pytesseract.image_to_string(image, config=f"--psm {psm}")
            filtered = clean_text(text)
            if len(filtered) > 8 and any(c.isalpha() for c in filtered):
                return filtered, []
        except Exception as e:
            st.error(f"Tesseract failed with PSM {psm}: {e}")
            continue
    return "", []

# ----------------------------------------
# UTILITIES (metrics, classification)
# ----------------------------------------
def classify_document(text, img, processed_img, ocr_results):
    words = text.split()
    real_words = [w for w in words if len(w) > 2 and re.match(r"[a-zA-Z]", w)]
    
    word_count = len(words)
    line_count = len(text.splitlines())
    char_count = len(text.replace(" ", "").replace("\n", ""))
    unique_words = len(set(real_words))

    img_width, img_height = img.size
    aspect_ratio = max(img_width, img_height) / min(img_width, img_height) if min(img_width, img_height) > 0 else 1
    
    document_score = 0
    handwritten_score = 0
    
    # Heuristics for document vs. picture
    if word_count > 50 and line_count > 10:
        document_score += 40
    if 1.4 < aspect_ratio < 1.6:
        document_score += 20
    if word_count < 10 or line_count < 3:
        return "Picture", 99, unique_words
    
    keywords = {'invoice', 'receipt', 'report', 'statement', 'bill', 'form'}
    for word in real_words:
        if word.lower() in keywords:
            document_score += 15
            break
    
    if document_score > handwritten_score:
        return "Document", min(100, 70 + document_score // 2), unique_words
    elif handwritten_score > document_score:
        return "Handwritten Note", min(100, 70 + handwritten_score // 2), unique_words
    else:
        return "Document", min(100, 70 + document_score // 2), unique_words

def word_frequency(text):
    words = re.findall(r'\b\w+\b', text.lower())
    freq = {}
    for word in words:
        freq[word] = freq.get(word, 0) + 1
    return freq

def top_n_words(freq, n=5):
    return sorted(freq.items(), key=lambda x: x[1], reverse=True)[:n]

# ----------------------------------------
# SESSION STATE MANAGEMENT
# ----------------------------------------
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'ocr_started' not in st.session_state:
    st.session_state.ocr_started = False
if 'last_uploaded_filename' not in st.session_state:
    st.session_state.last_uploaded_filename = None

# ----------------------------------------
# SIDEBAR FOR SETTINGS
# ----------------------------------------
st.sidebar.header("OCR Settings")
st.sidebar.markdown("Configure your OCR extraction preferences.")

# Added 'yo' for Yoruba
available_langs = ['en', 'ar', 'ru', 'ch_sim', 'ja', 'yo']
selected_langs = st.sidebar.multiselect(
    'Select language(s) for OCR',
    options=available_langs,
    default=['en'],
    help="Select the languages present in the image."
)

st.sidebar.markdown("---")
st.sidebar.header("Image Preprocessing")
deskew = st.sidebar.checkbox('Apply Deskewing', value=True, help="Corrects image rotation for better accuracy.")
contrast_factor = st.sidebar.slider(
    'Contrast Enhancement',
    min_value=1.0, max_value=4.0, value=2.0, step=0.1,
    help="Increases the contrast of the image."
)
st.sidebar.markdown("---")
st.sidebar.header("Tesseract Fallback")
tesseract_psm_options = [
    (1, '1: OCR a single column of text.'),
    (3, '3: Fully automatic page segmentation (default).'),
    (4, '4: Assume a single column of text of variable sizes.'),
    (5, '5: Assume a single uniform block of text.'),
    (6, '6: Assume a single uniform block of text.'),
    (7, '7: Treat the image as a single text line.'),
    (11, '11: Find as much text as possible in no particular order.'),
]
selected_psms = st.sidebar.multiselect(
    'Tesseract PSM Modes (Fallback)',
    options=[p[0] for p in tesseract_psm_options],
    format_func=lambda x: f"PSM {x}",
    default=[6, 11, 3],
    help="Select Tesseract page segmentation modes to try if EasyOCR fails."
)

# ----------------------------------------
# MAIN UI
# ----------------------------------------
st.markdown("""
<div class="ocr-container">
    <div class="header">
        <h1>OCR-TECH</h1>
        <p>Better Text Extraction from Images (Powered by EasyOCR, Tesseract, PraixTech)</p>
    </div>
    <div class="ocr-card">
        <div style="text-align: center;">
            <h4 style="color: var(--text-1) !important;">Upload an Image</h4>
            <p style="color: var(--text-2) !important; font-weight: 500;">Drag and drop a file or click below to choose a file.</p>
        </div>
        <div style="margin-top:1.5rem;">
""", unsafe_allow_html=True)
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")
st.markdown("""
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

if uploaded_file:
    if st.session_state.last_uploaded_filename != uploaded_file.name:
        st.session_state.last_uploaded_filename = uploaded_file.name
        st.session_state.uploaded_image = uploaded_file.getvalue()
        st.session_state.ocr_started = False
        st.rerun()

if st.session_state.uploaded_image:
    image = Image.open(BytesIO(st.session_state.uploaded_image)).convert("RGB")
    st.markdown('<div class="ocr-card" style="margin-top:2rem;">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Original Image", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
            <div style="display: flex; flex-direction: column; height: 100%; justify-content: space-between;">
                <div>
                    <h4 style="color: var(--text-1) !important;">Ready to process?</h4>
                    <p style="color: var(--text-1) !important;">Review the settings in the sidebar and press the button below to begin the OCR process.</p>
                </div>
                <div style="display:flex; justify-content:center; padding-top:1rem;">
            """, unsafe_allow_html=True)
        if st.button("Start OCR", key="start_ocr_button", use_container_width=True, help="Click to start the OCR process."):
            st.session_state.ocr_started = True
            st.rerun()
        st.markdown("""</div></div>""", unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

if st.session_state.ocr_started:
    image = Image.open(BytesIO(st.session_state.uploaded_image)).convert("RGB")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown('<div class="image-container">', unsafe_allow_html=True)
        st.image(image, caption="Original", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        processed_image_placeholder = st.empty()
    with col3:
        overlayed_image_placeholder = st.empty()
    
    text_output_placeholder = st.empty()
    
    start_time = time.time()
    
    with st.status("Running OCR...", expanded=True) as status_bar:
        # Preprocessing
        processed_image = preprocess_image(image, contrast_factor, deskew)
        processed_image_placeholder.image(processed_image, caption="Preprocessed Image", use_container_width=True)
        
        # Extraction
        status_bar.update(label="Extracting text...", state="running")
        extracted_text, ocr_results = extract_text(processed_image, selected_langs, selected_psms)
        
        # Bounding Boxes
        status_bar.update(label="Drawing bounding boxes...", state="running")
        if ocr_results:
            overlayed_image = draw_text_on_image(image, ocr_results)
            overlayed_image_placeholder.image(overlayed_image, caption="Text Overlay", use_container_width=True)
        else:
            overlayed_image_placeholder.image(image, caption="No OCR Results", use_container_width=True)
        
        status_bar.update(label="OCR process completed successfully!", state="complete")
    
    total_time = time.time() - start_time
    
    label, confidence, unique_words = classify_document(extracted_text, image, processed_image, ocr_results)
    word_count = len(extracted_text.split())
    line_count = len(extracted_text.splitlines())
    char_count = len(extracted_text.replace(" ", "").replace("\n", ""))
    freq = word_frequency(extracted_text)
    top_words = top_n_words(freq, 5)

    st.markdown(f"""
    <div class="results-grid" style="margin-top:2rem;">
        <div class="metric-card">
            <h4>Prediction</h4>
            <div class="metric-value" style="color:var(--brand);">{label}</div>
            <p style="color:var(--text-2) !important; font-size:0.83rem;">(Classified)</p>
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
            <p style="color:var(--text-2) !important; font-size:0.83rem;">(Words)</p>
        </div>
        <div class="metric-card">
            <h4>Processing</h4>
            <div class="metric-value">{total_time:.2f}s</div>
            <p style="color:var(--text-2) !important; font-size:0.83rem;">(Total Time)</p>
        </div>
        <div class="metric-card">
            <h4>Top Words</h4>
            <div class="metric-value">{', '.join([f"{w}({c})" for w, c in top_words]) or '-'}</div>
            <p style="color:var(--text-2) !important; font-size:0.83rem;">(Freq.)</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    quoted_text = quote(extracted_text)
    st.markdown(f"""
    <div class="text-output-card">
        <h4 style="color: var(--text-1) !important;">Extracted Text</h4>
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
    
st.markdown("""
<div style="text-align: center; margin-top: 1.5rem;">
    <p style="color:var(--text-2) !important; font-size:0.8rem;">OCR-TECH - ADELEKE, OLADOKUN, OLALEYE</p>
    <a href="https://github.com/Praiz22/ocr-tech" target="_blank" class="github-link">
        <span style="display:inline-flex; align-items:center; gap:5px; color:var(--text-2) !important; font-size:0.8rem; font-weight: 500;">
            Github Repo- Praiztech
            <svg class="github-icon-svg" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24">
                <path d="M12 0C5.373 0 0 5.373 0 12c0 5.302 3.438 9.799 8.205 11.387.6.11.82-.26.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.043-1.61-4.043-1.61-.546-1.387-1.334-1.758-1.334-1.758-1.087-.744.08-.729.08-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.492.998.108-.775.42-1.305.762-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.382 1.235-3.22-.125-.3-.535-1.52.117-3.176 0 0 1.005-.322 3.3-.997.96.26 1.98.39 3 .39 1.02 0 2.04-.13 3-.39 2.295.675 3.295.997 3.295.997.652 1.656.242 2.876.117 3.176.77.838 1.235 1.91 1.235 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .318.22.69.825.577C20.562 21.799 24 17.302 24 12c0-6.627-5.373-12-12-12z"/>
            </svg>
        </span>
    </a>
</div>
""", unsafe_allow_html=True)
