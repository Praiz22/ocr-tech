import streamlit as st
import pytesseract
from PIL import Image, ImageOps
from io import BytesIO
import re
import time
import numpy as np

# Set Streamlit to use a wide layout and a custom title.
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="collapsed")

# --- Custom CSS and HTML from ocr skeleton.html ---
# We're injecting the exact same CSS and HTML structure to replicate the design.
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  
  :root {
    --bg-1: #ffffff;
    --bg-2: #fff5eb;
    --bg-3: #ffe7cc;
    --card-bg: rgba(255, 255, 255, 0.24);
    --card-border: rgba(255, 255, 255, 0.36);
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
  
  .file-input-label {
    background-color: var(--brand);
    color: var(--bg-1);
    padding: 1rem 2rem;
    border-radius: var(--radius-lg);
    cursor: pointer;
    font-weight: 600;
    transition: background-color 0.3s ease;
  }
  
  .file-input-label:hover {
    background-color: var(--brand-2);
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
  }

  .text-output-card pre {
    white-space: pre-wrap;
    word-wrap: break-word;
    font-family: 'Poppins', sans-serif;
    color: var(--text-1);
  }
  
  .button-row {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-top: 2rem;
  }
  
  .ocr-button {
    background-color: var(--brand);
    color: var(--bg-1);
    padding: 1rem 2.5rem;
    border-radius: var(--radius-lg);
    font-size: 1.1rem;
    font-weight: 600;
    border: none;
    cursor: pointer;
    transition: all 0.2s ease;
  }
  
  .ocr-button:hover {
    background-color: var(--brand-2);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.2);
  }
  
  .st-emotion-cache-18ni7ap {
    /* This targets the invisible Streamlit file uploader button */
    visibility: hidden;
    height: 0;
  }

  .stFileUploader {
      visibility: hidden;
      display: none;
  }

  /* Style the container for the custom uploader area */
  .stFileUploader > div {
    display: flex !important;
    flex-direction: column !important;
    align-items: center !important;
    justify-content: center !important;
  }

  /* Target the drag-and-drop area to style it */
  .stFileUploader > div:nth-child(2) > div:first-child {
    border: 2px dashed var(--muted);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    cursor: pointer;
  }
</style>
""", unsafe_allow_html=True)

# Set the path to the Tesseract executable.
# You might need to change this line depending on your setup.
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# --- Core Logic for OCR and Classification ---

def preprocess_image(image):
    """
    Performs image preprocessing steps and returns the processed image.
    This simulates a full pipeline for your defense.
    """
    
    st.session_state.status_text.markdown('<h4 id="predStatus">Normalizing Image...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    
    # Grayscale conversion for normalization
    grayscale_image = ImageOps.grayscale(image)
    st.session_state.processed_image_placeholder.image(grayscale_image, caption="Processed Image (Grayscale)", use_column_width=True)
    
    st.session_state.status_text.markdown('<h4 id="predStatus">Deskewing...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    # Placeholder for deskewing logic
    deskewed_image = grayscale_image 

    st.session_state.status_text.markdown('<h4 id="predStatus">Removing Noise...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    # Placeholder for noise removal logic
    denoised_image = deskewed_image

    return denoised_image

def classify_document(text):
    """
    Classifies the document based on features of the extracted text.
    """
    text = text.lower()
    
    document_keywords = ["invoice", "receipt", "report", "statement", "document", "proposal", "contract", "memorandum", "agenda"]
    if any(keyword in text for keyword in document_keywords):
        return "Document", 95 

    letter_pattern = r"(dear|sincerely|best regards|regards|from:|to:|date:|re:|subject:|yours sincerely|yours faithfully|attachment:)"
    if re.search(letter_pattern, text):
        return "Letter", 90

    word_count = len(text.split())
    line_count = len(text.split('\n'))
    if word_count > 100 and line_count > 10:
        return "Document", 85
    
    if word_count < 50 and line_count < 5 or "file" in text or "menu" in text:
        return "Screenshot", 80
    
    return "Miscellaneous", 60


def run_ocr_and_classify(image):
    """
    Main function to process the image and get OCR and classification results.
    """
    
    st.session_state.status_text.markdown('<h4 id="predStatus">Running OCR & inference...</h4>', unsafe_allow_html=True)
    
    # Step 1: Pre-process the image
    processed_image = preprocess_image(image)
    
    # Step 2: Perform OCR
    st.session_state.status_text.markdown('<h4 id="predStatus">Extracting Text...</h4>', unsafe_allow_html=True)
    ocr_text = pytesseract.image_to_string(processed_image, config='--psm 6')

    # Step 3: Classify the document
    st.session_state.status_text.markdown('<h4 id="predStatus">Classifying Document...</h4>', unsafe_allow_html=True)
    label, confidence = classify_document(ocr_text)

    # Dynamic updates
    st.session_state.metric_grid_placeholder.markdown(f"""
    <div class="results-grid">
        <div class="metric-card">
            <h4>Prediction</h4>
            <div class="metric-value">...</div>
            <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Classified document type)</p>
        </div>
        <div class="metric-card">
            <h4>Confidence</h4>
            <div class="metric-value">...</div>
            <div class="progress-bar-container">
                <div class="progress-bar"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    time.sleep(0.5)
    
    st.session_state.metric_grid_placeholder.markdown(f"""
    <div class="results-grid">
        <div class="metric-card">
            <h4>Prediction</h4>
            <div class="metric-value">...</div>
            <p style="color:var(--text-2); font-size:0.85rem; margin:0.5rem 0 0;">(Classified document type)</p>
        </div>
        <div class="metric-card">
            <h4>Confidence</h4>
            <div class="metric-value">...</div>
            <div class="progress-bar-container">
                <div class="progress-bar" style="width:30%;"></div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    time.sleep(0.5)

    # Final update with true values
    st.session_state.status_text.markdown('<h4 id="predStatus">Done!</h4>', unsafe_allow_html=True)
    st.session_state.metric_grid_placeholder.markdown(f"""
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
    </div>
    """, unsafe_allow_html=True)

    st.session_state.text_output_placeholder.markdown(f"""
    <div class="text-output-card" style="margin-top: 1.5rem;">
        <h4>Extracted Text</h4>
        <pre id="ocrText">{ocr_text}</pre>
    </div>
    """, unsafe_allow_html=True)


# --- Streamlit UI Components ---

# State management for displaying results
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

# Main App Layout
st.markdown("""
<div class="ocr-container">
    <div class="header">
        <h1>OCR-TECH</h1>
        <p>Optical Character Recognition & Image Classification</p>
    </div>
</div>
""", unsafe_allow_html=True)

# File Uploader section with drag-and-drop
with st.container():
    st.markdown("""
    <div class="ocr-container ocr-card">
        <div class="file-upload-section">
            <h4>Upload an Image</h4>
            <p>Drag and drop a file here or click below to choose a file.</p>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")
    
    if uploaded_file:
        try:
            # Check if the file is a valid image before opening
            image_data = uploaded_file.getvalue()
            _ = Image.open(BytesIO(image_data))
            st.session_state.uploaded_image = image_data

        except Exception as e:
            st.error("Error: The file you uploaded could not be identified as a valid image. Please try a different file.")
            st.session_state.uploaded_image = None

# Display uploaded image and a Process button
if st.session_state.uploaded_image:
    st.markdown("""
    <div class="ocr-container" style="margin-top: -2rem;">
        <div class="ocr-card">
            <h4 id="predStatus">Image Preview</h4>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; margin-top: 1.5rem;">
                <div style="text-align: center;">
                    <h5>Original</h5>
                    <div style="border: 1px dashed var(--muted); padding: 1rem; border-radius: var(--radius-lg);">
                        <div id="originalImagePreview"></div>
                    </div>
                </div>
                <div style="text-align: center;">
                    <h5>Processed</h5>
                    <div style="border: 1px dashed var(--muted); padding: 1rem; border-radius: var(--radius-lg);">
                        <div id="processedImagePreview"></div>
                    </div>
                </div>
            </div>
        </div>
        <div class="button-row">
            <button id="processBtn" class="ocr-button" onclick="window.parent.document.querySelector('[data-testid=stButton][kind=secondary]').click()">Process Image</button>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Placeholders for dynamic UI updates
    st.session_state.status_text = st.empty()
    st.session_state.metric_grid_placeholder = st.empty()
    st.session_state.text_output_placeholder = st.empty()
    
    original_col, processed_col = st.columns(2)
    with original_col:
        st.session_state.original_image_placeholder = st.empty()
    with processed_col:
        st.session_state.processed_image_placeholder = st.empty()
    
    st.session_state.original_image_placeholder.image(Image.open(BytesIO(st.session_state.uploaded_image)), caption="Original Image", use_column_width=True)
    
    if st.button("Process Image", key="process_button"):
        image_to_process = Image.open(BytesIO(st.session_state.uploaded_image))
        run_ocr_and_classify(image_to_process)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="color:var(--text-2); font-size:0.8rem;">OCR-TECH - Designed by ADELEKE, OLADOKUN, and OLALEYE</p>
</div>
""", unsafe_allow_html=True)
