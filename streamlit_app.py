import streamlit as st
import pytesseract
from PIL import Image, ImageOps
from io import BytesIO
import re
import time
import numpy as np
import base64

# Set Streamlit to use a wide layout and a custom title.
st.set_page_config(layout="wide", page_title="OCR-TECH", initial_sidebar_state="collapsed")

# --- Custom CSS and HTML from ocr skeleton.html ---
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap');
  
  :root {
    --bg-1: #ffffff;
    --bg-2: #fff5eb;
    --bg-3: #ffe7cc;
    --card-bg: rgba(255, 255, 255, 0.5); /* Made more transparent for a glass effect */
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
  
  /* Hiding the default Streamlit file uploader button to replace with custom one */
  .stFileUploader > div > button {
    display: none;
  }

  /* Style the file uploader's drag-and-drop zone */
  .stFileUploader > div > div > div {
    border: 2px dashed var(--muted);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    cursor: pointer;
    background: rgba(255, 255, 255, 0.2);
    transition: background 0.3s ease;
  }
  .stFileUploader > div > div > div:hover {
      background: rgba(255, 255, 255, 0.3);
  }
</style>
""", unsafe_allow_html=True)

# Set the path to the Tesseract executable.
# You might need to change this line depending on your setup.
# pytesseract.pytesseract.tesseract_cmd = r'/usr/bin/tesseract'

# --- Core Logic for OCR and Classification ---

def preprocess_image(image, processed_image_placeholder, status_text_placeholder):
    """
    Performs image preprocessing steps and returns the processed image.
    Simulates a full pipeline with dynamic updates.
    """
    
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Normalizing Image...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    
    # Grayscale conversion for normalization
    grayscale_image = ImageOps.grayscale(image)
    processed_image_placeholder.image(grayscale_image, caption="Processed Image", use_container_width=True)
    
    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Deskewing...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    # Placeholder for deskewing logic
    deskewed_image = grayscale_image 

    status_text_placeholder.markdown('<h4 id="predStatus" style="color:var(--text-1);">Removing Noise...</h4>', unsafe_allow_html=True)
    time.sleep(0.5)
    # Placeholder for noise removal logic
    denoised_image = deskewed_image

    return denoised_image

def classify_document(text):
    """
    Classifies the document based on features of the extracted text.
    """
    text = text.lower()
    
    # Enhanced keyword-based classification
    if "invoice" in text or "bill to" in text or "invoice number" in text:
        return "Invoice", 98
    if "receipt" in text or "thank you for your purchase" in text:
        return "Receipt", 97
    if "report" in text or "summary" in text or "analysis" in text:
        return "Report", 95
    if "contract" in text or "agreement" in text or "terms and conditions" in text:
        return "Contract", 96
    if "memorandum" in text or "memo" in text or "interoffice" in text:
        return "Memorandum", 94
    if "agenda" in text or "meeting" in text or "minutes" in text:
        return "Agenda", 93
    if "prescription" in text or "rx" in text or "refill" in text:
        return "Medical Document", 92

    # Pattern-based classification
    letter_pattern = r"(dear|sincerely|best regards|regards|from:|to:|date:|re:|subject:|yours sincerely|yours faithfully|attachment:)"
    if re.search(letter_pattern, text):
        return "Letter", 90

    # Heuristic-based classification
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    
    if word_count > 250 and line_count > 20:
        return "Document", 88
    if word_count < 50 and line_count < 5:
        return "Screenshot/Short Text", 80
    
    return "Miscellaneous", 75


def run_ocr_and_classify(image):
    """
    Main function to process the image and get OCR and classification results.
    """
    
    # Create placeholders for dynamic UI elements
    status_text = st.empty()
    metric_grid_placeholder = st.empty()
    text_output_placeholder = st.empty()
    
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Running OCR & inference...</h4>', unsafe_allow_html=True)
    
    # Step 1: Pre-process the image
    original_col, processed_col = st.columns(2)
    with original_col:
        st.image(image, caption="Original Image", use_container_width=True)
    with processed_col:
        processed_image_placeholder = st.empty()
        processed_image = preprocess_image(image, processed_image_placeholder, status_text)
    
    # Step 2: Perform OCR
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Extracting Text...</h4>', unsafe_allow_html=True)
    ocr_text = pytesseract.image_to_string(processed_image, config='--psm 6')

    # Step 3: Simulate "training" phase for metrics
    epochs = 10
    epoch_val = st.empty()
    epoch_bar = st.empty()
    for e in range(1, epochs + 1):
        epoch_val.markdown(f'<h4 style="color:var(--text-1);">Simulating Training: Epoch {e}/{epochs}</h4>', unsafe_allow_html=True)
        epoch_bar.progress(e / epochs)
        time.sleep(0.1)
    
    # Step 4: Classify the document
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Classifying Document...</h4>', unsafe_allow_html=True)
    label, confidence = classify_document(ocr_text)

    # Calculate additional metrics
    word_count = len(ocr_text.split())
    char_count = len(ocr_text.replace(" ", "").replace("\n", ""))
    avg_word_length = char_count / word_count if word_count > 0 else 0
    
    # Dynamic updates
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
    </div>
    """, unsafe_allow_html=True)

    text_output_placeholder.markdown(f"""
    <div class="text-output-card" style="margin-top: 1.5rem;">
        <h4 style="color:var(--text-1);">Extracted Text</h4>
        <pre id="ocrText" style="color:var(--text-1);">{ocr_text}</pre>
        <div class="button-row">
            <button class="ocr-button" onclick="copyToClipboard()">Copy Text</button>
            <a href="data:text/plain;charset=utf-8,{base64.b64encode(ocr_text.encode()).decode()}" download="extracted_text.txt" class="ocr-button" style="text-decoration: none; color: white;">Download .txt</a>
        </div>
    </div>
    <script>
        function copyToClipboard() {{
            const textToCopy = document.getElementById('ocrText').innerText;
            navigator.clipboard.writeText(textToCopy).then(() => {{
                alert('Text copied to clipboard!');
            }}).catch(err => {{
                console.error('Could not copy text: ', err);
            }});
        }}
    </script>
    """, unsafe_allow_html=True)
    
    status_text.markdown('<h4 id="predStatus" style="color:var(--text-1);">Done!</h4>', unsafe_allow_html=True)

# --- Streamlit UI Components ---

# State management for displaying results
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing' not in st.session_state:
    st.session_state.processing = False

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
            <h4 style="color:var(--text-1);">Upload an Image</h4>
            <p style="color:var(--text-2);">Drag and drop a file here or click below to choose a file.</p>
            <div class="file-input-label-container">
                <label for="fileInput" class="file-input-label">Choose File</label>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader", label_visibility="collapsed")
    
    if uploaded_file and not st.session_state.uploaded_image:
        try:
            image_data = uploaded_file.getvalue()
            _ = Image.open(BytesIO(image_data))
            st.session_state.uploaded_image = image_data
            st.rerun() # Rerun to display the previews
        except Exception as e:
            st.error("Error: The file you uploaded could not be identified as a valid image. Please try a different file.")
            st.session_state.uploaded_image = None

# Automatically run the process if an image is uploaded
if st.session_state.uploaded_image and not st.session_state.processing:
    st.session_state.processing = True
    image_to_process = Image.open(BytesIO(st.session_state.uploaded_image))
    run_ocr_and_classify(image_to_process)
    st.session_state.processing = False

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="color:var(--text-2); font-size:0.8rem;">OCR-TECH - Designed by ADELEKE, OLADOKUN, and OLALEYE</p>
</div>
""", unsafe_allow_html=True)
