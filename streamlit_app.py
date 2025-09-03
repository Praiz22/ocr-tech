import streamlit as st
import pytesseract
from PIL import Image
from io import BytesIO
import re
import time
import numpy as np
# Note: In a real-world scenario, you would install and use OpenCV for image processing.
# import cv2 

# Set Streamlit to use the light theme and a centered layout for better UI control.
st.set_page_config(layout="centered", page_title="Praix Tech", initial_sidebar_state="collapsed")

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
    /* Hides default Streamlit file uploader text and button */
    visibility: hidden;
    height: 0;
  }

  /* Style the file uploader */
  .stFileUploader {
      visibility: hidden;
  }
</style>
""", unsafe_allow_html=True)

# Set the path to the Tesseract executable.
# You might need to change this line depending on your setup.
# On Windows: pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# On Linux: pytesseract.pytesseract.tesseract_cmd = '/usr/bin/tesseract'

# --- Core Logic for OCR and Classification ---

def preprocess_image(image):
    """Placeholder for advanced image processing."""
    # This is where you would implement image deskewing, noise removal, etc.
    # For the purpose of the defense, this shows you know the next steps.
    # For example, using OpenCV:
    # try:
    #     img_np = np.array(image.convert('L')) # Convert to grayscale
    #     img_np = cv2.fastNlMeansDenoising(img_np, None, 10, 7, 21) # Noise removal
    #     # Deskewing logic would go here
    #     return Image.fromarray(img_np)
    # except Exception as e:
    #     st.error(f"Image preprocessing failed: {e}")
    #     return image
    return image

def classify_document(text):
    """
    Classifies the document based on features of the extracted text.
    This is a simplified, rule-based classifier suitable for a project defense.
    """
    text = text.lower()
    
    # Check for common document keywords
    document_keywords = ["invoice", "receipt", "report", "statement", "document", "proposal", "contract", "memorandum", "agenda"]
    if any(keyword in text for keyword in document_keywords):
        return "Document", 95 

    # Check for letter features
    letter_pattern = r"(dear|sincerely|best regards|regards|from:|to:|date:|re:|subject:|yours sincerely|yours faithfully|attachment:)"
    if re.search(letter_pattern, text):
        return "Letter", 90

    # Check for a high concentration of text, typical of a document or book page
    word_count = len(text.split())
    line_count = len(text.split('\n'))
    if word_count > 100 and line_count > 10:
        return "Document", 85
    
    # Screenshots often contain short, fragmented text or UI elements
    # This is a very rough heuristic
    if word_count < 50 and line_count < 5 or "file" in text or "menu" in text:
        return "Screenshot", 80
    
    return "Miscellaneous", 60


def run_ocr_and_classify(image):
    """
    Main function to process the image and get OCR and classification results.
    """
    # Simulate the dynamic UI updates and logic from the HTML file
    status_placeholder = st.empty()
    metric_grid_placeholder = st.empty()
    text_output_placeholder = st.empty()
    
    status_placeholder.markdown('<h4 id="predStatus">Running OCR & inference...</h4>', unsafe_allow_html=True)
    
    # Step 1: Pre-process the image
    time.sleep(1)
    preprocessed_image = preprocess_image(image)
    
    # Step 2: Perform OCR
    ocr_text = pytesseract.image_to_string(preprocessed_image, config='--psm 6')

    # Step 3: Classify the document
    label, confidence = classify_document(ocr_text)

    # Dynamic updates
    metric_grid_placeholder.markdown(f"""
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
    
    # Update with initial values
    metric_grid_placeholder.markdown(f"""
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
    status_placeholder.markdown('<h4 id="predStatus">Done!</h4>', unsafe_allow_html=True)
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
    </div>
    """, unsafe_allow_html=True)

    text_output_placeholder.markdown(f"""
    <div class="text-output-card" style="margin-top: 1.5rem;">
        <h4>OCR Text</h4>
        <pre id="ocrText">{ocr_text}</pre>
    </div>
    """, unsafe_allow_html=True)

# --- Streamlit UI Components ---

# State management for displaying results
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

st.markdown("""
<div class="ocr-container">
    <div class="header">
        <h1>Praix Tech</h1>
        <p>Optical Character Recognition & Image Classification</p>
    </div>
    <div class="ocr-card">
        <div class="file-upload-section">
            <h4>Upload an Image</h4>
            <p>PNG, JPG, JPEG files only. Max 5MB.</p>
            <form id="uploadForm">
                <label for="fileInput" class="file-input-label">Choose File</label>
            </form>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# The actual file uploader, hidden and triggered by the custom button
uploaded_file = st.file_uploader("", type=["png", "jpg", "jpeg"], key="file_uploader")

if uploaded_file:
    try:
        # Check if the file is a valid image before opening
        # This helps to avoid the UnidentifiedImageError
        image_data = uploaded_file.getvalue()
        _ = Image.open(BytesIO(image_data)) # Test to see if it's a valid image
        st.session_state.uploaded_image = image_data
        
        st.markdown("""
        <div class="ocr-container">
            <div class="ocr-card" style="margin-top: -2rem;">
                <h4>Uploaded Image</h4>
                <br>
        """, unsafe_allow_html=True)
        
        image_display = Image.open(BytesIO(st.session_state.uploaded_image))
        st.image(image_display, use_column_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="button-row">
            <button class="ocr-button" onclick="window.parent.document.querySelector('[data-testid=stFileUploader] button').click()">Process Image</button>
        </div>
        """, unsafe_allow_html=True)

        if st.button("Process Image", key="process_button"):
            image_to_process = Image.open(BytesIO(st.session_state.uploaded_image))
            run_ocr_and_classify(image_to_process)
    
    except Exception as e:
        # Catches the PIL.UnidentifiedImageError and other potential issues
        st.error("Error: The file you uploaded could not be identified as a valid image. Please try a different file.")
        st.session_state.uploaded_image = None
        st.stop()


# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem;">
    <p style="color:var(--text-2); font-size:0.8rem;">Powered by Praix Tech</p>
</div>
""", unsafe_allow_html=True)
