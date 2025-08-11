import pytesseract
import numpy as np
import cv2
import re

try:
    from sklearn.ensemble import RandomForestClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

_rf_model = None
_rf_label_map = None

def set_rf_model(model, label_map):
    """Sets the ML model and label map for classification."""
    global _rf_model, _rf_label_map
    _rf_model = model
    _rf_label_map = label_map

def extract_features(img_bgr, processed_bin, ocr_text):
    """
    Extracts a variety of features from the image for classification.

    Args:
        img_bgr (numpy.ndarray): The original BGR color image.
        processed_bin (numpy.ndarray): The preprocessed binary image.
        ocr_text (str): The text extracted by the OCR engine.

    Returns:
        dict: A dictionary of extracted features.
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Heuristic features
    color_var = float(np.var(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)) / 255.0)
    edge_d = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / (gray.shape[0] * gray.shape[1])
    text_pixels = float(np.count_nonzero(processed_bin == 0)) / (processed_bin.shape[0] * processed_bin.shape[1])
    aspect = float(processed_bin.shape[1]) / (processed_bin.shape[0] + 1e-6)
    
    # OCR features
    text_len = len(re.sub(r'[^a-zA-Z0-9]', '', ocr_text))
    text_ratio = text_len / (processed_bin.shape[0] * processed_bin.shape[1] + 1e-6)
    
    return {
        "text_ratio": text_ratio,
        "edge_density": edge_d,
        "color_variance": color_var,
        "text_pixels_ratio": text_pixels,
        "aspect_ratio": aspect,
        "text_length": text_len,
        "width": processed_bin.shape[1],
        "height": processed_bin.shape[0],
        "text": ocr_text,
    }

def classify_text(img_bgr, processed_bin, ocr_text):
    """
    Classifies an image based on extracted features and OCR text.
    
    This function now takes the extracted text as a parameter, avoiding redundant OCR.
    
    Args:
        img_bgr (numpy.ndarray): The original BGR color image.
        processed_bin (numpy.ndarray): The preprocessed binary image.
        ocr_text (str): The text extracted by the OCR engine.

    Returns:
        dict: A dictionary containing the classification result and scores.
    """
    # Extract features using the new function
    features = extract_features(img_bgr, processed_bin, ocr_text)
    
    text_len = features['text_length']
    text_ratio = features['text_ratio']
    edge_d = features['edge_density']
    color_var = features['color_variance']
    text_pixels = features['text_pixels_ratio']

    # Initialize a placeholder for the result
    heuristic_category = "Unknown"
    heuristic_score = 0.0
    ml_label = None
    ml_conf = 0.0

    # --- ML-based Classification (if model is available) ---
    if SKLEARN_AVAILABLE and _rf_model:
        try:
            # Reshape features for the model
            input_features = np.array([[text_len, text_ratio, edge_d, color_var, text_pixels]])
            
            ml_pred_idx = _rf_model.predict(input_features)[0]
            ml_proba = _rf_model.predict_proba(input_features)[0]

            ml_label = _rf_label_map[ml_pred_idx]
            ml_conf = ml_proba[ml_pred_idx]
            
            # Use ML result as the primary classification
            heuristic_category = ml_label
            heuristic_score = ml_conf
        except Exception:
            # Fallback to heuristics if ML model fails
            pass
    
    # --- Heuristic-based Classification (as a fallback or primary if no ML model) ---
    if not _rf_model or (ml_conf < 0.6 and not ocr_text.strip()):
        # Based on image features and text presence
        if text_len > 200 and text_pixels > 0.01:
            heuristic_category = "Text Document"
            heuristic_score = 0.9
        elif text_len > 50 and edge_d > 0.02 and color_var < 0.4:
            heuristic_category = "Screenshot / UI"
            heuristic_score = 0.8
        elif color_var > 0.45 and text_len < 30 and edge_d < 0.01:
            heuristic_category = "Photograph"
            heuristic_score = 0.95
        elif text_len > 0 and text_len <= 80 and color_var < 0.35:
            heuristic_category = "Scanned Note / Small Text"
            heuristic_score = 0.75
        else:
            if text_pixels > 0.005:
                heuristic_category = "Text Document"
                heuristic_score = 0.6
            else:
                heuristic_category = "Other / Image"
                heuristic_score = 0.4

    # Combine all results into a single dictionary
    result = {
        "category": heuristic_category,
        "score": min(1.0, heuristic_score),
        "text_ratio": features['text_ratio'],
        "edge_density": features['edge_density'],
        "color_variance": features['color_variance'],
        "text_pixels_ratio": features['text_pixels_ratio'],
        "aspect_ratio": features['aspect_ratio'],
        "width": features['width'],
        "height": features['height'],
        "text": features['text'],
    }

    if ml_label:
        result['ml_label'] = ml_label
        result['ml_conf'] = ml_conf

    return result
