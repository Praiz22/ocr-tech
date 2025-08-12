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
        img_bgr (np.array): The original BGR image.
        processed_bin (np.array): The preprocessed binary image from OCR.
        ocr_text (str): The text extracted from the image.
        
    Returns:
        dict: A dictionary containing all the extracted features.
    """
    height, width = processed_bin.shape[:2] # Ensure we handle grayscale and color
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # Heuristic features
    # Note: `color_var` is computed on the BGR image, not the processed binary one.
    color_var = float(np.var(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)) / 255.0)
    edge_d = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / (height * width + 1e-6)
    text_pixels = float(np.count_nonzero(processed_bin == 0)) / (height * width + 1e-6)
    aspect_ratio = float(width) / (height + 1e-6)
    
    # OCR features
    # We now use the provided ocr_text directly.
    text_len = len(re.findall(r'\b\w+\b', ocr_text))
    text_ratio = len(ocr_text.strip()) / (height * width + 1e-6)

    return {
        "text_len": text_len,
        "text_ratio": text_ratio,
        "edge_density": edge_d,
        "color_variance": color_var,
        "text_pixels_ratio": text_pixels,
        "aspect_ratio": aspect_ratio,
        "width": width,
        "height": height,
        "text": ocr_text
    }

def classify_text(img_bgr, processed_bin, ocr_text):
    """
    Classifies an image based on extracted features using either a machine learning
    model (if available) or a set of heuristic rules.

    Args:
        img_bgr (np.array): The original BGR image.
        processed_bin (np.array): The preprocessed binary image.
        ocr_text (str): The text extracted from the image.

    Returns:
        dict: A dictionary with the classification result, including category,
              confidence score, and all extracted features.
    """
    features = extract_features(img_bgr, processed_bin, ocr_text)

    # Use the ML model if available
    ml_label, ml_conf = None, None
    if SKLEARN_AVAILABLE and _rf_model:
        # Prepare feature vector for the model
        feature_vector = np.array([
            features["text_ratio"],
            features["edge_density"],
            features["color_variance"],
            features["text_pixels_ratio"],
            features["aspect_ratio"]
        ]).reshape(1, -1)
        
        try:
            prediction = _rf_model.predict(feature_vector)[0]
            ml_label = _rf_label_map[prediction]
            ml_conf = np.max(_rf_model.predict_proba(feature_vector)[0])
        except Exception:
            pass  # Fallback to heuristic if ML model fails

    # Heuristic classification
    heuristic_category = "Other / Image"
    heuristic_score = 0.5
    
    if features["text_len"] > 200 and features["text_pixels_ratio"] > 0.01:
        heuristic_category = "Text Document"
        heuristic_score = 0.9
    elif features["text_len"] > 50 and features["edge_density"] > 0.02 and features["color_variance"] < 0.4:
        heuristic_category = "Screenshot / UI"
        heuristic_score = 0.8
    elif features["color_variance"] > 0.45 and features["text_len"] < 30 and features["edge_density"] < 0.01:
        heuristic_category = "Photograph"
        heuristic_score = 0.95
    elif features["text_len"] > 0 and features["text_len"] <= 80 and features["color_variance"] < 0.35:
        heuristic_category = "Scanned Note / Small Text"
        heuristic_score = 0.75
    else:
        if features["text_pixels_ratio"] > 0.005:
            heuristic_category = "Text Document"
            heuristic_score = 0.6
        else:
            heuristic_category = "Other / Image"
            heuristic_score = 0.4
    
    final_result = {
        "category": ml_label or heuristic_category,
        "score": ml_conf or min(1.0, heuristic_score),
        "ml_label": ml_label,
        "ml_conf": ml_conf,
        **features # Add all extracted features to the final result dictionary
    }

    return final_result
