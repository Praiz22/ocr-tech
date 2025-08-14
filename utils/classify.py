import cv2
import numpy as np
import joblib

# Global variables for the ML model and its labels
rf_model = None
rf_label_map = None
REVERSE_ML_LABEL_MAP = {}

def set_rf_model(model, label_map):
    """Sets the global RF model and label map if they exist."""
    global rf_model, rf_label_map, REVERSE_ML_LABEL_MAP
    rf_model = model
    rf_label_map = label_map
    if rf_label_map:
        REVERSE_ML_LABEL_MAP = {v: k for k, v in rf_label_map.items()}

def _get_classification_features(img_cv, processed_img, ocr_text):
    """
    Calculates a set of features from the image and OCR text for classification.
    """
    h, w, _ = img_cv.shape
    aspect_ratio = w / h
    
    # Text-based features
    text_len = len(ocr_text.split())
    # This is a better way to calculate text_pixels_ratio as it uses the pre-processed binary image
    text_pixels_ratio = np.sum(processed_img > 0) / (h * w) if (h * w) > 0 else 0
    
    # Image-based features
    edges = cv2.Canny(processed_img, 100, 200)
    edge_density = np.sum(edges > 0) / (h * w) if (h * w) > 0 else 0
    color_variance = np.var(img_cv)
    
    return {
        'text_len': text_len,
        'text_pixels_ratio': text_pixels_ratio,
        'edge_density': edge_density,
        'color_variance': color_variance,
        'aspect_ratio': aspect_ratio,
        'width': w,
        'height': h
    }

def _classify_with_ml(features_dict):
    """
    Uses the pre-trained ML model to classify the image based on features.
    Returns (prediction_label, confidence) or (None, 0.0) if no model is found.
    """
    global rf_model, REVERSE_ML_LABEL_MAP
    if rf_model is None:
        return None, 0.0
    
    try:
        # Create a feature array in the correct order for the model
        features = [
            features_dict['text_len'],
            features_dict['text_pixels_ratio'],
            features_dict['edge_density'],
            features_dict['color_variance'],
            features_dict['aspect_ratio']
        ]
        
        prediction_index = rf_model.predict([features])[0]
        prediction_proba = rf_model.predict_proba([features])[0]
        
        # Get the label and confidence
        ml_label = REVERSE_ML_LABEL_MAP.get(prediction_index)
        ml_conf = np.max(prediction_proba)
        
        return ml_label, ml_conf
    except Exception as e:
        print(f"Error during ML classification: {e}")
        return None, 0.0

def _classify_with_heuristics(features_dict):
    """
    Uses a simple rule-based system to classify the document.
    """
    score = 0.0
    category = "Unknown"
    
    # These are simplified rules. Adjust them based on your specific data.
    if features_dict['text_pixels_ratio'] > 0.02 and features_dict['edge_density'] > 0.015:
        category = "Invoice"
        score = 0.8
    elif features_dict['text_len'] > 50 and features_dict['color_variance'] < 500:
        category = "Form"
        score = 0.7
    elif features_dict['text_len'] > 5:
        category = "Document"
        score = 0.6
    else:
        category = "Image"
        score = 0.5
        
    return category, score

def classify_text(img_cv, processed_img, ocr_text):
    """
    Main classification function. It uses a combination of ML and heuristics.
    """
    features = _get_classification_features(img_cv, processed_img, ocr_text)

    # First, try to classify with the ML model
    ml_label, ml_conf = _classify_with_ml(features)
    
    # Fallback to heuristics if ML fails or isn't available
    if ml_label:
        category = ml_label
        score = ml_conf
    else:
        category, score = _classify_with_heuristics(features)
    
    # Combine all results into a single dictionary
    result = {
        'text': ocr_text,
        'category': category,
        'score': score,
        'ml_label': ml_label,
        'ml_conf': ml_conf,
    }
    result.update(features)
    
    return result
