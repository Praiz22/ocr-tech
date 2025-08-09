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
    global _rf_model, _rf_label_map
    _rf_model = model
    _rf_label_map = label_map

def extract_features(img_bgr, processed_bin):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    color_var = float(np.var(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)) / 255.0)
    edge_d = float(np.count_nonzero(cv2.Canny(gray, 50, 150))) / (gray.shape[0] * gray.shape[1])
    text_pixels = float(np.count_nonzero(processed_bin == 0)) / (processed_bin.shape[0] * processed_bin.shape[1])
    aspect = float(processed_bin.shape[1]) / (processed_bin.shape[0] + 1e-6)
    ocr_text = pytesseract.image_to_string(processed_bin)
    text_len = len(ocr_text.strip())
    text_ratio = text_len / (processed_bin.shape[0] * processed_bin.shape[1] + 1e-6)
    return np.array([color_var, edge_d, text_pixels, aspect, text_ratio, text_len]), ocr_text

def rf_predict(features):
    if _rf_model is None or _rf_label_map is None:
        return None, None
    pred = _rf_model.predict([features])[0]
    proba = np.max(_rf_model.predict_proba([features]))
    label = _rf_label_map.get(pred, str(pred))
    return label, proba

def _keyword_category(text):
    text = text.lower()
    if re.search(r'\b(invoice|bill|receipt|statement)\b', text):
        return "Invoice/Receipt"
    if re.search(r'\b(form|application|document)\b', text):
        return "Form/Document"
    if re.search(r'\b(exam|test|result|score)\b', text):
        return "Exam/Result"
    if re.search(r'\b(screenshot|screen|ui)\b', text):
        return "Screenshot/UI"
    if re.search(r'\b(note|handwriting)\b', text):
        return "Handwritten Note"
    if re.search(r'\b(photo|picture|photograph)\b', text):
        return "Photograph"
    return None

def smart_classify(img_bgr, processed_bin):
    features, ocr_text = extract_features(img_bgr, processed_bin)
    color_var, edge_d, text_pixels, aspect, text_ratio, text_len = features
    
    # Prioritize ML model if available and confident
    ml_label, ml_conf = (None, None)
    if SKLEARN_AVAILABLE:
        ml_label, ml_conf = rf_predict(features)
        if ml_label and ml_conf and ml_conf > 0.85:
            return {
                "category": f"ML: {ml_label}",
                "score": ml_conf,
                "text": ocr_text,
                "ml_label": ml_label,
                "ml_conf": ml_conf,
                "heuristic_category": None,
                "heuristic_score": None,
                "text_ratio": text_ratio,
                "edge_density": edge_d,
                "color_variance": color_var,
                "text_pixels_ratio": text_pixels,
                "width": processed_bin.shape[1],
                "height": processed_bin.shape[0],
                "aspect_ratio": aspect,
            }

    # Heuristic-based classification
    cat = _keyword_category(ocr_text)
    score = 0.0

    if cat:
        score = 0.8
        if text_len > 100:
            score = 0.95
    else:
        # Based on image features
        if text_len > 150 and text_pixels > 0.01:
            cat = "Text Document"
            score = 0.85
        elif text_len > 30 and edge_d > 0.02:
            cat = "Screenshot / UI"
            score = 0.75
        elif color_var > 0.45 and text_len < 30 and edge_d < 0.01:
            cat = "Photograph"
            score = 0.9
        elif text_len > 0 and text_len <= 50 and color_var < 0.3:
            cat = "Scanned Note / Small Text"
            score = 0.7
        else:
            if text_pixels > 0.005:
                cat = "Text Document"
                score = 0.5
            else:
                cat = "Other / Image"
                score = 0.3

    return {
        "category": cat,
        "score": min(1.0, score),
        "text_ratio": text_ratio,
        "edge_density": edge_d,
        "color_variance": color_var,
        "text_pixels_ratio": text_pixels,
        "width": processed_bin.shape[1],
        "height": processed_bin.shape[0],
        "text": ocr_text,
        "aspect_ratio": aspect,
        "ml_label": ml_label,
        "ml_conf": ml_conf,
        "heuristic_category": cat,
        "heuristic_score": score
    }

def classify_text(img_bgr, processed_bin):
    return smart_classify(img_bgr, processed_bin)
