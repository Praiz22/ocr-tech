import pytesseract
import numpy as np
import cv2

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
    if "invoice" in text:
        return "Invoice"
    if "receipt" in text:
        return "Receipt"
    if "form" in text:
        return "Form"
    if "exam" in text or "score" in text:
        return "Exam/Result"
    if "screenshot" in text or "screen" in text:
        return "Screenshot"
    if "note" in text or "handwriting" in text:
        return "Handwritten Note"
    if "photo" in text:
        return "Photograph"
    return None

def smart_classify(img_bgr, processed_bin):
    features, ocr_text = extract_features(img_bgr, processed_bin)
    color_var, edge_d, text_pixels, aspect, text_ratio, text_len = features

    score = 0.0
    if text_len > 100 or text_pixels > 0.008 or text_ratio > 0.0008:
        score += 0.6
    if color_var < 0.25:
        score += 0.15
    if edge_d > 0.015 and text_len > 20:
        score += 0.10
    if color_var > 0.45 and text_len < 30:
        score += 0.05

    cat = _keyword_category(ocr_text)
    if not cat:
        if (text_len > 100 or text_pixels > 0.012):
            cat = "Text Document"
        elif (text_len > 20 and edge_d > 0.018):
            cat = "Screenshot / UI"
        elif (color_var > 0.4 and text_len < 30 and edge_d < 0.015):
            cat = "Photograph"
        elif (text_len > 0 and text_len <= 20 and color_var < 0.3):
            cat = "Scanned Note / Small Text"
        else:
            if text_pixels > 0.006:
                cat = "Text Document"
            else:
                cat = "Other / Image"
    score = min(1.0, score)

    ml_label, ml_conf = rf_predict(features) if SKLEARN_AVAILABLE else (None, None)
    if ml_label and ml_conf and ml_conf > 0.80:
        final_cat = f"ML: {ml_label}"
        final_conf = ml_conf
    else:
        final_cat = cat
        final_conf = score

    return {
        "category": final_cat,
        "score": final_conf,
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
