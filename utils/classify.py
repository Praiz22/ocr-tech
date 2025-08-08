# utils/classify.py
import pytesseract
import numpy as np
import cv2
# utils/classify.py
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np

mobilenet_model = MobileNetV2(weights='imagenet')

def dl_classify(img_bgr):
    # Resize to 224x224 for MobileNetV2, convert to array
    img_resized = cv2.resize(img_bgr, (224,224))
    arr = img_to_array(img_resized)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = mobilenet_model.predict(arr)
    decoded = decode_predictions(preds, top=3)[0]
    # Return top prediction
    return [(c[1], float(c[2])) for c in decoded]

def smart_classify(img_bgr, processed_bin):
    # ...existing heuristics...
    result = original_heuristics(img_bgr, processed_bin)
    # Deep model prediction
    top_preds = dl_classify(img_bgr)
    result['dl_pred'] = top_preds[0][0]
    result['dl_conf'] = top_preds[0][1]
    # Optionally adjust category based on DL confidence
    if result['dl_conf'] > 0.7:
        result['category'] = result['dl_pred']
        result['score'] = result['dl_conf']
    return result

def _ocr_text_from_binary(bin_img):
    """Run Tesseract OCR on a binary/preprocessed image and return the text."""
    try:
        text = pytesseract.image_to_string(bin_img)
    except Exception:
        text = ""
    return text or ""

def _edge_density(gray_img):
    """Return density of edges (Canny edges / total pixels)."""
    edges = cv2.Canny(gray_img, 50, 150)
    return float(np.count_nonzero(edges)) / (gray_img.shape[0] * gray_img.shape[1])

def _color_variance(img_bgr):
    """Quick proxy for 'photo vs document' — photos have higher color variance."""
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(np.var(img_lab) / 255.0)

def _text_pixel_ratio(bin_img):
    """Estimate ratio of dark pixels (text-like) to total."""
    # bin_img expected binary (0 or 255)
    text_pixels = np.count_nonzero(bin_img == 0)
    return float(text_pixels) / (bin_img.shape[0] * bin_img.shape[1])

def smart_classify(img_bgr, processed_bin):
    """
    img_bgr: Original BGR image (cv2)
    processed_bin: Preprocessed binary image suitable for OCR (0/255 grayscale)
    Returns: Dictionary with category, score, and metrics.
    """
    h, w = processed_bin.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    ocr_text = _ocr_text_from_binary(processed_bin)
    tlen = len(ocr_text.strip())
    tratio = tlen / (w * h + 1e-6)

    edge_d = _edge_density(gray)
    color_var = _color_variance(img_bgr)
    text_pixel_r = _text_pixel_ratio(processed_bin)
    aspect = w / (h + 1e-6)

    # Heuristic scoring
    score = 0.0
    if tlen > 100 or text_pixel_r > 0.008 or tratio > 0.0008:
        score += 0.6
    if color_var < 0.25:
        score += 0.15
    if edge_d > 0.015 and tlen > 20:
        score += 0.10
    if color_var > 0.45 and tlen < 30:
        score += 0.05

    # Choose category
    cat = "Other / Image"
    if (tlen > 100 or text_pixel_r > 0.012):
        cat = "Text Document"
    elif (tlen > 20 and edge_d > 0.018):
        cat = "Screenshot / UI"
    elif (color_var > 0.4 and tlen < 30 and edge_d < 0.015):
        cat = "Photograph"
    elif (tlen > 0 and tlen <= 20 and color_var < 0.3):
        cat = "Scanned Note / Small Text"
    else:
        if text_pixel_r > 0.006:
            cat = "Text Document"
        else:
            cat = "Other / Image"

    score = min(1.0, score)
    return {
        "category": cat,
        "score": score,
        "text_ratio": tratio,
        "edge_density": edge_d,
        "color_variance": color_var,
        "text_pixels_ratio": text_pixel_r,
        "width": w,
        "height": h,
        "text": ocr_text,
        "aspect_ratio": aspect
    }

def classify_text(img_bgr, processed_bin):
    """
    Return the full classification dictionary (category, score, metrics).
    This ensures streamlit_app.py can access category, score, and metrics.
    """
    return smart_classify(img_bgr, processed_bin)
