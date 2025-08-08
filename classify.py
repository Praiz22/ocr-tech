import pytesseract
import numpy as np
import cv2

def _ocr_text_from_binary(bin_img):
    try:
        text = pytesseract.image_to_string(bin_img)
    except Exception:
        text = ""
    return text or ""

def _edge_density(gray_img):
    edges = cv2.Canny(gray_img, 50, 150)
    return float(np.count_nonzero(edges)) / (gray_img.shape[0] * gray_img.shape[1])

def _color_variance(img_bgr):
    img_lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return float(np.var(img_lab) / 255.0)

def _text_pixel_ratio(bin_img):
    text_pixels = np.count_nonzero(bin_img == 0)
    return float(text_pixels) / (bin_img.shape[0] * bin_img.shape[1])

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

    # Improved category logic: text-based keyword search first
    cat = _keyword_category(ocr_text)
    if not cat:
        # Fall back to heuristics if no keyword match
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
    return smart_classify(img_bgr, processed_bin)
