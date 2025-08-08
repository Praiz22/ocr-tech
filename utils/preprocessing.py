# utils/preprocessing.py
import cv2
import numpy as np

def preprocess_image(img_bgr):
    """
    Take BGR image, return cleaned binary image for OCR (uint8).
    """
    # convert to gray
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    # resize if huge (speeds up)
    max_dim = 1600
    h, w = gray.shape
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        gray = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_AREA)

    # denoise
    den = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    # adaptive threshold + morphological open to remove small blobs
    th = cv2.threshold(den, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,1))
    opened = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel)
    return opened

def prepare_for_display(pil_img):
    """Return a PIL image resized nicely for Streamlit display (keeps aspect ratio)."""
    max_width = 900
    w, h = pil_img.size
    if w > max_width:
        ratio = max_width / w
        return pil_img.resize((int(w*ratio), int(h*ratio)))
    return pil_img
