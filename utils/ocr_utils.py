# utils.py
import cv2
import numpy as np

def preprocess_image(image):
    # Full preprocessing pipeline (called if needed, but in app we do step-by-step for UI)
    img = cv2.medianBlur(image, 5)  # Noise removal
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    deskewed = deskew_image(thresh)
    normalized = cv2.resize(deskewed, (224, 224)) / 255.0
    return normalized

def deskew_image(image):
    # Simple deskew logic (find contours, angle, rotate)
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated
