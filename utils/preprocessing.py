import cv2
import numpy as np

def preprocess_image(img_bgr):
    """
    Applies image preprocessing steps for better OCR results.
    """
    if img_bgr is None or img_bgr.size == 0:
        raise ValueError("Input image is empty.")

    # Convert to grayscale
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Adaptive Thresholding to handle varying lighting conditions
    processed_img = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )

    return processed_img
