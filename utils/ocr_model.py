# utils/ocr_model.py
import torch
import streamlit as st
import numpy as np

def load_ocr_model(model_path):
    try:
        model = torch.load(model_path, weights_only=False, map_location=torch.device('cpu'))
        model.eval()
        return model
    except Exception as e:
        st.warning(f"Model loading failed: {str(e)}. Using dummy model.")
        class DummyModel(torch.nn.Module):
            def forward(self, x):
                return torch.randn(1, 3), torch.randn(1, 10)  # Classification + OCR
        return DummyModel()

def classify_image(model, image, return_confidence=False):
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        class_logits, _ = model(image)
        probs = torch.softmax(class_logits, dim=1)
        confidence, class_idx = torch.max(probs, dim=1)
        class_label = ["pictures", "screenshots", "documents"][class_idx.item()]

    return (class_label, confidence.item()) if return_confidence else class_label

def predict_text(model, image):
    if len(image.shape) == 2:
        image = image[..., np.newaxis]
    image = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)

    with torch.no_grad():
        _, ocr_logits = model(image)
        extracted_text = "Sample extracted text from OCR"  # Replace with CTC decoding
        confidence = 0.95
    return extracted_text, confidence
