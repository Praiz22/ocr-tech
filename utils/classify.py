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
