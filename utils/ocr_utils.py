# utils/ocr_utils.py
import cv2
import numpy as np
import pytesseract
import math
from pytesseract import Output

def _ocr_and_score(img, psm=6, lang='eng'):
    config = f'--oem 3 --psm {psm}'
    try:
        data = pytesseract.image_to_data(img, config=config, lang=lang, output_type=Output.DICT)
    except Exception:
        text = pytesseract.image_to_string(img, config=config, lang=lang)
        return text, -1.0, 0, {}
    words, confs = [], []
    for i, w in enumerate(data.get('text', [])):
        if w.strip():
            try:
                c = float(data['conf'][i])
            except:
                c = -1.0
            if c != -1.0:
                words.append(w.strip())
                confs.append(c)
    text = " ".join(words)
    avg_conf = float(np.mean(confs)) if confs else -1.0
    return text, avg_conf, len(words), data

def _clahe(gray):
    return cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8)).apply(gray)

def _unsharp(gray):
    blur = cv2.GaussianBlur(gray, (0,0), sigmaX=3)
    return cv2.addWeighted(gray, 1.5, blur, -0.5, 0)

def _adaptive_thresh(gray, block=31, C=2):
    return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, C)

def _otsu_thresh(gray):
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def _binarize_variant(img_bgr, variant_id):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if variant_id == 0:
        return _adaptive_thresh(_clahe(gray), 35, 3)
    if variant_id == 1:
        return _otsu_thresh(cv2.medianBlur(gray, 3))
    if variant_id == 2:
        b = cv2.bilateralFilter(gray, 9, 75, 75)
        return _adaptive_thresh(_clahe(b), 25, 2)
    if variant_id == 3:
        return _adaptive_thresh(_unsharp(gray), 31, 2)
    if variant_id == 4:
        h, w = gray.shape
        up = cv2.resize(gray, (w*2, h*2), interpolation=cv2.INTER_CUBIC)
        return _adaptive_thresh(_clahe(up), 25, 2)
    if variant_id == 5:
        return cv2.bitwise_not(_adaptive_thresh(_clahe(gray), 31, 2))
    return _otsu_thresh(gray)

def _deskew(img_bin):
    coords = np.column_stack(np.where(img_bin > 0))
    if len(coords) < 10:
        return img_bin
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return img_bin
    h, w = img_bin.shape
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    return cv2.warpAffine(img_bin, M, (w, h),
                          flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

def _score(avg_conf, wc):
    if avg_conf < 0:
        return math.log1p(wc) * 0.1
    return avg_conf * (1.0 + 0.02 * math.log1p(wc))

def ocr_ensemble(img_bgr, languages='eng', psm_list=(3, 6, 11), variants=(0,1,2,3,4,5)):
    best = None
    for variant in variants:
        proc = _deskew(_binarize_variant(img_bgr, variant))
        for psm in psm_list:
            text, conf, wc, _ = _ocr_and_score(proc, psm, languages)
            score = _score(conf, wc)
            if best is None or score > best['score']:
                best = {'text': text, 'conf': conf, 'wc': wc,
                        'variant': variant, 'psm': psm,
                        'score': score, 'proc_img': proc}
    return best['text'], best['proc_img'], best
