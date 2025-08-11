# utils/ocr_utils.py
import cv2
import numpy as np
import pytesseract
import math
from pytesseract import Output

def _ocr_and_score(img, psm=6, lang='eng'):
    """
    Run tesseract on img and compute a score from confidences & word count.
    Returns (text, avg_conf, word_count, data_dict)
    """
    config = f'--oem 3 --psm {psm}'
    try:
        data = pytesseract.image_to_data(img, config=config, lang=lang, output_type=Output.DICT)
    except Exception:
        # fallback to plain string if image_to_data fails
        text = pytesseract.image_to_string(img, config=config, lang=lang)
        return text, -1.0, 0, {}
    # join text
    words = []
    confs = []
    n = len(data.get('text', []))
    for i in range(n):
        w = (data['text'][i] or "").strip()
        try:
            c = float(data['conf'][i])
        except Exception:
            c = -1.0
        if w != "" and c != -1.0:
            words.append(w)
            confs.append(c)
    text = " ".join(words)
    word_count = len(words)
    avg_conf = float(np.mean(confs)) if len(confs) > 0 else -1.0
    return text, avg_conf, word_count, data

def _clahe(img_gray):
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    return clahe.apply(img_gray)

def _unsharp(img_gray):
    blur = cv2.GaussianBlur(img_gray, (0,0), sigmaX=3)
    unsharp = cv2.addWeighted(img_gray, 1.5, blur, -0.5, 0)
    return np.clip(unsharp, 0, 255).astype(np.uint8)

def _adaptive_thresh(img_gray, block=31, C=2):
    return cv2.adaptiveThreshold(img_gray, 255,
                                 cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, block, C)

def _otsu_thresh(img_gray):
    _, th = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return th

def _binarize_variant(img_bgr, variant_id):
    """
    Return a single-channel binary/contrast-enhanced image depending on variant_id.
    variant_id options:
      0: CLAHE -> adaptive thresh
      1: Median blur -> Otsu
      2: Bilateral -> adaptive
      3: Unsharp + adaptive
      4: Upscaled CLAHE + adaptive (for small text)
      5: Inverted CLAHE+adaptive (some printed white-on-dark)
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    if variant_id == 0:
        c = _clahe(gray)
        return _adaptive_thresh(c, block=35, C=3)
    if variant_id == 1:
        g = cv2.medianBlur(gray, 3)
        return _otsu_thresh(g)
    if variant_id == 2:
        b = cv2.bilateralFilter(gray, d=9, sigmaColor=75, sigmaSpace=75)
        c = _clahe(b)
        return _adaptive_thresh(c, block=25, C=2)
    if variant_id == 3:
        u = _unsharp(gray)
        return _adaptive_thresh(u, block=31, C=2)
    if variant_id == 4:
        # upscale then CLAHE + adaptive (helps small text)
        h, w = gray.shape
        scale = 2.0
        up = cv2.resize(gray, (int(w*scale), int(h*scale)), interpolation=cv2.INTER_CUBIC)
        c = _clahe(up)
        return _adaptive_thresh(c, block=25, C=2)
    if variant_id == 5:
        c = _clahe(gray)
        th = _adaptive_thresh(c, block=31, C=2)
        return cv2.bitwise_not(th)
    # default fallback
    return _otsu_thresh(gray)

def _deskew_if_needed(bin_img):
    # compute angle and rotate if text lines exist
    coords = np.column_stack(np.where(bin_img > 0))
    if coords.shape[0] < 10:
        return bin_img
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.5:
        return bin_img
    (h, w) = bin_img.shape[:2]
    M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
    rotated = cv2.warpAffine(bin_img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated

def _score_result(avg_conf, word_count):
    """
    Combine confidence and word count into a single score.
    avg_conf: mean of confidences (0..100). -1 if unavailable.
    word_count: number of OCR words.
    """
    if avg_conf < 0:
        # fallback to word-count based score
        return math.log1p(word_count) * 0.1
    # weight avg_conf more, but reward more words
    return avg_conf * (1.0 + 0.02 * math.log1p(word_count))

def ocr_ensemble(img_bgr, languages='eng', psm_list=(3,6,11), variants=(0,1,2,3,4,5), debug=False):
    """
    Try multiple preprocessing variants and psm modes, return best text + debug info.
    Returns: (best_text, best_img_gray_or_bin, best_details_dict)
    best_details_dict contains keys:
       'avg_conf', 'word_count', 'variant', 'psm', 'score', 'raw_text', 'data' (tesseract data)
    If debug True, also returns candidate_details list.
    """
    best = None
    candidates = []
    for variant in variants:
        try:
            proc = _binarize_variant(img_bgr, variant)
        except Exception:
            continue
        # deskew small tilt on the processed binary
        proc = _deskew_if_needed(proc)
        # some OCR variants benefit from resized input; we'll pass binary/resized depending on psm later
        for psm in psm_list:
            # try using the binary directly (tesseract can handle binary & gray)
            text, avg_conf, word_count, data = _ocr_and_score(proc, psm=psm, lang=languages)
            score = _score_result(avg_conf, word_count)
            candidate = {
                'variant': variant,
                'psm': psm,
                'avg_conf': avg_conf,
                'word_count': word_count,
                'score': score,
                'raw_text': text,
                'data': data,
                'proc_image': proc
            }
            candidates.append(candidate)
            if best is None or candidate['score'] > best['score']:
                best = candidate
            # also try an upscaled version for small text
            try:
                up = cv2.resize(proc, (0,0), fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
                text2, avg_conf2, wc2, data2 = _ocr_and_score(up, psm=psm, lang=languages)
                score2 = _score_result(avg_conf2, wc2)
                cand2 = {
                    'variant': variant,
                    'psm': psm,
                    'avg_conf': avg_conf2,
                    'word_count': wc2,
                    'score': score2,
                    'raw_text': text2,
                    'data': data2,
                    'proc_image': up
                }
                candidates.append(cand2)
                if cand2['score'] > best['score']:
                    best = cand2
            except Exception:
                pass

    if best is None:
        return "", img_bgr, {'error': 'no candidates'}

    # Compose details to return
    details = {
        'avg_conf': best['avg_conf'],
        'word_count': best['word_count'],
        'variant': best['variant'],
        'psm': best['psm'],
        'score': best['score']
    }
    if debug:
        return best['raw_text'], best['proc_image'], details, candidates
    return best['raw_text'], best['proc_image'], details
