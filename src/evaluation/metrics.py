"""
metrics.py
==========
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Quantitative Evaluation Metrics
---------------------------------------
Implements classical image quality metrics and OCR text accuracy metrics:
  Image Quality:
    1. PSNR (Peak Signal-to-Noise Ratio)
    2. SSIM (Structural Similarity Index Measure)
    3. CNR (Contrast-to-Noise Ratio)
    4. Image Entropy (Information density)
    5. Edge Intensity (Tenengrad sharpness)
  OCR Accuracy:
    6. Levenshtein Edit Distance
    7. Character Error Rate (CER)
    8. String Accuracy (Exact Match)
    9. Normalized Similarity (0.0 to 1.0)
"""

import math
from typing import Dict, Any, Tuple, Optional
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim_fn


# ==========================================
# 1. IMAGE QUALITY METRICS
# ==========================================

def compute_psnr(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Computes Peak Signal-to-Noise Ratio (PSNR) in decibels (dB).
    
    Mathematical Concept:
    ---------------------
    PSNR measures reconstruction fidelity by comparing the maximum possible signal power
    to the power of corrupting noise (Mean Squared Error):
    
        MSE = (1 / MN) \u2211\u2211 [I_true(i,j) - I_test(i,j)]\u00b2
        PSNR = 20 \u00b7 log10(MAX_I / \u221aMSE)
        
    Higher PSNR indicates higher fidelity to the reference image.
    """
    if image_true.shape != image_test.shape:
        image_test = cv2.resize(image_test, (image_true.shape[1], image_true.shape[0]))

    img1 = image_true.astype(np.float64)
    img2 = image_test.astype(np.float64)

    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return 100.0  # Perfect reconstruction

    max_pixel = 255.0
    psnr = 20.0 * math.log10(max_pixel / math.sqrt(mse))
    return float(round(psnr, 2))


def compute_ssim(image_true: np.ndarray, image_test: np.ndarray) -> float:
    """
    Computes Structural Similarity Index Measure (SSIM).
    
    Mathematical Concept:
    ---------------------
    Unlike PSNR which treats error independently at each pixel, SSIM measures
    perceived visual quality across three comparative terms:
      - Luminance comparison
      - Contrast comparison
      - Structural correlation
      
    SSIM values range from -1.0 to +1.0 (1.0 = identical).
    """
    if image_true.shape != image_test.shape:
        image_test = cv2.resize(image_test, (image_true.shape[1], image_true.shape[0]))

    if len(image_true.shape) == 3:
        gray_true = cv2.cvtColor(image_true, cv2.COLOR_BGR2GRAY)
    else:
        gray_true = image_true

    if len(image_test.shape) == 3:
        gray_test = cv2.cvtColor(image_test, cv2.COLOR_BGR2GRAY)
    else:
        gray_test = image_test

    score, _ = ssim_fn(gray_true, gray_test, full=True, data_range=255)
    return float(round(score, 4))


def compute_cnr(image: np.ndarray) -> float:
    """
    Computes Contrast-to-Noise Ratio (CNR) between plate characters and plate background.
    
    Mathematical Concept:
    ---------------------
    Using Otsu thresholding, pixels are segmented into Foreground (characters)
    and Background (plate surface). CNR is the difference in means divided by pooled variance:
    
        CNR = |\u03bc_fg - \u03bc_bg| / \u221a(\u03c3_fg\u00b2 + \u03c3_bg\u00b2)
        
    Higher CNR means characters stand out more distinctly against plate background.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Segment characters from background using Otsu binarization
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    fg = gray[binary == 0]  # Dark character pixels
    bg = gray[binary == 255]  # Light background pixels

    if len(fg) == 0 or len(bg) == 0:
        return 0.0

    mu_fg, std_fg = np.mean(fg), np.std(fg)
    mu_bg, std_bg = np.mean(bg), np.std(bg)

    denom = math.sqrt(std_fg**2 + std_bg**2) + 1e-6
    cnr = abs(mu_bg - mu_fg) / denom
    return float(round(cnr, 2))


def compute_entropy(image: np.ndarray) -> float:
    """
    Computes Shannon Image Entropy.
    
    Mathematical Concept:
    ---------------------
    Measures the average information content / dynamic range distribution:
    
        H = - \u2211 P(i) \u00b7 log2(P(i))
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-8)  # Normalize probabilities

    # Filter out zero probabilities
    p = hist[hist > 0]
    entropy = -np.sum(p * np.log2(p))
    return float(round(entropy, 3))


def compute_edge_intensity(image: np.ndarray) -> float:
    """
    Computes average edge gradient intensity using the Sobel operator.
    Sharpened images exhibit higher edge intensity.
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(round(np.mean(mag), 2))


def evaluate_image_quality(
    image: np.ndarray,
    reference: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Computes all standard image quality metrics for an enhanced plate.
    If reference is provided, computes full-reference metrics (PSNR, SSIM).
    """
    metrics = {
        "entropy": compute_entropy(image),
        "cnr": compute_cnr(image),
        "edge_intensity": compute_edge_intensity(image)
    }

    if reference is not None:
        metrics["psnr_db"] = compute_psnr(reference, image)
        metrics["ssim"] = compute_ssim(reference, image)

    return metrics


# ==========================================
# 2. OCR ACCURACY METRICS
# ==========================================

def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Computes minimum Levenshtein edit distance between two strings
    (insertions, deletions, substitutions).
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n]


def compute_character_error_rate(ground_truth: str, prediction: str) -> float:
    """
    Computes Character Error Rate (CER):
    
        CER = Levenshtein(GT, Pred) / max(1, len(GT))
        
    0.0 represents perfect recognition; lower is better.
    """
    gt = ground_truth.upper().strip()
    pred = prediction.upper().strip()

    if len(gt) == 0:
        return 0.0 if len(pred) == 0 else 1.0

    edit_dist = levenshtein_distance(gt, pred)
    cer = edit_dist / len(gt)
    return float(round(cer, 4))


def compute_ocr_accuracy(ground_truth: str, prediction: str) -> Dict[str, Any]:
    """
    Computes full OCR accuracy evaluation comparing Ground Truth to Prediction.
    
    Returns:
        dict:
            - 'exact_match': 1 if exact match else 0
            - 'edit_distance': integer edit distance
            - 'cer': Character Error Rate (0.0 = perfect)
            - 'similarity': Normalized similarity score between 0.0 and 1.0
    """
    gt = ground_truth.upper().strip()
    pred = prediction.upper().strip()

    dist = levenshtein_distance(gt, pred)
    max_len = max(len(gt), len(pred), 1)
    similarity = max(0.0, 1.0 - (dist / max_len))
    cer = compute_character_error_rate(gt, pred)

    return {
        "exact_match": int(gt == pred),
        "edit_distance": dist,
        "cer": cer,
        "similarity": float(round(similarity, 4))
    }
