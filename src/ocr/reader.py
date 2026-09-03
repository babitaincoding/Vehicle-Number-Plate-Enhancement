"""
reader.py
=========
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Optical Character Recognition (OCR) Engine
--------------------------------------------------
Performs:
  1. License plate text preprocessing (resizing, CLAHE, Otsu/adaptive binarization)
  2. OCR text extraction with character confidence via EasyOCR
  3. Post-processing and Indian license plate regex validation/sanitization
"""

import re
from typing import Dict, Any, List, Optional, Tuple
import cv2
import numpy as np

import sys
import os
import warnings

# Suppress PyTorch internal deprecation and CPU pin_memory warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*torch.quantize_per_tensor.*")
warnings.filterwarnings("ignore", message=".*pin_memory.*")

# Ensure UTF-8 output encoding for Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
os.environ["PYTHONIOENCODING"] = "utf-8"

# Lazy global reader to prevent reloading the model on every inference
_EASYOCR_READER = None


def get_ocr_reader():
    """Returns a singleton instance of the EasyOCR Reader."""
    global _EASYOCR_READER
    if _EASYOCR_READER is None:
        try:
            import easyocr
            import torch
            use_gpu = torch.cuda.is_available()
            _EASYOCR_READER = easyocr.Reader(["en"], gpu=use_gpu, verbose=False)
        except Exception as e:
            print(f"[Warning] Failed to initialize EasyOCR: {e}")
            _EASYOCR_READER = None
    return _EASYOCR_READER



def preprocess_plate_for_ocr(
    plate_image: np.ndarray,
    target_height: int = 120,
    use_clahe: bool = True,
    binarize: bool = False
) -> np.ndarray:
    """
    Preprocesses a cropped license plate to optimize OCR text recognition.
    
    Steps:
      1. Upscales plate if height is below target_height (preserves aspect ratio)
      2. Converts to grayscale
      3. Applies CLAHE (Contrast Limited Adaptive Histogram Equalization)
         to normalize illumination across the plate
      4. Optional Otsu binarization with morphological cleaning
    """
    img = plate_image.copy()
    h, w = img.shape[:2]

    # 1. Upscale if too small
    if h < target_height:
        scale = target_height / float(h)
        new_w = int(w * scale)
        img = cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_CUBIC)

    # 2. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # 3. CLAHE
    if use_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)

    # 4. Binarization (optional)
    if binarize:
        # Otsu's thresholding after slight Gaussian blur
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # If the background is dark (mean < 128), invert to make characters black on white
        if np.mean(binary) < 128:
            binary = cv2.bitwise_not(binary)
        return binary

    return gray


def clean_plate_text(raw_text: str) -> str:
    """
    Cleans raw OCR output by removing whitespace, special characters, and non-alphanumeric noise.
    Converts to uppercase and strips the standard Indian HSRP 'IND' emblem prefix if present.
    """
    if not raw_text:
        return ""
    # Strip spaces, hyphens, dots, underscores, colons, newlines
    cleaned = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    # Remove Indian High Security Registration Plate (HSRP) 'IND' emblem tag
    if cleaned.startswith("IND") and len(cleaned) >= 8:
        cleaned = cleaned[3:]

    return cleaned



def format_indian_plate(text: str) -> Tuple[str, bool]:
    """
    Validates and formats text according to standard Indian license plate patterns:
      Format: [State: 2 letters] [District: 1-2 digits] [Series: 1-3 letters] [Number: 4 digits]
      Example: 'UP84AE9889', 'KL35H5834', 'MH12DE1433', 'DL3CAA1111'
      
    Returns:
        tuple: (formatted_text: str, is_valid: bool)
    """
    cleaned = clean_plate_text(text)
    
    # Standard Indian Plate Regex
    pattern = r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$"
    is_valid = bool(re.match(pattern, cleaned))

    return cleaned, is_valid


def recognize_plate_text(
    image: np.ndarray,
    preprocess: bool = True,
    binarize: bool = False
) -> Dict[str, Any]:
    """
    Recognizes character text from a cropped license plate image.
    
    Args:
        image: BGR or Grayscale cropped plate image
        preprocess: Whether to apply resizing and CLAHE
        binarize: Whether to binarize before OCR
        
    Returns:
        dict:
            - 'text': Sanitized plate string
            - 'raw_text': Unsanitized raw OCR output
            - 'confidence': Average OCR confidence score (0.0 to 1.0)
            - 'is_valid_format': Boolean indicating valid Indian format
            - 'detections': Detailed EasyOCR character bounding boxes & scores
    """
    if preprocess:
        proc_img = preprocess_plate_for_ocr(image, binarize=binarize)
    else:
        proc_img = image

    reader = get_ocr_reader()
    if reader is None:
        return {
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "is_valid_format": False,
            "detections": []
        }

    # Run EasyOCR
    # paragraph=False returns individual word / token detections
    results = reader.readtext(proc_img, paragraph=False)

    if not results:
        return {
            "text": "",
            "raw_text": "",
            "confidence": 0.0,
            "is_valid_format": False,
            "detections": []
        }

    detected_words = []
    confidences = []
    detections = []

    for bbox, text, conf in results:
        cleaned_word = clean_plate_text(text)
        if cleaned_word:
            detected_words.append(cleaned_word)
            confidences.append(float(conf))
            detections.append({
                "box": bbox,
                "text": cleaned_word,
                "confidence": round(float(conf), 4)
            })

    raw_text = "".join(detected_words)
    cleaned_text, is_valid = format_indian_plate(raw_text)
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    return {
        "text": cleaned_text,
        "raw_text": raw_text,
        "confidence": round(avg_confidence, 4),
        "is_valid_format": is_valid,
        "detections": detections
    }
