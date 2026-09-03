"""
OCR package exports.
"""

from .reader import (
    preprocess_plate_for_ocr,
    clean_plate_text,
    format_indian_plate,
    recognize_plate_text,
    get_ocr_reader
)

__all__ = [
    "preprocess_plate_for_ocr",
    "clean_plate_text",
    "format_indian_plate",
    "recognize_plate_text",
    "get_ocr_reader"
]
