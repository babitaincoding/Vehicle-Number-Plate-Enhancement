"""
Evaluation package exports.
"""

from .metrics import (
    compute_psnr,
    compute_ssim,
    compute_cnr,
    compute_entropy,
    compute_edge_intensity,
    evaluate_image_quality,
    levenshtein_distance,
    compute_character_error_rate,
    compute_ocr_accuracy
)
from .evaluator import (
    evaluate_single_plate,
    print_evaluation_summary
)

__all__ = [
    "compute_psnr",
    "compute_ssim",
    "compute_cnr",
    "compute_entropy",
    "compute_edge_intensity",
    "evaluate_image_quality",
    "levenshtein_distance",
    "compute_character_error_rate",
    "compute_ocr_accuracy",
    "evaluate_single_plate",
    "print_evaluation_summary"
]
