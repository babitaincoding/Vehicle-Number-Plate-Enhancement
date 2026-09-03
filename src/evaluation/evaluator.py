"""
evaluator.py
============
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Systematic Benchmark Evaluator
--------------------------------------
Runs comparative benchmarks comparing:
  - Raw / Degraded Plate
  - Noise Reduced (Bilateral / Gaussian / Median)
  - Laplacian Sharpened
  - High-Pass Sharpened
  - Unsharp Masked
  
Generates comprehensive tables evaluating both image quality (PSNR, SSIM, CNR)
and OCR character recognition accuracy (CER, Levenshtein, Confidence).
"""

from typing import Dict, Any, List, Optional
import cv2
import numpy as np

from ..enhancement.sharpening import laplacian_sharpen, highpass_sharpen, unsharp_mask
from ..enhancement.noise_reduction import apply_bilateral_filter, apply_median_filter, apply_gaussian_filter
from ..enhancement.blur_detector import detect_blur
from ..ocr.reader import recognize_plate_text
from .metrics import evaluate_image_quality, compute_ocr_accuracy


def evaluate_single_plate(
    plate_image: np.ndarray,
    ground_truth_text: Optional[str] = None,
    reference_image: Optional[np.ndarray] = None
) -> Dict[str, Any]:
    """
    Evaluates enhancement methods and OCR performance on a single license plate.
    
    Args:
        plate_image: Input plate (may be degraded, blurry, or raw crop)
        ground_truth_text: True plate number (e.g. 'UP84AE9889') if available
        reference_image: High-quality pristine reference plate for PSNR/SSIM if available
        
    Returns:
        dict containing comparative metrics and OCR results for each pipeline variation.
    """
    if reference_image is None:
        reference_image = plate_image

    # 1. Generate enhanced variations
    variations = {
        "Raw Input": plate_image,
        "Gaussian Denoised": apply_gaussian_filter(plate_image, kernel_size=3, sigma=1.0),
        "Median Denoised": apply_median_filter(plate_image, kernel_size=3),
        "Bilateral Denoised": apply_bilateral_filter(plate_image, d=7, sigma_color=50, sigma_space=50),
        "Laplacian Sharpened": laplacian_sharpen(plate_image, alpha=0.6),
        "High-Pass Sharpened": highpass_sharpen(plate_image, beta=0.8),
        "Unsharp Masked": unsharp_mask(plate_image, amount=1.5, threshold=3)
    }

    # Also test combination: Bilateral Denoised + Unsharp Masking (our recommended pipeline)
    bilateral = variations["Bilateral Denoised"]
    variations["Bilateral + USM (Hybrid)"] = unsharp_mask(bilateral, amount=1.6, threshold=2)

    results = {}

    for name, img in variations.items():
        # Blur & sharpness score
        blur_info = detect_blur(img)

        # Image quality metrics
        quality = evaluate_image_quality(img, reference=reference_image)

        # OCR recognition
        ocr_res = recognize_plate_text(img)

        entry = {
            "image": img,
            "blur_status": blur_info["status"],
            "laplacian_var": blur_info["laplacian_var"],
            "sharpness_index": blur_info["sharpness_index"],
            "entropy": quality["entropy"],
            "cnr": quality["cnr"],
            "edge_intensity": quality["edge_intensity"],
            "psnr_db": quality.get("psnr_db", 0.0),
            "ssim": quality.get("ssim", 0.0),
            "ocr_text": ocr_res["text"],
            "ocr_raw": ocr_res["raw_text"],
            "ocr_confidence": ocr_res["confidence"],
            "is_valid_format": ocr_res["is_valid_format"]
        }

        # OCR Accuracy metrics (if ground truth is supplied)
        if ground_truth_text:
            ocr_acc = compute_ocr_accuracy(ground_truth_text, ocr_res["text"])
            entry["exact_match"] = ocr_acc["exact_match"]
            entry["edit_distance"] = ocr_acc["edit_distance"]
            entry["cer"] = ocr_acc["cer"]
            entry["similarity"] = ocr_acc["similarity"]

        results[name] = entry

    return results


def print_evaluation_summary(results: Dict[str, Any], ground_truth: Optional[str] = None):
    """
    Prints a cleanly formatted ASCII table of the evaluation results for terminal or report.
    """
    print("\n" + "=" * 90)
    print(f"BENCHMARK EVALUATION SUMMARY {f'[Ground Truth: {ground_truth}]' if ground_truth else ''}")
    print("=" * 90)

    header = f"{'Method':<25} | {'Sharpness':<9} | {'CNR':<6} | {'PSNR(dB)':<8} | {'SSIM':<6} | {'OCR Text':<12} | {'Conf':<6}"
    if ground_truth:
        header += f" | {'CER':<6} | {'Match'}"
    print(header)
    print("-" * len(header))

    for name, data in results.items():
        row = f"{name:<25} | {data['laplacian_var']:<9.1f} | {data['cnr']:<6.2f} | {data['psnr_db']:<8.2f} | {data['ssim']:<6.3f} | {data['ocr_text']:<12} | {data['ocr_confidence']:<6.2f}"
        if ground_truth:
            match_str = "YES" if data.get("exact_match", 0) else "NO"
            row += f" | {data.get('cer', 0.0):<6.3f} | {match_str}"
        print(row)

    print("=" * 90 + "\n")
