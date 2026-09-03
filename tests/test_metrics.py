"""
test_metrics.py
===============
Unit tests for image quality (PSNR, SSIM, CNR) and OCR accuracy (CER, Levenshtein).
"""

import numpy as np
import pytest

from src.evaluation.metrics import (
    compute_psnr,
    compute_ssim,
    compute_cnr,
    compute_entropy,
    levenshtein_distance,
    compute_character_error_rate,
    compute_ocr_accuracy
)


def test_image_metrics():
    # Identical images should have high PSNR and SSIM == 1.0
    img1 = np.full((100, 100, 3), 128, dtype=np.uint8)
    img2 = img1.copy()

    assert compute_psnr(img1, img2) >= 99.0
    assert compute_ssim(img1, img2) == 1.0

    # Slight modification
    img_mod = img1.copy()
    img_mod[40:60, 40:60] = 0
    psnr_mod = compute_psnr(img1, img_mod)
    ssim_mod = compute_ssim(img1, img_mod)

    assert psnr_mod < 99.0
    assert ssim_mod < 1.0
    assert compute_entropy(img_mod) > 0.0


def test_ocr_metrics():
    # Perfect match
    gt = "UP84AE9889"
    pred = "UP84AE9889"

    assert levenshtein_distance(gt, pred) == 0
    assert compute_character_error_rate(gt, pred) == 0.0

    acc = compute_ocr_accuracy(gt, pred)
    assert acc["exact_match"] == 1
    assert acc["similarity"] == 1.0

    # Single character error
    pred_1err = "UP84AE9888"
    assert levenshtein_distance(gt, pred_1err) == 1
    assert pytest.approx(compute_character_error_rate(gt, pred_1err), 0.01) == 0.10
    acc_1err = compute_ocr_accuracy(gt, pred_1err)
    assert acc_1err["exact_match"] == 0
    assert acc_1err["similarity"] == 0.90
