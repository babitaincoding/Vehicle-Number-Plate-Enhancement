"""
test_enhancement.py
===================
Unit tests for blur detection, degradation, noise reduction, and sharpening modules.
"""

import numpy as np
import pytest

from src.enhancement.blur_detector import (
    compute_laplacian_variance,
    compute_tenengrad_score,
    compute_fft_sharpness,
    detect_blur
)
from src.enhancement.degradation import (
    apply_gaussian_blur,
    apply_motion_blur,
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    simulate_degradation_pipeline
)
from src.enhancement.noise_reduction import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter
)
from src.enhancement.sharpening import (
    laplacian_sharpen,
    highpass_sharpen,
    unsharp_mask,
    compare_sharpening_methods
)


@pytest.fixture
def test_plate_image():
    """Generates a synthetic plate-like test image with high-contrast sharp edges."""
    img = np.ones((80, 240, 3), dtype=np.uint8) * 255
    # Draw dark simulated characters (rectangles)
    img[20:60, 30:50] = 0
    img[20:60, 70:90] = 0
    img[20:60, 110:130] = 0
    img[20:60, 150:170] = 0
    return img


def test_blur_detection(test_plate_image):
    # Sharp image should have high Laplacian variance
    sharp_info = detect_blur(test_plate_image)
    assert sharp_info["status"] == "Sharp"
    assert sharp_info["laplacian_var"] > 50.0

    # Blurred image should drop in variance
    blurred = apply_gaussian_blur(test_plate_image, kernel_size=11, sigma=3.0)
    blur_info = detect_blur(blurred)
    assert blur_info["laplacian_var"] < sharp_info["laplacian_var"]


def test_degradations(test_plate_image):
    motion = apply_motion_blur(test_plate_image, length=9, angle=10.0)
    assert motion.shape == test_plate_image.shape
    assert motion.dtype == np.uint8

    noisy = apply_gaussian_noise(test_plate_image, sigma=15.0)
    assert noisy.shape == test_plate_image.shape
    assert noisy.dtype == np.uint8

    sp = apply_salt_and_pepper_noise(test_plate_image, amount=0.05)
    assert sp.shape == test_plate_image.shape

    combined = simulate_degradation_pipeline(test_plate_image, degradation_type="motion_and_noise")
    assert combined.shape == test_plate_image.shape


def test_noise_reduction(test_plate_image):
    noisy = apply_gaussian_noise(test_plate_image, sigma=20.0)
    denoised_bilateral = apply_bilateral_filter(noisy, d=7, sigma_color=50, sigma_space=50)
    denoised_median = apply_median_filter(noisy, kernel_size=3)
    denoised_gaussian = apply_gaussian_filter(noisy, kernel_size=3)

    assert denoised_bilateral.shape == test_plate_image.shape
    assert denoised_median.shape == test_plate_image.shape
    assert denoised_gaussian.shape == test_plate_image.shape


def test_sharpening_methods(test_plate_image):
    lap = laplacian_sharpen(test_plate_image, alpha=0.6)
    hpf = highpass_sharpen(test_plate_image, beta=0.8)
    usm = unsharp_mask(test_plate_image, amount=1.6, threshold=2)

    assert lap.shape == test_plate_image.shape
    assert hpf.shape == test_plate_image.shape
    assert usm.shape == test_plate_image.shape

    all_methods = compare_sharpening_methods(test_plate_image)
    assert len(all_methods) == 5
    assert "original" in all_methods
    assert "laplacian" in all_methods
    assert "high_pass" in all_methods
    assert "unsharp_mask" in all_methods
    assert "adaptive_enhanced" in all_methods
