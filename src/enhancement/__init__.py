"""
Enhancement package exports.
"""

from .blur_detector import (
    compute_laplacian_variance,
    compute_tenengrad_score,
    compute_fft_sharpness,
    detect_blur
)
from .degradation import (
    apply_gaussian_blur,
    apply_motion_blur,
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_contrast,
    simulate_degradation_pipeline
)
from .noise_reduction import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter,
    compare_noise_reduction
)
from .sharpening import (
    laplacian_sharpen,
    highpass_sharpen,
    unsharp_mask,
    adaptive_plate_enhance,
    compare_sharpening_methods
)

__all__ = [
    "compute_laplacian_variance",
    "compute_tenengrad_score",
    "compute_fft_sharpness",
    "detect_blur",
    "apply_gaussian_blur",
    "apply_motion_blur",
    "apply_gaussian_noise",
    "apply_salt_and_pepper_noise",
    "apply_low_contrast",
    "simulate_degradation_pipeline",
    "apply_gaussian_filter",
    "apply_median_filter",
    "apply_bilateral_filter",
    "compare_noise_reduction",
    "laplacian_sharpen",
    "highpass_sharpen",
    "unsharp_mask",
    "adaptive_plate_enhance",
    "compare_sharpening_methods"
]
