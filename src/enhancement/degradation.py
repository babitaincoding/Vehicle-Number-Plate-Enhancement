"""
degradation.py
==============
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Controlled Degradation Simulator
----------------------------------------
Simulates real-world environmental and camera degradations (motion blur, defocus,
sensor noise, low illumination) in a controlled, reproducible manner.
This allows scientific before/after benchmarking of our sharpening and de-noising filters.
"""

from typing import Tuple, Optional
import cv2
import numpy as np


def apply_gaussian_blur(image: np.ndarray, kernel_size: int = 7, sigma: float = 2.0) -> np.ndarray:
    """
    Simulates out-of-focus camera lens defocus blur using a 2D Gaussian kernel.
    
    Args:
        image: BGR or grayscale image
        kernel_size: Size of Gaussian kernel (must be odd, e.g. 5, 7, 9)
        sigma: Standard deviation of Gaussian distribution
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_motion_blur(image: np.ndarray, length: int = 15, angle: float = 0.0) -> np.ndarray:
    """
    Simulates linear motion blur caused by a moving vehicle relative to a stationary camera.
    
    Mathematical Concept:
    ---------------------
    Constructs a directional motion blur point spread function (PSF) kernel rotated
    by the specified angle in degrees, then convolves it with the image.
    
    Args:
        image: BGR or grayscale image
        length: Length of motion streak in pixels
        angle: Motion direction in degrees (0 = horizontal motion)
    """
    if length <= 1:
        return image.copy()

    # Create a 1D horizontal line kernel
    kernel = np.zeros((length, length), dtype=np.float32)
    center = length // 2
    kernel[center, :] = 1.0 / length

    # Rotate kernel to desired angle
    if angle != 0:
        rot_mat = cv2.getRotationMatrix2D((center, center), angle, 1.0)
        kernel = cv2.warpAffine(kernel, rot_mat, (length, length))
        # Re-normalize to ensure brightness conservation
        kernel_sum = np.sum(kernel)
        if kernel_sum > 0:
            kernel /= kernel_sum

    return cv2.filter2D(image, -1, kernel)


def apply_gaussian_noise(image: np.ndarray, mean: float = 0.0, sigma: float = 25.0) -> np.ndarray:
    """
    Simulates sensor electronic noise via Additive White Gaussian Noise (AWGN).
    
    Args:
        image: BGR or grayscale image (uint8)
        mean: Mean noise value
        sigma: Standard deviation of noise
    """
    noise = np.random.normal(mean, sigma, image.shape).astype(np.float32)
    noisy_img = image.astype(np.float32) + noise
    return np.clip(noisy_img, 0, 255).astype(np.uint8)


def apply_salt_and_pepper_noise(image: np.ndarray, amount: float = 0.04, s_vs_p: float = 0.5) -> np.ndarray:
    """
    Simulates transmission errors / dead pixels using Salt and Pepper (impulse) noise.
    
    Args:
        image: BGR or grayscale image
        amount: Proportion of total image pixels to corrupt (e.g. 0.04 = 4%)
        s_vs_p: Ratio of salt (white) vs pepper (black) noise
    """
    noisy = image.copy()
    num_salt = int(np.ceil(amount * image.size * s_vs_p))
    num_pepper = int(np.ceil(amount * image.size * (1.0 - s_vs_p)))

    # Add salt (white pixels)
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy[coords[0], coords[1], :] = 255
    else:
        noisy[coords[0], coords[1]] = 255

    # Add pepper (black pixels)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:
        noisy[coords[0], coords[1], :] = 0
    else:
        noisy[coords[0], coords[1]] = 0

    return noisy


def apply_low_contrast(image: np.ndarray, factor: float = 0.5, brightness_bias: int = -20) -> np.ndarray:
    """
    Simulates poor lighting or foggy conditions by reducing dynamic range.
    
    Args:
        image: Input image
        factor: Contrast multiplier (< 1.0 reduces contrast)
        brightness_bias: Brightness offset
    """
    adjusted = image.astype(np.float32) * factor + brightness_bias
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def simulate_degradation_pipeline(
    image: np.ndarray,
    degradation_type: str = "motion_and_noise",
    severity: str = "medium"
) -> np.ndarray:
    """
    High-level helper to simulate typical CCTV license plate degradation.
    
    Args:
        image: Input cropped plate image
        degradation_type: 'motion_blur' | 'gaussian_blur' | 'noise' | 'motion_and_noise' | 'severe'
        severity: 'low' | 'medium' | 'high'
    """
    # Map severity to parameter scaling
    scale = {"low": 0.6, "medium": 1.0, "high": 1.6}.get(severity, 1.0)

    out = image.copy()

    if degradation_type == "motion_blur":
        length = int(11 * scale)
        out = apply_motion_blur(out, length=length, angle=15.0)

    elif degradation_type == "gaussian_blur":
        ksize = int(5 * scale)
        if ksize % 2 == 0:
            ksize += 1
        out = apply_gaussian_blur(out, kernel_size=ksize, sigma=1.8 * scale)

    elif degradation_type == "noise":
        out = apply_gaussian_noise(out, sigma=20.0 * scale)
        out = apply_salt_and_pepper_noise(out, amount=0.02 * scale)

    elif degradation_type == "motion_and_noise":
        out = apply_motion_blur(out, length=int(9 * scale), angle=10.0)
        out = apply_gaussian_noise(out, sigma=15.0 * scale)

    elif degradation_type == "severe":
        out = apply_motion_blur(out, length=int(13 * scale), angle=20.0)
        out = apply_gaussian_noise(out, sigma=25.0 * scale)
        out = apply_salt_and_pepper_noise(out, amount=0.03 * scale)
        out = apply_low_contrast(out, factor=0.7, brightness_bias=-10)

    return out
