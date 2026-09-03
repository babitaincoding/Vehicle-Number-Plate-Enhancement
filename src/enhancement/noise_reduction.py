"""
noise_reduction.py
==================
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Noise Reduction Filters
-------------------------------
Implements and compares three foundational de-noising techniques:
  1. Gaussian Filter (Linear spatial smoothing)
  2. Median Filter (Non-linear rank filter for impulse/salt-and-pepper noise)
  3. Bilateral Filter (Edge-preserving non-linear filter ideal for license plates)
"""

from typing import Dict
import cv2
import numpy as np


def apply_gaussian_filter(image: np.ndarray, kernel_size: int = 3, sigma: float = 1.0) -> np.ndarray:
    """
    Applies 2D Gaussian smoothing to reduce high-frequency Gaussian noise.
    
    Mathematical Concept:
    ---------------------
    Convolves the image with a normalized 2D Gaussian bell-curve kernel.
    While effective against additive noise, it softens sharp character transitions.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)


def apply_median_filter(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Applies non-linear median filtering to remove salt-and-pepper / impulse noise.
    
    Mathematical Concept:
    ---------------------
    Replaces each pixel value with the median value within its neighborhood window.
    Because extreme outlier values (0 or 255) are excluded from the median, it removes
    impulse noise without significantly shifting character edge locations.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1
    return cv2.medianBlur(image, kernel_size)


def apply_bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75.0,
    sigma_space: float = 75.0
) -> np.ndarray:
    """
    Applies Bilateral edge-preserving filtering.
    
    Mathematical Concept:
    ---------------------
    Unlike standard Gaussian filtering that only weights pixels by geometric spatial distance,
    the Bilateral filter incorporates a radiometric range weight based on pixel intensity difference:
    
        BF[I]_p = 1/W_p \u2211_{q \u2208 S} G_{\u03c3_s}(\u2016p - q\u2016) \u00b7 G_{\u03c3_r}(|I_p - I_q|) \u00b7 I_q
        
    When evaluating neighboring pixels across a character boundary (e.g. black digit on white plate),
    the intensity difference |I_p - I_q| is large, so the range weight drops to near zero.
    Result: Smooths the plate background while preserving sharp character edges!
    
    Args:
        image: BGR or grayscale image
        d: Diameter of pixel neighborhood
        sigma_color: Filter sigma in the color/intensity space
        sigma_space: Filter sigma in the coordinate spatial space
    """
    return cv2.bilateralFilter(image, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


def compare_noise_reduction(image: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Runs all three noise reduction techniques for visual and metric comparison.
    
    Returns:
        dict of { 'original', 'gaussian', 'median', 'bilateral' }
    """
    return {
        "original": image.copy(),
        "gaussian": apply_gaussian_filter(image, kernel_size=3, sigma=1.0),
        "median": apply_median_filter(image, kernel_size=3),
        "bilateral": apply_bilateral_filter(image, d=7, sigma_color=50, sigma_space=50)
    }
