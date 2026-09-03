"""
blur_detector.py
================
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Blur Detection & Quality Assessment
-------------------------------------------
Analyzes the sharpness and degradation level of license plate images using
classical computer vision techniques:
  1. Variance of Laplacian (Spatial 2nd Derivative)
  2. Tenengrad Gradient Energy (Sobel 1st Derivative)
  3. Frequency Domain (2D FFT) High-Frequency Energy Ratio
"""

from typing import Dict, Any, Tuple
import cv2
import numpy as np


def compute_laplacian_variance(gray_image: np.ndarray) -> float:
    """
    Computes the variance of the Laplacian of an image.
    
    Mathematical Concept:
    ---------------------
    The Laplacian operator highlights regions of rapid intensity change (edges).
    If an image contains sharp edges (like digits on a number plate), the variance
    of the Laplacian will be high. If the image is blurry, edges are spread out and
    the variance will be low.
    
        Laplacian operator: \u2207\u00b2 I = (\u2202\u00b2I/\u2202x\u00b2) + (\u2202\u00b2I/\u2202y\u00b2)
        Variance = Var(\u2207\u00b2 I)
        
    Args:
        gray_image: 2D numpy array (grayscale image)
        
    Returns:
        float: Variance of the Laplacian response.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
    variance = float(laplacian.var())
    return variance


def compute_tenengrad_score(gray_image: np.ndarray) -> float:
    """
    Computes the Tenengrad gradient energy metric.
    
    Mathematical Concept:
    ---------------------
    Calculates the first derivatives in the X and Y directions using the Sobel operator.
    The Tenengrad criterion sums the squared gradient magnitude over all pixels.
    
        G_x = Sobel_x(I), G_y = Sobel_y(I)
        Tenengrad = Mean(G_x\u00b2 + G_y\u00b2)
        
    Args:
        gray_image: 2D numpy array (grayscale image)
        
    Returns:
        float: Average gradient energy.
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)

    gradient_mag_sq = sobel_x**2 + sobel_y**2
    score = float(np.mean(gradient_mag_sq))
    return score


def compute_fft_sharpness(gray_image: np.ndarray, radius_ratio: float = 0.15) -> float:
    """
    Computes frequency domain sharpness using 2D Fast Fourier Transform (FFT).
    
    Mathematical Concept:
    ---------------------
    Blurring attenuates high frequencies. By shifting the DC component to the center
    and calculating the proportion of energy that lies outside the central low-frequency
    disc, we obtain a measure of high-frequency detail.
    
    Args:
        gray_image: 2D numpy array
        radius_ratio: Fraction of the image radius to consider as low frequency
        
    Returns:
        float: Ratio of high frequency magnitude to total frequency magnitude (0.0 to 1.0)
    """
    if len(gray_image.shape) == 3:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)

    h, w = gray_image.shape
    cy, cx = h // 2, w // 2

    # Compute 2D Fourier Transform and shift DC to center
    fft2 = np.fft.fft2(gray_image)
    fft_shift = np.fft.fftshift(fft2)
    magnitude_spectrum = np.abs(fft_shift)

    # Create low-frequency mask
    y, x = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)
    radius = radius_ratio * min(h, w)
    low_freq_mask = dist_from_center <= radius

    total_energy = np.sum(magnitude_spectrum) + 1e-8
    high_freq_energy = np.sum(magnitude_spectrum[~low_freq_mask])

    return float(high_freq_energy / total_energy)


def detect_blur(
    image: np.ndarray,
    laplacian_threshold: float = 100.0,
    tenengrad_threshold: float = 300.0
) -> Dict[str, Any]:
    """
    Assesses whether an input license plate image is blurry or sharp.
    
    Args:
        image: BGR or Grayscale numpy image
        laplacian_threshold: Threshold below which image is considered blurry
        tenengrad_threshold: Supplementary gradient threshold
        
    Returns:
        dict: Detailed blur diagnostics containing:
            - 'is_blurry': bool
            - 'status': 'Blurry' | 'Sharp'
            - 'laplacian_var': float
            - 'tenengrad_score': float
            - 'fft_ratio': float
            - 'sharpness_index': float (0 to 100 normalized score)
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    lap_var = compute_laplacian_variance(gray)
    tenengrad = compute_tenengrad_score(gray)
    fft_ratio = compute_fft_sharpness(gray)

    # Heuristic classification: primary factor is Laplacian variance
    is_blurry = bool(lap_var < laplacian_threshold)

    # Normalize a sharpness index between 0 and 100 for intuitive presentation
    # (capped at 500 variance for standard 100 scale)
    sharpness_index = min(100.0, (lap_var / 500.0) * 100.0)

    return {
        "is_blurry": is_blurry,
        "status": "Blurry" if is_blurry else "Sharp",
        "laplacian_var": round(lap_var, 2),
        "tenengrad_score": round(tenengrad, 2),
        "fft_ratio": round(fft_ratio, 4),
        "sharpness_index": round(sharpness_index, 1)
    }
