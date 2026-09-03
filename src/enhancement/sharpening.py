"""
sharpening.py
=============
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: Image Sharpening & Edge Enhancement
-------------------------------------------
Implements and compares three classic edge sharpening algorithms:
  1. Laplacian Sharpening (2nd Spatial Derivative)
  2. High-Pass Spatial Filtering (HPF)
  3. Unsharp Masking (USM)
"""

from typing import Dict, Any, Optional
import cv2
import numpy as np


def laplacian_sharpen(
    image: np.ndarray,
    alpha: float = 0.6,
    kernel_type: str = "8-neighbor"
) -> np.ndarray:
    """
    Sharpens image edges using the discrete Laplacian second-derivative operator.
    
    Mathematical Concept:
    ---------------------
    The Laplacian highlights points of inflection and sharp discontinuities in intensity.
    By scaling and adding the Laplacian response back to the original image, edge contrast
    is boosted:
    
        I_sharp(x, y) = I(x, y) + \u03b1 \u00b7 (\u2207\u00b2 I(x, y))
        
    where \u2207\u00b2 is defined by the 4-neighbor or 8-neighbor discrete kernel.
    
    Args:
        image: BGR or grayscale numpy image
        alpha: Sharpening weight factor (typically 0.3 to 1.0)
        kernel_type: '4-neighbor' | '8-neighbor'
    """
    if kernel_type == "4-neighbor":
        # Positive center, negative orthogonal neighbors
        kernel = np.array([
            [0, -1,  0],
            [-1, 4, -1],
            [0, -1,  0]
        ], dtype=np.float32)
    else:
        # 8-neighbor includes diagonal gradients for omnidirectional sharpness
        kernel = np.array([
            [-1, -1, -1],
            [-1,  8, -1],
            [-1, -1, -1]
        ], dtype=np.float32)

    # Compute Laplacian response using float32 to avoid saturation clipping
    lap = cv2.filter2D(image.astype(np.float32), -1, kernel)

    # Add scaled Laplacian back to original image
    sharpened = image.astype(np.float32) + alpha * lap
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def highpass_sharpen(
    image: np.ndarray,
    beta: float = 0.8,
    blur_ksize: int = 5,
    sigma: float = 1.5
) -> np.ndarray:
    """
    Sharpens image using spatial High-Pass Filtering (HPF).
    
    Mathematical Concept:
    ---------------------
    A high-pass filter isolates high-frequency components (edges and fine details)
    by subtracting a low-pass (Gaussian smoothed) image from the original image:
    
        HPF(I) = I - LowPass(I)
        I_sharp = I + \u03b2 \u00b7 HPF(I)
        
    Args:
        image: BGR or grayscale image
        beta: Boosting coefficient for high-frequency details
        blur_ksize: Gaussian kernel size for the low-pass component
        sigma: Standard deviation for Gaussian blur
    """
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    # Low-pass filter (smooth out high frequencies)
    low_pass = cv2.GaussianBlur(image, (blur_ksize, blur_ksize), sigmaX=sigma, sigmaY=sigma)

    # High-pass filter (isolated edge details)
    high_pass = image.astype(np.float32) - low_pass.astype(np.float32)

    # Recombine to boost high frequencies
    sharpened = image.astype(np.float32) + beta * high_pass
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def unsharp_mask(
    image: np.ndarray,
    kernel_size: int = 5,
    sigma: float = 1.0,
    amount: float = 1.5,
    threshold: int = 0
) -> np.ndarray:
    """
    Applies classic photographic Unsharp Masking (USM).
    
    Mathematical Concept:
    ---------------------
    Derives its name from traditional darkroom photography: creating an 'unsharp'
    (blurred) negative mask and subtracting it from the original to amplify contrast
    along intensity transitions (Mach band effect).
    
        Mask = I - Blurred(I)
        I_sharp = I + amount \u00b7 Mask   (where |Mask| > threshold)
        
    Args:
        image: BGR or grayscale image
        kernel_size: Gaussian blur kernel size (must be odd)
        sigma: Gaussian blur standard deviation
        amount: Scaling strength of the unsharp mask (typically 1.0 - 2.5)
        threshold: Minimum pixel difference required before sharpening is applied.
                   Using a threshold prevents amplifying subtle sensor noise in smooth plate areas.
    """
    if kernel_size % 2 == 0:
        kernel_size += 1

    blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), sigmaX=sigma, sigmaY=sigma)
    img_f = image.astype(np.float32)
    blr_f = blurred.astype(np.float32)

    mask = img_f - blr_f

    if threshold > 0:
        # Only sharpen pixels where difference exceeds threshold to avoid noise amplification
        low_contrast_mask = np.abs(mask) < threshold
        mask[low_contrast_mask] = 0.0

    sharpened = img_f + amount * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def adaptive_plate_enhance(
    image: np.ndarray,
    clahe_clip: float = 2.5,
    usm_amount: float = 2.2,
    denoise: bool = False
) -> np.ndarray:
    """
    High-contrast visual enhancement pipeline designed specifically for degraded/blurry license plates.
    
    Mathematical Concept:
    ---------------------
    Human vision perceives sharpness and detail predominantly through luminance variations,
    not chromatic variations. By operating in the CIELAB color space:
      1. Lightness (L) channel is separated from chromaticity (A, B).
      2. CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied to the L channel
         to normalize illumination and boost local character-to-background contrast.
      3. An edge-amplifying unsharp mask is applied to the enhanced L channel.
      4. The channels are recombined and converted back to standard BGR.
      
    Result: Produces distinctly darker, crisper character strokes against a clean plate background,
    making the enhancement immediately visible to the human eye and highly legible to OCR engines.
    """
    img = image.copy()

    # Optional gentle edge-preserving de-noising if sensor noise is present
    if denoise:
        img = cv2.bilateralFilter(img, d=5, sigmaColor=35, sigmaSpace=35)

    # 1. Convert to CIELAB color space
    if len(img.shape) == 3:
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan, a_chan, b_chan = cv2.split(lab)
    else:
        l_chan = img.copy()
        a_chan, b_chan = None, None

    # 2. Apply CLAHE on Lightness (L) channel
    if clahe_clip > 0:
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
        l_eq = clahe.apply(l_chan)
    else:
        l_eq = l_chan

    # 3. Apply Unsharp Masking on the L channel
    blurred_l = cv2.GaussianBlur(l_eq, (5, 5), sigmaX=1.5, sigmaY=1.5)
    mask = l_eq.astype(np.float32) - blurred_l.astype(np.float32)
    l_sharp = np.clip(l_eq.astype(np.float32) + usm_amount * mask, 0, 255).astype(np.uint8)

    # 4. Reconstruct BGR image
    if a_chan is not None and b_chan is not None:
        merged_lab = cv2.merge([l_sharp, a_chan, b_chan])
        enhanced_bgr = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        return enhanced_bgr
    else:
        return l_sharp


def compare_sharpening_methods(
    image: np.ndarray,
    laplacian_alpha: float = 0.6,
    hpf_beta: float = 0.8,
    usm_amount: float = 1.5
) -> Dict[str, np.ndarray]:
    """
    Computes all three sharpening methods on an input plate image for side-by-side comparison.
    
    Returns:
        dict with:
            - 'original': Input image
            - 'laplacian': Laplacian sharpened result
            - 'high_pass': High-pass filtered result
            - 'unsharp_mask': Unsharp masked result
            - 'adaptive_enhanced': Luminance CLAHE + Sharpening result
    """
    return {
        "original": image.copy(),
        "laplacian": laplacian_sharpen(image, alpha=laplacian_alpha),
        "high_pass": highpass_sharpen(image, beta=hpf_beta),
        "unsharp_mask": unsharp_mask(image, amount=usm_amount),
        "adaptive_enhanced": adaptive_plate_enhance(image, clahe_clip=2.5, usm_amount=2.2)
    }

