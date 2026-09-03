"""
demo_visualizer.py
==================
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Visual Comparison & Report Generator
------------------------------------
Produces an 8-panel comprehensive visual comparative report comparing:
  1. Input / Pristine Plate
  2. Degraded Plate (Simulated Motion Blur + Sensor Noise)
  3. Gaussian Filtered (Linear smoothing)
  4. Median Filtered (Impulse noise reduction)
  5. Bilateral Filtered (Edge-preserving smoothing)
  6. Laplacian Sharpened (2nd Spatial Derivative)
  7. High-Pass Sharpened (High-frequency boost)
  8. Unsharp Masked (Photographic contrast amplification)

Also displays:
  - Laplacian Sharpness variance for each method
  - Contrast-to-Noise Ratio (CNR)
  - OCR text output
"""

import argparse
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

from src.enhancement.blur_detector import detect_blur
from src.enhancement.degradation import simulate_degradation_pipeline
from src.enhancement.noise_reduction import (
    apply_gaussian_filter,
    apply_median_filter,
    apply_bilateral_filter
)
from src.enhancement.sharpening import (
    laplacian_sharpen,
    highpass_sharpen,
    unsharp_mask
)
from src.evaluation.metrics import evaluate_image_quality
from src.ocr.reader import recognize_plate_text


def create_comprehensive_comparison(
    image_path: Path,
    output_path: Path,
    degrade_input: bool = True
) -> Path:
    """
    Renders and saves an 8-panel academic figure comparing all de-noising
    and sharpening techniques side-by-side.
    """
    original = cv2.imread(str(image_path))
    if original is None:
        raise FileNotFoundError(f"Cannot open image: {image_path}")

    # If requested, apply controlled degradation to demonstrate restoration capability
    if degrade_input:
        work_image = simulate_degradation_pipeline(original, degradation_type="motion_and_noise", severity="medium")
    else:
        work_image = original.copy()

    # Apply filtering and sharpening methods
    methods = [
        ("1. Input Plate", work_image),
        ("2. Gaussian Filter (3x3)", apply_gaussian_filter(work_image, kernel_size=3, sigma=1.0)),
        ("3. Median Filter (3x3)", apply_median_filter(work_image, kernel_size=3)),
        ("4. Bilateral Filter (Edge-Preserving)", apply_bilateral_filter(work_image, d=7, sigma_color=50, sigma_space=50)),
        ("5. Laplacian Sharpening (\u03b1=0.6)", laplacian_sharpen(work_image, alpha=0.6)),
        ("6. High-Pass Sharpening (\u03b2=0.8)", highpass_sharpen(work_image, beta=0.8)),
        ("7. Unsharp Masking (Amount=1.6)", unsharp_mask(work_image, amount=1.6, threshold=2)),
        ("8. Bilateral + Unsharp (Hybrid Pipeline)", unsharp_mask(apply_bilateral_filter(work_image, d=7, sigma_color=50, sigma_space=50), amount=1.6, threshold=2))
    ]

    fig, axes = plt.subplots(4, 2, figsize=(14, 12))
    axes = axes.flatten()

    for idx, (title, img) in enumerate(methods):
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blur_info = detect_blur(img)
        quality = evaluate_image_quality(img, reference=original)
        ocr_info = recognize_plate_text(img)

        text_str = ocr_info["text"] if ocr_info["text"] else "[Not Detected]"
        conf_str = f"{ocr_info['confidence']:.2f}"

        axes[idx].imshow(rgb_img)
        caption = (
            f"{title}\n"
            f"Laplacian Var: {blur_info['laplacian_var']} | CNR: {quality['cnr']} | "
            f"PSNR: {quality.get('psnr_db', 0):.1f} dB\n"
            f"OCR: '{text_str}' (Conf: {conf_str})"
        )
        axes[idx].set_title(caption, fontsize=9.5, fontweight="bold")
        axes[idx].axis("off")

    plt.suptitle(
        f"Vehicle Number Plate Enhancement & Sharpening Comparison\nSource: {image_path.name}",
        fontsize=13,
        fontweight="bold"
    )
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=220, bbox_inches="tight")
    plt.close(fig)

    print(f"Comparison report successfully saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Generate 8-panel enhancement comparison report.")
    parser.add_argument(
        "--image",
        default="data/sample_plates/plate_dc_auto_image_000024_fqvRhfiO6i_0_UP84AE9889.jpg",
        help="Path to plate image."
    )
    parser.add_argument(
        "--output",
        default="outputs/sharpening_comparison_report.png",
        help="Output image path."
    )
    parser.add_argument(
        "--no-degrade",
        action="store_true",
        help="Do not apply synthetic degradation to the input."
    )
    args = parser.parse_args()

    create_comprehensive_comparison(
        image_path=Path(args.image),
        output_path=Path(args.output),
        degrade_input=not args.no_degrade
    )


if __name__ == "__main__":
    main()
