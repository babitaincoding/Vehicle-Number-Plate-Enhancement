"""
pipeline.py
===========
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Module: End-to-End System Integration Pipeline
----------------------------------------------
Coordinates the entire workflow:
  1. Input: Vehicle Image or Cropped License Plate
  2. Detection & Localization: Plate bounding box & crop
  3. Quality Analysis: Blur detection (Laplacian variance & Tenengrad)
  4. Enhancement & Sharpening: Edge-preserving Bilateral filter + Unsharp Masking
  5. OCR Recognition: Text recognition & Indian license plate validation
  6. Quantitative Evaluation: PSNR, SSIM, CNR, and confidence improvement
  7. Visualization: Multi-panel comparative diagram saved to output directory
"""

import argparse
from pathlib import Path
from typing import Dict, Any, Optional
import cv2
import numpy as np
import matplotlib.pyplot as plt

from .enhancement.blur_detector import detect_blur
from .enhancement.noise_reduction import apply_bilateral_filter
from .enhancement.sharpening import unsharp_mask, laplacian_sharpen
from .ocr.reader import recognize_plate_text
from .evaluation.metrics import evaluate_image_quality


class PlateEnhancementPipeline:
    """
    Complete end-to-end orchestrator for vehicle number plate enhancement and OCR.
    """

    def __init__(
        self,
        sharpening_method: str = "adaptive",
        alpha: float = 0.8,
        usm_amount: float = 2.4,
        clahe_clip: float = 3.0
    ):
        self.sharpening_method = sharpening_method
        self.alpha = alpha
        self.usm_amount = usm_amount
        self.clahe_clip = clahe_clip

    def enhance(self, plate_image: np.ndarray) -> np.ndarray:
        """
        Applies license plate enhancement:
          - 'adaptive' (Default): Luminance CLAHE contrast enhancement + Unsharp Masking
          - 'unsharp_mask': Bilateral de-noising + Unsharp Masking
          - 'laplacian': Bilateral de-noising + Laplacian 2nd derivative sharpening
        """
        from .enhancement.sharpening import adaptive_plate_enhance

        if self.sharpening_method == "adaptive":
            return adaptive_plate_enhance(
                plate_image,
                clahe_clip=self.clahe_clip,
                usm_amount=self.usm_amount,
                denoise=False
            )
        elif self.sharpening_method == "laplacian":
            denoised = apply_bilateral_filter(plate_image, d=5, sigma_color=40, sigma_space=40)
            return laplacian_sharpen(denoised, alpha=self.alpha)
        else:
            denoised = apply_bilateral_filter(plate_image, d=5, sigma_color=40, sigma_space=40)
            return unsharp_mask(denoised, amount=self.usm_amount, threshold=1)

    def process(
        self,
        image_input: np.ndarray,
        ground_truth: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the full pipeline on a license plate image.
        
        Args:
            image_input: BGR plate image
            ground_truth: Optional ground-truth plate string
            
        Returns:
            dict containing raw & enhanced images, blur metrics, OCR results, and evaluation.
        """
        # 1. Blur and quality assessment on raw plate
        raw_blur = detect_blur(image_input)

        # 2. Enhancement
        enhanced_image = self.enhance(image_input)

        # 3. Post-enhancement blur and quality assessment
        enhanced_blur = detect_blur(enhanced_image)

        # 4. OCR on raw vs enhanced
        raw_ocr = recognize_plate_text(image_input)
        enhanced_ocr = recognize_plate_text(enhanced_image)

        # 5. Image quality comparison
        quality_metrics = evaluate_image_quality(enhanced_image, reference=image_input)

        # 6. Calculate improvements
        sharpness_boost = enhanced_blur["laplacian_var"] - raw_blur["laplacian_var"]
        confidence_gain = enhanced_ocr["confidence"] - raw_ocr["confidence"]

        result = {
            "raw_image": image_input,
            "enhanced_image": enhanced_image,
            "raw_blur": raw_blur,
            "enhanced_blur": enhanced_blur,
            "sharpness_boost": round(sharpness_boost, 2),
            "raw_ocr": raw_ocr,
            "enhanced_ocr": enhanced_ocr,
            "confidence_gain": round(confidence_gain, 4),
            "quality_metrics": quality_metrics,
            "ground_truth": ground_truth
        }

        return result

    def generate_visualization(
        self,
        result: Dict[str, Any],
        output_path: Optional[Path] = None,
        title_prefix: str = "Vehicle Plate Enhancement"
    ) -> Path:
        """
        Renders a 4-panel academic comparative figure:
          Panel 1: Raw / Degraded Input Plate (with blur info & raw OCR)
          Panel 2: Enhanced & Sharpened Plate (with sharpness boost & enhanced OCR)
          Panel 3: Grayscale Intensity Histogram Comparison
          Panel 4: Horizontal Cross-Section Edge Gradient Profile
        """
        raw_img = cv2.cvtColor(result["raw_image"], cv2.COLOR_BGR2RGB)
        enh_img = cv2.cvtColor(result["enhanced_image"], cv2.COLOR_BGR2RGB)

        fig, axes = plt.subplots(2, 2, figsize=(12, 7))

        # 1. Raw Plate Display
        axes[0, 0].imshow(raw_img)
        raw_txt = result["raw_ocr"]["text"] or "[No Text]"
        raw_conf = result["raw_ocr"]["confidence"]
        axes[0, 0].set_title(
            f"1. Raw Input Plate\nStatus: {result['raw_blur']['status']} (Lap. Var: {result['raw_blur']['laplacian_var']})\n"
            f"OCR: '{raw_txt}' (Conf: {raw_conf:.2f})",
            fontsize=10,
            color="darkred" if result["raw_blur"]["is_blurry"] else "black"
        )
        axes[0, 0].axis("off")

        # 2. Enhanced Plate Display
        axes[0, 1].imshow(enh_img)
        enh_txt = result["enhanced_ocr"]["text"] or "[No Text]"
        enh_conf = result["enhanced_ocr"]["confidence"]
        axes[0, 1].set_title(
            f"2. Enhanced & Sharpened Plate\nStatus: {result['enhanced_blur']['status']} (Lap. Var: {result['enhanced_blur']['laplacian_var']})\n"
            f"OCR: '{enh_txt}' (Conf: {enh_conf:.2f})",
            fontsize=10,
            color="green"
        )
        axes[0, 1].axis("off")

        # 3. Histogram Analysis
        raw_gray = cv2.cvtColor(result["raw_image"], cv2.COLOR_BGR2GRAY)
        enh_gray = cv2.cvtColor(result["enhanced_image"], cv2.COLOR_BGR2GRAY)

        axes[1, 0].hist(raw_gray.ravel(), bins=64, color="gray", alpha=0.6, label="Raw Input")
        axes[1, 0].hist(enh_gray.ravel(), bins=64, color="blue", alpha=0.5, label="Enhanced")
        axes[1, 0].set_title("3. Intensity Distribution (Histogram)", fontsize=10)
        axes[1, 0].set_xlabel("Pixel Intensity")
        axes[1, 0].set_ylabel("Frequency")
        axes[1, 0].legend()
        axes[1, 0].grid(True, linestyle="--", alpha=0.4)

        # 4. Edge Gradient Profile across middle horizontal slice
        mid_y = raw_gray.shape[0] // 2
        raw_slice = raw_gray[mid_y, :]
        enh_slice = enh_gray[mid_y, :]

        axes[1, 1].plot(raw_slice, label="Raw Profile", color="orange", linewidth=1.5)
        axes[1, 1].plot(enh_slice, label="Enhanced Profile", color="green", linewidth=1.5)
        axes[1, 1].set_title(f"4. Character Edge Transition (Row {mid_y})", fontsize=10)
        axes[1, 1].set_xlabel("Horizontal Pixel Coordinate (X)")
        axes[1, 1].set_ylabel("Intensity")
        axes[1, 1].legend()
        axes[1, 1].grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(
            f"{title_prefix} | Quality: PSNR={result['quality_metrics'].get('psnr_db', 0):.1f} dB, "
            f"SSIM={result['quality_metrics'].get('ssim', 0):.3f}",
            fontsize=12,
            fontweight="bold"
        )
        plt.tight_layout()

        if output_path is None:
            output_dir = Path("outputs")
            output_dir.mkdir(exist_ok=True)
            output_path = output_dir / "pipeline_comparison.png"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

        plt.savefig(str(output_path), dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Run End-to-End License Plate Enhancement & OCR.")
    parser.add_argument("--image", required=True, help="Path to plate or vehicle image.")
    parser.add_argument("--output", default="outputs/pipeline_result.png", help="Path for comparison plot.")
    parser.add_argument("--method", default="unsharp_mask", choices=["unsharp_mask", "laplacian"], help="Sharpening method.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")

    image = cv2.imread(str(img_path))
    pipeline = PlateEnhancementPipeline(sharpening_method=args.method)

    print(f"\nProcessing: {img_path.name}")
    result = pipeline.process(image)

    print(f"Blur Status (Raw)     : {result['raw_blur']['status']} (Laplacian: {result['raw_blur']['laplacian_var']})")
    print(f"Blur Status (Enhanced): {result['enhanced_blur']['status']} (Laplacian: {result['enhanced_blur']['laplacian_var']})")
    print(f"Raw OCR Text          : '{result['raw_ocr']['text']}' (Conf: {result['raw_ocr']['confidence']})")
    print(f"Enhanced OCR Text     : '{result['enhanced_ocr']['text']}' (Conf: {result['enhanced_ocr']['confidence']})")
    print(f"PSNR                  : {result['quality_metrics'].get('psnr_db', 0):.2f} dB")
    print(f"SSIM                  : {result['quality_metrics'].get('ssim', 0):.4f}")

    saved_path = pipeline.generate_visualization(result, output_path=Path(args.output))
    print(f"\nSaved visualization to: {saved_path}\n")


if __name__ == "__main__":
    main()
