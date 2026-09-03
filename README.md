# Vehicle Number Plate Image Enhancement and Sharpening System Using Computer Vision

## Overview

A computer vision system for detecting vehicle number plates, extracting the plate region, enhancing image quality, sharpening character details, and recognizing the plate number using OCR.

## Pipeline

````text
Vehicle Image
      ↓
Number Plate Detection
      ↓
Plate Extraction
      ↓
Blur / Quality Analysis
      ↓
Image Enhancement
      ↓
Laplacian / High-Pass Sharpening
      ↓
OCR
      ↓
Evaluation

Objectives
Detect vehicle number plates from images.
Extract the detected plate region.
Analyze and improve degraded plate images.
Enhance character edges using Laplacian and high-pass filtering.
Apply OCR to recognize vehicle numbers.
Evaluate detection, enhancement, and OCR performance.
Datasets
Indian Number Plates Dataset

Used for number plate detection and localization.

Indian License Plate Images

Used for plate image enhancement and OCR-related processing.

Technologies
Python
OpenCV
NumPy
YOLO (Ultralytics)
scikit-image
Matplotlib
Pillow
OCR
Number Plate Detection

A pretrained YOLO11n model is fine-tuned to detect a single class:

The detection dataset contains 27 annotated images and is divided into:

Train: 22
Validation: 3
Test: 2
Detection Results
Metric	Score
Precision	0.1981
Recall	0.3333
mAP@50	0.3423
mAP@50–95	0.2366

The detected bounding boxes are used to automatically crop the number plate region for subsequent enhancement and OCR.

## Part 2: Enhancement, Sharpening & Character Recognition (OCR)

Part 2 implements classical computer vision filters and OCR text recognition:
1. **Blur Detection**: Laplacian variance ($\text{Var}(\nabla^2 I)$), Tenengrad gradient energy, and 2D FFT frequency ratio.
2. **Controlled Degradation**: Synthetic generation of motion blur, Gaussian defocus, sensor noise, and impulse noise for scientific benchmarking.
3. **Noise Reduction**: Gaussian blur, Median filter, and edge-preserving Bilateral filtering.
4. **Sharpening Comparison**: Laplacian 2nd-derivative operator, spatial High-Pass Filtering (HPF), and classic Unsharp Masking (USM).
5. **OCR Recognition**: Text extraction with CLAHE preprocessing, adaptive binarization, HSRP 'IND' prefix stripping, and Indian license plate regex validation.
6. **Evaluation**: PSNR, SSIM, Contrast-to-Noise Ratio (CNR), Image Entropy, Character Error Rate (CER), and exact string match accuracy.

For detailed theory, mathematical derivations, and viva exam preparation, see **[docs/PART2_GUIDE.md](docs/PART2_GUIDE.md)**.

## How to Run

### 1. Launch the Interactive Web Demo (Streamlit)
```bash
streamlit run app.py
````

### 2. Run the End-to-End Pipeline CLI

```bash
python -m src.pipeline --image data/sample_plates/plate_dc_auto_image_000024_fqvRhfiO6i_0_UP84AE9889.jpg --output outputs/my_result.png
```

### 3. Generate 8-Panel Visual Sharpening Report

```bash
python demo_visualizer.py
```

### 4. Run Automated Unit Tests

```bash
python -m pytest tests/ -v
```

## Project Structure

```text
Vehicle-Number-Plate-Enhancement/
│
├── app.py                     # Streamlit interactive presentation dashboard
├── demo_visualizer.py         # 8-panel comparison report generator
│
├── data/
│   ├── sample_plates/         # Verified ready-to-test cropped plates with index
│   ├── number_plate_images_ocr/
│   └── number_plate_annos_ocr/
│
├── docs/
│   └── PART2_GUIDE.md         # Comprehensive academic documentation & viva prep
│
├── notebooks/
│   ├── 01_dataset_exploration.ipynb
│   └── 02_enhancement_sharpening_ocr.ipynb
│
├── src/
│   ├── detection/             # Part 1: YOLO training, evaluation & cropping
│   │   ├── train.py
│   │   ├── evaluate.py
│   │   └── detect_and_crop.py
│   ├── enhancement/           # Part 2: Blur detection, degradation, de-noising, sharpening
│   │   ├── blur_detector.py
│   │   ├── degradation.py
│   │   ├── noise_reduction.py
│   │   └── sharpening.py
│   ├── ocr/                   # Part 2: OCR reader, preprocessing & plate formatting
│   │   └── reader.py
│   ├── evaluation/            # Part 2: PSNR, SSIM, CNR, CER metrics & evaluator
│   │   ├── metrics.py
│   │   └── evaluator.py
│   ├── utils/                 # Dataset loading and sample extraction utilities
│   │   ├── dataset_loader.py
│   │   └── extract_samples.py
│   └── pipeline.py            # End-to-end integration orchestrator
│
├── tests/                     # Automated unit test suite
│   ├── test_enhancement.py
│   └── test_metrics.py
│
├── outputs/                   # Generated evaluation plots and visual reports
├── requirements.txt
└── README.md
```
