"""
app.py
======
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Interactive Web Demonstration App (Streamlit)
---------------------------------------------
A presentation-ready interactive dashboard allowing users/evaluators to:
  1. Test pre-loaded Indian license plates or upload custom vehicle images
  2. Simulate camera degradations (motion blur, defocus, noise)
  3. Interactively tune sharpening parameters (\u03b1, USM amount, Bilateral filtering)
  4. Compare Laplacian vs High-Pass vs Unsharp Masking side-by-side
  5. View live OCR recognition and quantitative metrics (PSNR, SSIM, CNR, CER)
"""

import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import json
import cv2
import numpy as np
import streamlit as st

from src.enhancement.blur_detector import detect_blur
from src.enhancement.degradation import (
    apply_motion_blur,
    apply_gaussian_blur,
    apply_gaussian_noise,
    apply_salt_and_pepper_noise,
    apply_low_contrast
)
from src.enhancement.noise_reduction import (
    apply_bilateral_filter,
    apply_gaussian_filter,
    apply_median_filter
)
from src.enhancement.sharpening import (
    laplacian_sharpen,
    highpass_sharpen,
    unsharp_mask,
    adaptive_plate_enhance
)
from src.ocr.reader import recognize_plate_text

from src.evaluation.metrics import evaluate_image_quality, compute_ocr_accuracy
from src.evaluation.evaluator import evaluate_single_plate


st.set_page_config(
    page_title="Vehicle Plate Enhancement System",
    page_icon="\U0001f697",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-box {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #3B82F6;
        margin-bottom: 10px;
    }
    .badge-success {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
    .badge-warning {
        background-color: #FEF08A;
        color: #854D0E;
        padding: 4px 8px;
        border-radius: 4px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_sample_plates_list():
    """Loads sample plates index from data/sample_plates/."""
    index_path = Path("data/sample_plates/samples_index.json")
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


# ==========================================
# SIDEBAR CONTROLS
# ==========================================
st.sidebar.title("\U0001f697 Control Panel")
st.sidebar.markdown("---")

input_source = st.sidebar.radio(
    "Select Input Source",
    ["Sample Dataset Plates", "Upload Custom Plate Image"]
)

ground_truth = None
current_image = None
image_title = ""

if input_source == "Sample Dataset Plates":
    samples = load_sample_plates_list()
    if samples:
        sample_options = [f"{s['ground_truth']} ({s['filename']})" for s in samples]
        selected_idx = st.sidebar.selectbox("Choose License Plate Sample", range(len(samples)), format_func=lambda i: sample_options[i])
        selected_sample = samples[selected_idx]
        plate_path = Path(selected_sample["path"])
        ground_truth = selected_sample["ground_truth"]
        image_title = f"Plate: {ground_truth}"
        if plate_path.exists():
            current_image = cv2.imread(str(plate_path))
    else:
        st.sidebar.warning("No sample plates found in data/sample_plates/")

else:
    uploaded_file = st.sidebar.file_uploader("Upload Plate Image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        current_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        image_title = uploaded_file.name
        gt_input = st.sidebar.text_input("Ground Truth Plate Number (Optional):", value="")
        if gt_input.strip():
            ground_truth = gt_input.strip().upper()

st.sidebar.markdown("---")
st.sidebar.subheader("\u2699\ufe0f Degradation Simulator")
simulate_deg = st.sidebar.checkbox("Apply Synthetic Camera Flaws", value=False)
deg_image = None

if simulate_deg and current_image is not None:
    deg_type = st.sidebar.selectbox(
        "Degradation Type",
        ["Motion Blur", "Defocus (Gaussian Blur)", "Sensor Noise (Gaussian)", "Salt & Pepper Noise", "Low Illumination"]
    )
    if deg_type == "Motion Blur":
        streak = st.sidebar.slider("Streak Length (px)", 3, 25, 11)
        deg_image = apply_motion_blur(current_image, length=streak, angle=15.0)
    elif deg_type == "Defocus (Gaussian Blur)":
        ksize = st.sidebar.slider("Kernel Size", 3, 15, 7, step=2)
        deg_image = apply_gaussian_blur(current_image, kernel_size=ksize, sigma=2.0)
    elif deg_type == "Sensor Noise (Gaussian)":
        sigma_noise = st.sidebar.slider("Noise Sigma", 5.0, 50.0, 20.0)
        deg_image = apply_gaussian_noise(current_image, sigma=sigma_noise)
    elif deg_type == "Salt & Pepper Noise":
        sp_amt = st.sidebar.slider("Impulse Ratio", 0.01, 0.10, 0.03, step=0.01)
        deg_image = apply_salt_and_pepper_noise(current_image, amount=sp_amt)
    elif deg_type == "Low Illumination":
        contrast = st.sidebar.slider("Contrast Multiplier", 0.2, 1.0, 0.5)
        deg_image = apply_low_contrast(current_image, factor=contrast, brightness_bias=-15)
else:
    if current_image is not None:
        deg_image = current_image.copy()

st.sidebar.markdown("---")
st.sidebar.subheader("🛠️ Enhancement Settings")

enhancement_mode = st.sidebar.selectbox(
    "Enhancement Preset",
    [
        "High-Contrast Adaptive Sharpening (Recommended)",
        "Unsharp Masking (USM)",
        "Laplacian Sharpening",
        "High-Pass Spatial Filter"
    ]
)

use_bilateral = st.sidebar.checkbox(
    "Apply Bilateral De-noising",
    value=False,
    help="Enable only if image has sensor noise/grain. Leave OFF for blurry images to avoid extra smoothing."
)

if use_bilateral:
    b_diameter = st.sidebar.slider("Bilateral Diameter", 3, 11, 5, step=2)
    b_sigma = st.sidebar.slider("Bilateral Sigma", 20, 80, 40)

if enhancement_mode == "High-Contrast Adaptive Sharpening (Recommended)":
    clahe_clip = st.sidebar.slider("Contrast Boost (CLAHE)", 1.0, 5.0, 3.0, step=0.5)
    usm_amt = st.sidebar.slider("Sharpening Strength", 1.0, 4.0, 2.5, step=0.2)
elif enhancement_mode == "Unsharp Masking (USM)":
    usm_amt = st.sidebar.slider("USM Amount", 0.5, 4.0, 2.2, step=0.1)
    usm_thresh = st.sidebar.slider("USM Threshold", 0, 10, 1)
elif enhancement_mode == "Laplacian Sharpening":
    lap_alpha = st.sidebar.slider("Laplacian Weight (α)", 0.2, 2.0, 0.8, step=0.1)
else:
    hpf_beta = st.sidebar.slider("High-Pass Boost (β)", 0.2, 2.5, 1.0, step=0.1)


# ==========================================
# MAIN INTERFACE
# ==========================================
st.markdown('<div class="main-header">Vehicle Number Plate Enhancement & Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Classical Computer Vision Pipeline: Detection &rarr; Blur Analysis &rarr; Bilateral Filtering &rarr; Sharpening &rarr; OCR</div>', unsafe_allow_html=True)

if current_image is None:
    st.info("Please select a sample plate from the sidebar or upload an image to begin.")
    st.stop()

# Execution of enhancement pipeline
processed_input = deg_image if deg_image is not None else current_image

# Stage 1: Noise reduction (optional)
if use_bilateral:
    stage1 = apply_bilateral_filter(processed_input, d=b_diameter, sigma_color=b_sigma, sigma_space=b_sigma)
else:
    stage1 = processed_input.copy()

# Stage 2: Sharpening
if enhancement_mode == "High-Contrast Adaptive Sharpening (Recommended)":
    enhanced = adaptive_plate_enhance(stage1, clahe_clip=clahe_clip, usm_amount=usm_amt, denoise=False)
elif enhancement_mode == "Unsharp Masking (USM)":
    enhanced = unsharp_mask(stage1, amount=usm_amt, threshold=usm_thresh)
elif enhancement_mode == "Laplacian Sharpening":
    enhanced = laplacian_sharpen(stage1, alpha=lap_alpha)
else:
    enhanced = highpass_sharpen(stage1, beta=hpf_beta)

# Tabs
tab1, tab2 = st.tabs([
    "\U0001f50d Live Pipeline",
    "\u2696\ufe0f Sharpening Comparison"
])

# ----------------------------------------------------
# TAB 1: LIVE PIPELINE
# ----------------------------------------------------
with tab1:
    col1, col2 = st.columns(2)

    # Blur diagnostics
    raw_blur = detect_blur(processed_input)
    enh_blur = detect_blur(enhanced)

    # OCR extraction
    with st.spinner("Running OCR text recognition..."):
        raw_ocr = recognize_plate_text(processed_input)
        enh_ocr = recognize_plate_text(enhanced)

    # Image quality metrics
    quality = evaluate_image_quality(enhanced, reference=current_image)

    with col1:
        st.subheader("1. Input / Degraded Plate")
        st.image(cv2.cvtColor(processed_input, cv2.COLOR_BGR2RGB), use_container_width=True)
        raw_display = raw_ocr.get("text") or raw_ocr.get("raw_text") or "[Not Detected]"
        st.markdown(f"""
        **Quality Diagnostics:**
        - **Blur Status**: {'🔴 Blurry' if raw_blur['is_blurry'] else '🟢 Sharp'}
        - **Laplacian Variance**: `{raw_blur['laplacian_var']:.1f}`
        - **Tenengrad Energy**: `{raw_blur['tenengrad_score']:.1f}`
        - **OCR Recognized**: `{raw_display}`
        - **OCR Confidence**: `{raw_ocr['confidence']:.2f}`
        """)

    with col2:
        st.subheader("2. Enhanced & Sharpened Plate")
        st.image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB), use_container_width=True)
        sharp_boost = enh_blur['laplacian_var'] - raw_blur['laplacian_var']
        enh_display = enh_ocr.get("text") or enh_ocr.get("raw_text") or "[Not Detected]"
        st.markdown(f"""
        **Quality Diagnostics:**
        - **Blur Status**: {'🔴 Blurry' if enh_blur['is_blurry'] else '🟢 Sharp'}
        - **Laplacian Variance**: `{enh_blur['laplacian_var']:.1f}` (**+{sharp_boost:.1f}**)
        - **Tenengrad Energy**: `{enh_blur['tenengrad_score']:.1f}`
        - **OCR Recognized**: `{enh_display}`
        - **OCR Confidence**: `{enh_ocr['confidence']:.2f}`
        """)

    show_binary = st.checkbox("Show Binarized Character Stroke Segmentation", value=False)
    if show_binary:
        b_col1, b_col2 = st.columns(2)
        with b_col1:
            st.caption("Raw Plate (Otsu Thresholded)")
            g_raw = cv2.cvtColor(processed_input, cv2.COLOR_BGR2GRAY) if len(processed_input.shape) == 3 else processed_input
            _, b_raw = cv2.threshold(g_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(b_raw, use_container_width=True)
        with b_col2:
            st.caption("Enhanced Plate (Otsu Thresholded)")
            g_enh = cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY) if len(enhanced.shape) == 3 else enhanced
            _, b_enh = cv2.threshold(g_enh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            st.image(b_enh, use_container_width=True)

    st.markdown("---")
    st.subheader("\U0001f4ca Performance & Improvement Summary")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("PSNR (dB)", f"{quality.get('psnr_db', 0):.2f} dB", help="Peak Signal to Noise Ratio compared to reference")
    m2.metric("SSIM", f"{quality.get('ssim', 0):.4f}", help="Structural Similarity Index (1.0 = identical)")
    m3.metric("CNR", f"{quality.get('cnr', 0):.2f}", help="Contrast to Noise Ratio between characters and plate surface")
    m4.metric("Sharpness Variance", f"{enh_blur['laplacian_var']:.0f}", f"{sharp_boost:+.0f}")
    conf_diff = enh_ocr['confidence'] - raw_ocr['confidence']
    m5.metric("OCR Confidence", f"{enh_ocr['confidence']:.2f}", f"{conf_diff:+.2f}")

    if ground_truth:
        st.markdown("---")
        st.subheader("\U0001f3af Ground Truth Verification")
        gt_acc = compute_ocr_accuracy(ground_truth, enh_ocr['text'])
        g1, g2, g3, g4 = st.columns(4)
        g1.markdown(f"**Ground Truth:** `{ground_truth}`")
        g2.markdown(f"**Enhanced OCR:** `{enh_ocr['text']}`")
        g3.markdown(f"**Exact Match:** {'\u2705 YES' if gt_acc['exact_match'] else '\u274c NO'}")
        g4.markdown(f"**Character Error Rate (CER):** `{gt_acc['cer']:.3f}`")

# ----------------------------------------------------
# TAB 2: SHARPENING COMPARISON
# ----------------------------------------------------
with tab2:
    st.subheader("Direct Comparison: 4 Enhancement Methods")
    st.markdown("Evaluating how different spatial and contrast enhancement operators affect character boundaries:")

    c_lap = laplacian_sharpen(stage1, alpha=0.8)
    c_hpf = highpass_sharpen(stage1, beta=1.0)
    c_usm = unsharp_mask(stage1, amount=2.0, threshold=1)
    c_ada = adaptive_plate_enhance(stage1, clahe_clip=3.0, usm_amount=2.5)

    col_a, col_b, col_c, col_d = st.columns(4)

    with col_a:
        st.markdown("**1. Laplacian 2nd Derivative**")
        st.image(cv2.cvtColor(c_lap, cv2.COLOR_BGR2RGB), use_container_width=True)
        b_lap = detect_blur(c_lap)
        q_lap = evaluate_image_quality(c_lap, reference=current_image)
        st.caption(f"Lap. Var: {b_lap['laplacian_var']:.1f}\n\nCNR: {q_lap['cnr']:.2f}")

    with col_b:
        st.markdown("**2. Spatial High-Pass Filter**")
        st.image(cv2.cvtColor(c_hpf, cv2.COLOR_BGR2RGB), use_container_width=True)
        b_hpf = detect_blur(c_hpf)
        q_hpf = evaluate_image_quality(c_hpf, reference=current_image)
        st.caption(f"Lap. Var: {b_hpf['laplacian_var']:.1f}\n\nCNR: {q_hpf['cnr']:.2f}")

    with col_c:
        st.markdown("**3. Unsharp Masking (USM)**")
        st.image(cv2.cvtColor(c_usm, cv2.COLOR_BGR2RGB), use_container_width=True)
        b_usm = detect_blur(c_usm)
        q_usm = evaluate_image_quality(c_usm, reference=current_image)
        st.caption(f"Lap. Var: {b_usm['laplacian_var']:.1f}\n\nCNR: {q_usm['cnr']:.2f}")

    with col_d:
        st.markdown("**4. Adaptive High-Contrast**")
        st.image(cv2.cvtColor(c_ada, cv2.COLOR_BGR2RGB), use_container_width=True)
        b_ada = detect_blur(c_ada)
        q_ada = evaluate_image_quality(c_ada, reference=current_image)
        st.caption(f"Lap. Var: {b_ada['laplacian_var']:.1f}\n\nCNR: {q_ada['cnr']:.2f}")

    st.info("""
    **Academic Findings:** 
    - **Laplacian** boosts rapid spatial transitions, effective for high-contrast edges.
    - **High-Pass Filtering** isolates frequency content by low-pass subtraction.
    - **Unsharp Masking** provides clean photographic edge amplification.
    - **Adaptive High-Contrast (CLAHE + USM)** provides the most striking visual clarity on severely blurred plates by normalizing luminance and boosting character stroke darkness.
    """)