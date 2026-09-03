"""
dataset_loader.py
=================
College Project: Vehicle Number Plate Image Enhancement and Sharpening System

Utility functions to locate, parse, and load vehicle images, number plate annotations,
and ground-truth plate text from the local dataset directories.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
from typing import List, Dict, Any, Optional
import cv2
import numpy as np


def find_dataset_root(base_dir: Optional[Path] = None) -> Path:
    """
    Locates the dataset directory within the project workspace.
    Supports either `data/` or `data/raw/indian-number-plates/`.
    """
    if base_dir is None:
        base_dir = Path(__file__).resolve().parents[2] / "data"

    # Check direct location first
    if (base_dir / "number_plate_images_ocr").exists():
        return base_dir

    # Check nested raw directory
    nested = base_dir / "raw" / "indian-number-plates"
    if nested.exists():
        return nested

    return base_dir


def parse_ocr_annotation(xml_path: Path) -> List[Dict[str, Any]]:
    """
    Parses an OCR annotation XML file (Pascal VOC style with custom text attributes).
    
    Returns a list of detected plate dictionaries containing:
        - 'box': (xmin, ymin, xmax, ymax)
        - 'text': ground-truth license plate string (e.g., 'UP84AE9889')
    """
    if not xml_path.exists():
        return []

    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    plates = []

    for obj in root.findall("object"):
        name_tag = obj.find("name")
        if name_tag is None or name_tag.text != "number_plate":
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", "0"))
        ymin = float(bndbox.findtext("ymin", "0"))
        xmax = float(bndbox.findtext("xmax", "0"))
        ymax = float(bndbox.findtext("ymax", "0"))

        plate_text = ""
        attrs = obj.find("attributes")
        if attrs is not None:
            for attr in attrs.findall("attribute"):
                attr_name = attr.findtext("name", "")
                if attr_name == "number_plate_text":
                    plate_text = attr.findtext("value", "").strip()

        plates.append({
            "box": (int(round(xmin)), int(round(ymin)), int(round(xmax)), int(round(ymax))),
            "text": plate_text
        })

    return plates


def load_ocr_dataset(max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    Loads pairs of (image, annotation) for the OCR dataset.
    
    Returns a list of dictionaries with:
        - 'image_path': Path to vehicle image
        - 'image_name': File stem
        - 'plates': List of {'box': (xmin, ymin, xmax, ymax), 'text': str}
    """
    data_dir = find_dataset_root()

    # Search recursively for images and annotations (handles potential nested folders)
    img_dir = data_dir / "number_plate_images_ocr"
    ann_dir = data_dir / "number_plate_annos_ocr"

    if not img_dir.exists() or not ann_dir.exists():
        return []

    img_files = list(img_dir.rglob("*.jpg")) + list(img_dir.rglob("*.png")) + list(img_dir.rglob("*.jpeg"))
    results = []

    for img_path in sorted(img_files):
        xml_name = f"{img_path.stem}.xml"
        matching_xmls = list(ann_dir.rglob(xml_name))

        if matching_xmls:
            xml_path = matching_xmls[0]
            plates = parse_ocr_annotation(xml_path)
            if plates:
                results.append({
                    "image_path": img_path,
                    "image_name": img_path.stem,
                    "xml_path": xml_path,
                    "plates": plates
                })

        if max_samples is not None and len(results) >= max_samples:
            break

    return results


def crop_plate_region(image: np.ndarray, box: tuple, margin: float = 0.05) -> np.ndarray:
    """
    Crops a license plate region from a vehicle image using bounding box coordinates.
    Optionally applies a margin percentage around the box for better character visibility.
    """
    xmin, ymin, xmax, ymax = box
    h, w = image.shape[:2]

    # Calculate padding margin
    box_w = xmax - xmin
    box_h = ymax - ymin
    pad_w = int(box_w * margin)
    pad_h = int(box_h * margin)

    x1 = max(0, xmin - pad_w)
    y1 = max(0, ymin - pad_h)
    x2 = min(w, xmax + pad_w)
    y2 = min(h, ymax + pad_h)

    crop = image[y1:y2, x1:x2]
    return crop
