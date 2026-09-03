"""
extract_sample_plates.py
========================
Extracts prominent license plate crops from the dataset and saves them in `data/sample_plates/`
along with a metadata JSON index. This enables instant demonstration and standalone testing.
"""

from pathlib import Path
import json
import cv2
from src.utils.dataset_loader import load_ocr_dataset, crop_plate_region


def main():
    output_dir = Path("data/sample_plates")
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_ocr_dataset()
    print(f"Loaded {len(records)} vehicle images with plate annotations.")

    saved_samples = []

    for item in records:
        img_path = item["image_path"]
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for i, plate in enumerate(item["plates"]):
            box = plate["box"]
            text = plate["text"]
            w = box[2] - box[0]
            h = box[3] - box[1]

            # Filter for reasonably sized, clear plates for clean demonstration
            if w > 80 and h > 25 and len(text) >= 4:
                crop = crop_plate_region(img, box, margin=0.06)
                if crop.size > 0:
                    filename = f"plate_{item['image_name']}_{i}_{text}.jpg"
                    save_path = output_dir / filename
                    cv2.imwrite(str(save_path), crop)

                    saved_samples.append({
                        "filename": filename,
                        "path": str(save_path),
                        "ground_truth": text,
                        "original_image": item["image_name"],
                        "box": box,
                        "dimensions": [int(crop.shape[1]), int(crop.shape[0])]
                    })

    # Save index
    meta_path = output_dir / "samples_index.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(saved_samples, f, indent=2)

    print(f"Extracted {len(saved_samples)} sample plate crops into: {output_dir}")
    print(f"Index written to: {meta_path}")


if __name__ == "__main__":
    main()
