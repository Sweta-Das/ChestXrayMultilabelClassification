"""Create side-by-side NIH bbox vs Grad-CAM comparison images.

This script reads the bbox annotations from `ref/ref.csv`, loads the matching
X-ray image, draws the ground-truth bounding box, runs model inference, and
generates a Grad-CAM heatmap for the annotated disease.

Output layout per sample:
  1. Original image with ground-truth bbox
  2. Grad-CAM heatmap
  3. Grad-CAM overlay
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from inference import predict
from utils.gradcam_pth import generate_multi_disease_heatmaps as generate_pth_heatmaps


DISEASE_LABELS = [
    "Atelectasis",
    "Cardiomegaly",
    "Effusion",
    "Infiltration",
    "Mass",
    "Nodule",
    "Pneumonia",
    "Pneumothorax",
    "Consolidation",
    "Edema",
    "Emphysema",
    "Fibrosis",
    "Pleural Thickening",
    "Hernia",
]

LABEL_ALIASES = {
    "Infiltrate": "Infiltration",
}


@dataclass
class Sample:
    image_index: str
    finding_label: str
    bbox_x: float
    bbox_y: float
    bbox_w: float
    bbox_h: float

    @property
    def canonical_label(self) -> str:
        return LABEL_ALIASES.get(self.finding_label, self.finding_label)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate side-by-side bbox vs Grad-CAM comparison images."
    )
    parser.add_argument("--csv", default="ref/ref.csv", help="Path to bbox CSV")
    parser.add_argument(
        "--image-dir",
        default=None,
        help="Directory containing the NIH images. Defaults to the CSV folder.",
    )
    parser.add_argument(
        "--output-dir",
        default="ref/comparisons",
        help="Directory to write comparison PNGs and metadata JSON files.",
    )
    parser.add_argument(
        "--model-path",
        default="CTransCNN/models/epoch_45.pth",
        help="Path to the PyTorch checkpoint used for Grad-CAM.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Maximum number of rows to process.",
    )
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional label filter. Example: --labels Atelectasis Effusion",
    )
    parser.add_argument(
        "--panel-size",
        type=int,
        default=512,
        help="Square size for each panel in the final comparison image.",
    )
    parser.add_argument(
        "--bbox-source-size",
        type=int,
        nargs=2,
        default=(1024, 1024),
        metavar=("WIDTH", "HEIGHT"),
        help="Pixel size that the CSV bbox coordinates are defined in.",
    )
    return parser.parse_args()


def load_samples(csv_path: Path) -> list[Sample]:
    samples: list[Sample] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if not header:
            return samples

        for row in reader:
            if len(row) < 6:
                continue
            try:
                sample = Sample(
                    image_index=row[0].strip(),
                    finding_label=row[1].strip(),
                    bbox_x=float(row[2]),
                    bbox_y=float(row[3]),
                    bbox_w=float(row[4]),
                    bbox_h=float(row[5]),
                )
            except ValueError:
                continue
            samples.append(sample)
    return samples


def decode_data_url(data_url: str) -> Image.Image:
    if data_url.startswith("data:image"):
        _, b64_data = data_url.split(",", 1)
        raw = io.BytesIO(__import__("base64").b64decode(b64_data))
        return Image.open(raw).convert("RGB")
    return Image.open(data_url).convert("RGB")


def draw_bbox(image: Image.Image, sample: Sample) -> Image.Image:
    return draw_bbox_scaled(image, sample, image.size)


def draw_bbox_scaled(
    image: Image.Image,
    sample: Sample,
    target_size: tuple[int, int],
    source_size: tuple[int, int] = (1024, 1024),
    label_text: Optional[str] = None,
) -> Image.Image:
    image = image.convert("RGB")
    draw = ImageDraw.Draw(image)
    x1, y1, x2, y2 = scale_bbox_to_size(sample, source_size, target_size)

    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=max(3, image.width // 256))
    label = label_text or sample.finding_label
    font = ImageFont.load_default()
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_w = text_bbox[2] - text_bbox[0]
    text_h = text_bbox[3] - text_bbox[1]
    pad = 4
    bg = [
        x1,
        max(0, y1 - text_h - pad * 2),
        x1 + text_w + pad * 2,
        max(0, y1),
    ]
    draw.rectangle(bg, fill=(255, 0, 0))
    draw.text((x1 + pad, max(0, y1 - text_h - pad)), label, fill=(255, 255, 255), font=font)
    return image


def scale_bbox_to_size(
    sample: Sample,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> tuple[int, int, int, int]:
    src_w, src_h = source_size
    tgt_w, tgt_h = target_size
    scale_x = tgt_w / float(src_w)
    scale_y = tgt_h / float(src_h)
    x1 = int(round(sample.bbox_x * scale_x))
    y1 = int(round(sample.bbox_y * scale_y))
    x2 = int(round((sample.bbox_x + sample.bbox_w) * scale_x))
    y2 = int(round((sample.bbox_y + sample.bbox_h) * scale_y))
    x1 = max(0, min(tgt_w - 1, x1))
    y1 = max(0, min(tgt_h - 1, y1))
    x2 = max(0, min(tgt_w - 1, x2))
    y2 = max(0, min(tgt_h - 1, y2))
    return x1, y1, x2, y2


def bbox_heatmap_energy_ratio(image: Image.Image, sample: Sample) -> float:
    gray = image.convert("L")
    arr = np.asarray(gray, dtype=np.float32)
    x1, y1, x2, y2 = scale_bbox_to_size(sample, (1024, 1024), image.size)
    bbox_energy = float(arr[y1 : y2 + 1, x1 : x2 + 1].sum())
    total_energy = float(arr.sum())
    if total_energy <= 0:
        return 0.0
    return bbox_energy / total_energy


def make_panel(image: Image.Image, title: str, panel_size: int) -> Image.Image:
    title_height = int(panel_size * 0.12)
    canvas = Image.new("RGB", (panel_size, panel_size + title_height), (20, 24, 31))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.rectangle([0, 0, panel_size, title_height], fill=(31, 41, 55))
    draw.text((16, 10), title, fill=(255, 255, 255), font=font)

    fitted = ImageOps.contain(image.convert("RGB"), (panel_size - 24, panel_size - 24))
    x = (panel_size - fitted.width) // 2
    y = title_height + ((panel_size - 24) - fitted.height) // 2
    canvas.paste(fitted, (x, y))
    return canvas


def compose_triptych(left: Image.Image, middle: Image.Image, right: Image.Image) -> Image.Image:
    width = left.width + middle.width + right.width
    height = max(left.height, middle.height, right.height)
    canvas = Image.new("RGB", (width, height), (15, 23, 42))
    x = 0
    for img in (left, middle, right):
        y = (height - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width
    return canvas


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    image_dir = Path(args.image_dir).resolve() if args.image_dir else csv_path.parent
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    bbox_source_size = tuple(args.bbox_source_size)

    samples = load_samples(csv_path)
    if args.labels:
        allowed = {label.strip() for label in args.labels if label.strip()}
        samples = [sample for sample in samples if sample.canonical_label in allowed]

    if not samples:
        raise SystemExit("No matching samples found.")

    processed = 0
    skipped = 0

    for sample in samples:
        if processed >= args.limit:
            break

        image_path = image_dir / sample.image_index
        if not image_path.exists():
            skipped += 1
            continue

        label = sample.canonical_label
        if label not in DISEASE_LABELS:
            skipped += 1
            continue

        class_index = DISEASE_LABELS.index(label)

        try:
            probs = predict(str(image_path))
            heatmaps = generate_pth_heatmaps(
                model_path=args.model_path,
                image_file=str(image_path),
                predictions=probs,
                disease_labels=DISEASE_LABELS,
                target_index=class_index,
                top_k=1,
                threshold=0.0,
            )
        except Exception as exc:
            print(f"Skipping {sample.image_index}: {type(exc).__name__}: {exc}", file=sys.stderr)
            skipped += 1
            continue

        if label not in heatmaps:
            skipped += 1
            continue

        result = heatmaps[label]
        original = Image.open(image_path).convert("RGB")
        original_with_bbox = draw_bbox_scaled(
            original.copy(),
            sample,
            original.size,
            source_size=bbox_source_size,
        )
        heatmap_img = decode_data_url(result["heatmap"])
        overlay_img = decode_data_url(result["overlay"])
        heatmap_with_bbox = draw_bbox_scaled(
            heatmap_img.copy(),
            sample,
            heatmap_img.size,
            source_size=bbox_source_size,
            label_text=f"{sample.finding_label} bbox",
        )
        overlay_with_bbox = draw_bbox_scaled(
            overlay_img.copy(),
            sample,
            overlay_img.size,
            source_size=bbox_source_size,
            label_text=f"{sample.finding_label} bbox",
        )

        bbox_score = bbox_heatmap_energy_ratio(heatmap_img, sample)

        panel_left = make_panel(
            ImageOps.contain(original_with_bbox, (args.panel_size, args.panel_size)),
            f"Original + GT BBox | {sample.image_index}",
            args.panel_size,
        )
        panel_mid = make_panel(
            ImageOps.contain(heatmap_with_bbox, (args.panel_size, args.panel_size)),
            f"Grad-CAM Heatmap | {label} | score={bbox_score:.2f}",
            args.panel_size,
        )
        panel_right = make_panel(
            ImageOps.contain(overlay_with_bbox, (args.panel_size, args.panel_size)),
            f"Grad-CAM Overlay | p={result['probability']:.3f} | score={bbox_score:.2f}",
            args.panel_size,
        )

        triptych = compose_triptych(panel_left, panel_mid, panel_right)
        out_name = f"{Path(sample.image_index).stem}__{label}.png"
        out_path = output_dir / out_name
        triptych.save(out_path)

        meta = {
            "image_index": sample.image_index,
            "label": sample.finding_label,
            "canonical_label": label,
            "bbox": {
                "x": sample.bbox_x,
                "y": sample.bbox_y,
                "w": sample.bbox_w,
                "h": sample.bbox_h,
            },
            "gradcam_probability": result["probability"],
            "bbox_heatmap_energy_ratio": bbox_score,
            "output_path": str(out_path),
        }
        (output_dir / f"{Path(sample.image_index).stem}__{label}.json").write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )
        processed += 1
        print(f"Saved {out_path}")

    print(f"Done. Processed {processed}, skipped {skipped}. Output: {output_dir}")


if __name__ == "__main__":
    main()
