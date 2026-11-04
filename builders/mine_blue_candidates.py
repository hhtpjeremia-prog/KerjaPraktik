#!/usr/bin/env python3
"""Cari kandidat lampu biru/cyan di crop NB02 untuk anotasi terarah kelas Berwarna."""

import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_common import SPLIT_NAME_BY_CAMERA

CHUNK_DIR = PROJECT_ROOT / "output" / "nb02" / "chunks"
CROP_DIR = PROJECT_ROOT / "output" / "nb02" / "crops"
STORE_PATH = PROJECT_ROOT / "output" / "annotation" / "annotations.json"
MANIFEST_PATH = PROJECT_ROOT / "output" / "annotation" / "sample_manifest.json"
CANDIDATE_CSV_PATH = PROJECT_ROOT / "output" / "annotation" / "blue_candidates.csv"
CONTACT_SHEET_PATH = PROJECT_ROOT / "output" / "annotation" / "blue_candidates.png"

BLUE_HUE_LOW = 95
BLUE_HUE_HIGH = 135
COLORED_SATURATION_MIN = 80
COLORED_VALUE_MIN = 80
CORE_VALUE_MIN = 200
BLUE_RATIO_MIN = 0.25
BLUE_CORE_RATIO_MIN = 0.50
FRAME_CAST_VALUE_MIN = 220
FRAME_CAST_SATURATION_MIN = 40
FRAME_CAST_SATURATION_MAX = 200
AWB_CAST_SUSPECT_RATIO = 0.75
STORE_MATCH_IOU = 0.50
CONTACT_SHEET_LIMIT = 40
CONTACT_SHEET_COLUMNS = 8
THUMBNAIL_SIDE_INCHES = 1.6
CONTEXT_SCALE = 1.8


def load_chunk_boxes(chunk_directory: Path) -> list[dict]:
    """Kumpulkan seluruh box pseudo-label beserta identitas frame-nya."""
    candidate_rows = []
    for chunk_path in sorted(chunk_directory.glob("*.jsonl")):
        for line in chunk_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            for detection in record["detections"]:
                candidate_rows.append({
                    "frame_stem": record["crop_stem"],
                    "camera_id": record["camera_id"],
                    "box": detection["box"],
                    "pseudo_score": detection["score"],
                    "hint_vehicle_type": detection.get("vehicle_type"),
                })
    return candidate_rows


def load_manifest_stems(manifest_path: Path) -> set[str]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {record["crop_stem"] for record in manifest_payload["frames"]}


def load_store_boxes(store_path: Path) -> dict[str, list[dict]]:
    store_payload = json.loads(store_path.read_text(encoding="utf-8"))
    store_boxes = {}
    for frame_stem, frame_entry in store_payload["frames"].items():
        store_boxes[frame_stem] = [
            box_entry for box_entry in frame_entry["boxes"] if not box_entry["deleted"]
        ]
    return store_boxes


def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_left = max(box_a[0], box_b[0])
    intersection_top = max(box_a[1], box_b[1])
    intersection_right = min(box_a[2], box_b[2])
    intersection_bottom = min(box_a[3], box_b[3])
    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        return 0.0
    intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def match_store_class(box: list[float], store_entries: list[dict] | None) -> str:
    """Ambil kelas yang sudah dianotasi untuk box ini, bila ada dan cocok."""
    if not store_entries:
        return ""
    best_iou = 0.0
    best_class = ""
    for box_entry in store_entries:
        overlap = compute_iou(box, box_entry["box"])
        if overlap > best_iou:
            best_iou = overlap
            best_class = box_entry["class_name"]
    return best_class if best_iou >= STORE_MATCH_IOU else ""


def clamp_patch(hsv_image: np.ndarray, box: list[float]) -> np.ndarray:
    image_height, image_width = hsv_image.shape[:2]
    x_low = int(max(0, min(image_width - 1, round(box[0]))))
    x_high = int(max(x_low + 1, min(image_width, round(box[2]))))
    y_low = int(max(0, min(image_height - 1, round(box[1]))))
    y_high = int(max(y_low + 1, min(image_height, round(box[3]))))
    return hsv_image[y_low:y_high, x_low:x_high]


def measure_blue(hsv_image: np.ndarray, box: list[float]) -> dict:
    """Hitung fraksi piksel biru/cyan memakai ambang definisi 2026-10-28."""
    patch = clamp_patch(hsv_image, box)
    if patch.size == 0:
        return {}
    hue = patch[:, :, 0]
    saturation = patch[:, :, 1]
    value = patch[:, :, 2]
    bright_enough = (saturation >= COLORED_SATURATION_MIN) & (value >= COLORED_VALUE_MIN)
    blue_mask = bright_enough & (hue >= BLUE_HUE_LOW) & (hue <= BLUE_HUE_HIGH)
    core_mask = value >= CORE_VALUE_MIN
    core_bright_count = int(core_mask.sum())
    indicators = {}
    for threshold in (200, 240):
        indicators[f"v{threshold}"] = float((value >= threshold).mean())
    return {
        "blue_ratio": float(blue_mask.mean()),
        "blue_core_ratio": float((blue_mask & core_mask).sum() / core_bright_count) if core_bright_count else 0.0,
        "colored_ratio": float(bright_enough.mean()),
        "blue_hue_median": int(np.median(hue[blue_mask])) if blue_mask.any() else -1,
        "blue_saturation_median": int(np.median(saturation[blue_mask])) if blue_mask.any() else -1,
        **indicators,
    }


def measure_frame_blue_share(hsv_image: np.ndarray) -> float:
    """Fraksi piksel terang seluruh frame yang berhue biru; tinggi berarti AWB kamera bergeser biru."""
    hue = hsv_image[:, :, 0]
    saturation = hsv_image[:, :, 1]
    value = hsv_image[:, :, 2]
    bright_mask = (
        (value >= FRAME_CAST_VALUE_MIN)
        & (saturation >= FRAME_CAST_SATURATION_MIN)
        & (saturation < FRAME_CAST_SATURATION_MAX)
    )
    if not bright_mask.any():
        return 0.0
    blue_mask = bright_mask & (hue >= BLUE_HUE_LOW) & (hue <= BLUE_HUE_HIGH)
    return float(blue_mask.sum() / bright_mask.sum())


def build_contact_sheet(candidate_frame: pd.DataFrame, output_path: Path) -> None:
    """Simpan lembar kontak kandidat agar bisa dinilai manual sebelum anotasi."""
    selected = candidate_frame.head(CONTACT_SHEET_LIMIT)
    row_count = int(np.ceil(len(selected) / CONTACT_SHEET_COLUMNS))
    figure, axes = plt.subplots(
        row_count,
        CONTACT_SHEET_COLUMNS,
        figsize=(THUMBNAIL_SIDE_INCHES * CONTACT_SHEET_COLUMNS, THUMBNAIL_SIDE_INCHES * row_count),
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis in flat_axes:
        axis.axis("off")
    for axis, (_, candidate_row) in zip(flat_axes, selected.iterrows()):
        crop_image = cv2.cvtColor(cv2.imread(str(CROP_DIR / f"{candidate_row['frame_stem']}.png")), cv2.COLOR_BGR2RGB)
        image_height, image_width = crop_image.shape[:2]
        box = [candidate_row["box_x1"], candidate_row["box_y1"], candidate_row["box_x2"], candidate_row["box_y2"]]
        center_x = (box[0] + box[2]) / 2.0
        center_y = (box[1] + box[3]) / 2.0
        half_width = (box[2] - box[0]) * CONTEXT_SCALE / 2.0
        half_height = (box[3] - box[1]) * CONTEXT_SCALE / 2.0
        x_low = int(max(0, min(image_width - 1, round(center_x - half_width))))
        x_high = int(max(x_low + 1, min(image_width, round(center_x + half_width))))
        y_low = int(max(0, min(image_height - 1, round(center_y - half_height))))
        y_high = int(max(y_low + 1, min(image_height, round(center_y + half_height))))
        axis.imshow(crop_image[y_low:y_high, x_low:x_high])
        axis.add_patch(Rectangle(
            (box[0] - x_low, box[1] - y_low),
            box[2] - box[0],
            box[3] - box[1],
            fill=False,
            edgecolor="#00FFFF",
            linewidth=1.2,
        ))
        axis.set_title(
            f"{candidate_row['camera_id'].replace('bogor-kab-', '')}\n"
            f"inti biru {candidate_row['blue_core_ratio']:.2f}"
            f"{' · AWB?' if candidate_row['awb_suspect'] else ''}",
            fontsize=6,
        )
    figure.suptitle("Kandidat lampu biru/cyan (hue 95-135, inti terang didominasi biru)")
    figure.tight_layout()
    figure.savefig(output_path, dpi=140)
    plt.close(figure)


def main() -> None:
    chunk_boxes = load_chunk_boxes(CHUNK_DIR)
    manifest_stems = load_manifest_stems(MANIFEST_PATH)
    store_boxes = load_store_boxes(STORE_PATH)

    measurement_rows = []
    for frame_stem, frame_boxes in pd.DataFrame(chunk_boxes).groupby("frame_stem", sort=True):
        image_path = CROP_DIR / f"{frame_stem}.png"
        if not image_path.exists():
            continue
        hsv_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2HSV)
        frame_blue_share = measure_frame_blue_share(hsv_image)
        for _, box_row in frame_boxes.iterrows():
            measurements = measure_blue(hsv_image, box_row["box"])
            if not measurements:
                continue
            measurement_rows.append({
                "frame_stem": frame_stem,
                "camera_id": box_row["camera_id"],
                "split": SPLIT_NAME_BY_CAMERA.get(box_row["camera_id"], ""),
                "box_x1": round(float(box_row["box"][0]), 2),
                "box_y1": round(float(box_row["box"][1]), 2),
                "box_x2": round(float(box_row["box"][2]), 2),
                "box_y2": round(float(box_row["box"][3]), 2),
                "pseudo_score": round(float(box_row["pseudo_score"]), 4),
                "hint_vehicle_type": box_row["hint_vehicle_type"],
                "frame_blue_share": round(frame_blue_share, 3),
                "awb_suspect": frame_blue_share >= AWB_CAST_SUSPECT_RATIO,
                "in_manifest": frame_stem in manifest_stems,
                "manifest_status": "di manifest" if frame_stem in manifest_stems else "di luar manifest",
                "current_class": match_store_class(box_row["box"], store_boxes.get(frame_stem)),
                **measurements,
            })

    measured_frame = pd.DataFrame(measurement_rows)
    if measured_frame.empty:
        raise SystemExit("tidak ada box terukur; jalankan NB02 lebih dulu")
    loose_frame = measured_frame[measured_frame["blue_ratio"] >= BLUE_RATIO_MIN].copy()
    candidate_frame = loose_frame[loose_frame["blue_core_ratio"] >= BLUE_CORE_RATIO_MIN].copy()
    candidate_frame = candidate_frame.sort_values(
        ["blue_core_ratio", "blue_ratio"], ascending=[False, False]
    )
    candidate_frame.to_csv(CANDIDATE_CSV_PATH, index=False)
    if not candidate_frame.empty:
        build_contact_sheet(candidate_frame, CONTACT_SHEET_PATH)

    print(f"box diukur        : {len(measured_frame)} dari {measured_frame['frame_stem'].nunique()} frame")
    print(f"kandidat longgar  : {len(loose_frame)} box (biru >= {BLUE_RATIO_MIN:.0%} luas box)")
    print(f"kandidat ketat    : {len(candidate_frame)} box "
          f"(inti terang didominasi biru >= {BLUE_CORE_RATIO_MIN:.0%})")
    print(f"  di manifest     : {int(candidate_frame['in_manifest'].sum())} box")
    print(f"  di luar manifest: {int((~candidate_frame['in_manifest']).sum())} box")
    print(f"  sudah berlabel  : {int((candidate_frame['current_class'] != '').sum())} box")
    if not candidate_frame.empty:
        split_counts = candidate_frame["split"].value_counts()
        print(f"  per split       : {split_counts.to_dict()}")
        print(f"  AWB diduga geser: {int(candidate_frame['awb_suspect'].sum())} box")
    print(f"CSV               : {CANDIDATE_CSV_PATH}")
    if not candidate_frame.empty:
        print(f"lembar kontak     : {CONTACT_SHEET_PATH}")
        print("\n=== kelas yang sudah ada pada kandidat ===")
        class_counts = candidate_frame.loc[candidate_frame["current_class"] != "", "current_class"].value_counts()
        print(class_counts.to_string() if len(class_counts) else "belum ada")
        print("\n=== 15 kandidat teratas ===")
        print(candidate_frame[[
            "camera_id", "split", "frame_stem", "blue_core_ratio", "blue_ratio", "blue_hue_median",
            "frame_blue_share", "awb_suspect", "manifest_status", "current_class",
        ]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
