#!/usr/bin/env python3
"""Probe fotometrik gold set: uji apakah fitur piksel memisahkan silau dari normal."""

import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = PROJECT_ROOT / "output" / "annotation" / "annotations.json"
CROP_DIR = PROJECT_ROOT / "output" / "nb02" / "crops"
PROBE_PATH = PROJECT_ROOT / "output" / "annotation" / "label_probe.csv"

BRIGHT_VALUE_THRESHOLDS = (180, 200, 240)
WHITE_SATURATION_MAX = 40
COLORED_SATURATION_MIN = 80
COLORED_VALUE_MIN = 80
HUE_BUCKET_SIZE = 10
RING_SCALE = 1.6
CUTOFF_STEP = 0.01


def clamp_region(
    box: list[float],
    scale: float,
    image_width: int,
    image_height: int,
) -> tuple[slice, slice]:
    """Ambil jendela di sekitar box, diperbesar `scale`, dipotong ke batas gambar."""
    center_x = (box[0] + box[2]) / 2.0
    center_y = (box[1] + box[3]) / 2.0
    half_width = (box[2] - box[0]) * scale / 2.0
    half_height = (box[3] - box[1]) * scale / 2.0
    x_low = int(max(0, round(center_x - half_width)))
    x_high = int(min(image_width, round(center_x + half_width)))
    y_low = int(max(0, round(center_y - half_height)))
    y_high = int(min(image_height, round(center_y + half_height)))
    return slice(y_low, y_high), slice(x_low, x_high)


def measure_box(hsv_image: np.ndarray, box: list[float]) -> dict:
    """Hitung fitur fotometrik di dalam box dan di ring sekeliling box."""
    image_height, image_width = hsv_image.shape[:2]
    inner_rows, inner_columns = clamp_region(box, 1.0, image_width, image_height)
    patch = hsv_image[inner_rows, inner_columns]
    if patch.size == 0:
        return {}
    saturation = patch[:, :, 1]
    value = patch[:, :, 2]
    colored_mask = (saturation >= COLORED_SATURATION_MIN) & (value >= COLORED_VALUE_MIN)
    dominant_hue = -1
    if colored_mask.any():
        colored_hue = patch[:, :, 0][colored_mask]
        hue_histogram = np.bincount(colored_hue // HUE_BUCKET_SIZE)
        dominant_hue = int(hue_histogram.argmax()) * HUE_BUCKET_SIZE
    measurements = {
        "colored_ratio": float(colored_mask.mean()),
        "dominant_hue": dominant_hue,
        "white_bloom_ratio": float(((value >= 200) & (saturation < WHITE_SATURATION_MAX)).mean()),
    }
    for threshold in BRIGHT_VALUE_THRESHOLDS:
        measurements[f"v{threshold}"] = float((value >= threshold).mean())
    ring = measure_ring(hsv_image, box)
    measurements.update(ring)
    return measurements


def measure_ring(hsv_image: np.ndarray, box: list[float]) -> dict:
    """Hitung fraksi piksel terang di ring antara box dan jendela yang diperbesar."""
    image_height, image_width = hsv_image.shape[:2]
    outer_rows, outer_columns = clamp_region(box, RING_SCALE, image_width, image_height)
    inner_rows, inner_columns = clamp_region(box, 1.0, image_width, image_height)
    outer_value = hsv_image[outer_rows, outer_columns, 2]
    if outer_value.size == 0:
        return {}
    ring_mask = np.ones(outer_value.shape, dtype=bool)
    ring_mask[
        inner_rows.start - outer_rows.start:inner_rows.stop - outer_rows.start,
        inner_columns.start - outer_columns.start:inner_columns.stop - outer_columns.start,
    ] = False
    if not ring_mask.any():
        return {}
    ring_value = outer_value[ring_mask]
    return {
        f"ring_v{threshold}": float((ring_value >= threshold).mean())
        for threshold in (180, 200)
    }


def load_measurement_frame(store_payload: dict, crop_directory: Path) -> pd.DataFrame:
    """Ukur seluruh box aktif pada frame yang sudah reviewed."""
    measurement_rows = []
    for frame_stem, frame_entry in store_payload["frames"].items():
        if not frame_entry["reviewed"]:
            continue
        active_boxes = [box_entry for box_entry in frame_entry["boxes"] if not box_entry["deleted"]]
        if not active_boxes:
            continue
        image_path = crop_directory / f"{frame_stem}.png"
        if not image_path.exists():
            continue
        hsv_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2HSV)
        for box_entry in active_boxes:
            measurements = measure_box(hsv_image, box_entry["box"])
            if measurements:
                measurement_rows.append({
                    "frame_stem": frame_stem,
                    "camera_id": frame_stem.split("_")[1],
                    "class_name": box_entry["class_name"],
                    **measurements,
                })
    return pd.DataFrame(measurement_rows)


def find_best_cutoff(
    feature_name: str,
    measured_frame: pd.DataFrame,
    positive_mask: pd.Series,
    negative_mask: pd.Series,
) -> dict:
    """Cari ambang dengan Youden index tertinggi untuk satu fitur."""
    feature_values = measured_frame[feature_name]
    positive_values = feature_values[positive_mask].dropna()
    negative_values = feature_values[negative_mask].dropna()
    cutoff_grid = np.arange(float(feature_values.min()), float(feature_values.max()), CUTOFF_STEP)
    best_row = {"feature": feature_name, "cutoff": float("nan"), "recall": 0.0,
                "false_positive": 1.0, "youden": -1.0}
    for cutoff in cutoff_grid:
        recall = float((positive_values >= cutoff).mean())
        false_positive = float((negative_values >= cutoff).mean())
        youden = recall - false_positive
        if youden > best_row["youden"]:
            best_row = {"feature": feature_name, "cutoff": round(float(cutoff), 2),
                        "recall": round(recall, 3), "false_positive": round(false_positive, 3),
                        "youden": round(youden, 3)}
    return best_row


def main() -> None:
    store_payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    measured_frame = load_measurement_frame(store_payload, CROP_DIR)
    if measured_frame.empty:
        raise SystemExit("tidak ada box terukur; jalankan NB02b dan anotasi minimal 1 frame")
    PROBE_PATH.parent.mkdir(parents=True, exist_ok=True)
    measured_frame.to_csv(PROBE_PATH, index=False)

    print(f"box terukur: {len(measured_frame)} dari {measured_frame['frame_stem'].nunique()} frame")
    print(f"CSV: {PROBE_PATH}")

    print("\n=== jumlah & median fitur per kelas ===")
    class_summary = measured_frame.groupby("class_name").agg(
        boxes=("class_name", "size"),
        v180=("v180", "median"),
        v200=("v200", "median"),
        v240=("v240", "median"),
        white_bloom=("white_bloom_ratio", "median"),
        colored_ratio=("colored_ratio", "median"),
    ).round(3).sort_values("boxes", ascending=False)
    print(class_summary.to_string())

    silau_mask = measured_frame["class_name"].str.contains("Silau")
    normal_mask = measured_frame["class_name"].str.contains("Normal")
    reflection_mask = measured_frame["class_name"] == "PantulanCahaya"
    print(f"\nsilau n={int(silau_mask.sum())} · normal n={int(normal_mask.sum())} · "
          f"pantulan n={int(reflection_mask.sum())}")

    print("\n=== daya pisah silau vs normal ===")
    separation_rows = [
        find_best_cutoff(feature_name, measured_frame, silau_mask, normal_mask)
        for feature_name in ["v240", "v200", "v180", "white_bloom_ratio", "ring_v180", "ring_v200"]
        if feature_name in measured_frame.columns
    ]
    separation_frame = pd.DataFrame(separation_rows)
    for row in separation_rows:
        reflection_share = float(
            (measured_frame.loc[reflection_mask, row["feature"]] >= row["cutoff"]).mean()
        )
        separation_frame.loc[
            separation_frame["feature"] == row["feature"], "pantulan_ikut"
        ] = round(reflection_share, 3)
    print(separation_frame.to_string(index=False))

    print("\n=== box berlabel berwarna ===")
    colored_frame = measured_frame[measured_frame["class_name"].str.contains("Berwarna")]
    if colored_frame.empty:
        print("tidak ada")
    else:
        print(colored_frame[[
            "class_name", "dominant_hue", "colored_ratio", "v200", "v240",
        ]].to_string(index=False))

    print("\n=== box dengan piksel berwarna kuat (colored_ratio >= 0.25) ===")
    saturated_frame = measured_frame[measured_frame["colored_ratio"] >= 0.25]
    print(f"jumlah {len(saturated_frame)} dari {len(measured_frame)}")
    print(saturated_frame["class_name"].value_counts().to_string())
    print("hue bucket:", saturated_frame["dominant_hue"].value_counts().sort_index().to_dict())


if __name__ == "__main__":
    main()
