#!/usr/bin/env python3
"""Self-check logika hitung bersama: kontrak angka, penyambungan fragmen, dan guard ambang tracker."""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from attribute_classifier import FEATURE_SIZE, extract_pixel_features
from dashboard.data import (
    CAMERA_STATUS_COUNTED,
    STATE_EMPTY,
    build_camera_status_table,
    build_camera_table,
    format_count,
    load_counting_snapshot,
)
from pipeline_common import (
    BOX_COUNT_CONTRACT,
    CLASS_NAMES,
    COUNT_CONTRACT,
    UNMEASURABLE_COUNT_CONTRACT_KEYS,
    UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS,
    TRAINING_CLASS_NAMES,
    box_overlap_iou,
    filter_store_frames,
    stitch_track_fragments,
    training_class_name,
    validate_tracker_thresholds,
)

TRACK_COLUMNS = [
    "video_stem", "unit_id", "camera_id", "class_name", "hits", "first_frame", "last_frame", "mean_score",
    "first_x1", "first_y1", "first_x2", "first_y2", "last_x1", "last_y1", "last_x2", "last_y2",
]

SAME_PLACE_BOX = (100.0, 100.0, 160.0, 140.0)
NEAR_PLACE_BOX = (104.0, 102.0, 164.0, 142.0)
FAR_PLACE_BOX = (500.0, 300.0, 560.0, 340.0)


def build_track_row(
    unit_id: int,
    class_name: str,
    first_frame: int,
    last_frame: int,
    first_box: tuple[float, float, float, float],
    last_box: tuple[float, float, float, float],
) -> dict:
    return {
        "video_stem": "video_uji",
        "unit_id": unit_id,
        "camera_id": "bogor-kab-bappenda2",
        "class_name": class_name,
        "hits": last_frame - first_frame + 1,
        "first_frame": first_frame,
        "last_frame": last_frame,
        "mean_score": 0.5,
        "first_x1": first_box[0],
        "first_y1": first_box[1],
        "first_x2": first_box[2],
        "first_y2": first_box[3],
        "last_x1": last_box[0],
        "last_y1": last_box[1],
        "last_x2": last_box[2],
        "last_y2": last_box[3],
    }


def count_vehicles(track_rows: list[dict]) -> int:
    track_frame = pd.DataFrame(track_rows, columns=TRACK_COLUMNS)
    return int(stitch_track_fragments(track_frame)["vehicle_id"].nunique())


def check_contract_rollup_is_complete() -> None:
    assert COUNT_CONTRACT["jumlah_kendaraan_mobil"] == [
        class_name for class_name in CLASS_NAMES if class_name.startswith("Mobil")
    ]
    assert COUNT_CONTRACT["jumlah_kendaraan_motor"] == [
        class_name for class_name in CLASS_NAMES if class_name.startswith("Motor")
    ]
    for contract_key, contract_classes in COUNT_CONTRACT.items():
        unknown_classes = [class_name for class_name in contract_classes if class_name not in CLASS_NAMES]
        assert not unknown_classes, f"{contract_key} memuat kelas tak dikenal: {unknown_classes}"
    assert "MobilSilauBerwarna" in COUNT_CONTRACT["jumlah_lampu_silau_mobil"]
    assert "MobilSilauBerwarna" in COUNT_CONTRACT["jumlah_lampu_berwarna_mobil"]
    assert COUNT_CONTRACT["jumlah_lampu_silau_dan_berwarna_mobil"] == ["MobilSilauBerwarna"]
    assert COUNT_CONTRACT["jumlah_lampu_silau_dan_berwarna_motor"] == ["MotorSilauBerwarna"]


def check_box_contract_partitions_all_classes() -> None:
    assert BOX_COUNT_CONTRACT["jumlah_box_mobil"] == COUNT_CONTRACT["jumlah_kendaraan_mobil"]
    assert BOX_COUNT_CONTRACT["jumlah_box_motor"] == COUNT_CONTRACT["jumlah_kendaraan_motor"]
    partition_total = (
        len(BOX_COUNT_CONTRACT["jumlah_box_mobil"])
        + len(BOX_COUNT_CONTRACT["jumlah_box_motor"])
        + 1
    )
    assert partition_total == len(CLASS_NAMES)
    assert "PantulanCahaya" not in sum(BOX_COUNT_CONTRACT.values(), [])


def check_mobil_fragment_not_stitched_to_motor() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MobilSilau", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MotorSilau", 8, 12, SAME_PLACE_BOX, SAME_PLACE_BOX),
    ])
    assert vehicles_counted == 2, f"fragmen lintas jenis tersambung: {vehicles_counted} kendaraan"


def check_overlapping_fragments_rejected() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MobilSilau", 1, 10, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MobilSilau", 10, 14, SAME_PLACE_BOX, SAME_PLACE_BOX),
    ])
    assert vehicles_counted == 2, f"fragmen tumpang waktu tersambung: {vehicles_counted} kendaraan"


def check_distant_fragments_rejected() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MotorNormal", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MotorNormal", 8, 12, FAR_PLACE_BOX, FAR_PLACE_BOX),
    ])
    assert vehicles_counted == 2, f"fragmen berjauhan tersambung: {vehicles_counted} kendaraan"


def check_gap_beyond_limit_rejected() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MobilNormal", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MobilNormal", 56, 60, SAME_PLACE_BOX, SAME_PLACE_BOX),
    ])
    assert vehicles_counted == 2, f"fragmen melewati batas jeda tersambung: {vehicles_counted} kendaraan"


def check_adjacent_fragments_stitched() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MobilNormal", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MobilNormal", 8, 12, NEAR_PLACE_BOX, NEAR_PLACE_BOX),
    ])
    assert vehicles_counted == 1, f"fragmen bersambung tidak menyatu: {vehicles_counted} kendaraan"


def check_chain_of_fragments_stitched() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MotorBerwarna", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "MotorBerwarna", 8, 12, NEAR_PLACE_BOX, NEAR_PLACE_BOX),
        build_track_row(3, "MotorBerwarna", 15, 19, SAME_PLACE_BOX, SAME_PLACE_BOX),
    ])
    assert vehicles_counted == 1, f"rantai fragmen tidak menyatu: {vehicles_counted} kendaraan"


def check_reflection_stays_separate() -> None:
    vehicles_counted = count_vehicles([
        build_track_row(1, "MobilSilau", 1, 5, SAME_PLACE_BOX, SAME_PLACE_BOX),
        build_track_row(2, "PantulanCahaya", 8, 12, SAME_PLACE_BOX, SAME_PLACE_BOX),
    ])
    assert vehicles_counted == 2, f"pantulan menyatu ke kendaraan: {vehicles_counted} kendaraan"


def check_tracker_threshold_guard() -> None:
    validate_tracker_thresholds({"new_track_thresh": 0.05}, {"MobilSilau": 0.05, "MotorNormal": 0.30})
    validate_tracker_thresholds({"new_track_thresh": 0.02}, {"MobilSilau": 0.05, "MotorNormal": 0.30})
    try:
        validate_tracker_thresholds({"new_track_thresh": 0.35}, {"MobilSilau": 0.05, "MotorNormal": 0.30})
    except ValueError:
        return
    raise AssertionError("guard ambang tracker tidak menolak new_track_thresh yang terlalu tinggi")


def check_iou_bounds() -> None:
    assert box_overlap_iou(list(SAME_PLACE_BOX), list(SAME_PLACE_BOX)) == 1.0
    assert box_overlap_iou(list(SAME_PLACE_BOX), list(FAR_PLACE_BOX)) == 0.0


def check_training_class_names() -> None:
    assert TRAINING_CLASS_NAMES == [
        "MobilNormal",
        "MobilSilau",
        "MotorNormal",
        "MotorSilau",
        "PantulanCahaya",
    ]
    assert training_class_name("MobilBerwarna") == "MobilNormal"
    assert training_class_name("MobilSilauBerwarna") == "MobilSilau"
    assert training_class_name("MotorBerwarna") == "MotorNormal"
    assert training_class_name("MotorSilauBerwarna") == "MotorSilau"
    assert training_class_name("PantulanCahaya") == "PantulanCahaya"
    unmeasurable = sorted(UNMEASURABLE_COUNT_CONTRACT_KEYS)
    assert unmeasurable == [], unmeasurable
    unmeasurable_box = sorted(UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS)
    assert unmeasurable_box == [
        "jumlah_box_lampu_berwarna_mobil",
        "jumlah_box_lampu_berwarna_motor",
        "jumlah_box_lampu_silau_dan_berwarna_mobil",
        "jumlah_box_lampu_silau_dan_berwarna_motor",
    ], unmeasurable_box


def check_attributes_features_are_finite() -> None:
    grey_frame = Image.new("RGB", (40, 40), (128, 128, 128))
    grey_features = extract_pixel_features(grey_frame, [0.0, 0.0, 40.0, 40.0])
    assert grey_features.shape == (FEATURE_SIZE,), grey_features.shape
    assert np.isfinite(grey_features).all(), "fitur NaN saat crop tanpa piksel berwarna"
    blue_frame = Image.new("RGB", (40, 40), (20, 40, 220))
    blue_features = extract_pixel_features(blue_frame, [0.0, 0.0, 40.0, 40.0])
    assert np.isfinite(blue_features).all(), "fitur NaN saat crop berwarna"
    hue_histogram_total = float(blue_features[-16:].sum())
    assert abs(hue_histogram_total - 1.0) < 1e-5, hue_histogram_total


def check_store_filter_keeps_only_active_frames() -> None:
    store_frames = {"frame_a": {"reviewed": True}, "frame_b": {"reviewed": False}, "frame_c": {"reviewed": True}}
    kept_frames = filter_store_frames(store_frames, {"frame_a", "frame_c"})
    assert sorted(kept_frames) == ["frame_a", "frame_c"], sorted(kept_frames)
    assert filter_store_frames({}, {"frame_a"}) == {}


def check_dashboard_reports_empty_state() -> None:
    with tempfile.TemporaryDirectory() as empty_directory:
        snapshot = load_counting_snapshot(Path(empty_directory))
    assert snapshot.state == STATE_EMPTY, snapshot.state
    assert len(snapshot.missing_artifacts) == 2, snapshot.missing_artifacts
    assert snapshot.camera_table is None
    assert snapshot.summary is None


def check_dashboard_count_formatting_is_honest() -> None:
    assert format_count("n/a") == "n/a"
    assert format_count(None) == "n/a"
    assert format_count(float("nan")) == "n/a"
    assert format_count(0) == "0"
    assert format_count(999) == "999"
    assert format_count(1_234) == "1.2k"
    assert format_count(1_000) == "1k"
    assert format_count(38_000) == "38k"
    assert format_count(1_400_000) == "1.4M"


def check_dashboard_camera_table_sums_per_camera() -> None:
    per_video_frame = pd.DataFrame(
        [
            {
                "video_stem": "video_a",
                "camera_id": "bogor-kab-pdam2",
                "vehicles_counted": 4,
                "jumlah_kendaraan_mobil": 3,
                "jumlah_kendaraan_motor": 1,
                "jumlah_lampu_berwarna_mobil": 1,
            },
            {
                "video_stem": "video_b",
                "camera_id": "bogor-kab-pdam2",
                "vehicles_counted": 2,
                "jumlah_kendaraan_mobil": 2,
                "jumlah_kendaraan_motor": 0,
                "jumlah_lampu_berwarna_mobil": 2,
            },
            {
                "video_stem": "video_c",
                "camera_id": "bogor-kab-bappenda2",
                "vehicles_counted": 5,
                "jumlah_kendaraan_mobil": 1,
                "jumlah_kendaraan_motor": 4,
                "jumlah_lampu_berwarna_mobil": "n/a",
            },
        ]
    )
    camera_table = build_camera_table(per_video_frame)
    camera_totals = camera_table.set_index("camera_id")
    assert camera_totals.loc["bogor-kab-pdam2", "vehicles_counted"] == 6
    assert camera_totals.loc["bogor-kab-pdam2", "videos_counted"] == 2
    assert camera_totals.loc["bogor-kab-bappenda2", "jumlah_kendaraan_motor"] == 4
    assert "jumlah_lampu_berwarna_mobil" in camera_table.columns
    status_frame = build_camera_status_table(camera_table, ["test"])
    status_by_camera = status_frame.set_index("camera_id")
    assert status_by_camera.loc["bogor-kab-pdam2", "status"] == CAMERA_STATUS_COUNTED
    assert "di luar split hitung (train)" in status_by_camera.loc["bogor-kab-bappenda1", "status"]
    assert pd.isna(status_by_camera.loc["bogor-kab-bappenda1", "vehicles_counted"])


def main() -> None:
    checks = [
        check_contract_rollup_is_complete,
        check_box_contract_partitions_all_classes,
        check_mobil_fragment_not_stitched_to_motor,
        check_overlapping_fragments_rejected,
        check_distant_fragments_rejected,
        check_gap_beyond_limit_rejected,
        check_adjacent_fragments_stitched,
        check_chain_of_fragments_stitched,
        check_reflection_stays_separate,
        check_tracker_threshold_guard,
        check_iou_bounds,
        check_training_class_names,
        check_attributes_features_are_finite,
        check_store_filter_keeps_only_active_frames,
        check_dashboard_reports_empty_state,
        check_dashboard_count_formatting_is_honest,
        check_dashboard_camera_table_sums_per_camera,
    ]
    for check in checks:
        check()
        print(f"OK  {check.__name__}")
    print(f"Semua {len(checks)} pemeriksaan lolos.")


if __name__ == "__main__":
    main()
