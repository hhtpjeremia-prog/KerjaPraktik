#!/usr/bin/env python3
"""Jalankan smoke test NB02 sampai NB06 di output/smoke/ tanpa menyentuh artefak asli."""

import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SMOKE_ROOT = PROJECT_ROOT / "output" / "smoke"
SMOKE_MANIFEST_PATH = SMOKE_ROOT / "annotation" / "sample_manifest.json"
SMOKE_STORE_PATH = SMOKE_ROOT / "annotation" / "annotations.json"

os.environ["PIPELINE_SMOKE"] = "1"
os.environ["MPLBACKEND"] = "Agg"
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_common import BOX_COUNT_CONTRACT, CLASS_NAMES, COUNT_CONTRACT, SPLIT_NAME_BY_CAMERA

BOXES_PER_FRAME = 3
BOX_WIDTH_RATIO = 0.18
BOX_HEIGHT_RATIO = 0.18
BOX_X_START_RATIO = 0.08
BOX_X_STEP_RATIO = 0.20
BOX_Y_START_RATIO = 0.15
BOX_Y_STEP_RATIO = 0.15
MIN_BOX_SPAN_PIXELS = 5

NOTEBOOK_STEPS = [
    {"name": "NB02_pseudo_label.ipynb", "skip_cells": []},
    {"name": "NB02b_annotation.ipynb", "skip_cells": [4]},
    {"name": "NB03_gold_threshold.ipynb", "skip_cells": []},
    {"name": "NB04_train_yolo.ipynb", "skip_cells": []},
    {"name": "NB05_evaluation.ipynb", "skip_cells": []},
    {"name": "NB06_tracking_count.ipynb", "skip_cells": []},
]


def strip_magics(cell_source: str) -> str:
    return "".join(
        "pass\n" if line.lstrip().startswith("%") else line for line in cell_source.splitlines(keepends=True)
    )


def build_fake_boxes(image_width: int, image_height: int, frame_position: int) -> list[dict]:
    boxes = []
    for slot in range(BOXES_PER_FRAME):
        x_low = image_width * (BOX_X_START_RATIO + BOX_X_STEP_RATIO * slot)
        y_low = image_height * (BOX_Y_START_RATIO + BOX_Y_STEP_RATIO * slot)
        x_high = min(image_width - 1.0, x_low + image_width * BOX_WIDTH_RATIO)
        y_high = min(image_height - 1.0, y_low + image_height * BOX_HEIGHT_RATIO)
        if x_high - x_low < MIN_BOX_SPAN_PIXELS or y_high - y_low < MIN_BOX_SPAN_PIXELS:
            continue
        class_name = CLASS_NAMES[(frame_position * BOXES_PER_FRAME + slot) % len(CLASS_NAMES)]
        boxes.append({
            "box": [round(x_low, 2), round(y_low, 2), round(x_high, 2), round(y_high, 2)],
            "class_name": class_name,
            "source": "manual",
            "deleted": False,
            "hint_vehicle_type": None,
        })
    return boxes


def write_fake_annotation_store() -> None:
    manifest_payload = json.loads(SMOKE_MANIFEST_PATH.read_text(encoding="utf-8"))
    store_frames = {}
    for frame_position, frame_record in enumerate(manifest_payload["frames"]):
        store_frames[frame_record["frame_stem"]] = {
            "boxes": build_fake_boxes(frame_record["image_width"], frame_record["image_height"], frame_position),
            "reviewed": True,
            "image_width": frame_record["image_width"],
            "image_height": frame_record["image_height"],
        }
    store_payload = {
        "frames": store_frames,
        "split_by_camera": dict(SPLIT_NAME_BY_CAMERA),
    }
    SMOKE_STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SMOKE_STORE_PATH.write_text(json.dumps(store_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[smoke] store anotasi palsu: {len(store_frames)} frame", flush=True)


def execute_notebook(step: dict) -> dict:
    notebook_path = PROJECT_ROOT / step["name"]
    notebook_payload = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook_payload["cells"] if cell["cell_type"] == "code"]
    namespace = {"__name__": "__smoke__", "__file__": str(notebook_path)}
    print(f"\n[smoke] === {step['name']} ({len(code_cells)} sel kode) ===", flush=True)
    for cell_index, cell in enumerate(code_cells):
        if cell_index in step["skip_cells"]:
            print(f"[smoke] sel {cell_index}: DILEWATI", flush=True)
            continue
        cell_source = strip_magics("".join(cell["source"]))
        try:
            exec(compile(cell_source, f"{step['name']}[{cell_index}]", "exec"), namespace)
        except Exception as cell_error:
            raise RuntimeError(f"{step['name']} sel {cell_index} gagal: {cell_error}") from cell_error
        print(f"[smoke] sel {cell_index}: OK", flush=True)
        if step["name"] == "NB02b_annotation.ipynb" and cell_index == 1:
            write_fake_annotation_store()
    return namespace


def assert_artifact(relative_path: str) -> Path:
    artifact_path = SMOKE_ROOT / relative_path
    if not artifact_path.exists():
        raise FileNotFoundError(f"artefak smoke tidak terbentuk: {artifact_path}")
    print(f"[smoke] artefak ada: {relative_path}", flush=True)
    return artifact_path


def check_counting_contract(
    totals: dict,
    class_counts: dict,
    contract: dict,
    mobil_key: str,
    motor_key: str,
    total_units: int,
) -> None:
    for contract_key, contract_classes in contract.items():
        if totals[contract_key] == "n/a":
            continue
        expected_total = sum(class_counts[class_name] for class_name in contract_classes)
        if totals[contract_key] != expected_total:
            raise AssertionError(f"{contract_key}: {totals[contract_key]} != {expected_total}")
    partitioned_units = totals[mobil_key] + totals[motor_key] + class_counts["PantulanCahaya"]
    if partitioned_units != total_units:
        raise AssertionError(f"partisi kelas ({partitioned_units}) != unit terhitung ({total_units})")
    print("[smoke] kontrak hitung konsisten", flush=True)


def main() -> None:
    if SMOKE_ROOT.exists():
        shutil.rmtree(SMOKE_ROOT)
        print(f"[smoke] smoke root lama dihapus: {SMOKE_ROOT}", flush=True)
    os.chdir(PROJECT_ROOT)

    attribute_model_source = PROJECT_ROOT / "output" / "attribute_classifier" / "best.pt"
    attribute_model_target = SMOKE_ROOT / "attribute_classifier" / "best.pt"
    if not attribute_model_source.exists():
        raise FileNotFoundError(
            f"classifier atribut tidak ada: {attribute_model_source}. "
            "Jalankan builders/train_attribute_classifier.py lebih dulu."
        )
    attribute_model_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(attribute_model_source, attribute_model_target)
    print("[smoke] classifier atribut disalin ke smoke root", flush=True)

    for step in NOTEBOOK_STEPS:
        execute_notebook(step)

    assert_artifact("nb02/nb02_summary.json")
    assert_artifact("nb02/chunks/chunk_0000.jsonl")
    assert_artifact("annotation/annotations_coco.json")
    assert_artifact("annotation/yolo/split_map.csv")
    assert_artifact("nb03/gold_audit.json")
    assert_artifact("nb03/photometry_by_class.png")
    assert_artifact("nb04/val_metrics.json")
    assert_artifact("nb04/runs/nb04_yolo11n_smoke/weights/best.pt")
    assert_artifact("nb05/thresholds.json")
    assert_artifact("nb05/counting_bar.png")
    assert_artifact("nb06/track_summary.csv")
    assert_artifact("nb06/roi_mapping_audit.csv")
    assert_artifact("nb06/counting_bar.png")

    counting_payload = json.loads(assert_artifact("nb05/counting_summary.json").read_text(encoding="utf-8"))
    if counting_payload.get("counting_unit") != "box_per_frame":
        raise AssertionError("NB05 harus menandai hitungannya sebagai box_per_frame")
    check_counting_contract(
        counting_payload["box_totals"],
        counting_payload["class_counts"],
        BOX_COUNT_CONTRACT,
        "jumlah_box_mobil",
        "jumlah_box_motor",
        counting_payload["counted_boxes"],
    )
    tracking_payload = json.loads(assert_artifact("nb06/counting_summary.json").read_text(encoding="utf-8"))
    check_counting_contract(
        tracking_payload["counting_totals"],
        tracking_payload["class_counts"],
        COUNT_CONTRACT,
        "jumlah_kendaraan_mobil",
        "jumlah_kendaraan_motor",
        tracking_payload["vehicles_counted"],
    )
    if tracking_payload["frames_processed"] == 0:
        raise AssertionError("NB06 memproses 0 frame")
    if tracking_payload["videos_counted"] == 0:
        raise AssertionError("NB06 menghitung 0 video")

    print("\n[smoke] === jumlah box per frame (NB05) ===", flush=True)
    for contract_key, contract_value in counting_payload["box_totals"].items():
        print(f"[smoke] {contract_key}: {contract_value}", flush=True)
    print(f"[smoke] frame dihitung: {counting_payload['frames_counted']}", flush=True)
    print(f"[smoke] box lolos ambang: {counting_payload['counted_boxes']}", flush=True)
    print("\n[smoke] === 8 angka kendaraan unik (NB06) ===", flush=True)
    for contract_key, contract_value in tracking_payload["counting_totals"].items():
        print(f"[smoke] {contract_key}: {contract_value}", flush=True)
    print(
        f"[smoke] video ditemukan: {tracking_payload['videos_found']} · "
        f"video dihitung: {tracking_payload['videos_counted']}",
        flush=True,
    )
    print(f"[smoke] frame tracking: {tracking_payload['frames_processed']}", flush=True)
    print(
        f"[smoke] track mentah: {tracking_payload['tracks_total']} · "
        f"fragmen disambung: {tracking_payload['fragments_stitched']} · "
        f"kendaraan dihitung: {tracking_payload['vehicles_counted']}",
        flush=True,
    )
    print("[smoke] SELESAI TANPA ERROR", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as fatal_error:
        print(f"[smoke] GAGAL: {fatal_error}", flush=True)
        sys.exit(1)
