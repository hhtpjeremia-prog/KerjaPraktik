#!/usr/bin/env python3
"""Build NB05_evaluation.ipynb: kalibrasi confidence YOLO dan baseline bbox per frame."""

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_gate import verify_code_cells


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB05_evaluation.ipynb"

EXPECTED_CODE_CELL_COUNT = 6
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB05 — Kalibrasi Threshold Detector

Tahap terakhir: tentukan ambang *confidence* per kelas di split `val`, lalu pakai ambang itu untuk
menghitung keluaran yang diminta riset.

**Posisi:** `NB02 → NB02b → NB03 → NB04 (latih) → NB05 (kalibrasi + hitung)`

## Evaluasi dan baseline

1. **Kalibrasi (split `val`, ada ground truth).** Untuk setiap kelas, ambang confidence disapu dari
   `CONF_FLOOR` sampai `CONF_CEILING`; ambang terpilih = F1 terbaik. Prediksi di-matching ke gold
   dengan IoU ≥ `MATCH_IOU`, greedy per kelas, prediksi ber-confidence lebih tinggi didahulukan.
   Ambang **tidak pernah** dipilih dari `train` maupun dari kamera test.
2. **Penghitungan (kamera `test`).** Model dijalankan pada frame kamera test,
   box difilter dengan ambang hasil kalibrasi, lalu jumlah bbox Mobil dan Motor dilaporkan terpisah.

## Hitungan baseline NB05

NB05 **bukan** sumber counting utama. Sel hitung di sini menghitung **box per frame** pada frame
test tersampling untuk memeriksa cakupan dan keseimbangan kelas pada ambang terpilih. Counting
dashboard dari video kontinu diterbitkan NB06 sebagai bbox tidak standar per frame.

| Keluaran | Penjumlahan kelas |
|---|---|
| Jumlah bbox mobil | deteksi kelas `Mobil` lolos threshold |
| Jumlah bbox motor | deteksi kelas `Motor` lolos threshold |

NB05 mengevaluasi detector tipe kendaraan. Klasifikasi kondisi lampu dan hitungan bbox tidak standar
dihasilkan pada tahap berikutnya oleh NB04b dan NB06.

## Batas yang harus ditulis di laporan

Jika NB02c sudah mengekspor anotasi holdout yang direview, sel terakhir menghitung metrik detector
test (precision, recall, mAP@50, mAP@50:95, serta error ukuran/koordinat bbox). Jika belum ada GT,
metrik berstatus unavailable dan tidak dibuat-buat. Angka hitung tetap **jumlah box pada
frame tersampling unik**, bukan estimasi arus lalu lintas per jam dan bukan jumlah kendaraan unik.

## Mode smoke

`PIPELINE_SMOKE=1` → root `output/smoke/`, ambang grid dipersempit. Angka yang keluar dari smoke
**bukan** hasil riset.

Output: `output/nb05/{val_calibration.csv, thresholds.json, counting_by_frame.csv,
counting_by_camera.csv, counting_summary.json, counting_bar.png}`.
"""

CONFIG_SOURCE = '''from pathlib import Path
import json
import os

import numpy as np
import pandas as pd
import cv2
import torch
from IPython.display import Markdown, display

from pipeline_common import BOX_COUNT_CONTRACT, TRAINING_CLASS_NAMES, training_class_name

from pipeline_common import resolve_project_root
RISET_ROOT = resolve_project_root(Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())))
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT

NB02_ROOT = DATA_ROOT / "nb02"
CHUNK_DIR = NB02_ROOT / "chunks"
CROP_DIR = NB02_ROOT / "crops"
ANNOTATION_ROOT = DATA_ROOT / "annotation"
ANNOTATION_STORE_PATH = ANNOTATION_ROOT / "annotations.json"
SAMPLE_MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"
NB04_ROOT = DATA_ROOT / "nb04"
DATASET_IMAGE_DIR = NB04_ROOT / "dataset" / "images"
VAL_METRICS_PATH = NB04_ROOT / "val_metrics.json"
NB05_ROOT = DATA_ROOT / "nb05"
VAL_CALIBRATION_PATH = NB05_ROOT / "val_calibration.csv"
THRESHOLDS_PATH = NB05_ROOT / "thresholds.json"
COUNTING_BY_FRAME_PATH = NB05_ROOT / "counting_by_frame.csv"
COUNTING_BY_CAMERA_PATH = NB05_ROOT / "counting_by_camera.csv"
COUNTING_SUMMARY_PATH = NB05_ROOT / "counting_summary.json"
COUNTING_FIGURE_PATH = NB05_ROOT / "counting_bar.png"

NB05_ROOT.mkdir(parents=True, exist_ok=True)

TEST_SPLIT_NAME = "test"
WEIGHTS_PATH = NB04_ROOT / "runs" / ("nb04_yolo11n_smoke" if SMOKE_MODE else "nb04_yolo11n") / "weights" / "best.pt"

IMAGE_SIZE = 640
BATCH_SIZE = 8
NMS_IOU = 0.70
MATCH_IOU = 0.50
CONF_FLOOR = 0.05
CONF_CEILING = 0.90
CONF_STEP = 0.05 if not SMOKE_MODE else 0.20
FALLBACK_CONF = 0.25
MIN_VAL_BOXES_PER_CLASS = 1 if SMOKE_MODE else 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

display(Markdown(
    f"NB05 siap. Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · device: **{DEVICE}** · "
    f"bobot: `{WEIGHTS_PATH}` (ada: **{WEIGHTS_PATH.exists()}**) · kelas: {len(TRAINING_CLASS_NAMES)}"
))'''

CALIBRATE_SOURCE = '''from ultralytics import YOLO

if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"bobot tidak ada: {WEIGHTS_PATH}. Jalankan NB04 lebih dulu.")


def load_json_document(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"berkas wajib tidak ada: {path}")
    with path.open(encoding="utf-8") as source_file:
        return json.load(source_file)


def assert_class_order_matches_model(model, expected_names: list[str]) -> None:
    model_names = [model.names[class_index] for class_index in sorted(model.names)]
    if model_names != expected_names:
        raise ValueError(
            f"urutan kelas bobot {model_names} tidak sama dengan kontrak {expected_names}. "
            "Latih ulang di NB04 dengan TRAINING_CLASS_NAMES yang sama."
        )


def box_iou_matrix(prediction_boxes: np.ndarray, target_boxes: np.ndarray) -> np.ndarray:
    prediction_x1 = prediction_boxes[:, 0][:, None]
    prediction_y1 = prediction_boxes[:, 1][:, None]
    prediction_x2 = prediction_boxes[:, 2][:, None]
    prediction_y2 = prediction_boxes[:, 3][:, None]
    target_x1 = target_boxes[:, 0][None, :]
    target_y1 = target_boxes[:, 1][None, :]
    target_x2 = target_boxes[:, 2][None, :]
    target_y2 = target_boxes[:, 3][None, :]
    intersection_width = np.clip(np.minimum(prediction_x2, target_x2) - np.maximum(prediction_x1, target_x1), 0, None)
    intersection_height = np.clip(np.minimum(prediction_y2, target_y2) - np.maximum(prediction_y1, target_y1), 0, None)
    intersection_area = intersection_width * intersection_height
    prediction_area = np.clip(prediction_x2 - prediction_x1, 0, None) * np.clip(prediction_y2 - prediction_y1, 0, None)
    target_area = np.clip(target_x2 - target_x1, 0, None) * np.clip(target_y2 - target_y1, 0, None)
    union_area = prediction_area + target_area - intersection_area
    return np.where(union_area > 0, intersection_area / np.maximum(union_area, 1e-9), 0.0)


def match_predictions(prediction_frame: pd.DataFrame, target_frame: pd.DataFrame, iou_threshold: float) -> pd.DataFrame:
    matched_flags = np.zeros(len(prediction_frame), dtype=bool)
    if prediction_frame.empty or target_frame.empty:
        prediction_frame = prediction_frame.copy()
        prediction_frame["matched"] = matched_flags
        return prediction_frame
    for class_name in TRAINING_CLASS_NAMES:
        class_predictions = prediction_frame[prediction_frame["class_name"] == class_name].sort_values("score", ascending=False)
        class_targets = target_frame[target_frame["class_name"] == class_name]
        if class_predictions.empty or class_targets.empty:
            continue
        prediction_boxes = class_predictions[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(dtype=float)
        target_boxes = class_targets[["box_x1", "box_y1", "box_x2", "box_y2"]].to_numpy(dtype=float)
        iou_matrix = box_iou_matrix(prediction_boxes, target_boxes)
        used_target_flags = np.zeros(len(class_targets), dtype=bool)
        for prediction_position, prediction_index in enumerate(class_predictions.index):
            candidate_iou = np.where(used_target_flags, -1.0, iou_matrix[prediction_position])
            best_target = int(np.argmax(candidate_iou))
            if candidate_iou[best_target] >= iou_threshold:
                used_target_flags[best_target] = True
                matched_flags[prediction_frame.index.get_loc(prediction_index)] = True
    prediction_frame = prediction_frame.copy()
    prediction_frame["matched"] = matched_flags
    return prediction_frame


def load_gold_targets(store_payload: dict, frame_stem: str) -> pd.DataFrame:
    frame_entry = store_payload["frames"][frame_stem]
    target_rows = [
        {
            "class_name": training_class_name(box_entry["class_name"]),
            "box_x1": float(box_entry["box"][0]),
            "box_y1": float(box_entry["box"][1]),
            "box_x2": float(box_entry["box"][2]),
            "box_y2": float(box_entry["box"][3]),
        }
        for box_entry in frame_entry["boxes"]
        if not box_entry["deleted"] and training_class_name(box_entry["class_name"]) is not None
    ]
    return pd.DataFrame(target_rows, columns=["class_name", "box_x1", "box_y1", "box_x2", "box_y2"])


def resolve_val_frames(manifest_payload: dict, store_payload: dict) -> list[dict]:
    camera_split = store_payload.get("split_by_camera", {})
    val_frames = []
    for frame_record in manifest_payload["frames"]:
        frame_entry = store_payload["frames"].get(frame_record["frame_stem"])
        if frame_entry is None or not frame_entry["reviewed"]:
            continue
        split_name = camera_split.get(frame_record["camera_id"], frame_record.get("split", "unknown"))
        if split_name != "val":
            continue
        val_frames.append({"frame_stem": frame_record["frame_stem"], "image_path": frame_record["image_path"]})
    return sorted(val_frames, key=lambda frame_record: frame_record["frame_stem"])


sample_manifest_payload = load_json_document(SAMPLE_MANIFEST_PATH)
annotation_store_payload = load_json_document(ANNOTATION_STORE_PATH)
val_frame_records = resolve_val_frames(sample_manifest_payload, annotation_store_payload)
if not val_frame_records:
    raise ValueError("tidak ada frame val berlabel gold; kalibrasi tidak bisa dilakukan.")

calibration_model = YOLO(str(WEIGHTS_PATH))
assert_class_order_matches_model(calibration_model, TRAINING_CLASS_NAMES)

prediction_rows = []
for frame_record, prediction_result in zip(
    val_frame_records,
    calibration_model.predict(
        source=[frame_record["image_path"] for frame_record in val_frame_records],
        imgsz=IMAGE_SIZE,
        conf=CONF_FLOOR,
        iou=NMS_IOU,
        device=DEVICE,
        batch=BATCH_SIZE,
        verbose=False,
        stream=True,
    ),
):
    frame_boxes = prediction_result.boxes
    if frame_boxes is None or len(frame_boxes) == 0:
        continue
    box_coordinates = frame_boxes.xyxy.cpu().numpy()
    box_scores = frame_boxes.conf.cpu().numpy()
    box_classes = frame_boxes.cls.cpu().numpy().astype(int)
    for box_index in range(len(box_scores)):
        prediction_rows.append({
            "frame_stem": frame_record["frame_stem"],
            "class_name": TRAINING_CLASS_NAMES[box_classes[box_index]],
            "score": float(box_scores[box_index]),
            "box_x1": float(box_coordinates[box_index][0]),
            "box_y1": float(box_coordinates[box_index][1]),
            "box_x2": float(box_coordinates[box_index][2]),
            "box_y2": float(box_coordinates[box_index][3]),
        })

prediction_frame = pd.DataFrame(
    prediction_rows, columns=["frame_stem", "class_name", "score", "box_x1", "box_y1", "box_x2", "box_y2"]
)

matched_frames = []
for frame_record in val_frame_records:
    frame_predictions = prediction_frame[prediction_frame["frame_stem"] == frame_record["frame_stem"]]
    frame_targets = load_gold_targets(annotation_store_payload, frame_record["frame_stem"])
    matched_frames.append({
        "frame_stem": frame_record["frame_stem"],
        "predictions": match_predictions(frame_predictions, frame_targets, MATCH_IOU),
        "targets": frame_targets,
    })

display(Markdown(
    f"Prediksi val: **{len(prediction_frame)} box** pada **{len(val_frame_records)} frame** "
    f"(conf ≥ {CONF_FLOOR}, NMS IoU {NMS_IOU}, match IoU {MATCH_IOU})."
))'''

THRESHOLD_SOURCE = '''def sweep_class_threshold(class_name: str) -> pd.DataFrame:
    matched_scores = []
    unmatched_scores = []
    target_total = 0
    for matched_frame in matched_frames:
        frame_predictions = matched_frame["predictions"]
        frame_targets = matched_frame["targets"]
        target_total += int((frame_targets["class_name"] == class_name).sum())
        class_predictions = frame_predictions[frame_predictions["class_name"] == class_name]
        if class_predictions.empty:
            continue
        matched_scores.extend(class_predictions.loc[class_predictions["matched"], "score"].tolist())
        unmatched_scores.extend(class_predictions.loc[~class_predictions["matched"], "score"].tolist())
    matched_score_array = np.array(matched_scores, dtype=float)
    unmatched_score_array = np.array(unmatched_scores, dtype=float)
    sweep_rows = []
    for confidence_threshold in np.arange(CONF_FLOOR, CONF_CEILING + 1e-9, CONF_STEP):
        true_positive = int((matched_score_array >= confidence_threshold).sum())
        false_positive = int((unmatched_score_array >= confidence_threshold).sum())
        false_negative = target_total - true_positive
        precision = true_positive / max(true_positive + false_positive, 1)
        recall = true_positive / max(true_positive + false_negative, 1)
        f1_score = 2 * precision * recall / max(precision + recall, 1e-9)
        sweep_rows.append({
            "class_name": class_name,
            "confidence_threshold": round(float(confidence_threshold), 2),
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "precision": precision,
            "recall": recall,
            "f1": f1_score,
            "target_total": target_total,
        })
    return pd.DataFrame(sweep_rows)


sweep_frame = pd.concat([sweep_class_threshold(class_name) for class_name in TRAINING_CLASS_NAMES], ignore_index=True)
sweep_frame.to_csv(VAL_CALIBRATION_PATH, index=False)

calibration_rows = []
selected_thresholds = {}
for class_name in TRAINING_CLASS_NAMES:
    class_sweep = sweep_frame[sweep_frame["class_name"] == class_name]
    target_total = int(class_sweep["target_total"].max()) if not class_sweep.empty else 0
    if target_total == 0:
        selected_thresholds[class_name] = FALLBACK_CONF
        calibration_rows.append({
            "class_name": class_name,
            "gold_boxes": 0,
            "selected_threshold": FALLBACK_CONF,
            "precision": float("nan"),
            "recall": float("nan"),
            "f1": float("nan"),
            "note": "tidak ada gold di val untuk kelas ini; dipakai fallback",
        })
        continue
    best_row = class_sweep.sort_values(["f1", "confidence_threshold"], ascending=[False, True]).iloc[0]
    selected_thresholds[class_name] = round(float(best_row["confidence_threshold"]), 2)
    calibration_rows.append({
        "class_name": class_name,
        "gold_boxes": target_total,
        "selected_threshold": selected_thresholds[class_name],
        "precision": round(float(best_row["precision"]), 4),
        "recall": round(float(best_row["recall"]), 4),
        "f1": round(float(best_row["f1"]), 4),
        "note": "ok" if target_total >= MIN_VAL_BOXES_PER_CLASS else "gold val sedikit, ambang tidak stabil",
    })

calibration_frame = pd.DataFrame(calibration_rows)
threshold_payload = {
    "smoke_mode": SMOKE_MODE,
    "weights": str(WEIGHTS_PATH),
    "calibration_split": "val",
    "match_iou": MATCH_IOU,
    "nms_iou": NMS_IOU,
    "conf_floor": CONF_FLOOR,
    "conf_step": CONF_STEP,
    "val_frames": len(val_frame_records),
    "selected_thresholds": selected_thresholds,
    "per_class": calibration_frame.to_dict(orient="records"),
}
THRESHOLDS_PATH.write_text(json.dumps(threshold_payload, indent=2, ensure_ascii=False), encoding="utf-8")

display(Markdown("### Kalibrasi ambang confidence per kelas (split `val`)"))
display(calibration_frame)
watch_classes = calibration_frame[calibration_frame["note"] != "ok"]["class_name"].tolist()
if watch_classes:
    display(Markdown(
        "⚠️ Ambang untuk kelas berikut tidak stabil karena gold val tipis atau kosong: "
        + ", ".join(f"`{class_name}`" for class_name in watch_classes)
    ))
display(Markdown(f"Ambang terpilih ditulis ke `{THRESHOLDS_PATH}`."))'''

COUNT_SOURCE = '''def load_camera_split_by_frame(chunk_directory: Path) -> pd.DataFrame:
    chunk_rows = []
    for chunk_path in sorted(chunk_directory.glob("chunk_*.jsonl")):
        with chunk_path.open(encoding="utf-8") as chunk_file:
            chunk_rows.extend(json.loads(chunk_line) for chunk_line in chunk_file if chunk_line.strip())
    if not chunk_rows:
        raise FileNotFoundError(f"tidak ada baris chunk di {chunk_directory}")
    chunk_frame = pd.DataFrame(chunk_rows)
    chunk_frame["frame_stem"] = chunk_frame["crop_stem"]
    return chunk_frame[["frame_stem", "camera_id"]]


camera_split = annotation_store_payload.get("split_by_camera") or {
    frame_record["camera_id"]: frame_record.get("split", "unknown")
    for frame_record in sample_manifest_payload["frames"]
}
chunk_frame = load_camera_split_by_frame(CHUNK_DIR)
chunk_frame["split"] = chunk_frame["camera_id"].map(camera_split).fillna("unknown")
counting_frames = chunk_frame[chunk_frame["split"] == TEST_SPLIT_NAME].copy()
if counting_frames.empty:
    raise ValueError(
        f"tidak ada frame kamera split {TEST_SPLIT_NAME} di {CHUNK_DIR}. "
        "Pastikan peta split kamera memuat kamera test dan chunk NB02 sudah lengkap."
    )
counting_frames["image_path"] = counting_frames["frame_stem"].map(lambda stem: str(CROP_DIR / f"{stem}.png"))
existing_crop_mask = counting_frames["image_path"].map(
    lambda path_text: Path(path_text).exists()
).astype(bool)
counting_frames = counting_frames[existing_crop_mask]
if counting_frames.empty:
    raise ValueError(f"tidak ada crop yang bisa dibaca untuk split {TEST_SPLIT_NAME} di {CROP_DIR}.")
counting_frames = counting_frames.sort_values("frame_stem").reset_index(drop=True)

counting_rows = []
for counting_frame_record, prediction_result in zip(
    counting_frames.to_dict(orient="records"),
    calibration_model.predict(
        source=counting_frames["image_path"].tolist(),
        imgsz=IMAGE_SIZE,
        conf=CONF_FLOOR,
        iou=NMS_IOU,
        device=DEVICE,
        batch=BATCH_SIZE,
        verbose=False,
        stream=True,
    ),
):
    frame_boxes = prediction_result.boxes
    if frame_boxes is None or len(frame_boxes) == 0:
        continue
    box_coordinates = frame_boxes.xyxy.cpu().numpy()
    box_scores = frame_boxes.conf.cpu().numpy()
    box_classes = frame_boxes.cls.cpu().numpy().astype(int)
    for box_index in range(len(box_scores)):
        class_name = TRAINING_CLASS_NAMES[box_classes[box_index]]
        if float(box_scores[box_index]) < selected_thresholds[class_name]:
            continue
        counting_rows.append({
            "frame_stem": counting_frame_record["frame_stem"],
            "camera_id": counting_frame_record["camera_id"],
            "class_name": class_name,
            "score": round(float(box_scores[box_index]), 6),
            "box_x1": round(float(box_coordinates[box_index][0]), 2),
            "box_y1": round(float(box_coordinates[box_index][1]), 2),
            "box_x2": round(float(box_coordinates[box_index][2]), 2),
            "box_y2": round(float(box_coordinates[box_index][3]), 2),
        })

counted_box_frame = pd.DataFrame(
    counting_rows, columns=["frame_stem", "camera_id", "class_name", "score", "box_x1", "box_y1", "box_x2", "box_y2"]
)
counted_box_frame.to_csv(COUNTING_BY_FRAME_PATH, index=False)

class_counts = {class_name: int((counted_box_frame["class_name"] == class_name).sum()) for class_name in TRAINING_CLASS_NAMES}

per_camera_rows = []
for camera_id in sorted(counting_frames["camera_id"].unique()):
    camera_slice = counted_box_frame[counted_box_frame["camera_id"] == camera_id]
    camera_class_counts = {class_name: int((camera_slice["class_name"] == class_name).sum()) for class_name in TRAINING_CLASS_NAMES}
    per_camera_rows.append({
        "camera_id": camera_id,
        "frames": int(counting_frames.loc[counting_frames["camera_id"] == camera_id, "frame_stem"].nunique()),
        **{
            contract_key: sum(camera_class_counts[class_name] for class_name in class_names)
            for contract_key, class_names in BOX_COUNT_CONTRACT.items()
        },
    })
per_camera_frame = pd.DataFrame(per_camera_rows)
per_camera_frame.to_csv(COUNTING_BY_CAMERA_PATH, index=False)

display(Markdown(
    f"Penghitungan pada **{len(counting_frames)} frame kamera test** "
    f"({', '.join(sorted(counting_frames['camera_id'].unique()))}) · "
    f"**{len(counted_box_frame)} box lolos ambang**."
))
display(Markdown("### Jumlah box per kelas (bukan jumlah kendaraan unik — itu NB06)"))
display(pd.DataFrame([{"kelas": key, "jumlah_bbox": value} for key, value in class_counts.items()]))
display(Markdown("### Rincian per kamera test"))
display(per_camera_frame)'''

SUMMARY_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labelled_totals = {f"bbox {class_name.lower()}": class_counts[class_name] for class_name in TRAINING_CLASS_NAMES}

figure, axis = plt.subplots(figsize=(12, 5))
plot_totals = {key: value for key, value in labelled_totals.items() if isinstance(value, int)}
bar_positions = range(len(plot_totals))
axis.bar(bar_positions, list(plot_totals.values()), color="#1E90FF")
axis.set_xticks(list(bar_positions))
axis.set_xticklabels(list(plot_totals.keys()), rotation=30, ha="right")
axis.set_ylabel("jumlah box")
axis.set_title(
    f"Hasil hitung NB05 — {len(counting_frames)} frame kamera test"
    + (" (SMOKE, bukan hasil riset)" if SMOKE_MODE else "")
)
for position, value in zip(bar_positions, plot_totals.values()):
    axis.text(position, value, str(value), ha="center", va="bottom", fontsize=9)
axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
figure.savefig(COUNTING_FIGURE_PATH, dpi=120)
plt.close(figure)

counting_summary_payload = {
    "smoke_mode": SMOKE_MODE,
    "weights": str(WEIGHTS_PATH),
    "counting_split": TEST_SPLIT_NAME,
    "frames_counted": int(len(counting_frames)),
    "counted_boxes": int(len(counted_box_frame)),
    "class_counts": class_counts,
    "box_totals": {
        contract_key: sum(class_counts[class_name] for class_name in class_names)
        for contract_key, class_names in BOX_COUNT_CONTRACT.items()
    },
    "selected_thresholds": selected_thresholds,
    "counting_unit": "box_per_frame",
    "unit_note": (
        "satu box per frame; kendaraan yang sama terlihat di beberapa frame dihitung beberapa kali. "
        "Angka kendaraan unik hanya terbit dari NB06."
    ),
    "val_metrics_path": str(VAL_METRICS_PATH),
    "val_metrics_exists": VAL_METRICS_PATH.exists(),
}
COUNTING_SUMMARY_PATH.write_text(
    json.dumps(counting_summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
)

val_metrics_note = "tidak ditemukan"
if VAL_METRICS_PATH.exists():
    val_metrics_payload = json.loads(VAL_METRICS_PATH.read_text(encoding="utf-8"))
    val_overall = val_metrics_payload["overall"]
    val_metrics_note = (
        f"mAP50 {val_overall['mAP50']:.3f} · mAP50-95 {val_overall['mAP50-95']:.3f} · "
        f"precision {val_overall['precision']:.3f} · recall {val_overall['recall']:.3f} "
        f"({val_metrics_payload['val_frames']} frame val)"
    )

class_lines = "\\n".join(
    f"| `{class_name}` | {class_counts[class_name]} |" for class_name in TRAINING_CLASS_NAMES
)

display(Markdown(
    f"""### Ringkasan NB05

Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · kamera test: {', '.join(sorted(counting_frames['camera_id'].unique()))} · frame dihitung: **{len(counting_frames)}**

**Box per kelas (setelah ambang kalibrasi)**

| Kelas | Box |
|---|---|
{class_lines}

**Bukti akurasi (split val, label gold)** — {val_metrics_note}

Gambar: `{COUNTING_FIGURE_PATH}` · ringkasan: `{COUNTING_SUMMARY_PATH}`

**Kalimat yang jujur untuk laporan:** angka di atas adalah jumlah objek yang lolos ambang model pada
frame kamera test, tanpa ground truth di split itu. Ketelitian model diukur di split `val`; angka
hitung bukan estimasi arus lalu lintas dan tidak boleh diekstrapolasi ke per jam.
"""
))'''


TEST_EVALUATION_SOURCE = '''from ultralytics import YOLO
from ultralytics.utils import YAML

GT_ROOT = OUTPUT_ROOT / "ground_truth" / "yolo"
GT_IMAGES, GT_LABELS = GT_ROOT / "images", GT_ROOT / "labels"
TEST_METRICS_PATH = NB05_ROOT / "test_metrics.json"
test_images = sorted([*GT_IMAGES.glob("*.png"), *GT_IMAGES.glob("*.jpg"), *GT_IMAGES.glob("*.jpeg")])
if not test_images:
    test_metrics_payload = {"status": "unavailable", "reason": "Belum ada anotasi NB02c yang direview dan diekspor."}
    display(Markdown("**Metrik test belum tersedia:** review dan export anotasi holdout di NB02c terlebih dahulu."))
else:
    label_errors = []
    for image_path in test_images:
        label_path = GT_LABELS / f"{image_path.stem}.txt"
        if not label_path.exists():
            label_errors.append(f"label hilang: {label_path.name}")
            continue
        for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split()
            try:
                values = [float(value) for value in fields]
            except ValueError:
                label_errors.append(f"nilai bukan numerik: {label_path.name}:{line_number}")
                continue
            if len(values) != 5 or not np.isfinite(values).all() or values[0] != int(values[0]) or int(values[0]) not in range(len(TRAINING_CLASS_NAMES)):
                label_errors.append(f"format/ID kelas tidak sesuai: {label_path.name}:{line_number} ({line.strip()})")
                continue
            _, center_x, center_y, box_width, box_height = values
            if not (0 <= center_x <= 1 and 0 <= center_y <= 1 and 0 < box_width <= 1 and 0 < box_height <= 1
                    and center_x - box_width / 2 >= -1e-6 and center_y - box_height / 2 >= -1e-6
                    and center_x + box_width / 2 <= 1 + 1e-6 and center_y + box_height / 2 <= 1 + 1e-6):
                label_errors.append(f"koordinat tidak ternormalisasi/keluar batas: {label_path.name}:{line_number} ({line.strip()})")
    if label_errors:
        raise ValueError(
            f"Ground truth YOLO tidak valid ({len(label_errors)} masalah); evaluasi dibatalkan. "
            f"Contoh: {label_errors[:8]}. Regenerasi dari store NB02c dengan repair_ground_truth_export.py."
        )
    (GT_ROOT / "labels.cache").unlink(missing_ok=True)
    test_yaml = NB05_ROOT / "test_ground_truth.yaml"
    YAML.save(test_yaml, data={"path": str(GT_ROOT), "train": "images", "val": "images", "test": "images",
        "nc": len(TRAINING_CLASS_NAMES), "names": {i: n for i, n in enumerate(TRAINING_CLASS_NAMES)}})
    test_model = YOLO(str(WEIGHTS_PATH))
    result = test_model.val(data=str(test_yaml), split="test", imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device=DEVICE, plots=True)
    metrics = {"precision": float(result.box.mp), "recall": float(result.box.mr),
               "mAP50": float(result.box.map50), "mAP50-95": float(result.box.map)}
    errors, iou_errors = [], []
    for image_path in test_images:
        label_path = GT_LABELS / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_path.exists(): continue
        h, w = image.shape[:2]
        gt = []
        for line in label_path.read_text(encoding="utf-8").splitlines():
            row = [float(x) for x in line.split()]
            if len(row) == 5: gt.append((int(row[0]), np.array(row[1:])))
        pred_result = test_model.predict(str(image_path), imgsz=IMAGE_SIZE, conf=FALLBACK_CONF, device=DEVICE, verbose=False)[0]
        pred = []
        if pred_result.boxes is not None:
            for b in pred_result.boxes:
                x1,y1,x2,y2 = b.xyxy[0].cpu().numpy()
                pred.append((int(b.cls.item()), np.array([(x1+x2)/(2*w),(y1+y2)/(2*h),(x2-x1)/w,(y2-y1)/h])))
        pairs=[]
        for pi,(pc,pb) in enumerate(pred):
            for gi,(gc,gb) in enumerate(gt):
                if pc != gc: continue
                pxy=[pb[0]-pb[2]/2,pb[1]-pb[3]/2,pb[0]+pb[2]/2,pb[1]+pb[3]/2]
                gxy=[gb[0]-gb[2]/2,gb[1]-gb[3]/2,gb[0]+gb[2]/2,gb[1]+gb[3]/2]
                inter=max(0,min(pxy[2],gxy[2])-max(pxy[0],gxy[0]))*max(0,min(pxy[3],gxy[3])-max(pxy[1],gxy[1]))
                iou=inter/max(pb[2]*pb[3]+gb[2]*gb[3]-inter,1e-12)
                if iou >= MATCH_IOU: pairs.append((iou,pi,gi))
        used_p,used_g=set(),set()
        for iou,pi,gi in sorted(pairs,reverse=True):
            if pi in used_p or gi in used_g: continue
            used_p.add(pi);used_g.add(gi);errors.append(pred[pi][1]-gt[gi][1]);iou_errors.append(iou)
    if errors:
        e=np.asarray(errors)
        metrics.update({"bbox_coordinate_mse_normalized":float(np.mean(e**2)),"bbox_coordinate_rmse_normalized":float(np.sqrt(np.mean(e**2))),
                        "bbox_size_mae_normalized":float(np.mean(np.abs(e[:,2:4]))),"bbox_iou_error_mean":float(1-np.mean(iou_errors)),"bbox_matched_count":len(errors)})
    test_metrics_payload={"status":"computed","split":"test","label_source":"reviewed NB02c holdout","frames":len(test_images),"overall":metrics}
    display(Markdown(f"**Detector test** — precision {metrics['precision']:.3f} · recall {metrics['recall']:.3f} · mAP@50 {metrics['mAP50']:.3f} · mAP@50:95 {metrics['mAP50-95']:.3f}"))
    if errors: display(Markdown(f"BBox size MAE normalized {metrics['bbox_size_mae_normalized']:.4f}; coordinate RMSE normalized {metrics['bbox_coordinate_rmse_normalized']:.4f}; IoU error {metrics['bbox_iou_error_mean']:.4f}; matched {len(errors)}."))
TEST_METRICS_PATH.write_text(json.dumps(test_metrics_payload,indent=2,ensure_ascii=False),encoding="utf-8")'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(CALIBRATE_SOURCE),
        nbformat.v4.new_code_cell(THRESHOLD_SOURCE),
        nbformat.v4.new_code_cell(COUNT_SOURCE),
        nbformat.v4.new_code_cell(SUMMARY_SOURCE),
        nbformat.v4.new_code_cell(TEST_EVALUATION_SOURCE),
    ]
    return nbformat.v4.new_notebook(
        cells=notebook_cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
    )


def main() -> None:
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    verify_code_cells(OUTPUT_NOTEBOOK, EXPECTED_CODE_CELL_COUNT, EXPECTED_MARKDOWN_CELL_COUNT)


if __name__ == "__main__":
    main()
