#!/usr/bin/env python3
"""Build NB06_tracking_count.ipynb: hitung kendaraan unik dengan ByteTrack di dalam ROI."""

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_gate import verify_code_cells


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB06_tracking_count.ipynb"

EXPECTED_CODE_CELL_COUNT = 5
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB06 — Hitung Kendaraan Unik (ByteTrack, hanya di dalam ROI)

NB05 menghitung **box per frame sampel**, sehingga satu kendaraan bisa terhitung berkali-kali.
NB06 menghitung **kendaraan unik**: satu kendaraan = satu track, dihitung sekali sejak muncul di
dalam ROI sampai keluar.

**Posisi:** `NB02 → NB02b → NB03 → NB04 → NB05 (kalibrasi) → NB06 (hitung unik)`

## Kenapa stage terpisah, bukan menumpang NB05

Tracking butuh **frame berurutan**. NB02 hanya menyimpan 1 dari setiap 15 frame (0,6 detik), dan
pada jarak itu asosiasi antar frame sudah rapuh. Karena itu NB06 membaca **video asli** langsung
(`output/videos/`), bukan frame sampel, dan menerapkan crop ROI yang sama persis dengan NB02.

## Rantai koordinat (titik paling rawan)

| Ruang | Isi |
|---|---|
| frame asli | video 25 fps, ukuran per kamera |
| crop ROI | bbox polygon + area luar polygon dihitamkan, `thumbnail(640)` |
| deteksi | YOLO + ByteTrack dijalankan **di crop**, koordinat crop |
| keluaran NB06 | box dipetakan balik ke koordinat frame asli: `frame = offset + crop / scale` |

`pipeline_common.build_roi_crop` mengembalikan `offset_x`, `offset_y`, dan `scale` bersama crop.
Skala sekarang 1,0 karena semua crop lebih kecil dari 640 px (thumbnail tidak memperbesar), tetapi
rumusnya tetap umum supaya tidak diam-diam salah kalau ROI berubah. Audit pemetaan disimpan di
`roi_mapping_audit.csv` per video.

## Aturan hitung

1. YOLO + ByteTrack dijalankan per frame di dalam crop ROI; deteksi di luar polygon tidak pernah
   terlihat model karena area itu dihitamkan.
2. Deteksi difilter dengan ambang confidence per kelas hasil kalibrasi NB05. Ambang **tidak**
   dikalibrasi ulang di sini.
3. Satu `track_id` = satu kendaraan. Track yang terputus karena kendaraan sempat hilang dari
   pandangan **disambung kembali** sebelum dihitung (jenis sama, jeda 1–50 frame, box bertumpuk atau
   pusat berdekatan), lalu `MIN_TRACK_HITS` diterapkan pada kendaraan hasil sambungan. Bukti setiap
   penggabungan ada di `fragment_merges.csv`.
4. Kelas per track = **suara mayoritas** di antara frame yang lolos ambang; seri diputus oleh skor
   rata-rata lebih tinggi. Alasannya: kondisi lampu satu kendaraan bisa terbaca berbeda antar frame,
   dan satu kendaraan hanya boleh menyumbang satu angka.
5. Angka tetap dijumlahkan dari kelas dengan kontrak MASTER §10.6. `PantulanCahaya` tetap
   hard-negative dan tidak pernah dihitung.

## Ambang internal tracker

ByteTrack bawaan memakai `new_track_thresh: 0.25`; box di bawah 0.25 tidak akan pernah membentuk
track baru, hanya bisa menempel ke track yang sudah ada. Sementara ambang per kelas hasil NB05 boleh
serendah `CONF_FLOOR = 0.05`. Selisih itu membuat kelas langka tidak pernah punya track dan jumlahnya
kurang terhitung secara sistematis. Karena itu NB06 memakai `builders/bytetrack_nb06.yaml`
(`new_track_thresh 0.05`, `track_high_thresh 0.05`, `track_low_thresh 0.02`, `track_buffer 30`),
yaitu lantai yang sama dengan lantai detektor. Sel JOBS menolak jalan bila `new_track_thresh` lebih
tinggi dari ambang kelas terendah. Tracker hanya mengurus asosiasi antar frame; penyaringan akhir
tetap dilakukan ambang per kelas hasil NB05 saat agregasi track.

## Satu rekaman per kamera

NB01 tidak pernah menimpa video: nama berkasnya bertimestamp, jadi satu kamera bisa punya beberapa
rekaman. NB06 memilih **satu rekaman terbaru per kamera** (stem terbesar) dan menampilkan tabel
"video ditemukan vs video dihitung" berisi rekaman yang dilewati beserta alasannya. Tanpa aturan ini,
menjalankan NB01 dua kali akan menghitung dua rekaman kamera yang sama dan menggandakan total test.
Untuk memakai rekaman tertentu secara eksplisit, hapus/pindahkan rekaman lain dari `output/videos`.

Nama kamera diambil dari berkas (`stem.split("_")[1]`) dan **wajib** ada di peta split
`pipeline_common`. Nama yang tidak dikenal membuat sel JOBS berhenti, bukan dilewati diam-diam.

## Batas yang wajib ditulis di laporan

- Track di-reset per rekaman, jadi kendaraan yang sama di dua rekaman/kamera berbeda terhitung dua
  kali. Tidak ada identitas lintas sesi.
- Kendaraan yang sudah berada di dalam ROI pada frame pertama video tetap dihitung sekali
  (muncul track baru), walaupun momen masuknya tidak terekam.
- Kendaraan yang keluar lalu masuk lagi akan membentuk track baru dan **terhitung dua kali**.
  Ini konsekuensi tak terhindarkan dari definisi "satu masuk, satu hitung".
- Model hanya dilatih pada gold set kamera train/val; kamera test tidak punya ground truth,
  sehingga angka NB06 adalah hasil aplikasi model, bukan akurasi terverifikasi.

## Mode smoke

`PIPELINE_SMOKE=1` → `output/smoke/`, hanya video pertama dari split yang dihitung, maksimum
`SMOKE_MAX_FRAMES_PER_VIDEO` frame. Tujuannya menguji alur tracking, bukan mengukur jumlah.

Output: `output/nb06/{tracking_detections.csv, track_summary.csv, counting_by_video.csv,
counting_timeline.csv, overlay_manifest.json, counting_summary.json, roi_mapping_audit.csv,
counting_bar.png, overlays/*.mp4}`. Overlay
dibuat untuk satu rekaman terbaru setiap kamera; agregasi hitungan tetap hanya memakai split test.
"""

CONFIG_SOURCE = '''from pathlib import Path
import json
import os

import cv2
import pandas as pd
import torch
from IPython.display import Markdown, display

from pipeline_common import (
    CLASS_NAMES,
    TRAINING_CLASS_NAMES,
    COUNT_CONTRACT,
    UNMEASURABLE_COUNT_CONTRACT_KEYS,
    FRAGMENT_DISTANCE_RATIO,
    FRAGMENT_MAX_GAP_FRAMES,
    FRAGMENT_MIN_IOU,
    SPLIT_NAME_BY_CAMERA,
    build_roi_crop,
    mask_frame_to_roi,
    load_roi_config,
    stitch_track_fragments,
    validate_tracker_thresholds,
)
from attribute_classifier import decode_attributes, extract_pixel_features, load_classifier

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT

VIDEO_DIR = OUTPUT_ROOT / "videos"
ROI_DIR = OUTPUT_ROOT / "roi"
NB04_ROOT = DATA_ROOT / "nb04"
NB05_ROOT = DATA_ROOT / "nb05"
NB06_ROOT = DATA_ROOT / "nb06"
RUN_NAME = "nb04_yolo11n_smoke" if SMOKE_MODE else "nb04_yolo11n"
WEIGHTS_PATH = NB04_ROOT / "runs" / RUN_NAME / "weights" / "best.pt"
ATTRIBUTE_MODEL_PATH = DATA_ROOT / "attribute_classifier" / "best.pt"
THRESHOLDS_PATH = NB05_ROOT / "thresholds.json"
TRACK_ROWS_PATH = NB06_ROOT / "tracking_detections.csv"
TRACK_SUMMARY_PATH = NB06_ROOT / "track_summary.csv"
COUNTING_BY_VIDEO_PATH = NB06_ROOT / "counting_by_video.csv"
COUNTING_SUMMARY_PATH = NB06_ROOT / "counting_summary.json"
COUNTING_TIMELINE_PATH = NB06_ROOT / "counting_timeline.csv"
OVERLAY_MANIFEST_PATH = NB06_ROOT / "overlay_manifest.json"
EVENT_CLIPS_DIR = NB06_ROOT / "event_clips"
EVENT_CLIP_MANIFEST_PATH = NB06_ROOT / "event_clip_manifest.json"
MAPPING_AUDIT_PATH = NB06_ROOT / "roi_mapping_audit.csv"
FRAGMENT_MERGES_PATH = NB06_ROOT / "fragment_merges.csv"
FIGURE_PATH = NB06_ROOT / "counting_bar.png"
OVERLAY_DIR = NB06_ROOT / "overlays"
EVENT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_LABELS = os.environ.get("NB06_OVERLAY_LABELS", "0") == "1"

NB06_ROOT.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

COUNTED_SPLITS = ["test"]
TRACKER_CONFIG_PATH = RISET_ROOT / "builders" / "bytetrack_nb06.yaml"
TRACKER_CONFIG = str(TRACKER_CONFIG_PATH)
IMAGE_SIZE = 640
CONF_FLOOR = 0.05
NMS_IOU = 0.70
FRAME_STRIDE = 1
MIN_TRACK_HITS = 2 if SMOKE_MODE else 3
SMOKE_MAX_FRAMES_PER_VIDEO = 60
VIDEO_EXTENSION = ".mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_COLORS_BGR = {
    "Normal": (235, 235, 235),
    "Silau": (0, 165, 255),
    "Berwarna": (255, 210, 0),
    "SilauBerwarna": (255, 0, 255),
    "PantulanCahaya": (150, 150, 150),
}

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

display(Markdown(
    f"NB06 siap. Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · device: **{DEVICE}** · "
    f"bobot: `{WEIGHTS_PATH}` (ada: **{WEIGHTS_PATH.exists()}**) · split dihitung: {COUNTED_SPLITS} · "
    f"minimal {MIN_TRACK_HITS} kemunculan per track · overlay label: **{OVERLAY_LABELS}**"
))'''

JOBS_SOURCE = '''def load_json_document(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"berkas wajib tidak ada: {path}. Jalankan tahap sebelumnya lebih dulu.")
    with path.open(encoding="utf-8") as source_file:
        return json.load(source_file)


def camera_id_from_video(video_path: Path) -> str:
    return video_path.stem.split("_")[1]


def probe_video(video_path: Path) -> dict:
    video_capture = cv2.VideoCapture(str(video_path))
    if not video_capture.isOpened():
        raise ValueError(f"video tidak bisa dibuka: {video_path}")
    frame_count = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_second = float(video_capture.get(cv2.CAP_PROP_FPS)) or 25.0
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_capture.release()
    return {
        "frame_count": frame_count,
        "fps": frames_per_second,
        "width": frame_width,
        "height": frame_height,
    }


if not WEIGHTS_PATH.exists():
    raise FileNotFoundError(f"bobot tidak ada: {WEIGHTS_PATH}. Jalankan NB04 lebih dulu.")
if not ATTRIBUTE_MODEL_PATH.exists():
    raise FileNotFoundError(
        f"classifier atribut tidak ada: {ATTRIBUTE_MODEL_PATH}. "
        "Jalankan builders/train_attribute_classifier.py setelah gold train/val cukup."
    )
if not TRACKER_CONFIG_PATH.exists():
    raise FileNotFoundError(f"konfigurasi tracker tidak ada: {TRACKER_CONFIG_PATH}")
thresholds_payload = load_json_document(THRESHOLDS_PATH)
selected_thresholds = thresholds_payload["selected_thresholds"]
missing_thresholds = [class_name for class_name in TRAINING_CLASS_NAMES if class_name not in selected_thresholds]
if missing_thresholds:
    raise ValueError(f"ambang confidence belum ada untuk kelas: {missing_thresholds}")

from ultralytics.utils import YAML

tracker_settings = YAML.load(TRACKER_CONFIG)
validate_tracker_thresholds(tracker_settings, selected_thresholds)

roi_config = load_roi_config(ROI_DIR)
video_paths = sorted(VIDEO_DIR.glob(f"*{VIDEO_EXTENSION}"))
if not video_paths:
    raise FileNotFoundError(f"tidak ada video di {VIDEO_DIR}")

unmapped_cameras = sorted({camera_id_from_video(video_path) for video_path in video_paths} - set(SPLIT_NAME_BY_CAMERA))
if unmapped_cameras:
    raise ValueError(
        f"nama video memuat kamera yang tidak ada di peta split: {unmapped_cameras}. "
        "Perbaiki penamaan rekaman atau tambahkan kameranya di pipeline_common."
    )

latest_video_by_camera = {}
for video_path in video_paths:
    camera_id = camera_id_from_video(video_path)
    current_latest = latest_video_by_camera.get(camera_id)
    if current_latest is None or video_path.stem > current_latest.stem:
        latest_video_by_camera[camera_id] = video_path

discovery_rows = []
for video_path in video_paths:
    camera_id = camera_id_from_video(video_path)
    split_name = SPLIT_NAME_BY_CAMERA[camera_id]
    if split_name not in COUNTED_SPLITS:
        discovery_note = f"split {split_name}, tidak dihitung"
    elif latest_video_by_camera[camera_id] != video_path:
        discovery_note = f"rekaman lama, dipakai {latest_video_by_camera[camera_id].stem}"
    else:
        discovery_note = "dihitung"
    discovery_rows.append({
        "video_stem": video_path.stem,
        "camera_id": camera_id,
        "split": split_name,
        "status": discovery_note,
    })
discovery_frame = pd.DataFrame(discovery_rows)

scan_jobs = []
for camera_id, video_path in sorted(latest_video_by_camera.items()):
    if camera_id not in roi_config:
        raise ValueError(f"ROI tidak ada untuk kamera {camera_id} di roi_config.json")
    video_metadata = probe_video(video_path)
    scan_jobs.append({
        "video_stem": video_path.stem,
        "video_path": str(video_path),
        "camera_id": camera_id,
        "split": SPLIT_NAME_BY_CAMERA[camera_id],
        "frame_limit": min(video_metadata["frame_count"], SMOKE_MAX_FRAMES_PER_VIDEO) if SMOKE_MODE else video_metadata["frame_count"],
        **video_metadata,
    })

if SMOKE_MODE:
    scan_jobs = scan_jobs[:1]

if not scan_jobs:
    raise ValueError(f"tidak ada video dengan split {COUNTED_SPLITS} di {VIDEO_DIR}")

scan_frame = pd.DataFrame(scan_jobs)[["video_stem", "camera_id", "split", "frame_count", "frame_limit", "fps", "width", "height"]]
display(Markdown("### Video ditemukan vs video dihitung"))
display(discovery_frame)
display(Markdown("### Video yang diproses"))
display(scan_frame)
display(Markdown(
    f"Total frame diproses: **{int(scan_frame['frame_limit'].sum())}** · "
    f"estimasi durasi: **{scan_frame['frame_limit'].sum() / scan_frame['fps'].mean() / 60:.1f} menit** video"
))'''

TRACK_SOURCE = '''from PIL import Image
from ultralytics import YOLO
from attribute_classifier import decode_attributes, extract_pixel_features, load_classifier

RAW_BOX_DECIMALS = 2
SCORE_DECIMALS = 4
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.5
LABEL_THICKNESS = 1
LABEL_PADDING = 4


def rectangles_overlap(first: tuple[int, int, int, int], second: tuple[int, int, int, int]) -> bool:
    return not (
        first[2] <= second[0]
        or second[2] <= first[0]
        or first[3] <= second[1]
        or second[3] <= first[1]
    )


def draw_debug_label(
    frame: object,
    text: str,
    color: tuple[int, int, int],
    box: tuple[int, int, int, int],
) -> None:
    frame_height, frame_width = frame.shape[:2]
    x1, y1, x2, y2 = box
    box_width = max(x2 - x1, 1)
    box_height = max(y2 - y1, 1)
    font_scale = LABEL_FONT_SCALE
    (text_width, text_height), baseline = cv2.getTextSize(text, LABEL_FONT, font_scale, LABEL_THICKNESS)
    if text_width + LABEL_PADDING * 2 > box_width:
        font_scale = max(0.25, LABEL_FONT_SCALE * box_width / (text_width + LABEL_PADDING * 2))
        (text_width, text_height), baseline = cv2.getTextSize(text, LABEL_FONT, font_scale, LABEL_THICKNESS)
    label_width = text_width + LABEL_PADDING * 2
    label_height = text_height + baseline + LABEL_PADDING * 2
    left = max(0, min(x1 + LABEL_PADDING, frame_width - label_width))
    top = max(0, min(y1 + LABEL_PADDING, frame_height - label_height))
    right, bottom = left + label_width, top + label_height
    cv2.rectangle(frame, (left, top), (right, bottom), (15, 15, 15), cv2.FILLED)
    cv2.putText(
        frame, text, (left + LABEL_PADDING, bottom - baseline - LABEL_PADDING),
        LABEL_FONT, font_scale, color, LABEL_THICKNESS, cv2.LINE_AA,
    )


attribute_model, attribute_mean, attribute_std, attribute_metadata = load_classifier(ATTRIBUTE_MODEL_PATH, DEVICE)


def classify_vehicle_attributes(crop_image: Image.Image, crop_box: list[float]) -> str:
    features = extract_pixel_features(crop_image, crop_box)
    normalised = (features - attribute_mean) / attribute_std
    with torch.no_grad():
        logits = attribute_model(torch.from_numpy(normalised).to(DEVICE).unsqueeze(0)).squeeze(0).cpu().numpy()
        attributes = decode_attributes(logits, attribute_metadata.get("thresholds"))
    if attributes["silau"] and attributes["berwarna"]:
        return "SilauBerwarna"
    if attributes["silau"]:
        return "Silau"
    if attributes["berwarna"]:
        return "Berwarna"
    return "Normal"


def process_video_tracking(job: dict) -> tuple[list[dict], dict, list[dict]]:
    tracking_model = YOLO(str(WEIGHTS_PATH))
    video_capture = cv2.VideoCapture(job["video_path"])
    tracking_rows = []
    audit_rows = []
    untracked_detection_count = 0
    processed_frame_count = 0
    frame_index = 0
    overlay_path = OVERLAY_DIR / f"{job['video_stem']}_overlay.mp4"
    overlay_writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"mp4v"),
        float(job["fps"]), (int(job["width"]), int(job["height"])),
    )
    if not overlay_writer.isOpened():
        video_capture.release()
        raise RuntimeError(f"gagal membuat video overlay: {overlay_path}")
    try:
        while True:
            read_ok, frame_array = video_capture.read()
            if not read_ok or (job["frame_limit"] and processed_frame_count >= job["frame_limit"]):
                break
            if frame_index % FRAME_STRIDE == 0:
                frame_image = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
                detector_image = mask_frame_to_roi(frame_image, roi_config[job["camera_id"]])
                if processed_frame_count == 0:
                    crop_image, offset_x, offset_y, scale = build_roi_crop(
                        frame_image, roi_config[job["camera_id"]]
                    )
                    audit_rows.append({
                        "video_stem": job["video_stem"],
                        "camera_id": job["camera_id"],
                        "frame_width": frame_image.size[0],
                        "frame_height": frame_image.size[1],
                        "crop_width": crop_image.size[0],
                        "crop_height": crop_image.size[1],
                        "offset_x": offset_x,
                        "offset_y": offset_y,
                        "scale": round(float(scale), 6),
                    })
                tracking_result = tracking_model.track(
                    source=detector_image,
                    persist=True,
                    tracker=TRACKER_CONFIG,
                    conf=CONF_FLOOR,
                    iou=NMS_IOU,
                    imgsz=IMAGE_SIZE,
                    device=DEVICE,
                    verbose=False,
                )[0]
                processed_frame_count += 1
                frame_boxes = tracking_result.boxes
                overlay_frame = frame_array.copy()
                if frame_boxes is not None and len(frame_boxes) > 0:
                    box_coordinates = frame_boxes.xyxy.cpu().numpy()
                    box_scores = frame_boxes.conf.cpu().numpy()
                    box_classes = frame_boxes.cls.cpu().numpy().astype(int)
                    box_track_ids = frame_boxes.id.cpu().numpy().astype(int) if frame_boxes.id is not None else None
                    for box_index in range(len(box_scores)):
                        crop_box = [float(coordinate) for coordinate in box_coordinates[box_index]]
                        frame_box = crop_box
                        detector_class_name = TRAINING_CLASS_NAMES[box_classes[box_index]]
                        x1, y1, x2, y2 = [int(round(value)) for value in frame_box]
                        track_id = int(box_track_ids[box_index]) if box_track_ids is not None else None
                        if track_id is None:
                            untracked_detection_count += 1
                            continue
                        if detector_class_name.startswith(("Mobil", "Motor")):
                            vehicle_type = "Mobil" if detector_class_name.startswith("Mobil") else "Motor"
                            attribute_suffix = classify_vehicle_attributes(frame_image, frame_box)
                            output_class_name = vehicle_type + attribute_suffix
                        else:
                            output_class_name = detector_class_name
                        event_suffix = output_class_name.removeprefix(vehicle_type) if detector_class_name.startswith(("Mobil", "Motor")) else "PantulanCahaya"
                        overlay_color = EVENT_COLORS_BGR.get(event_suffix, EVENT_COLORS_BGR["PantulanCahaya"])
                        cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), overlay_color, 2)
                        if OVERLAY_LABELS:
                            overlay_label = f"{box_scores[box_index] * 100:.0f}%"
                            draw_debug_label(
                                overlay_frame, overlay_label, overlay_color,
                                (x1, y1, x2, y2),
                            )
                        tracking_rows.append({
                            "video_stem": job["video_stem"],
                            "camera_id": job["camera_id"],
                            "frame_index": frame_index,
                            "track_id": track_id,
                            "class_name": output_class_name,
                            "score": round(float(box_scores[box_index]), SCORE_DECIMALS),
                            "frame_x1": round(frame_box[0], RAW_BOX_DECIMALS),
                            "frame_y1": round(frame_box[1], RAW_BOX_DECIMALS),
                            "frame_x2": round(frame_box[2], RAW_BOX_DECIMALS),
                            "frame_y2": round(frame_box[3], RAW_BOX_DECIMALS),
                        })
                overlay_writer.write(overlay_frame)
            frame_index += 1
    finally:
        video_capture.release()
        overlay_writer.release()
        del tracking_model
        if DEVICE == "cuda":
            torch.cuda.empty_cache()
    video_summary = {
        "video_stem": job["video_stem"],
        "camera_id": job["camera_id"],
        "frames_processed": processed_frame_count,
        "detections": len(tracking_rows),
        "untracked_detections": untracked_detection_count,
        "overlay_path": str(overlay_path),
    }
    return tracking_rows, video_summary, audit_rows


all_tracking_rows = []
video_summaries = []
mapping_audit_rows = []
for job in scan_jobs:
    tracking_rows, video_summary, audit_rows = process_video_tracking(job)
    all_tracking_rows.extend(tracking_rows)
    video_summaries.append(video_summary)
    mapping_audit_rows.extend(audit_rows)
    display(Markdown(
        f"`{job['video_stem']}` selesai — {video_summary['frames_processed']} frame, "
        f"{video_summary['detections']} deteksi, {video_summary['untracked_detections']} tanpa track id."
    ))

tracking_frame = pd.DataFrame(
    all_tracking_rows,
    columns=[
        "video_stem", "camera_id", "frame_index", "track_id", "class_name", "score",
        "frame_x1", "frame_y1", "frame_x2", "frame_y2",
    ],
)
tracking_frame.to_csv(TRACK_ROWS_PATH, index=False)
mapping_audit_frame = pd.DataFrame(mapping_audit_rows)
mapping_audit_frame.to_csv(MAPPING_AUDIT_PATH, index=False)
video_summary_frame = pd.DataFrame(video_summaries)

display(Markdown("### Hasil tracking per video"))
display(video_summary_frame)
display(Markdown("### Audit pemetaan crop ke frame asli"))
display(mapping_audit_frame)
if int(video_summary_frame["frames_processed"].sum()) == 0:
    raise ValueError("tidak ada frame yang berhasil diproses; periksa video dan path ROI.")
total_untracked = int(video_summary_frame["untracked_detections"].sum())
total_tracked = int(video_summary_frame["detections"].sum())
if total_untracked > 0:
    display(Markdown(
        f"**{total_untracked} deteksi tidak mendapat `track_id`** dan dibuang, bukan digabung ke satu "
        "track palsu. Penyebab paling umum: confidence model seluruhnya di bawah `new_track_thresh` "
        f"tracker (`{TRACKER_CONFIG_PATH.name}`), sehingga tracker tidak pernah memulai track."
    ))
if not SMOKE_MODE and total_tracked == 0 and total_untracked > 0:
    raise ValueError(
        "semua deteksi gagal mendapat track_id, sehingga hitungan kendaraan akan nol secara palsu. "
        "Periksa: bobot NB04 benar-benar terlatih, dan ambang confidence per kelas di "
        "thresholds.json berada di atas new_track_thresh tracker."
    )'''

COUNT_SOURCE = '''def qualify_by_class_threshold(detection_frame: pd.DataFrame) -> pd.DataFrame:
    if detection_frame.empty:
        return detection_frame.copy()
    threshold_series = detection_frame["class_name"].map(selected_thresholds)
    return detection_frame[detection_frame["score"] >= threshold_series].copy()


def resolve_track_class(track_frame: pd.DataFrame) -> tuple[str, float]:
    class_stats = track_frame.groupby("class_name")["score"].agg(["count", "mean"])
    class_stats = class_stats.sort_values(["count", "mean"], ascending=[False, False])
    return str(class_stats.index[0]), float(class_stats.iloc[0]["mean"])


def summarise_units(detection_frame: pd.DataFrame, grouping_column: str) -> pd.DataFrame:
    unit_rows = []
    for (video_stem, unit_id), unit_frame in detection_frame.groupby(["video_stem", grouping_column], sort=True):
        ordered_frame = unit_frame.sort_values("frame_index")
        first_row = ordered_frame.iloc[0]
        last_row = ordered_frame.iloc[-1]
        resolved_class, mean_score = resolve_track_class(ordered_frame)
        unit_rows.append({
            "video_stem": video_stem,
            "unit_id": int(unit_id),
            "camera_id": first_row["camera_id"],
            "class_name": resolved_class,
            "hits": int(len(ordered_frame)),
            "first_frame": int(first_row["frame_index"]),
            "last_frame": int(last_row["frame_index"]),
            "mean_score": round(mean_score, 4),
            "first_x1": float(first_row["frame_x1"]),
            "first_y1": float(first_row["frame_y1"]),
            "first_x2": float(first_row["frame_x2"]),
            "first_y2": float(first_row["frame_y2"]),
            "last_x1": float(last_row["frame_x1"]),
            "last_y1": float(last_row["frame_y1"]),
            "last_x2": float(last_row["frame_x2"]),
            "last_y2": float(last_row["frame_y2"]),
        })
    return pd.DataFrame(
        unit_rows,
        columns=[
            "video_stem", "unit_id", "camera_id", "class_name", "hits",
            "first_frame", "last_frame", "mean_score",
            "first_x1", "first_y1", "first_x2", "first_y2",
            "last_x1", "last_y1", "last_x2", "last_y2",
        ],
    )


test_tracking_frame = tracking_frame[
    tracking_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).isin(COUNTED_SPLITS)
].copy()
qualified_frame = qualify_by_class_threshold(test_tracking_frame)
track_frame = summarise_units(qualified_frame, "track_id")
stitch_lookup = stitch_track_fragments(track_frame)
track_output = track_frame.merge(
    stitch_lookup,
    left_on=["video_stem", "unit_id"],
    right_on=["video_stem", "track_id"],
).drop(columns=["unit_id"])
track_output.to_csv(TRACK_SUMMARY_PATH, index=False)

vehicle_detections = qualified_frame.merge(
    stitch_lookup, on=["video_stem", "track_id"], how="left"
)
if vehicle_detections["vehicle_id"].isna().any():
    raise ValueError("ada track tanpa pemetaan vehicle_id; penyambungan fragmen tidak lengkap")
vehicle_frame = summarise_units(vehicle_detections, "vehicle_id")
counted_vehicle_frame = vehicle_frame[vehicle_frame["hits"] >= MIN_TRACK_HITS].copy()

timeline_frame = counted_vehicle_frame.rename(columns={"unit_id": "vehicle_id"})[
    ["video_stem", "camera_id", "vehicle_id", "class_name", "hits", "first_frame", "last_frame", "mean_score"]
].copy()
timeline_frame["fps"] = timeline_frame["video_stem"].map(
    {row.video_stem: float(row.fps) for row in scan_frame.itertuples()}
)
timeline_frame["first_timestamp_seconds"] = timeline_frame["first_frame"] / timeline_frame["fps"]
timeline_frame["last_timestamp_seconds"] = timeline_frame["last_frame"] / timeline_frame["fps"]
timeline_frame.to_csv(COUNTING_TIMELINE_PATH, index=False)

class_counts = {
    class_name: int((counted_vehicle_frame["class_name"] == class_name).sum())
    for class_name in CLASS_NAMES
}
counting_totals = {
    contract_key: "n/a"
    if contract_key in UNMEASURABLE_COUNT_CONTRACT_KEYS
    else int(sum(class_counts[class_name] for class_name in contract_classes))
    for contract_key, contract_classes in COUNT_CONTRACT.items()
}

per_video_rows = []
for video_stem, video_slice in counted_vehicle_frame.groupby("video_stem", sort=True):
    video_class_counts = {
        class_name: int((video_slice["class_name"] == class_name).sum())
        for class_name in CLASS_NAMES
    }
    per_video_rows.append({
        "video_stem": video_stem,
        "camera_id": video_slice["camera_id"].iloc[0],
        "vehicles_counted": int(len(video_slice)),
        **video_class_counts,
        **{
            contract_key: "n/a"
            if contract_key in UNMEASURABLE_COUNT_CONTRACT_KEYS
            else int(sum(video_class_counts[class_name] for class_name in contract_classes))
            for contract_key, contract_classes in COUNT_CONTRACT.items()
        },
    })
per_video_frame = pd.DataFrame(per_video_rows)
per_video_frame.to_csv(COUNTING_BY_VIDEO_PATH, index=False)

overlay_manifest = []
for video_summary in video_summaries:
    matching_job = next(job for job in scan_jobs if job["video_stem"] == video_summary["video_stem"])
    overlay_manifest.append({
        "video_stem": video_summary["video_stem"],
        "camera_id": video_summary["camera_id"],
        "split": matching_job["split"],
        "overlay_path": video_summary["overlay_path"],
        "fps": float(matching_job["fps"]),
        "frames_processed": int(video_summary["frames_processed"]),
        "duration_seconds": round(float(video_summary["frames_processed"]) / float(matching_job["fps"]), 3),
    })
OVERLAY_MANIFEST_PATH.write_text(
    json.dumps(overlay_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

event_clip_manifest = []
for overlay_info in overlay_manifest:
    video_events = timeline_frame[timeline_frame["video_stem"] == overlay_info["video_stem"]]
    if video_events.empty:
        continue
    overlay_capture = cv2.VideoCapture(overlay_info["overlay_path"])
    if not overlay_capture.isOpened():
        raise RuntimeError(f"overlay tidak bisa dibuka untuk clip event: {overlay_info['overlay_path']}")
    fps = float(overlay_info["fps"])
    total_frames = int(overlay_info["frames_processed"])
    for event in video_events.itertuples(index=False):
        start_frame = max(0, int(event.first_frame) - int(fps))
        end_frame = min(total_frames - 1, int(event.last_frame) + int(fps))
        class_directory = EVENT_CLIPS_DIR / str(event.class_name)
        class_directory.mkdir(parents=True, exist_ok=True)
        clip_path = class_directory / f"{event.video_stem}_vehicle_{int(event.vehicle_id):05d}.mp4"
        overlay_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_writer = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps,
            (int(overlay_capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(overlay_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))),
        )
        if not clip_writer.isOpened():
            overlay_capture.release()
            raise RuntimeError(f"clip event tidak bisa dibuat: {clip_path}")
        written_frames = 0
        for _ in range(start_frame, end_frame + 1):
            read_ok, clip_frame = overlay_capture.read()
            if not read_ok:
                break
            clip_writer.write(clip_frame)
            written_frames += 1
        clip_writer.release()
        if written_frames:
            event_clip_manifest.append({
                "class_name": str(event.class_name),
                "camera_id": str(event.camera_id),
                "video_stem": str(event.video_stem),
                "vehicle_id": int(event.vehicle_id),
                "first_frame": int(event.first_frame),
                "last_frame": int(event.last_frame),
                "hits": int(event.hits),
                "mean_score": float(event.mean_score),
                "start_frame": start_frame,
                "end_frame": end_frame,
                "clip_path": str(clip_path),
                "duration_seconds": round(written_frames / fps, 3),
            })
    overlay_capture.release()
EVENT_CLIP_MANIFEST_PATH.write_text(
    json.dumps(event_clip_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

fragment_merges = track_output.groupby(["video_stem", "vehicle_id"]).agg(
    member_tracks=("track_id", "count"),
    total_hits=("hits", "sum"),
    first_frame=("first_frame", "min"),
    last_frame=("last_frame", "max"),
).reset_index()
fragment_merges = fragment_merges[fragment_merges["member_tracks"] > 1]
fragment_merges.to_csv(FRAGMENT_MERGES_PATH, index=False)

tracks_total = int(len(track_frame))
vehicles_total = int(len(vehicle_frame))
fragments_stitched = tracks_total - vehicles_total
duplication_factor = len(qualified_frame) / max(len(counted_vehicle_frame), 1)

display(Markdown(
    f"Deteksi lolos ambang: **{len(qualified_frame)}** box · "
    f"track mentah: **{tracks_total}** · setelah penyambungan fragmen: **{vehicles_total}** kendaraan "
    f"(**{fragments_stitched}** fragmen disambung) · "
    f"dihitung (>= {MIN_TRACK_HITS} kemunculan): **{len(counted_vehicle_frame)}** · "
    f"rasio duplikasi vs hitung per frame: **{duplication_factor:.1f}x**"
))
display(Markdown(f"### Fragmen yang disambung ({len(fragment_merges)} kendaraan)"))
display(fragment_merges.head(20))
display(Markdown("### Keluaran wajib (kontrak MASTER §10.6, satuan kendaraan unik)"))
display(pd.DataFrame([{"keluaran": key, "jumlah": value} for key, value in counting_totals.items()]))
display(Markdown("### Rincian per video"))
display(per_video_frame)'''

SUMMARY_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labelled_totals = {
    "kendaraan mobil": counting_totals["jumlah_kendaraan_mobil"],
    "kendaraan motor": counting_totals["jumlah_kendaraan_motor"],
    "silau mobil": counting_totals["jumlah_lampu_silau_mobil"],
    "silau motor": counting_totals["jumlah_lampu_silau_motor"],
    "berwarna mobil": counting_totals["jumlah_lampu_berwarna_mobil"],
    "berwarna motor": counting_totals["jumlah_lampu_berwarna_motor"],
    "silau+berwarna mobil": counting_totals["jumlah_lampu_silau_dan_berwarna_mobil"],
    "silau+berwarna motor": counting_totals["jumlah_lampu_silau_dan_berwarna_motor"],
}

figure, axis = plt.subplots(figsize=(12, 5))
plot_totals = {key: value for key, value in labelled_totals.items() if isinstance(value, int)}
bar_positions = range(len(plot_totals))
axis.bar(bar_positions, list(plot_totals.values()), color="#1F7A4D")
axis.set_xticks(list(bar_positions))
axis.set_xticklabels(list(plot_totals.keys()), rotation=30, ha="right")
axis.set_ylabel("jumlah kendaraan unik")
axis.set_title(
    f"NB06 — kendaraan unik di dalam ROI ({len(counted_vehicle_frame)} kendaraan)"
    + (" (SMOKE, bukan hasil riset)" if SMOKE_MODE else "")
)
for position, value in zip(bar_positions, plot_totals.values()):
    axis.text(position, value, str(value), ha="center", va="bottom", fontsize=9)
axis.grid(axis="y", alpha=0.3)
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=120)
plt.close(figure)

counting_summary_payload = {
    "smoke_mode": SMOKE_MODE,
    "weights": str(WEIGHTS_PATH),
    "tracker": TRACKER_CONFIG,
    "counted_splits": COUNTED_SPLITS,
    "unit": "kendaraan unik per video (satu track dihitung satu kali)",
    "frames_processed": int(video_summary_frame["frames_processed"].sum()),
    "raw_detections": int(len(tracking_frame)),
    "untracked_detections": total_untracked,
    "thresholded_detections": int(len(qualified_frame)),
    "tracks_total": tracks_total,
    "vehicles_total": vehicles_total,
    "fragments_stitched": fragments_stitched,
    "vehicles_counted": int(len(counted_vehicle_frame)),
    "videos_found": int(len(discovery_frame)),
    "videos_counted": int(len(scan_jobs)),
    "min_track_hits": MIN_TRACK_HITS,
    "max_fragment_gap_frames": FRAGMENT_MAX_GAP_FRAMES,
    "min_fragment_iou": FRAGMENT_MIN_IOU,
    "duplication_factor_vs_per_frame": round(float(duplication_factor), 2),
    "class_counts": class_counts,
    "counting_totals": counting_totals,
    "selected_thresholds": selected_thresholds,
    "counting_timeline": str(COUNTING_TIMELINE_PATH),
    "overlay_manifest": str(OVERLAY_MANIFEST_PATH),
}
COUNTING_SUMMARY_PATH.write_text(
    json.dumps(counting_summary_payload, indent=2, ensure_ascii=False), encoding="utf-8"
)

contract_lines = "\\n".join(
    f"| {contract_key} | {counting_totals[contract_key]} |" for contract_key in COUNT_CONTRACT
)
class_lines = "\\n".join(
    f"| `{class_name}` | {class_counts[class_name]} |" for class_name in TRAINING_CLASS_NAMES
)

display(Markdown(
    f"""### Ringkasan NB06

Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · video: {len(scan_jobs)} · frame diproses: **{int(video_summary_frame['frames_processed'].sum())}**

| Tahap | Jumlah |
|---|---|
| track mentah dari ByteTrack | {tracks_total} |
| fragmen disambung | {fragments_stitched} |
| kendaraan setelah penyambungan | {vehicles_total} |
| kendaraan dihitung (>= {MIN_TRACK_HITS} kemunculan) | {len(counted_vehicle_frame)} |

| Keluaran | Jumlah |
|---|---|
{contract_lines}

**Kendaraan per kelas**

| Kelas | Jumlah |
|---|---|
{class_lines}

Gambar: `{FIGURE_PATH}` · ringkasan: `{COUNTING_SUMMARY_PATH}` · track: `{TRACK_SUMMARY_PATH}` · fragmen: `{FRAGMENT_MERGES_PATH}`

**Cara membaca angka ini:** satu baris = satu kendaraan yang dilacak ByteTrack di dalam ROI,
bukan satu box per frame. Rasio duplikasi dibanding hitung per frame adalah
**{duplication_factor:.1f}x**, itu gambaran seberapa besar overcount kalau tracking dilewati.

**Penyambungan fragmen.** Track yang terputus karena kendaraan sempat hilang dari pandangan
(lebih dari buffer tracker) disambung kembali bila jarak waktunya <= {FRAGMENT_MAX_GAP_FRAMES} frame
dan box-nya bertumpuk (IoU >= {FRAGMENT_MIN_IOU}) atau pusatnya sangat dekat. Hanya fragmen dengan
jenis sama (mobil/motor) yang boleh disambung. Bukti setiap penggabungan ada di `fragment_merges.csv`
supaya bisa diperiksa manual.

**Batas:** track di-reset per video (tidak ada identitas lintas sesi); kendaraan yang keluar lalu
masuk lagi terhitung dua kali; kamera test tidak punya ground truth sehingga angka ini adalah hasil
aplikasi model, bukan akurasi terverifikasi.
"""
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(JOBS_SOURCE),
        nbformat.v4.new_code_cell(TRACK_SOURCE),
        nbformat.v4.new_code_cell(COUNT_SOURCE),
        nbformat.v4.new_code_cell(SUMMARY_SOURCE),
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
