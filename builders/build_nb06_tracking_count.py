#!/usr/bin/env python3
"""Build NB06_tracking_count.ipynb: count event kendaraan tidak standar saat track melintasi garis."""

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_gate import verify_code_cells


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB06_tracking_count.ipynb"

EXPECTED_CODE_CELL_COUNT = 6
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB06 â€” Hitung Crossing Kendaraan Tidak Standar

NB05 menghitung **box per frame sampel**, sehingga satu kendaraan bisa terhitung berkali-kali.
NB06 menghitung **kendaraan tidak standar satu kali saat track melintasi counting line**. Sebagai
metrik pendamping, jumlah bbox tidak standar per frame tetap disimpan terpisah untuk mengaudit aktivitas
deteksi; metrik pendamping itu bukan total kendaraan.

**Posisi:** `NB02 â†’ NB02b â†’ NB03 â†’ NB04 â†’ NB05 (kalibrasi) â†’ NB06 (hitung unik)`

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
   pandangan **disambung kembali** sebelum dihitung (jenis sama, jeda 1â€“50 frame, box bertumpuk atau
   pusat berdekatan), lalu `MIN_TRACK_HITS` diterapkan pada kendaraan hasil sambungan. Bukti setiap
   penggabungan ada di `fragment_merges.csv`.
4. Kelas per track = **suara mayoritas** di antara frame yang lolos ambang; seri diputus oleh skor
   rata-rata lebih tinggi. Alasannya: kondisi lampu satu kendaraan bisa terbaca berbeda antar frame,
   dan satu kendaraan hanya boleh menyumbang satu angka.
5. YOLO dan CNN memakai lima kelas yang sama. YOLO memberi bbox/track; CNN mengklasifikasikan crop bbox menjadi kelas akhir. PantulanCahaya tetap kelas terlatih, tetapi tidak dihitung sebagai kendaraan dan tidak digambar sebagai event.
   Kelas ini tidak dihitung dan tidak digambar pada overlay.
6. Angka utama adalah satu crossing per vehicle track tidak standar. Garis horizontal awal berada pada
   72% tinggi frame dan dipotong mengikuti ROI. Track dihitung ketika center bbox berpindah sisi garis
   melewati hysteresis 1.5% tinggi frame, dan center x harus berada di rentang garis dalam ROI;
   arah dan frame crossing dicatat. Track yang mulai sudah di sisi seberang tanpa perpindahan
   teramati tidak dihitung. Hitungan bbox per frame tetap tersedia
   sebagai metrik pendamping.

## Ambang internal tracker

ByteTrack bawaan memakai `new_track_thresh: 0.25`; box di bawah 0.25 tidak akan pernah membentuk
track baru, hanya bisa menempel ke track yang sudah ada. Sementara ambang per kelas hasil NB05 boleh
serendah `CONF_FLOOR = 0.05`. Selisih itu membuat kelas langka tidak pernah punya track dan jumlahnya
kurang terhitung secara sistematis. Karena itu NB06 memakai `builders/bytetrack_nb06.yaml`
(`new_track_thresh 0.05`, `track_high_thresh 0.05`, `track_low_thresh 0.02`, `track_buffer 30`),
yaitu lantai yang sama dengan lantai detektor. Sel JOBS menolak jalan bila `new_track_thresh` lebih
tinggi dari ambang kelas terendah. Tracker hanya mengurus asosiasi antar frame; penyaringan akhir
tetap dilakukan ambang per kelas hasil NB05 saat agregasi track. Sebelum update ByteTrack, NB06
membuang deteksi dengan koordinat non-finite atau lebar/tinggi < 1 px. Jika update Kalman tetap
singular, tracker di-reset sambil mempertahankan urutan ID, dicoba sekali lagi, lalu frame dicatat
sebagai terlewat bila masih gagal. Ringkasan video memuat jumlah deteksi yang dibuang, recovery,
dan frame yang terlewat.

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

`PIPELINE_SMOKE=1` â†’ `output/smoke/`, hanya video pertama dari split yang dihitung, maksimum
`SMOKE_MAX_FRAMES_PER_VIDEO` frame. Tujuannya menguji alur tracking, bukan mengukur jumlah.

Output: `output/nb06/{tracking_detections.csv, track_summary.csv, counting_by_video.csv,
counting_timeline.csv, overlay_manifest.json, counting_summary.json, roi_mapping_audit.csv,
counting_bar.png, overlays/*.mp4}`. Overlay
dibuat untuk satu rekaman terbaru setiap kamera; agregasi hitungan tetap hanya memakai split test.

### Overlay mask SAM2 (wajib)

SAM2 membuat mask dari semua bbox YOLO per frame. Mask menjadi input photometry/bentuk CNN dan selalu
ditampilkan di overlay: kelas TidakStandar merah transparan, kelas Standar biru transparan. CNN
menggabungkan crop RGB, 77 fitur numerik berbasis bbox+mask, dan one-hot cluster e-means. Centroid,
scaler, dan checkpoint harus berasal dari NB04b yang sama. Proses SAM2 menambah waktu dan memori,
terutama di CPU.
"""

CONFIG_SOURCE = '''from pathlib import Path
import json
import os

import cv2
import numpy as np
import pandas as pd
import torch
from IPython.display import Markdown, display

from pipeline_common import (
    CLASS_NAMES,
    TRAINING_CLASS_NAMES,
    detect_center_line_crossing,
    COUNT_CONTRACT,
    FRAGMENT_DISTANCE_RATIO,
    FRAGMENT_MAX_GAP_FRAMES,
    FRAGMENT_MIN_IOU,
    SPLIT_NAME_BY_CAMERA,
    build_roi_crop,
    load_roi_config,
    stitch_track_fragments,
    validate_tracker_thresholds,
)
from attribute_classifier import IMAGE_TRANSFORM, decode_attributes, extract_mask_aware_features, load_classifier
from feature_clustering import assign_e_means_clusters, cluster_one_hot

from pipeline_common import resolve_project_root
RISET_ROOT = resolve_project_root(Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())))
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
BBOX_TIMELINE_PATH = NB06_ROOT / "bbox_per_frame_timeline.csv"
OVERLAY_MANIFEST_PATH = NB06_ROOT / "overlay_manifest.json"
EVENT_CLIPS_DIR = NB06_ROOT / "event_clips"
EVENT_CROPS_DIR = NB06_ROOT / "event_crops"
EVENT_CLIP_MANIFEST_PATH = NB06_ROOT / "event_clip_manifest.json"
MAPPING_AUDIT_PATH = NB06_ROOT / "roi_mapping_audit.csv"
FRAGMENT_MERGES_PATH = NB06_ROOT / "fragment_merges.csv"
FIGURE_PATH = NB06_ROOT / "counting_bar.png"
OVERLAY_DIR = NB06_ROOT / "overlays"
EVENT_CLIPS_DIR.mkdir(parents=True, exist_ok=True)
EVENT_CROPS_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_LABELS = os.environ.get("NB06_OVERLAY_LABELS", "1") == "1"
SAM2_MODEL_ID = os.environ.get("NB06_SAM2_MODEL", "facebook/sam2.1-hiera-tiny")

NB06_ROOT.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

COUNTED_SPLITS = ["test"]
COUNTING_LINE_Y_RATIO_BY_CAMERA = {
    camera_id: 0.72 for camera_id in SPLIT_NAME_BY_CAMERA
}
COUNTING_LINE_HYSTERESIS_RATIO = 0.015
TRACKER_CONFIG_PATH = RISET_ROOT / "builders" / "bytetrack_nb06.yaml"
TRACKER_CONFIG = str(TRACKER_CONFIG_PATH)
IMAGE_SIZE = 640
CONF_FLOOR = 0.05
NMS_IOU = 0.70
MIN_TRACK_BOX_SIDE_PX = 1.0
FRAME_STRIDE = 1
MIN_TRACK_HITS = 2 if SMOKE_MODE else 3
SMOKE_MAX_FRAMES_PER_VIDEO = 60
VIDEO_EXTENSION = ".mp4"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EVENT_COLORS_BGR = {
    "TidakStandar": (0, 0, 255),
}

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 40)

display(Markdown(
    f"NB06 siap. Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** Â· device: **{DEVICE}** Â· "
    f"bobot: `{WEIGHTS_PATH}` (ada: **{WEIGHTS_PATH.exists()}**) Â· split dihitung: {COUNTED_SPLITS} Â· "
    f"audit/galeri kendaraan unik: minimal {MIN_TRACK_HITS} kemunculan per track Â· "
    f"overlay label: **{OVERLAY_LABELS}** Â· overlay mask SAM2: **ON ({SAM2_MODEL_ID})**"
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
effective_thresholds = {
    class_name: float(
        selected_thresholds.get(
            class_name,
            selected_thresholds.get(class_name, 0.25),
        )
    )
    for class_name in CLASS_NAMES
}
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

def resolve_counting_line(camera_id: str, width: int, height: int) -> tuple[int, int, int]:
    """Horizontal tripwire at a camera-specific normalized image height, clipped to ROI polygon."""
    if camera_id not in COUNTING_LINE_Y_RATIO_BY_CAMERA:
        raise ValueError(f"Counting line belum dikonfigurasi untuk {camera_id}")
    y_line = int(round(height * COUNTING_LINE_Y_RATIO_BY_CAMERA[camera_id]))
    polygon = roi_config[camera_id].get("polygon")
    if not polygon:
        raise ValueError(f"ROI polygon diperlukan untuk menentukan counting line: {camera_id}")
    intersections = []
    for index, point_a in enumerate(polygon):
        point_b = polygon[(index + 1) % len(polygon)]
        x1, y1 = map(float, point_a)
        x2, y2 = map(float, point_b)
        if (y1 <= y_line < y2) or (y2 <= y_line < y1):
            ratio = (y_line - y1) / (y2 - y1)
            intersections.append(x1 + ratio * (x2 - x1))
    if len(intersections) < 2:
        raise ValueError(f"Garis hitung y={y_line} tidak memotong ROI valid kamera {camera_id}")
    x_start = max(0, int(round(min(intersections))))
    x_end = min(width - 1, int(round(max(intersections))))
    if x_end - x_start < 8:
        raise ValueError(f"Counting line terlalu pendek setelah dipotong ROI: {camera_id}")
    return y_line, x_start, x_end


for scan_job in scan_jobs:
    line_y, line_x1, line_x2 = resolve_counting_line(
        scan_job["camera_id"], scan_job["width"], scan_job["height"]
    )
    scan_job.update({"count_line_y": line_y, "count_line_x1": line_x1, "count_line_x2": line_x2})

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
    f"Total frame diproses: **{int(scan_frame['frame_limit'].sum())}** Â· "
    f"estimasi durasi: **{scan_frame['frame_limit'].sum() / scan_frame['fps'].mean() / 60:.1f} menit** video"
))'''

TRACK_SOURCE = '''from PIL import Image
from ultralytics import YOLO
from attribute_classifier import decode_attributes, extract_mask_aware_features, load_classifier
from feature_clustering import assign_e_means_clusters, cluster_one_hot

from transformers import Sam2Model, Sam2Processor
sam2_processor = Sam2Processor.from_pretrained(SAM2_MODEL_ID)
sam2_model = Sam2Model.from_pretrained(SAM2_MODEL_ID).to(DEVICE).eval()

RAW_BOX_DECIMALS = 2
SCORE_DECIMALS = 4
LABEL_FONT = cv2.FONT_HERSHEY_SIMPLEX
LABEL_FONT_SCALE = 0.38
LABEL_THICKNESS = 1
LABEL_PADDING = 2


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


def get_sam2_masks(crop_image: Image.Image, boxes_xyxy: np.ndarray) -> list[np.ndarray]:
    if len(boxes_xyxy) == 0:
        return []
    sam_inputs = sam2_processor(
        images=crop_image,
        input_boxes=[[list(map(float, row)) for row in boxes_xyxy]],
        return_tensors="pt",
    ).to(DEVICE)
    with torch.inference_mode():
        sam_outputs = sam2_model(**sam_inputs)
    masks = sam2_processor.post_process_masks(
        sam_outputs.pred_masks.cpu(), sam_inputs["original_sizes"].cpu()
    )[0]
    quality = sam_outputs.iou_scores[0].detach().cpu()
    return [masks[index, int(quality[index].argmax())].numpy().astype(bool) for index in range(len(boxes_xyxy))]


def blend_mask_on_frame(
    frame: np.ndarray, crop_mask: np.ndarray, offset_x: int, offset_y: int,
    scale: float, color_bgr: tuple[int, int, int], alpha: float = 0.30,
) -> None:
    full_height, full_width = frame.shape[:2]
    roi_width = max(1, int(round(crop_mask.shape[1] / scale)))
    roi_height = max(1, int(round(crop_mask.shape[0] / scale)))
    full_mask = cv2.resize(crop_mask.astype(np.uint8), (roi_width, roi_height), interpolation=cv2.INTER_NEAREST).astype(bool)
    x1, y1 = max(0, offset_x), max(0, offset_y)
    x2, y2 = min(full_width, offset_x + roi_width), min(full_height, offset_y + roi_height)
    if x2 <= x1 or y2 <= y1:
        return
    local_mask = full_mask[y1 - offset_y:y2 - offset_y, x1 - offset_x:x2 - offset_x]
    region = frame[y1:y2, x1:x2]
    tint = np.asarray(color_bgr, dtype=np.float32)
    region[local_mask] = np.clip(region[local_mask].astype(np.float32) * (1 - alpha) + tint * alpha, 0, 255).astype(np.uint8)


attribute_model, attribute_mean, attribute_std, cluster_centroids, attribute_metadata = load_classifier(ATTRIBUTE_MODEL_PATH, DEVICE)
if attribute_metadata.get("model_type") != "resnet18_sam2_emeans_fusion_5class":
    raise RuntimeError(
        "Checkpoint classifier tidak kompatibel. Latih ulang dengan "
        "`python builders/train_attribute_classifier.py` sebelum menjalankan NB06."
    )
if tuple(attribute_metadata.get("attribute_names", ())) != tuple(TRAINING_CLASS_NAMES):
    raise RuntimeError("Checkpoint CNN belum memakai urutan lima kelas aktif; latih ulang NB04b.")


def classify_vehicle_condition(roi_image: Image.Image, roi_box: list[float], roi_mask: np.ndarray) -> str:
    crop = extract_image_crop(roi_image, roi_box)
    image_tensor = IMAGE_TRANSFORM(crop).unsqueeze(0).to(DEVICE)
    raw_features = extract_mask_aware_features(roi_image, roi_box, roi_mask)
    normalized_features = (raw_features - attribute_mean) / np.maximum(attribute_std, 1e-6)
    feature_tensor = torch.from_numpy(normalized_features.astype(np.float32)).unsqueeze(0).to(DEVICE)
    cluster_id = assign_e_means_clusters(normalized_features[None, :], cluster_centroids)
    cluster_tensor = torch.from_numpy(cluster_one_hot(cluster_id, len(cluster_centroids))).to(DEVICE)
    with torch.no_grad():
        logits = attribute_model(image_tensor, feature_tensor, cluster_tensor).squeeze(0).cpu().numpy()
    attributes = decode_attributes(logits, attribute_metadata.get("thresholds"))
    return next(class_name for class_name, is_selected in attributes.items() if is_selected)


def process_video_tracking(job: dict) -> tuple[list[dict], dict, list[dict]]:
    tracking_model = YOLO(str(WEIGHTS_PATH))
    model_names = [tracking_model.names[index] for index in sorted(tracking_model.names)]
    if model_names != TRAINING_CLASS_NAMES:
        raise RuntimeError(
            f"Checkpoint YOLO memakai kelas {model_names}; kontrak aktif {TRAINING_CLASS_NAMES}. "
            "Latih ulang NB04 sebelum NB06."
        )
    vehicle_class_ids = [i for i, name in enumerate(TRAINING_CLASS_NAMES) if name != "PantulanCahaya"]
    video_capture = cv2.VideoCapture(job["video_path"])
    tracking_rows = []
    audit_rows = []
    condition_by_track: dict[int, str] = {}
    untracked_detection_count = 0
    invalid_tracker_detection_count = 0
    tracker_recovery_count = 0
    tracker_skipped_frame_count = 0
    processed_frame_count = 0
    frame_index = 0
    overlay_path = OVERLAY_DIR / f"{job['video_stem']}_overlay.mp4"
    overlay_writer = cv2.VideoWriter(
        str(overlay_path), cv2.VideoWriter_fourcc(*"avc1"),
        float(job["fps"]), (int(job["width"]), int(job["height"])),
    )
    if not overlay_writer.isOpened():
        video_capture.release()
        raise RuntimeError(f"gagal membuat video overlay: {overlay_path}")

    def sanitize_tracker_detections(predictor) -> None:
        nonlocal invalid_tracker_detection_count
        for result_index, result in enumerate(predictor.results):
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue
            xyxy = boxes.xyxy
            widths = xyxy[:, 2] - xyxy[:, 0]
            heights = xyxy[:, 3] - xyxy[:, 1]
            valid = torch.isfinite(xyxy).all(dim=1) & torch.isfinite(boxes.conf)
            valid &= (widths >= MIN_TRACK_BOX_SIDE_PX) & (heights >= MIN_TRACK_BOX_SIDE_PX)
            if not bool(valid.all()):
                invalid_tracker_detection_count += int((~valid).sum().item())
                predictor.results[result_index] = result[valid]

    # This callback runs after detector postprocessing but before Ultralytics updates ByteTrack.
    # A zero-area/non-finite box can make the XYAH Kalman covariance singular.
    tracking_model.add_callback("on_predict_postprocess_end", sanitize_tracker_detections)

    def reset_tracker_preserving_ids() -> bool:
        trackers = getattr(tracking_model.predictor, "trackers", None)
        if not trackers:
            return False
        from ultralytics.trackers.basetrack import BaseTrack

        next_track_id = BaseTrack._count
        for tracker in trackers:
            tracker.reset()
        # BYTETracker.reset() resets the process-wide ID counter; preserve it to avoid
        # reusing an ID that already exists in this video's rows.
        BaseTrack._count = next_track_id
        return True

    try:
        while True:
            read_ok, frame_array = video_capture.read()
            if not read_ok or (job["frame_limit"] and processed_frame_count >= job["frame_limit"]):
                break
            if frame_index % FRAME_STRIDE == 0:
                frame_image = Image.fromarray(cv2.cvtColor(frame_array, cv2.COLOR_BGR2RGB))
                crop_image, offset_x, offset_y, scale = build_roi_crop(
                    frame_image, roi_config[job["camera_id"]]
                )
                if scale <= 0:
                    raise ValueError(f"skala ROI tidak valid untuk {job['video_stem']}: {scale}")
                if processed_frame_count == 0:
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
                track_arguments = {
                    "source": crop_image,
                    "persist": True,
                    "tracker": TRACKER_CONFIG,
                    "classes": vehicle_class_ids,
                    "conf": CONF_FLOOR,
                    "iou": NMS_IOU,
                    "imgsz": IMAGE_SIZE,
                    "device": DEVICE,
                    "verbose": False,
                }
                try:
                    tracking_result = tracking_model.track(**track_arguments)[0]
                except np.linalg.LinAlgError as error:
                    if "singular" not in str(error).lower() or not reset_tracker_preserving_ids():
                        raise
                    tracker_recovery_count += 1
                    try:
                        tracking_result = tracking_model.track(**track_arguments)[0]
                    except np.linalg.LinAlgError as retry_error:
                        if "singular" not in str(retry_error).lower():
                            raise
                        reset_tracker_preserving_ids()
                        tracker_skipped_frame_count += 1
                        tracking_result = None
                processed_frame_count += 1
                frame_boxes = tracking_result.boxes if tracking_result is not None else None
                overlay_frame = frame_array.copy()
                line_start = (int(job["count_line_x1"]), int(job["count_line_y"]))
                line_end = (int(job["count_line_x2"]), int(job["count_line_y"]))
                cv2.line(overlay_frame, line_start, line_end, (0, 220, 255), 2, cv2.LINE_AA)
                cv2.putText(
                    overlay_frame, "GARIS HITUNG", (line_start[0], max(18, line_start[1] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 220, 255), 1, cv2.LINE_AA,
                )
                if frame_boxes is not None and len(frame_boxes) > 0:
                    box_coordinates = frame_boxes.xyxy.cpu().numpy()
                    box_scores = frame_boxes.conf.cpu().numpy()
                    box_classes = frame_boxes.cls.cpu().numpy().astype(int)
                    box_track_ids = frame_boxes.id.cpu().numpy().astype(int) if frame_boxes.id is not None else None
                    segmentation_masks = get_sam2_masks(crop_image, box_coordinates)
                    for box_index in range(len(box_scores)):
                        crop_box = [float(coordinate) for coordinate in box_coordinates[box_index]]
                        frame_box = [
                            offset_x + crop_box[0] / scale,
                            offset_y + crop_box[1] / scale,
                            offset_x + crop_box[2] / scale,
                            offset_y + crop_box[3] / scale,
                        ]
                        detector_class_name = TRAINING_CLASS_NAMES[box_classes[box_index]]
                        x1, y1, x2, y2 = [int(round(value)) for value in frame_box]
                        track_id = int(box_track_ids[box_index]) if box_track_ids is not None else None
                        if track_id is None:
                            untracked_detection_count += 1
                            blend_mask_on_frame(
                                overlay_frame, segmentation_masks[box_index], offset_x, offset_y,
                                scale, (255, 170, 45),
                            )
                            continue
                        if track_id not in condition_by_track:
                            condition_by_track[track_id] = classify_vehicle_condition(
                                crop_image, crop_box, segmentation_masks[box_index]
                            )
                        output_class_name = condition_by_track[track_id]
                        is_nonstandard = output_class_name in {"MobilTidakStandar", "MotorTidakStandar"}
                        is_vehicle = output_class_name in {
                            "MobilStandar", "MobilTidakStandar", "MotorStandar", "MotorTidakStandar"
                        }
                        overlay_color = EVENT_COLORS_BGR["TidakStandar"] if is_nonstandard else (255, 170, 45)
                        if is_vehicle:
                            blend_mask_on_frame(
                                overlay_frame, segmentation_masks[box_index], offset_x, offset_y,
                                scale, overlay_color,
                            )
                        if is_nonstandard:
                            cv2.rectangle(overlay_frame, (x1, y1), (x2, y2), overlay_color, 2)
                            if OVERLAY_LABELS:
                                overlay_label = f"ID {track_id} Â· {box_scores[box_index] * 100:.0f}%"
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
        "invalid_tracker_detections_removed": invalid_tracker_detection_count,
        "tracker_recoveries": tracker_recovery_count,
        "tracker_frames_skipped_after_recovery": tracker_skipped_frame_count,
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
        f"`{job['video_stem']}` selesai â€” {video_summary['frames_processed']} frame, "
        f"{video_summary['detections']} deteksi, {video_summary['untracked_detections']} tanpa track id, "
        f"{video_summary['invalid_tracker_detections_removed']} box degenerat dibuang, "
        f"{video_summary['tracker_recoveries']} recovery Kalman, "
        f"{video_summary['tracker_frames_skipped_after_recovery']} frame terlewat."
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
    threshold_series = detection_frame["class_name"].map(effective_thresholds)
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


def detect_line_crossings(detection_frame: pd.DataFrame) -> pd.DataFrame:
    """Emit one event per non-standard vehicle track when its bbox center crosses its camera line."""
    line_by_video = {job["video_stem"]: job for job in scan_jobs}
    crossing_rows = []
    event_classes = {"MobilTidakStandar", "MotorTidakStandar"}
    for (video_stem, vehicle_id), unit_frame in detection_frame.groupby(["video_stem", "vehicle_id"], sort=True):
        ordered = unit_frame.sort_values("frame_index")
        if len(ordered) < MIN_TRACK_HITS:
            continue
        event_class, mean_score = resolve_track_class(ordered)
        if event_class not in event_classes:
            continue
        job = line_by_video[video_stem]
        crossing = detect_center_line_crossing(
            ordered.to_dict(orient="records"),
            line_y=float(job["count_line_y"]),
            line_x1=float(job["count_line_x1"]),
            line_x2=float(job["count_line_x2"]),
            hysteresis=max(3.0, float(job["height"]) * COUNTING_LINE_HYSTERESIS_RATIO),
            min_hits=MIN_TRACK_HITS,
        )
        if crossing is None:
            continue
        row = crossing["row"]
        crossing_rows.append({
            "video_stem": video_stem,
            "camera_id": row["camera_id"],
            "vehicle_id": int(vehicle_id),
            "track_id": int(row["track_id"]),
            "class_name": event_class,
            "score": round(float(row["score"]), 4),
            "mean_track_score": round(mean_score, 4),
            "crossing_frame": int(row["frame_index"]),
            "crossing_direction": crossing["crossing_direction"],
            "line_y": int(job["count_line_y"]),
            "frame_x1": float(row["frame_x1"]),
            "frame_y1": float(row["frame_y1"]),
            "frame_x2": float(row["frame_x2"]),
            "frame_y2": float(row["frame_y2"]),
        })
    return pd.DataFrame(crossing_rows)


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

presentation_qualified_frame = qualify_by_class_threshold(tracking_frame)
presentation_track_frame = summarise_units(presentation_qualified_frame, "track_id")
presentation_stitch_lookup = stitch_track_fragments(presentation_track_frame)
presentation_vehicle_detections = presentation_qualified_frame.merge(
    presentation_stitch_lookup, on=["video_stem", "track_id"], how="left"
)
presentation_vehicle_frame = summarise_units(presentation_vehicle_detections, "vehicle_id")
presentation_counted_vehicle_frame = presentation_vehicle_frame[
    presentation_vehicle_frame["hits"] >= MIN_TRACK_HITS
].copy()
timeline_columns = [
    "video_stem", "camera_id", "vehicle_id", "class_name", "hits",
    "first_frame", "last_frame", "mean_score",
    "first_x1", "first_y1", "first_x2", "first_y2",
    "last_x1", "last_y1", "last_x2", "last_y2",
]
presentation_timeline_frame = presentation_counted_vehicle_frame.rename(columns={"unit_id": "vehicle_id"})[
    timeline_columns
].copy()
presentation_timeline_frame["fps"] = presentation_timeline_frame["video_stem"].map(
    {row.video_stem: float(row.fps) for row in scan_frame.itertuples()}
)
presentation_timeline_frame["first_timestamp_seconds"] = presentation_timeline_frame["first_frame"] / presentation_timeline_frame["fps"]
presentation_timeline_frame["last_timestamp_seconds"] = presentation_timeline_frame["last_frame"] / presentation_timeline_frame["fps"]
presentation_timeline_frame["scope"] = presentation_timeline_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).map(
    lambda split_name: "official" if split_name in COUNTED_SPLITS else "presentation_only"
)
event_timeline_frame = presentation_timeline_frame

crossing_frame = detect_line_crossings(presentation_vehicle_detections)
if crossing_frame.empty:
    crossing_frame = pd.DataFrame(columns=[
        "video_stem", "camera_id", "vehicle_id", "track_id", "class_name", "score",
        "mean_track_score", "crossing_frame", "crossing_direction", "line_y",
        "frame_x1", "frame_y1", "frame_x2", "frame_y2",
    ])
fps_by_video = {row.video_stem: float(row.fps) for row in scan_frame.itertuples()}
crossing_frame["fps"] = crossing_frame["video_stem"].map(fps_by_video)
crossing_frame["timestamp_seconds"] = crossing_frame["crossing_frame"] / crossing_frame["fps"]
crossing_frame["scope"] = crossing_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).map(
    lambda split_name: "official" if split_name in COUNTED_SPLITS else "presentation_only"
)
crossing_frame["count"] = 1
crossing_frame.to_csv(COUNTING_TIMELINE_PATH, index=False)

nonstandard_frame = tracking_frame[
    tracking_frame["class_name"].isin({"MobilTidakStandar", "MotorTidakStandar"})
].copy()
bbox_counted_frame = qualify_by_class_threshold(nonstandard_frame)
timeline_frame = bbox_counted_frame.rename(columns={"frame_index": "counting_frame"}).copy()
timeline_frame["bbox_id"] = (
    timeline_frame["video_stem"].astype(str)
    + "__frame_" + timeline_frame["counting_frame"].astype(str)
    + "__track_" + timeline_frame["track_id"].astype(str)
)
timeline_frame["fps"] = timeline_frame["video_stem"].map(
    {row.video_stem: float(row.fps) for row in scan_frame.itertuples()}
)
timeline_frame["timestamp_seconds"] = timeline_frame["counting_frame"] / timeline_frame["fps"]
timeline_frame["scope"] = timeline_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).map(
    lambda split_name: "official" if split_name in COUNTED_SPLITS else "presentation_only"
)
timeline_frame.to_csv(NB06_ROOT / "bbox_per_frame_timeline.csv", index=False)

official_bbox_frame = crossing_frame[
    crossing_frame["scope"] == "official"
].copy()
official_qualified_frame = qualified_frame[
    qualified_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).isin(COUNTED_SPLITS)
].copy()
class_counts = {
    class_name: int((official_qualified_frame["class_name"] == class_name).sum())
    for class_name in CLASS_NAMES
}
event_class_counts = {
    class_name: int((official_bbox_frame["class_name"] == class_name).sum())
    for class_name in CLASS_NAMES
}
counting_totals = {
    contract_key: int(sum(event_class_counts[class_name] for class_name in contract_classes))
    for contract_key, contract_classes in COUNT_CONTRACT.items()
}

per_video_rows = []
for video_job in sorted(
    (job for job in scan_jobs if job["split"] in COUNTED_SPLITS),
    key=lambda job: job["video_stem"],
):
    video_stem = video_job["video_stem"]
    video_slice = official_qualified_frame[official_qualified_frame["video_stem"] == video_stem]
    event_slice = official_bbox_frame[official_bbox_frame["video_stem"] == video_stem]
    video_class_counts = {
        class_name: int((video_slice["class_name"] == class_name).sum())
        for class_name in CLASS_NAMES
    }
    per_video_rows.append({
        "video_stem": video_stem,
        "camera_id": video_job["camera_id"],
        "crossings_counted": int(len(event_slice)),
        "bbox_per_frame_detections": int(len(bbox_counted_frame[bbox_counted_frame["video_stem"] == video_stem])),
        **video_class_counts,
        **{
            contract_key: int(sum(
                (event_slice["class_name"] == class_name).sum()
                for class_name in contract_classes
            ))
            for contract_key, contract_classes in COUNT_CONTRACT.items()
        },
    })
per_video_columns = [
    "video_stem", "camera_id", "crossings_counted", "bbox_per_frame_detections",
    *CLASS_NAMES, *COUNT_CONTRACT.keys()
]
per_video_frame = pd.DataFrame(per_video_rows, columns=per_video_columns)
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
        "counting_line": {
            "y_pixel": int(matching_job["count_line_y"]),
            "x_start": int(matching_job["count_line_x1"]),
            "x_end": int(matching_job["count_line_x2"]),
            "y_ratio": COUNTING_LINE_Y_RATIO_BY_CAMERA[matching_job["camera_id"]],
        },
        "segmentation_overlay": {
            "enabled": True,
            "model": SAM2_MODEL_ID,
            "alpha": 0.30,
            "classes": ["MobilTidakStandar", "MotorTidakStandar"],
        },
    })
OVERLAY_MANIFEST_PATH.write_text(
    json.dumps(overlay_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
)

event_clip_manifest = []
video_path_by_stem = {job["video_stem"]: job["video_path"] for job in scan_jobs}
for overlay_info in overlay_manifest:
    video_events = event_timeline_frame[event_timeline_frame["video_stem"] == overlay_info["video_stem"]]
    if video_events.empty:
        continue
    overlay_capture = cv2.VideoCapture(overlay_info["overlay_path"])
    if not overlay_capture.isOpened():
        raise RuntimeError(f"overlay tidak bisa dibuka untuk clip event: {overlay_info['overlay_path']}")
    fps = float(overlay_info["fps"])
    total_frames = int(overlay_info["frames_processed"])
    source_capture = cv2.VideoCapture(video_path_by_stem[overlay_info["video_stem"]])
    if not source_capture.isOpened():
        overlay_capture.release()
        raise RuntimeError(f"video sumber tidak bisa dibuka untuk crop event: {video_path_by_stem[overlay_info['video_stem']]}")
    for event in video_events.itertuples(index=False):
        start_frame = max(0, int(event.first_frame) - int(fps))
        end_frame = min(total_frames - 1, int(event.last_frame) + int(fps))
        class_directory = EVENT_CLIPS_DIR / str(event.class_name)
        class_directory.mkdir(parents=True, exist_ok=True)
        crop_directory = EVENT_CROPS_DIR / str(event.class_name)
        crop_directory.mkdir(parents=True, exist_ok=True)
        clip_path = class_directory / f"{event.video_stem}_vehicle_{int(event.vehicle_id):05d}.mp4"
        crop_path = crop_directory / f"{event.video_stem}_vehicle_{int(event.vehicle_id):05d}.jpg"
        source_capture.set(cv2.CAP_PROP_POS_FRAMES, int(event.first_frame))
        crop_ok, source_frame = source_capture.read()
        if crop_ok:
            crop_x1 = max(0, min(int(round(event.first_x1)), source_frame.shape[1] - 1))
            crop_y1 = max(0, min(int(round(event.first_y1)), source_frame.shape[0] - 1))
            crop_x2 = max(crop_x1 + 1, min(int(round(event.first_x2)), source_frame.shape[1]))
            crop_y2 = max(crop_y1 + 1, min(int(round(event.first_y2)), source_frame.shape[0]))
            crop_image = source_frame[crop_y1:crop_y2, crop_x1:crop_x2]
            cv2.imwrite(str(crop_path), crop_image)
        overlay_capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        clip_writer = cv2.VideoWriter(
            str(clip_path), cv2.VideoWriter_fourcc(*"avc1"), fps,
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
                "crop_path": str(crop_path) if crop_path.exists() else "",
                "timestamp_seconds": round(float(event.first_frame) / fps, 3),
                "duration_seconds": round(written_frames / fps, 3),
            })
    overlay_capture.release()
    source_capture.release()
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
    f"Deteksi lolos ambang: **{len(qualified_frame)}** box Â· "
    f"track mentah: **{tracks_total}** Â· setelah penyambungan fragmen: **{vehicles_total}** kendaraan "
    f"(**{fragments_stitched}** fragmen disambung) Â· "
    f"audit kendaraan unik (>= {MIN_TRACK_HITS} kemunculan): **{len(counted_vehicle_frame)}** Â· "
    f"rasio duplikasi vs hitung per frame: **{duplication_factor:.1f}x**"
))
display(Markdown(f"### Fragmen yang disambung ({len(fragment_merges)} kendaraan)"))
display(fragment_merges.head(20))
display(Markdown("### Keluaran utama (track tidak standar yang melintasi garis)"))
display(pd.DataFrame([{"keluaran": key, "jumlah": value} for key, value in counting_totals.items()]))
display(Markdown("### Rincian per video"))
display(per_video_frame)'''

SUMMARY_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

labelled_totals = {
    "mobil melintasi garis": counting_totals["jumlah_kendaraan_mobil_tidak_standar_melintasi_garis"],
    "motor melintasi garis": counting_totals["jumlah_kendaraan_motor_tidak_standar_melintasi_garis"],
}

figure, axis = plt.subplots(figsize=(12, 5))
plot_totals = {key: value for key, value in labelled_totals.items() if isinstance(value, int)}
bar_positions = range(len(plot_totals))
axis.bar(bar_positions, list(plot_totals.values()), color="#1F7A4D")
axis.set_xticks(list(bar_positions))
axis.set_xticklabels(list(plot_totals.keys()), rotation=30, ha="right")
axis.set_ylabel("kendaraan tidak standar melintasi garis")
axis.set_title(
    f"NB06 â€” crossing kendaraan tidak standar ({len(official_bbox_frame)} track)"
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
    "segmentation_overlay": {"enabled": True, "model": SAM2_MODEL_ID, "alpha": 0.30},
    "counting_line_y_ratio_by_camera": COUNTING_LINE_Y_RATIO_BY_CAMERA,
    "counted_splits": COUNTED_SPLITS,
    "unit": "vehicle line crossing (satu track dihitung satu kali per video saat melintasi garis)",
    "frames_processed": int(video_summary_frame["frames_processed"].sum()),
    "raw_detections": int(len(tracking_frame)),
    "untracked_detections": total_untracked,
    "thresholded_detections": int(len(qualified_frame)),
    "tracks_total": tracks_total,
    "vehicles_total": vehicles_total,
    "fragments_stitched": fragments_stitched,
    "line_crossings_counted": int(len(official_bbox_frame)),
    "bbox_per_frame_nonstandard_detections": int(len(bbox_counted_frame[bbox_counted_frame["camera_id"].map(SPLIT_NAME_BY_CAMERA).isin(COUNTED_SPLITS)])),
    "unique_vehicles_audit": int(len(counted_vehicle_frame)),
    "presentation_line_crossings": int((crossing_frame["scope"] == "presentation_only").sum()),
    "videos_found": int(len(discovery_frame)),
    "videos_counted": int(len(scan_jobs)),
    "min_track_hits": MIN_TRACK_HITS,
    "max_fragment_gap_frames": FRAGMENT_MAX_GAP_FRAMES,
    "min_fragment_iou": FRAGMENT_MIN_IOU,
    "duplication_factor_vs_per_frame": round(float(duplication_factor), 2),
    "class_counts": class_counts,
    "event_class_counts": event_class_counts,
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
    f"| `{class_name}` | {class_counts[class_name]} |" for class_name in CLASS_NAMES
)

display(Markdown(
    f"""### Ringkasan NB06

Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** Â· video: {len(scan_jobs)} Â· frame diproses: **{int(video_summary_frame['frames_processed'].sum())}**

| Tahap | Jumlah |
|---|---|
| track mentah dari ByteTrack | {tracks_total} |
| fragmen disambung | {fragments_stitched} |
| kendaraan setelah penyambungan | {vehicles_total} |
| kendaraan tidak standar melintasi garis (official test) | {len(official_bbox_frame)} |

| Keluaran | Jumlah |
|---|---|
{contract_lines}

**Kendaraan per kelas**

| Kelas | Jumlah |
|---|---|
{class_lines}

Gambar: `{FIGURE_PATH}` Â· ringkasan: `{COUNTING_SUMMARY_PATH}` Â· track: `{TRACK_SUMMARY_PATH}` Â· fragmen: `{FRAGMENT_MERGES_PATH}`

**Cara membaca angka ini:** satu baris pada counting utama = satu vehicle track tidak standar
yang melintasi garis hitung. Setiap kendaraan dihitung paling banyak satu kali per video. Arah
crossing dicatat (`up`/`down`). Bbox-per-frame disimpan di `bbox_per_frame_timeline.csv` sebagai
metrik pendamping untuk menilai aktivitas detektor.

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


CONVERT_OVERLAY_SOURCE = '''import os
import json
from pathlib import Path

import cv2
from IPython.display import Markdown, display


RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT
NB06_ROOT = DATA_ROOT / "nb06"
OVERLAY_MANIFEST_PATH = NB06_ROOT / "overlay_manifest.json"
BROWSER_OVERLAY_DIR = NB06_ROOT / "overlays_browser"

if not OVERLAY_MANIFEST_PATH.exists():
    raise FileNotFoundError(
        f"Manifest belum ditemukan: {OVERLAY_MANIFEST_PATH}. "
        "Jalankan NB06 sampai cell summary terlebih dahulu."
    )

overlay_manifest = json.loads(OVERLAY_MANIFEST_PATH.read_text(encoding="utf-8"))
manifest_is_list = isinstance(overlay_manifest, list)
if manifest_is_list:
    overlay_rows = overlay_manifest
else:
    overlay_rows = overlay_manifest.get("overlays", [])
BROWSER_OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

converted_rows = []
for item in overlay_rows:
    source = Path(item["overlay_path"])
    if not source.is_absolute():
        source = (RISET_ROOT / source).resolve()
    if not source.exists():
        raise FileNotFoundError(f"Overlay sumber tidak ditemukan: {source}")

    target = BROWSER_OVERLAY_DIR / source.name
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Overlay tidak bisa dibaca: {source}")

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    expected_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(target), cv2.VideoWriter_fourcc(*"avc1"), fps, (width, height)
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(
            "Writer AVC1 tidak tersedia pada environment ini. "
            "Cek instalasi OpenCV/codec sebelum mengulang cell."
        )

    written_frames = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        writer.write(frame)
        written_frames += 1
    capture.release()
    writer.release()

    if written_frames == 0 or not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(f"Konversi gagal atau menghasilkan file kosong: {target}")
    if expected_frames and written_frames != expected_frames:
        raise RuntimeError(
            f"Jumlah frame berubah untuk {source.name}: "
            f"{expected_frames} menjadi {written_frames}"
        )

    converted_item = dict(item)
    converted_item["overlay_path"] = str(target)
    converted_item["codec"] = "avc1"
    converted_item["source_overlay_path"] = str(source)
    converted_rows.append(converted_item)

if manifest_is_list:
    manifest_to_write = converted_rows
else:
    manifest_to_write = dict(overlay_manifest)
    manifest_to_write["overlays"] = converted_rows
    manifest_to_write["browser_compatible_dir"] = str(BROWSER_OVERLAY_DIR)
OVERLAY_MANIFEST_PATH.write_text(
    json.dumps(manifest_to_write, indent=2, ensure_ascii=False), encoding="utf-8"
)

display(Markdown(
    f"**Selesai:** {len(converted_rows)} overlay dikonversi ke AVC1 di "
    f"`{BROWSER_OVERLAY_DIR}`. Video asli tetap ada di folder `overlays`."
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(JOBS_SOURCE),
        nbformat.v4.new_code_cell(TRACK_SOURCE),
        nbformat.v4.new_code_cell(COUNT_SOURCE),
        nbformat.v4.new_code_cell(SUMMARY_SOURCE),
        nbformat.v4.new_code_cell(CONVERT_OVERLAY_SOURCE),
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
