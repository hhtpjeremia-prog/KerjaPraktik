#!/usr/bin/env python3
"""Build NB02_pseudo_label.ipynb for Riset-HampirMatang (ROI crop + lamp localisation pseudo-labels)."""

from pathlib import Path

import nbformat

from notebook_gate import verify_code_cells

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB02_pseudo_label.ipynb"

EXPECTED_CODE_CELL_COUNT = 6
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB02 — Crop ROI + Pseudo-Label Lokasi Lampu

Tahap NB02 mengubah frame sampling NB01 menjadi **dua bahan mentah** untuk seluruh pipeline:

1. **`crops/`** — setiap frame unik di-crop ke ROI kamera (polygon/rect milik pemilik riset) lalu di-`fit` 640 px. Inilah gambar yang dilihat anotator di NB02b dan yang dipakai NB04.
2. **`chunks/`** — pseudo-label **lokasi** lampu (bukan kondisi) hasil Grounding DINO-tiny + pengelompokan lampu satu kendaraan + vote jenis kendaraan.

## Mengapa NB02 tidak memprediksi kondisi lampu

Target riset adalah **kondisi** lampu: silau, berwarna, silau+berwarna. Itu **atribut visual**, bukan objek — Grounding DINO mendeteksi objek, sehingga prompt kondisi (`blinding headlight`, `blue headlight`, ...) tidak dapat dipercaya sebagai label. Mempaksakannya berarti menanam label salah ke dataset latih.

Karena itu pembagian tugasnya tegas:

| Peran | Sumber |
|---|---|
| Lokasi lampu (box) | DINO (pseudo) + koreksi manusia |
| Jenis kendaraan (mobil/motor) | vote FasterRCNN + geometri pasangan lampu (pseudo) |
| **Kondisi lampu** (silau/berwarna/silau+berwarna) | **anotasi manusia saja** (NB02b) |

Pseudo-label kondisi sengaja **tidak** dibuat. `MobilNormal`/`MotorNormal` juga tidak pernah ditebak DINO, karena kendaraan tanpa deteksi kondisi belum tentu normal.

## Mekanisme

- **Satu prompt gabungan**: `vehicle headlight. motorcycle headlight.` dijalankan satu kali per crop. Keduanya satu kelas internal (`LampuKendaraan`); memisahkannya jadi dua forward pass hanya menggandakan waktu tanpa mengubah hasil.
- **Merge lampu tumpang tindih**: IoU ≥ 0.50 digabung, skor per prompt disimpan.
- **Kelompokkan lampu satu kendaraan**: jarak pusat ≤ 80 px, offset vertikal ≤ 25 px, rasio lebar ≤ 2.5 (`GROUP_*`).
- **Satu box per kelompok (union)** = satu kendaraan. `lamp_count ≥ 2` → `Mobil`; selain itu ikut vote FasterRCNN; bila keduanya tidak ada → `None` (anotator yang memutuskan). Di preview, `None` **selalu ditulis `UNSURE`** agar tidak terbaca sebagai nama kelas; di `nb02_summary.json` nilainya tetap `TidakPasti`.
- **`signals_agree = false`** menandai geometri dan vote bertentangan → prioritas review manusia.
- **Threshold box 0.25 / text 0.20, verifier score 0.30, coverage vote 0.30**. Threshold box dan verifier dilonggarkan 2026-10-29 setelah audit gold 170 frame: recall proposal DINO hanya ±61%, jadi proposal dibuat lebih permisif dan penolakan diserahkan ke anotator. Kalibrasi berbasis gold dikerjakan NB03.
- **Dedup**: jarak Hamming ≤ 4 terhadap satu frame sebelumnya dalam video yang sama.
- **Resume aman**: chunk per 100 frame; bila prompt/threshold/ROI/parameter berubah, signature konfigurasi berubah dan NB02 **menolak resume** (hapus `output/nb02/chunks` dulu).
- **Tidak ada `data.yaml` dan tidak ada label YOLO di sini.** Dataset latih dibangun NB04 dari gold store manusia, bukan dari pseudo-label kondisi.
- **Semua frame unik dapat crop**, termasuk yang tidak ber-deteksi — tanpa itu anotator tidak akan pernah melihat frame yang dilewatkan DINO.
- Setiap frame diisolasi `try/except`; kegagalan satu frame dicatat, tidak menghentikan fase.
- Device otomatis: CUDA bila tersedia, selain itu CPU.

Output: `output/nb02/crops/*.png`, `output/nb02/chunks/*.jsonl`, `output/nb02/previews/*.png`, `output/nb02/nb02_summary.json`.
"""

CONFIG_SOURCE = '''from pathlib import Path
import csv
import hashlib
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from IPython.display import Markdown, display

from pipeline_common import build_roi_crop, SPLIT_NAME_BY_CAMERA

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT
FRAMES_DIR = OUTPUT_ROOT / "frames"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
ROI_DIR = OUTPUT_ROOT / "roi"
NB02_ROOT = DATA_ROOT / "nb02"
CHUNK_DIR = NB02_ROOT / "chunks"
CROP_DIR = NB02_ROOT / "crops"
PREVIEW_DIR = NB02_ROOT / "previews"

for output_directory in (CHUNK_DIR, CROP_DIR, PREVIEW_DIR):
    output_directory.mkdir(parents=True, exist_ok=True)

DETECTION_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
DETECTION_PROMPTS = ("vehicle headlight. motorcycle headlight.",)
DETECTION_THRESHOLD = 0.25
TEXT_THRESHOLD = 0.20
DETECTION_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VEHICLE_VERIFIER_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
VEHICLE_SCORE_THRESHOLD = 0.30
VEHICLE_VOTE_MIN_COVERAGE = 0.3
VEHICLE_CLASS_REMAP = {
    "car": "Mobil",
    "bus": "Mobil",
    "truck": "Mobil",
    "motorcycle": "Motor",
}

MAX_RESIZE_SIDE = 640
MIN_BOX_PIXELS = 5
MAX_BOX_AREA_RATIO = 0.40
CHUNK_FRAME_COUNT = 100
SMOKE_FRAME_LIMIT = 12
DUPLICATE_HASH_DISTANCE = 4
DUPLICATE_WINDOW_FRAMES = 1
HASH_PLACEHOLDER = -1
PREVIEW_FRAMES_PER_CAMERA = 3

LAMP_MERGE_IOU_THRESHOLD = 0.5
GROUP_MAX_CENTER_DISTANCE_PIXELS = 80
GROUP_MAX_VERTICAL_OFFSET_PIXELS = 25
GROUP_SIZE_RATIO_LIMIT = 2.5
MULTI_LAMP_MIN_COUNT = 2

VEHICLE_TYPE_MOBIL = "Mobil"
VEHICLE_TYPE_MOTOR = "Motor"

display(Markdown(
    f"NB02 siap — lokalisasi lampu tanpa prediksi kondisi. Device: **{DETECTION_DEVICE}** · "
    f"prompt: {len(DETECTION_PROMPTS)} · root: {RISET_ROOT.name}"
))'''

FRAME_PLAN_SOURCE = '''def read_table_rows(table_path: Path) -> list[dict]:
    with table_path.open(encoding="utf-8") as table_file:
        return list(csv.DictReader(table_file))


def load_roi_config(roi_directory: Path) -> dict:
    with (roi_directory / "roi_config.json").open(encoding="utf-8") as roi_file:
        return json.load(roi_file)


def extract_camera_id(video_stem: str) -> str:
    return video_stem.split("_")[1]


def extract_session_ts(video_stem: str) -> str:
    return "_".join(video_stem.split("_")[2:])


def frame_order_key(frame_path_text: str) -> int:
    return int(Path(frame_path_text).stem.split("_")[1])


def hamming_distance(first_hash: int, second_hash: int) -> int:
    return bin(first_hash ^ second_hash).count("1")


def compute_duplicate_flags(frame_hashes: list[int], window_frames: int, distance_threshold: int) -> list[bool]:
    duplicate_flags = []
    for frame_index, current_hash in enumerate(frame_hashes):
        is_duplicate = False
        for previous_hash in frame_hashes[max(0, frame_index - window_frames) : frame_index]:
            if previous_hash == HASH_PLACEHOLDER:
                continue
            if hamming_distance(previous_hash, current_hash) <= distance_threshold:
                is_duplicate = True
                break
        duplicate_flags.append(is_duplicate)
    return duplicate_flags


def build_frame_records(frames_manifest: list[dict], dedup_manifest: list[dict], roi_config: dict) -> list[dict]:
    hash_by_frame_path = {row["frame_path"]: int(row["frame_hash"]) for row in dedup_manifest}
    video_groups: dict[str, list[dict]] = {}
    for frame_row in frames_manifest:
        video_path_text = frame_row["video_path"]
        video_stem = Path(video_path_text).stem
        video_groups.setdefault(video_path_text, []).append(
            {
                "video_path": video_path_text,
                "video_stem": video_stem,
                "camera_id": extract_camera_id(video_stem),
                "session_ts": extract_session_ts(video_stem),
                "frame_path": frame_row["frame_path"],
            }
        )
    frame_records: list[dict] = []
    for video_path_text, video_rows in video_groups.items():
        video_rows.sort(key=lambda video_row: frame_order_key(video_row["frame_path"]))
        duplicate_flags = compute_duplicate_flags(
            [hash_by_frame_path.get(video_row["frame_path"], HASH_PLACEHOLDER) for video_row in video_rows],
            DUPLICATE_WINDOW_FRAMES,
            DUPLICATE_HASH_DISTANCE,
        )
        for video_row, is_duplicate in zip(video_rows, duplicate_flags):
            video_row["is_duplicate"] = is_duplicate
            video_row["roi_entry"] = roi_config.get(video_row["camera_id"], {})
            frame_records.append(video_row)
    return frame_records


def select_smoke_frames(frame_records: list[dict], frame_limit: int) -> list[dict]:
    frames_by_camera: dict[str, list[dict]] = {}
    for frame_record in frame_records:
        frames_by_camera.setdefault(frame_record["camera_id"], []).append(frame_record)
    camera_ids = sorted(frames_by_camera)
    selected_frames = []
    frame_position = 0
    while len(selected_frames) < frame_limit and any(
        frame_position < len(frames_by_camera[camera_id]) for camera_id in camera_ids
    ):
        for camera_id in camera_ids:
            if frame_position < len(frames_by_camera[camera_id]) and len(selected_frames) < frame_limit:
                selected_frames.append(frames_by_camera[camera_id][frame_position])
        frame_position += 1
    return sorted(selected_frames, key=lambda frame_record: frame_record["frame_path"])


CURATED_FRAMES_MANIFEST = MANIFEST_DIR / "curated_frames_manifest.csv"
CURATED_DEDUP_MANIFEST = MANIFEST_DIR / "curated_dedup_manifest.csv"
INPUT_FRAMES_MANIFEST = CURATED_FRAMES_MANIFEST if CURATED_FRAMES_MANIFEST.exists() else MANIFEST_DIR / "frames_manifest.csv"
INPUT_DEDUP_MANIFEST = CURATED_DEDUP_MANIFEST if CURATED_DEDUP_MANIFEST.exists() else MANIFEST_DIR / "dedup_manifest.csv"
frames_manifest_rows = read_table_rows(INPUT_FRAMES_MANIFEST)
dedup_manifest_rows = read_table_rows(INPUT_DEDUP_MANIFEST)
roi_config = load_roi_config(ROI_DIR)
frame_records = build_frame_records(frames_manifest_rows, dedup_manifest_rows, roi_config)
unique_frame_records = [frame_record for frame_record in frame_records if not frame_record["is_duplicate"]]
skipped_duplicate_count = len(frame_records) - len(unique_frame_records)
unique_frame_total_before_smoke = len(unique_frame_records)
if SMOKE_MODE:
    unique_frame_records = select_smoke_frames(unique_frame_records, SMOKE_FRAME_LIMIT)

camera_summary: dict[str, dict] = {}
for frame_record in frame_records:
    camera_entry = camera_summary.setdefault(
        frame_record["camera_id"], {"frames_total": 0, "frames_unique": 0}
    )
    camera_entry["frames_total"] += 1
    if not frame_record["is_duplicate"]:
        camera_entry["frames_unique"] += 1

display(Markdown(
    f"Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · frame total: **{len(frame_records)}** · "
    f"unik: **{unique_frame_total_before_smoke}** · diproses: **{len(unique_frame_records)}** · "
    f"duplikat di-skip: **{skipped_duplicate_count}**"
))
for camera_id, camera_entry in sorted(camera_summary.items()):
    display(Markdown(
        f"- `{camera_id}` → split **{SPLIT_NAME_BY_CAMERA.get(camera_id, 'tidak terdaftar')}** · "
        f"{camera_entry['frames_total']} frame ({camera_entry['frames_unique']} unik)"
    ))'''

CROP_SOURCE = '''def crop_file_stem(frame_record: dict) -> str:
    return frame_record["video_stem"] + "_" + Path(frame_record["frame_path"]).stem


def write_crop_images(frame_records: list[dict], crop_directory: Path) -> dict:
    crop_image_count = 0
    failed_crop_stems = []
    for frame_record in frame_records:
        crop_stem = crop_file_stem(frame_record)
        try:
            frame_image = Image.open(Path(frame_record["frame_path"])).convert("RGB")
            crop_image = build_roi_crop(frame_image, frame_record["roi_entry"], MAX_RESIZE_SIDE)[0]
            crop_image.save(crop_directory / f"{crop_stem}.png")
            crop_image_count += 1
        except Exception as crop_error:
            failed_crop_stems.append(f"{crop_stem}: {crop_error}")
    return {"crop_images": crop_image_count, "failed_crop_stems": failed_crop_stems}


crop_summary = write_crop_images(unique_frame_records, CROP_DIR)
display(Markdown(
    f"Crop selesai: **{crop_summary['crop_images']}** gambar di `{CROP_DIR.name}/` · "
    f"gagal: **{len(crop_summary['failed_crop_stems'])}**"
))
for failed_crop in crop_summary["failed_crop_stems"][:10]:
    display(Markdown(f"- gagal: `{failed_crop}`"))'''

DETECTION_SOURCE = '''def intersection_over_union(box_a: list[float], box_b: list[float]) -> float:
    overlap_x1 = max(box_a[0], box_b[0])
    overlap_y1 = max(box_a[1], box_b[1])
    overlap_x2 = min(box_a[2], box_b[2])
    overlap_y2 = min(box_a[3], box_b[3])
    intersection_area = max(0.0, overlap_x2 - overlap_x1) * max(0.0, overlap_y2 - overlap_y1)
    if intersection_area <= 0.0:
        return 0.0
    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = box_a_area + box_b_area - intersection_area
    return intersection_area / union_area if union_area > 0.0 else 0.0


def box_coverage(outer_box: list[float], inner_box: list[float]) -> float:
    overlap_x1 = max(outer_box[0], inner_box[0])
    overlap_y1 = max(outer_box[1], inner_box[1])
    overlap_x2 = min(outer_box[2], inner_box[2])
    overlap_y2 = min(outer_box[3], inner_box[3])
    intersection_area = max(0.0, overlap_x2 - overlap_x1) * max(0.0, overlap_y2 - overlap_y1)
    inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
    return intersection_area / inner_area if inner_area > 0.0 else 0.0


def run_dino_detection(model, processor, image: Image.Image, prompt: str, box_threshold: float, text_threshold: float, device: str) -> dict:
    inputs = processor(images=[image], text=prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        outputs = model(**inputs)
    target_size = torch.tensor([[image.size[1], image.size[0]]], device=device)
    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=box_threshold,
        text_threshold=text_threshold,
        target_sizes=target_size,
    )
    return results[0]


def clip_and_filter_boxes(box_list, score_list, image_width: int, image_height: int) -> list[dict]:
    filtered_detections = []
    image_area = image_width * image_height
    for box, score in zip(box_list, score_list):
        x1_value, y1_value, x2_value, y2_value = [float(boundary) for boundary in box.tolist()]
        x1_value = max(0.0, min(x1_value, image_width))
        y1_value = max(0.0, min(y1_value, image_height))
        x2_value = max(0.0, min(x2_value, image_width))
        y2_value = max(0.0, min(y2_value, image_height))
        if x2_value - x1_value < MIN_BOX_PIXELS or y2_value - y1_value < MIN_BOX_PIXELS:
            continue
        box_area_ratio = ((x2_value - x1_value) * (y2_value - y1_value)) / image_area
        if box_area_ratio > MAX_BOX_AREA_RATIO:
            continue
        filtered_detections.append({"score": float(score), "box": [x1_value, y1_value, x2_value, y2_value]})
    return filtered_detections


def detect_lamp_candidates(model, processor, image: Image.Image, device: str) -> list[dict]:
    lamp_candidates = []
    for prompt in DETECTION_PROMPTS:
        detection_result = run_dino_detection(
            model, processor, image, prompt, DETECTION_THRESHOLD, TEXT_THRESHOLD, device
        )
        filtered_detections = clip_and_filter_boxes(
            detection_result["boxes"],
            detection_result["scores"],
            image.size[0],
            image.size[1],
        )
        for filtered_detection in filtered_detections:
            lamp_candidates.append(
                {
                    "prompt": prompt,
                    "score": filtered_detection["score"],
                    "box": filtered_detection["box"],
                }
            )
    return lamp_candidates


def merge_overlapping_lamps(lamp_candidates: list[dict]) -> list[dict]:
    merged_lamps: list[dict] = []
    for candidate in sorted(lamp_candidates, key=lambda candidate_item: candidate_item["score"], reverse=True):
        matching_lamp = None
        for merged_lamp in merged_lamps:
            if intersection_over_union(candidate["box"], merged_lamp["box"]) >= LAMP_MERGE_IOU_THRESHOLD:
                matching_lamp = merged_lamp
                break
        if matching_lamp is None:
            merged_lamps.append({"box": candidate["box"], "score": candidate["score"]})
        else:
            matching_lamp["score"] = max(matching_lamp["score"], candidate["score"])
    return merged_lamps


def lamp_center(lamp: dict) -> tuple[float, float]:
    box = lamp["box"]
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def lamps_are_same_vehicle(first_lamp: dict, second_lamp: dict) -> bool:
    first_center_x, first_center_y = lamp_center(first_lamp)
    second_center_x, second_center_y = lamp_center(second_lamp)
    center_distance = abs(first_center_x - second_center_x)
    vertical_offset = abs(first_center_y - second_center_y)
    first_width = first_lamp["box"][2] - first_lamp["box"][0]
    second_width = second_lamp["box"][2] - second_lamp["box"][0]
    width_ratio = max(first_width, second_width) / max(1.0, min(first_width, second_width))
    return (
        center_distance <= GROUP_MAX_CENTER_DISTANCE_PIXELS
        and vertical_offset <= GROUP_MAX_VERTICAL_OFFSET_PIXELS
        and width_ratio <= GROUP_SIZE_RATIO_LIMIT
    )


def group_lamps_into_vehicles(merged_lamps: list[dict]) -> list[list[dict]]:
    vehicle_groups: list[list[dict]] = []
    for lamp in sorted(merged_lamps, key=lambda lamp_item: lamp_center(lamp_item)[0]):
        target_group = None
        for vehicle_group in vehicle_groups:
            if any(lamps_are_same_vehicle(lamp, group_lamp) for group_lamp in vehicle_group):
                target_group = vehicle_group
                break
        if target_group is None:
            vehicle_groups.append([lamp])
        else:
            target_group.append(lamp)
    return vehicle_groups


def group_bounding_box(vehicle_group: list[dict]) -> list[float]:
    group_x1 = min(lamp["box"][0] for lamp in vehicle_group)
    group_y1 = min(lamp["box"][1] for lamp in vehicle_group)
    group_x2 = max(lamp["box"][2] for lamp in vehicle_group)
    group_y2 = max(lamp["box"][3] for lamp in vehicle_group)
    return [group_x1, group_y1, group_x2, group_y2]


def load_vehicle_verifier():
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    try:
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
        verifier_model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=VEHICLE_SCORE_THRESHOLD)
        verifier_model.to(VEHICLE_VERIFIER_DEVICE).eval()
        return verifier_model, list(weights.meta["categories"])
    except Exception as verifier_error:
        raise RuntimeError(f"Verifier kendaraan gagal dimuat: {verifier_error}") from verifier_error


def detect_vehicles(verifier_model, verifier_categories: list | None, image: Image.Image) -> list[dict] | None:
    import torchvision.transforms.functional as vision_functional
    image_tensor = vision_functional.to_tensor(image).to(VEHICLE_VERIFIER_DEVICE)
    with torch.inference_mode():
        prediction = verifier_model([image_tensor])[0]
    vehicle_detections = []
    for box, score, label in zip(prediction["boxes"], prediction["scores"], prediction["labels"]):
        category_name = verifier_categories[int(label)]
        if category_name not in VEHICLE_CLASS_REMAP:
            continue
        vehicle_detections.append(
            {
                "vehicle_type": VEHICLE_CLASS_REMAP[category_name],
                "score": float(score),
                "box": [float(value) for value in box.tolist()],
            }
        )
    return vehicle_detections


def vehicle_vote_for_box(vehicle_detections: list[dict] | None, group_box: list[float]) -> str | None:
    if not vehicle_detections:
        return None
    best_vote = None
    best_coverage = 0.0
    for vehicle_detection in vehicle_detections:
        coverage = box_coverage(vehicle_detection["box"], group_box)
        if coverage > best_coverage:
            best_coverage = coverage
            best_vote = vehicle_detection["vehicle_type"]
    if best_coverage >= VEHICLE_VOTE_MIN_COVERAGE:
        return best_vote
    return None


def resolve_vehicle_type(lamp_count: int, vehicle_vote: str | None) -> str | None:
    if lamp_count >= MULTI_LAMP_MIN_COUNT:
        return VEHICLE_TYPE_MOBIL
    return vehicle_vote


def type_signals_agree(lamp_count: int, vehicle_vote: str | None) -> bool:
    if lamp_count < MULTI_LAMP_MIN_COUNT or vehicle_vote is None:
        return True
    return vehicle_vote == VEHICLE_TYPE_MOBIL


def build_group_detections(merged_lamps: list[dict], vehicle_detections: list[dict] | None = None) -> list[dict]:
    group_detections = []
    for vehicle_group in group_lamps_into_vehicles(merged_lamps):
        group_box = group_bounding_box(vehicle_group)
        lamp_count = len(vehicle_group)
        vehicle_vote = vehicle_vote_for_box(vehicle_detections, group_box)
        group_detections.append(
            {
                "box": group_box,
                "score": max(lamp["score"] for lamp in vehicle_group),
                "lamp_count": lamp_count,
                "vehicle_type": resolve_vehicle_type(lamp_count, vehicle_vote),
                "vehicle_vote": vehicle_vote,
                "signals_agree": type_signals_agree(lamp_count, vehicle_vote),
            }
        )
    return group_detections


def compute_config_signature(roi_config: dict) -> str:
    signature_payload = {
        "detection_model_id": DETECTION_MODEL_ID,
        "detection_prompts": list(DETECTION_PROMPTS),
        "detection_threshold": DETECTION_THRESHOLD,
        "text_threshold": TEXT_THRESHOLD,
        "max_resize_side": MAX_RESIZE_SIDE,
        "min_box_pixels": MIN_BOX_PIXELS,
        "max_box_area_ratio": MAX_BOX_AREA_RATIO,
        "duplicate_window_frames": DUPLICATE_WINDOW_FRAMES,
        "duplicate_hash_distance": DUPLICATE_HASH_DISTANCE,
        "lamp_merge_iou_threshold": LAMP_MERGE_IOU_THRESHOLD,
        "group_max_center_distance_pixels": GROUP_MAX_CENTER_DISTANCE_PIXELS,
        "group_max_vertical_offset_pixels": GROUP_MAX_VERTICAL_OFFSET_PIXELS,
        "group_size_ratio_limit": GROUP_SIZE_RATIO_LIMIT,
        "multi_lamp_min_count": MULTI_LAMP_MIN_COUNT,
        "chunk_frame_count": CHUNK_FRAME_COUNT,
        "vehicle_score_threshold": VEHICLE_SCORE_THRESHOLD,
        "vehicle_vote_min_coverage": VEHICLE_VOTE_MIN_COVERAGE,
        "vehicle_class_remap": VEHICLE_CLASS_REMAP,
        "roi_config": roi_config,
    }
    signature_text = json.dumps(signature_payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(signature_text.encode("utf-8")).hexdigest()[:16]


def ensure_chunk_signature(chunk_directory: Path, config_signature: str) -> None:
    signature_path = chunk_directory / "_config_signature.txt"
    if signature_path.exists() and signature_path.read_text(encoding="utf-8").strip() != config_signature:
        raise SystemExit(
            "Signature konfigurasi berubah — hapus isi output/nb02/chunks lalu jalankan ulang NB02."
        )
    signature_path.write_text(config_signature, encoding="utf-8")


def write_chunk_jsonl(chunk_path: Path, chunk_rows: list[dict]) -> None:
    temporary_path = chunk_path.with_name(chunk_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as chunk_file:
        for chunk_row in chunk_rows:
            chunk_file.write(json.dumps(chunk_row, ensure_ascii=False) + "\\n")
    os.replace(temporary_path, chunk_path)


def count_chunk_rows(chunk_directory: Path) -> int:
    row_total = 0
    for chunk_path in sorted(chunk_directory.glob("chunk_*.jsonl")):
        with chunk_path.open(encoding="utf-8") as chunk_file:
            row_total += sum(1 for chunk_line in chunk_file if chunk_line.strip())
    return row_total


def process_detection_phase(frame_records: list[dict], chunk_directory: Path, crop_directory: Path) -> dict:
    from transformers.models.grounding_dino import GroundingDinoForObjectDetection, GroundingDinoProcessor

    config_signature = compute_config_signature(roi_config)
    ensure_chunk_signature(chunk_directory, config_signature)
    processor = GroundingDinoProcessor.from_pretrained(DETECTION_MODEL_ID)
    model = GroundingDinoForObjectDetection.from_pretrained(DETECTION_MODEL_ID).to(DETECTION_DEVICE)
    model.eval()
    vehicle_verifier, vehicle_categories = load_vehicle_verifier()
    total_chunks = (len(frame_records) + CHUNK_FRAME_COUNT - 1) // CHUNK_FRAME_COUNT
    processed_frame_count = 0
    failed_frame_stems = []
    verifier_error_count = 0
    try:
        for chunk_index in range(total_chunks):
            chunk_path = chunk_directory / f"chunk_{chunk_index:04d}.jsonl"
            if chunk_path.exists():
                with chunk_path.open(encoding="utf-8") as existing_chunk_file:
                    processed_frame_count += sum(1 for chunk_line in existing_chunk_file if chunk_line.strip())
                display(Markdown(f"chunk {chunk_index + 1}/{total_chunks} sudah ada — dilewati (resume)."))
                continue
            chunk_records = frame_records[chunk_index * CHUNK_FRAME_COUNT : (chunk_index + 1) * CHUNK_FRAME_COUNT]
            chunk_rows = []
            for frame_record in chunk_records:
                crop_stem = crop_file_stem(frame_record)
                crop_path = crop_directory / f"{crop_stem}.png"
                if not crop_path.exists():
                    failed_frame_stems.append(f"{crop_stem}: crop tidak ada")
                    continue
                fitted_image = Image.open(crop_path).convert("RGB")
                try:
                    vehicle_detections = detect_vehicles(vehicle_verifier, vehicle_categories, fitted_image)
                except Exception as verifier_error:
                    verifier_error_count += 1
                    display(Markdown(f"Verifier kendaraan gagal: `{crop_stem}` — {verifier_error}"))
                    vehicle_detections = None
                try:
                    lamp_candidates = detect_lamp_candidates(model, processor, fitted_image, DETECTION_DEVICE)
                    group_detections = build_group_detections(
                        merge_overlapping_lamps(lamp_candidates), vehicle_detections
                    )
                except Exception as frame_error:
                    display(Markdown(f"Frame gagal dideteksi: `{crop_stem}` — {frame_error}"))
                    group_detections = []
                    failed_frame_stems.append(f"{crop_stem}: {frame_error}")
                chunk_rows.append(
                    {
                        "chunk_index": chunk_index,
                        "crop_stem": crop_stem,
                        "frame_path": frame_record["frame_path"],
                        "video_stem": frame_record["video_stem"],
                        "camera_id": frame_record["camera_id"],
                        "session_ts": frame_record["session_ts"],
                        "image_width": fitted_image.size[0],
                        "image_height": fitted_image.size[1],
                        "detections": group_detections,
                    }
                )
                processed_frame_count += 1
            write_chunk_jsonl(chunk_path, chunk_rows)
            detection_count = sum(len(chunk_row["detections"]) for chunk_row in chunk_rows)
            display(Markdown(
                f"chunk {chunk_index + 1}/{total_chunks} selesai — {len(chunk_rows)} frame, "
                f"{detection_count} deteksi (total {processed_frame_count}/{len(frame_records)} frame)."
            ))
    finally:
        del model
        if vehicle_verifier is not None:
            del vehicle_verifier
        torch.cuda.empty_cache()
    expected_row_total = sum(
        1 for frame_record in frame_records
        if (crop_directory / f"{crop_file_stem(frame_record)}.png").exists()
    )
    counted_row_total = count_chunk_rows(chunk_directory)
    if counted_row_total != expected_row_total:
        raise RuntimeError(
            f"Chunk NB02 tidak lengkap: {counted_row_total} baris vs {expected_row_total} frame ber-crop. "
            "Hapus chunk yang rusak lalu jalankan ulang NB02; jangan lanjut ke NB02b dengan chunk parsial."
        )
    return {
        "config_signature": config_signature,
        "processed_frames": processed_frame_count,
        "failed_frame_stems": failed_frame_stems,
        "verifier_error_count": verifier_error_count,
    }


detection_summary = process_detection_phase(unique_frame_records, CHUNK_DIR, CROP_DIR)
display(Markdown(
    f"Deteksi selesai: **{detection_summary['processed_frames']}** frame · "
    f"gagal: **{len(detection_summary['failed_frame_stems'])}** · signature `{detection_summary['config_signature']}`"
))'''

VISUAL_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from IPython.display import Markdown, display

BOX_COLOR_BY_TYPE = {"Mobil": "lime", "Motor": "cyan", None: "red"}


def load_all_chunk_rows(chunk_directory: Path) -> list[dict]:
    chunk_rows = []
    for chunk_path in sorted(chunk_directory.glob("chunk_*.jsonl")):
        with chunk_path.open(encoding="utf-8") as chunk_file:
            chunk_rows.extend(json.loads(chunk_line) for chunk_line in chunk_file if chunk_line.strip())
    return chunk_rows


def annotate_detections(axis, detections: list[dict]) -> None:
    for detection in detections:
        box = detection["box"]
        box_color = BOX_COLOR_BY_TYPE.get(detection.get("vehicle_type"), "red")
        axis.add_patch(plt.Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1], fill=False, edgecolor=box_color, linewidth=2
        ))
        label_text = f"{detection.get('vehicle_type') or 'UNSURE'} lamp={detection['lamp_count']} {detection['score']:.2f}"
        if not detection.get("signals_agree", True):
            label_text += " [konflik]"
        axis.text(
            box[0] + 3, box[1] + 12, label_text, color=box_color, fontsize=7,
            bbox={"fc": "black", "alpha": 0.65},
        )


all_chunk_rows = load_all_chunk_rows(CHUNK_DIR)
rows_by_camera: dict[str, list[dict]] = {}
for chunk_row in all_chunk_rows:
    rows_by_camera.setdefault(chunk_row["camera_id"], []).append(chunk_row)
display(Markdown(
    f"Total frame terproses: **{len(all_chunk_rows)}** · kamera: **{len(rows_by_camera)}**\\n\\n"
    "hijau = Mobil · cyan = Motor · merah = jenis belum pasti (anotator memutuskan)."
))

for camera_id, camera_rows in sorted(rows_by_camera.items()):
    selected_rows = camera_rows[:PREVIEW_FRAMES_PER_CAMERA]
    figure, axes = plt.subplots(1, len(selected_rows), figsize=(7 * len(selected_rows), 7))
    if len(selected_rows) == 1:
        axes = [axes]
    for axis, chunk_row in zip(axes, selected_rows):
        crop_image = Image.open(CROP_DIR / f"{chunk_row['crop_stem']}.png").convert("RGB")
        axis.imshow(np.array(crop_image))
        annotate_detections(axis, chunk_row["detections"])
        axis.set_title(f"{chunk_row['image_width']}x{chunk_row['image_height']} | {len(chunk_row['detections'])} deteksi")
        axis.axis("off")
    figure.suptitle(f"{camera_id} — pseudo-label lokasi lampu")
    figure.tight_layout()
    figure.savefig(PREVIEW_DIR / f"nb02_visual_{camera_id}.png", dpi=110)
    plt.close(figure)
display(Markdown("Preview tersimpan di `output/nb02/previews/`."))'''

SUMMARY_SOURCE = '''from IPython.display import Markdown, display

summary_detections_by_camera: dict[str, int] = {}
summary_detections_by_type: dict[str, int] = {}
summary_detections_by_split: dict[str, int] = {}
summary_frames_with_detection = 0
summary_frames_without_detection = 0
summary_conflict_count = 0
summary_type_unknown_count = 0
summary_detection_count = 0

for chunk_row in all_chunk_rows:
    if chunk_row["detections"]:
        summary_frames_with_detection += 1
    else:
        summary_frames_without_detection += 1
    for detection in chunk_row["detections"]:
        summary_detection_count += 1
        vehicle_type = detection.get("vehicle_type") or "TidakPasti"
        summary_detections_by_type[vehicle_type] = summary_detections_by_type.get(vehicle_type, 0) + 1
        if vehicle_type == "TidakPasti":
            summary_type_unknown_count += 1
        summary_detections_by_camera[chunk_row["camera_id"]] = summary_detections_by_camera.get(chunk_row["camera_id"], 0) + 1
        split_name = SPLIT_NAME_BY_CAMERA.get(chunk_row["camera_id"], "tidak terdaftar")
        summary_detections_by_split[split_name] = summary_detections_by_split.get(split_name, 0) + 1
        if not detection.get("signals_agree", True):
            summary_conflict_count += 1

nb02_summary = {
    "frame_total": len(frame_records),
    "frame_unique": len(unique_frame_records),
    "frame_skipped_duplicate": skipped_duplicate_count,
    "crop_images": crop_summary["crop_images"],
    "failed_crop_count": len(crop_summary["failed_crop_stems"]),
    "detection_model_id": DETECTION_MODEL_ID,
    "detection_prompts": list(DETECTION_PROMPTS),
    "box_threshold": DETECTION_THRESHOLD,
    "text_threshold": TEXT_THRESHOLD,
    "detection_device": DETECTION_DEVICE,
    "vehicle_verifier_device": VEHICLE_VERIFIER_DEVICE,
    "vehicle_score_threshold": VEHICLE_SCORE_THRESHOLD,
    "max_resize_side": MAX_RESIZE_SIDE,
    "config_signature": detection_summary["config_signature"],
    "processed_frames": detection_summary["processed_frames"],
    "failed_detection_count": len(detection_summary["failed_frame_stems"]),
    "verifier_error_count": detection_summary["verifier_error_count"],
    "frames_with_detection": summary_frames_with_detection,
    "frames_without_detection": summary_frames_without_detection,
    "detection_count": summary_detection_count,
    "type_unknown_count": summary_type_unknown_count,
    "signal_conflict_count": summary_conflict_count,
    "detections_by_camera": summary_detections_by_camera,
    "detections_by_type": summary_detections_by_type,
    "detections_by_split": summary_detections_by_split,
}
with (NB02_ROOT / "nb02_summary.json").open("w", encoding="utf-8") as summary_file:
    json.dump(nb02_summary, summary_file, ensure_ascii=False, indent=2)

summary_rows = [
    ("Frame total (sampling)", nb02_summary["frame_total"]),
    ("Frame unik", nb02_summary["frame_unique"]),
    ("Frame duplikat di-skip", nb02_summary["frame_skipped_duplicate"]),
    ("Gambar crop", nb02_summary["crop_images"]),
    ("Frame dengan >=1 deteksi", nb02_summary["frames_with_detection"]),
    ("Frame tanpa deteksi", nb02_summary["frames_without_detection"]),
    ("Total deteksi (kendaraan)", nb02_summary["detection_count"]),
    ("Jenis belum pasti", nb02_summary["type_unknown_count"]),
    ("Sinyal bertentangan", nb02_summary["signal_conflict_count"]),
    ("Verifier gagal (crop)", nb02_summary["verifier_error_count"]),
]
summary_table_lines = ["| Metrik | Nilai |", "|---|---|"]
for metric_name, metric_value in summary_rows:
    summary_table_lines.append(f"| {metric_name} | {metric_value} |")
display(Markdown("\\n".join(summary_table_lines)))

type_table_lines = ["| Jenis (vote) | Deteksi |", "|---|---|"]
for vehicle_type, type_count in sorted(nb02_summary["detections_by_type"].items()):
    type_table_lines.append(f"| {vehicle_type} | {type_count} |")
display(Markdown("### Distribusi jenis kendaraan\\n" + "\\n".join(type_table_lines)))

camera_table_lines = ["| Kamera | Split | Deteksi |", "|---|---|---|"]
for camera_id, camera_count in sorted(nb02_summary["detections_by_camera"].items()):
    camera_table_lines.append(
        f"| {camera_id} | {SPLIT_NAME_BY_CAMERA.get(camera_id, 'tidak terdaftar')} | {camera_count} |"
    )
display(Markdown("### Distribusi per kamera\\n" + "\\n".join(camera_table_lines)))

display(Markdown(
    "**Catatan penting:** kolom jenis di atas hanya **sinyal bantu**, bukan label final. "
    "Kondisi lampu (silau / berwarna / silau+berwarna) **tidak** diprediksi NB02 dan wajib diisi manusia di NB02b. "
    f"Dataset latih dibangun NB04 dari gold store, bukan dari pseudo-label ini."
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(FRAME_PLAN_SOURCE),
        nbformat.v4.new_code_cell(CROP_SOURCE),
        nbformat.v4.new_code_cell(DETECTION_SOURCE),
        nbformat.v4.new_code_cell(VISUAL_SOURCE),
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
