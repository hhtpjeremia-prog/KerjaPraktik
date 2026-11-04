#!/usr/bin/env python3
"""Build NB02b_annotation.ipynb for Riset-HampirMatang (manual annotation of lamp condition)."""

from pathlib import Path

import nbformat

from notebook_gate import verify_code_cells

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB02b_annotation.ipynb"

EXPECTED_CODE_CELL_COUNT = 6
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB02b — Anotasi Manual Kondisi Lampu

Tahap ini mengisi satu-satunya hal yang **tidak boleh** ditebak model: **kondisi lampu**.
Grounding DINO di NB02 hanya memberi *lokasi* lampu; di sini manusia menetapkan kelasnya.

**Posisi:** `NB01 (akuisisi) → NB02 (crop + pseudo-lokasi) → NB02b (anotasi) → NB03 → NB04 → NB05`

**Sumber gambar:** `output/nb02/crops/*.png` — frame yang sudah ROI-crop + fit 640 px.
Semua frame unik tersedia di sini, termasuk yang dilewatkan DINO, sehingga frame tanpa
deteksi tetap bisa dianotasi dari nol.

## Sembilan kelas (hotkey `1`–`9`)

| Hotkey | Kelas | Arti |
|---|---|---|
| 1 | `MobilNormal` | mobil, lampu tidak silau dan tidak berwarna |
| 2 | `MobilSilau` | mobil, lampu menyilaukan, warna dasar normal |
| 3 | `MobilBerwarna` | mobil, lampu depan biru/cyan, tidak menyilaukan |
| 4 | `MobilSilauBerwarna` | mobil, lampu depan biru/cyan **dan** menyilaukan |
| 5 | `MotorNormal` | motor, lampu normal |
| 6 | `MotorSilau` | motor, lampu menyilaukan |
| 7 | `MotorBerwarna` | motor, lampu depan biru/cyan |
| 8 | `MotorSilauBerwarna` | motor, lampu depan biru/cyan **dan** menyilaukan |
| 9 | `PantulanCahaya` | pantulan/glare di aspal — **hard-negative, tidak pernah dihitung** |

## Definisi operasional (diputuskan 2026-10-28)

**Yang dinilai hanya lampu depan.** Lampu rem, sein, dan lampu belakang tidak dinilai
warnanya. Cahaya yang jatuh di aspal atau bodi adalah `PantulanCahaya`, bukan lampu.
ROI setiap kamera sudah dibatasi ke arah jalan, jadi frame hanya memuat tampak depan
kendaraan — tidak ada kasus "hanya tampak belakang" yang perlu diputuskan.

**`Silau` dinilai dari bentuk luapan cahaya, bukan dari seberapa terang lampunya.**

| Yang terlihat | Keputusan |
|---|---|
| halo/lingkaran putih keluar melewati batas fisik lampu sampai menutupi bodi, plat, atau aspal di sekitar lampu | `*Silau` |
| streak/flare memanjang dari inti lampu | `*Silau` |
| lampu terang tapi batas lampunya masih jelas, tanpa luapan | **bukan** silau |
| cahaya putih atau berwarna yang jatuh di aspal, bukan di lampu | `PantulanCahaya` |

**`Berwarna` hanya biru/cyan.** Piksel berwarna = `S >= 80` dan `V >= 80` dengan hue
OpenCV 95–135, dan minimal **25%** luas box harus piksel tersebut. Merah, oranye,
kuning, hijau, dan ungu **tidak** dihitung pada revisi ini. Ragu antara dua kelas →
`BelumDitentukan`, jangan menebak.

> Ambang fotometrik sengaja **tidak** dipakai untuk silau: pengukuran 2026-10-28 pada
> 1.125 box menunjukkan tidak ada fitur piksel yang memisahkan silau dari normal
> (MASTER §13.9a). Acuan visual berupa set contoh patch akan disimpan di
> `output/annotation/reference_patches/`.

**Kelas berwarna tetap dianotasi, tetapi tidak masuk latihan.** Keputusan pemilik riset
2026-10-28: data lampu biru tidak mencukupi (mining 1.548 crop hanya menghasilkan 2 box
`train` dan 0 box `val`), jadi label YOLO memakai 5 kelas saja. Saat export,
`*Berwarna` → `*Normal` dan `*SilauBerwarna` → `*Silau` padanannya. Anotasi 9 kelas tetap
disimpan di `annotations.json` agar riset warna bisa dilanjutkan tanpa anotasi ulang.

**Satu box = satu kendaraan.** Untuk mobil, box melingkupi sepasang lampu (union); untuk
motor, box melingkupi lampunya. Kelas `Normal` wajib ada supaya jumlah kendaraan total
bisa dihitung, bukan hanya kendaraan yang lampunya menyimpang.

## Status "belum ditentukan"

Setiap box pseudo masuk sebagai `BelumDitentukan` — **bukan** ditebak `MobilNormal`.
Alasannya: kendaraan yang lampunya tidak terdeteksi menyimpang belum tentu normal, dan
label `Normal` yang salah lebih merusak daripada label kosong. Box seperti ini ditandai
`?` dan **memblokir export** sampai Anda menetapkannya.

Sinyal bantu dari NB02 (`vehicle_type` = vote FasterRCNN / geometri pasangan) ditampilkan
sebagai petunjuk `? (Mobil)` — mempercepat keputusan tanpa mengisinya otomatis.

## Alur kerja

1. **Sampling** stratified per kamera (`SAMPLE_TOTAL_FRAMES`, seed tetap), manifest disimpan
   sekali dan aman untuk resume. **Kamera split `test` dikecualikan** agar gold-set bebas
   kebocoran.
2. **UI anotasi** (matplotlib widget + ipywidgets): drag = box baru, klik = pilih,
   `1`–`9` = set kelas, `x` = hapus, `u` = undo, `n`/`p` = navigasi, `r` = reviewed + next.
3. **Autosave** tiap mutasi ke `output/annotation/annotations.json` (tulis atomik).
4. **Export**: `annotations_coco.json` (untuk fine-tune DINO) + gold-set YOLO
   (`output/annotation/yolo/{images,labels}`) + `split_map.csv`.

> `data.yaml` **tidak** ditulis di sini. Dataset latih dimaterialisasi NB04 dari gold store
> dengan split per kamera; menulis `train: images` di sini hanya mengundang kebocoran.

Output: `output/annotation/{sample_manifest.json, annotations.json, annotations_coco.json,
annotation_summary.json, yolo/}`.

> Jalankan **per sel, berurutan**, di VS Code dengan kernel `.venv312`. Sel UI butuh `ipympl`.
"""

CONFIG_SOURCE = '''from pathlib import Path
import json
import os

from PIL import Image
from IPython.display import Markdown, display

from pipeline_common import CLASS_NAMES, PENDING_CLASS_NAME, SPLIT_NAME_BY_CAMERA, TRAINING_CLASS_NAMES, filter_store_frames, training_class_name

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT
NB02_ROOT = DATA_ROOT / "nb02"
CHUNK_DIR = NB02_ROOT / "chunks"
CROP_DIR = NB02_ROOT / "crops"
NB02_SUMMARY_PATH = NB02_ROOT / "nb02_summary.json"
ANNOTATION_ROOT = DATA_ROOT / "annotation"
ANNOTATION_YOLO_ROOT = ANNOTATION_ROOT / "yolo"
ANNOTATION_YOLO_IMAGE_DIR = ANNOTATION_YOLO_ROOT / "images"
ANNOTATION_YOLO_LABEL_DIR = ANNOTATION_YOLO_ROOT / "labels"
SAMPLE_MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"
ANNOTATION_STORE_PATH = ANNOTATION_ROOT / "annotations.json"
COCO_EXPORT_PATH = ANNOTATION_ROOT / "annotations_coco.json"
SPLIT_MAP_PATH = ANNOTATION_YOLO_ROOT / "split_map.csv"
SUMMARY_PATH = ANNOTATION_ROOT / "annotation_summary.json"

for output_dir in (ANNOTATION_ROOT, ANNOTATION_YOLO_IMAGE_DIR, ANNOTATION_YOLO_LABEL_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

STORE_CLASS_NAMES = CLASS_NAMES + [PENDING_CLASS_NAME]
REFLECTION_CLASS_NAME = "PantulanCahaya"
MOBIL_CLASS_NAMES = [class_name for class_name in CLASS_NAMES if class_name.startswith("Mobil")]
MOTOR_CLASS_NAMES = [class_name for class_name in CLASS_NAMES if class_name.startswith("Motor")]
VEHICLE_CLASS_NAMES = MOBIL_CLASS_NAMES + MOTOR_CLASS_NAMES

CLASS_COLORS = {
    "MobilNormal": "#39FF14",
    "MobilSilau": "#FFD400",
    "MobilBerwarna": "#00CFFF",
    "MobilSilauBerwarna": "#FF00FF",
    "MotorNormal": "#9ACD32",
    "MotorSilau": "#FF8C00",
    "MotorBerwarna": "#1E90FF",
    "MotorSilauBerwarna": "#FF4500",
    "PantulanCahaya": "#B0B0B0",
    "BelumDitentukan": "#FFFFFF",
}
CLASS_TOGGLE_LABELS = [f"{class_index + 1}:{class_name}" for class_index, class_name in enumerate(CLASS_NAMES)]
CLASS_TOGGLE_LABELS.append(f"? {PENDING_CLASS_NAME}")
CLASS_COLOR_BY_TOGGLE = {
    toggle_label: CLASS_COLORS[class_name]
    for toggle_label, class_name in zip(CLASS_TOGGLE_LABELS, STORE_CLASS_NAMES)
}
CLASS_NAME_BY_TOGGLE = dict(zip(CLASS_TOGGLE_LABELS, STORE_CLASS_NAMES))
CLASS_TOGGLE_BY_NAME = dict(zip(STORE_CLASS_NAMES, CLASS_TOGGLE_LABELS))

SAMPLE_TOTAL_FRAMES = 12 if SMOKE_MODE else 400
RANDOM_SEED = 20261020
MIN_SAMPLE_GAP_FRAMES = 30
SAMPLE_EXCLUDED_SPLITS = ["test"]

BOX_SOURCE_PSEUDO = "pseudo"
BOX_SOURCE_MANUAL = "manual"
BOX_SOURCE_EDITED = "edited"

MIN_BOX_SPAN_PIXELS = 4
FIGURE_SIZE = (9.5, 9.5)
STATUS_UPDATE_EVERY_FRAMES = 25
PREVIEW_FRAME_COUNT = 3

display(Markdown(
    f"NB02b siap. Root: `{RISET_ROOT.name}` · target sampling: **{SAMPLE_TOTAL_FRAMES} frame** · "
    f"split dikecualikan: **{', '.join(SAMPLE_EXCLUDED_SPLITS)}** · {len(CLASS_NAMES)} kelas + status belum ditentukan"
))'''

SAMPLING_SOURCE = '''from pathlib import Path
import json
import random
import shutil
from datetime import datetime

from IPython.display import Markdown, display


def load_all_chunk_rows(chunk_dir: Path) -> list[dict]:
    chunk_rows = []
    for chunk_path in sorted(chunk_dir.glob("chunk_*.jsonl")):
        with chunk_path.open(encoding="utf-8") as chunk_file:
            chunk_rows.extend(json.loads(chunk_line) for chunk_line in chunk_file if chunk_line.strip())
    return chunk_rows


def assert_detection_phase_complete(chunk_frame_total: int, summary_path: Path) -> None:
    if not summary_path.exists():
        raise RuntimeError(
            f"Fase deteksi NB02 belum selesai: {summary_path.name} belum ada. "
            f"Manifest anotasi tidak boleh dibentuk sebelum seluruh chunk NB02 selesai."
        )
    with summary_path.open(encoding="utf-8") as summary_file:
        nb02_summary = json.load(summary_file)
    expected_frame_total = nb02_summary["frame_unique"]
    if chunk_frame_total != expected_frame_total:
        raise RuntimeError(
            f"Chunk NB02 belum lengkap: {chunk_frame_total} baris vs {expected_frame_total} frame unik "
            f"di {summary_path.name}. Manifest tidak boleh dibentuk dari chunk parsial."
        )


def resolve_frame_stem(chunk_row: dict) -> str:
    return chunk_row.get("crop_stem") or chunk_row["video_stem"] + "_" + Path(chunk_row["frame_path"]).stem


def resolve_camera_split(camera_id: str, split_by_camera: dict[str, str]) -> str:
    return split_by_camera.get(camera_id, "unknown")


def split_candidates(candidates: list[dict], split_by_camera: dict[str, str], excluded_splits: list[str]) -> tuple[list[dict], list[dict]]:
    eligible_candidates = []
    excluded_candidates = []
    for candidate in candidates:
        split_name = resolve_camera_split(candidate["camera_id"], split_by_camera)
        annotated_candidate = {**candidate, "split": split_name}
        if split_name in excluded_splits:
            excluded_candidates.append(annotated_candidate)
        else:
            eligible_candidates.append(annotated_candidate)
    return eligible_candidates, excluded_candidates


def build_image_candidates(chunk_rows: list[dict], crop_dir: Path) -> list[dict]:
    candidates = []
    for chunk_row in chunk_rows:
        frame_stem = resolve_frame_stem(chunk_row)
        image_path = crop_dir / f"{frame_stem}.png"
        if image_path.exists():
            candidates.append({**chunk_row, "frame_stem": frame_stem, "image_path": str(image_path)})
    return candidates


def allocate_sample_quota(camera_counts: dict[str, int], total_frames: int) -> dict[str, int]:
    eligible_total = sum(camera_counts.values())
    if eligible_total == 0:
        return {camera_id: 0 for camera_id in camera_counts}
    target_total = min(total_frames, eligible_total)
    quotas = {
        camera_id: target_total * camera_count // eligible_total
        for camera_id, camera_count in camera_counts.items()
    }
    remainder = target_total - sum(quotas.values())
    fractional_order = sorted(
        camera_counts,
        key=lambda camera_id: (-(camera_counts[camera_id] % eligible_total), camera_id),
    )
    for camera_id in fractional_order[:remainder]:
        quotas[camera_id] += 1
    return quotas


def sample_frame_number(frame_record: dict) -> int:
    return int(Path(frame_record["frame_path"]).stem.split("_")[-1])


def stratified_sample_frame_records(
    candidates: list[dict],
    total_frames: int,
    random_seed: int,
    min_gap_frames: int,
    protected_frame_stems: set[str] | None = None,
) -> list[dict]:
    protected_frame_stems = protected_frame_stems or set()
    candidates_by_camera: dict[str, list[dict]] = {}
    for candidate in candidates:
        candidates_by_camera.setdefault(candidate["camera_id"], []).append(candidate)
    camera_counts = {camera_id: len(camera_rows) for camera_id, camera_rows in sorted(candidates_by_camera.items())}
    quotas = allocate_sample_quota(camera_counts, total_frames)
    random_generator = random.Random(random_seed)
    sampled_records = []
    for camera_id, camera_rows in sorted(candidates_by_camera.items()):
        ordered_rows = sorted(camera_rows, key=lambda candidate: candidate["frame_stem"])
        camera_quota = quotas[camera_id]
        protected_rows = [row for row in ordered_rows if row["frame_stem"] in protected_frame_stems]
        camera_quota = max(camera_quota, len(protected_rows))
        if camera_quota >= len(ordered_rows):
            selected_rows = ordered_rows
        else:
            shuffled_rows = ordered_rows.copy()
            random_generator.shuffle(shuffled_rows)
            selected_rows = protected_rows.copy()
            for candidate in shuffled_rows:
                if candidate["frame_stem"] in protected_frame_stems:
                    continue
                candidate_frame_number = sample_frame_number(candidate)
                if all(
                    abs(candidate_frame_number - sample_frame_number(selected)) >= min_gap_frames
                    for selected in selected_rows
                ):
                    selected_rows.append(candidate)
                if len(selected_rows) == camera_quota:
                    break
            if len(selected_rows) < camera_quota:
                raise ValueError(
                    f"quota sampling {camera_id} tidak tercapai dengan jarak minimum "
                    f"{min_gap_frames} frame: {len(selected_rows)}/{camera_quota}. "
                    "Kurangi SAMPLE_TOTAL_FRAMES atau jarak minimum."
                )
        sampled_records.extend(sorted(selected_rows, key=lambda selected: selected["frame_stem"]))
    return sampled_records


def write_json_atomic(target_path: Path, payload: dict) -> None:
    temporary_path = target_path.with_name(target_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as target_file:
        json.dump(payload, target_file, ensure_ascii=False, indent=2)
    temporary_path.replace(target_path)


def initialize_annotation_entry(sample_record: dict) -> dict:
    pseudo_boxes = []
    for detection in sample_record.get("detections", []):
        box = detection["box"]
        pseudo_boxes.append({
            "box": [round(float(box[0]), 2), round(float(box[1]), 2), round(float(box[2]), 2), round(float(box[3]), 2)],
            "class_name": PENDING_CLASS_NAME,
            "source": BOX_SOURCE_PSEUDO,
            "deleted": False,
            "hint_vehicle_type": detection.get("vehicle_type"),
        })
    return {
        "boxes": pseudo_boxes,
        "reviewed": False,
        "image_width": sample_record["image_width"],
        "image_height": sample_record["image_height"],
    }


def build_or_load_sample_manifest(
    chunk_dir: Path,
    crop_dir: Path,
    manifest_path: Path,
    total_frames: int,
    random_seed: int,
    split_by_camera: dict[str, str],
    excluded_splits: list[str],
) -> tuple[list[dict], dict]:
    chunk_rows = load_all_chunk_rows(chunk_dir)
    assert_detection_phase_complete(len(chunk_rows), NB02_SUMMARY_PATH)
    all_candidates = build_image_candidates(chunk_rows, crop_dir)
    unmapped_cameras = sorted({candidate["camera_id"] for candidate in all_candidates} - set(split_by_camera))
    if unmapped_cameras:
        raise ValueError(f"kamera tanpa split per kamera di pipeline_common: {unmapped_cameras}")
    candidates, excluded_candidates = split_candidates(all_candidates, split_by_camera, excluded_splits)
    candidates_by_camera: dict[str, list[dict]] = {}
    for candidate in candidates:
        candidates_by_camera.setdefault(candidate["camera_id"], []).append(candidate)
    eligible_by_camera = {camera_id: len(camera_rows) for camera_id, camera_rows in sorted(candidates_by_camera.items())}
    excluded_by_camera: dict[str, int] = {}
    for candidate in excluded_candidates:
        excluded_by_camera[candidate["camera_id"]] = excluded_by_camera.get(candidate["camera_id"], 0) + 1
    protected_frame_stems: set[str] = set()
    if manifest_path.exists() and ANNOTATION_STORE_PATH.exists():
        with manifest_path.open(encoding="utf-8") as old_manifest_file:
            old_manifest_payload = json.load(old_manifest_file)
        with ANNOTATION_STORE_PATH.open(encoding="utf-8") as annotation_file:
            old_annotation_payload = json.load(annotation_file)
        protected_frame_stems = {
            record["frame_stem"]
            for record in old_manifest_payload.get("frames", [])
            if old_annotation_payload.get("frames", {}).get(record["frame_stem"], {}).get("reviewed")
        }
    manifest_reused = manifest_path.exists()
    if manifest_reused and protected_frame_stems:
        with manifest_path.open(encoding="utf-8") as old_manifest_file:
            existing_manifest_settings = json.load(old_manifest_file)
        if existing_manifest_settings.get("min_sample_gap_frames") != MIN_SAMPLE_GAP_FRAMES:
            manifest_reused = False
    if manifest_reused:
        with manifest_path.open(encoding="utf-8") as manifest_file:
            manifest_payload = json.load(manifest_file)
        sample_records = manifest_payload["frames"]
        installed_excluded_splits = manifest_payload.get("excluded_splits", [])
        if sorted(installed_excluded_splits) != sorted(excluded_splits):
            raise ValueError(
                f"Manifest lama memakai excluded_splits {installed_excluded_splits}, konfigurasi sekarang {excluded_splits}. "
                f"Hapus {manifest_path} lalu jalankan ulang sel ini."
            )
        installed_min_gap = manifest_payload.get("min_sample_gap_frames")
        if installed_min_gap != MIN_SAMPLE_GAP_FRAMES and not protected_frame_stems:
            raise ValueError(
                f"Manifest lama memakai min_sample_gap_frames={installed_min_gap}, konfigurasi sekarang "
                f"{MIN_SAMPLE_GAP_FRAMES}. Hapus {manifest_path} lalu jalankan ulang sel ini."
            )
        installed_class_names = manifest_payload.get("class_names", [])
        if installed_class_names != CLASS_NAMES:
            raise ValueError(
                f"Manifest lama memakai kelas {installed_class_names}, konfigurasi sekarang {CLASS_NAMES}. "
                f"Hapus {manifest_path} lalu jalankan ulang sel ini."
            )
        installed_total_frames = manifest_payload.get("sample_total_frames")
        if installed_total_frames != total_frames:
            raise ValueError(
                f"Manifest lama memakai sample_total_frames {installed_total_frames}, konfigurasi sekarang {total_frames}. "
                f"Hapus {manifest_path} lalu jalankan ulang sel ini."
            )
        installed_random_seed = manifest_payload.get("random_seed")
        if installed_random_seed != random_seed:
            raise ValueError(
                f"Manifest lama memakai random_seed {installed_random_seed}, konfigurasi sekarang {random_seed}. "
                f"Hapus {manifest_path} lalu jalankan ulang sel ini."
            )
    else:
        sample_records = stratified_sample_frame_records(
            candidates, total_frames, random_seed, MIN_SAMPLE_GAP_FRAMES, protected_frame_stems
        )
        manifest_payload = {
            "sample_total_frames": total_frames,
            "random_seed": random_seed,
            "min_sample_gap_frames": MIN_SAMPLE_GAP_FRAMES,
            "excluded_splits": excluded_splits,
            "class_names": CLASS_NAMES,
            "pending_class_name": PENDING_CLASS_NAME,
            "chunk_frame_total": len(chunk_rows),
            "candidate_frame_total": len(candidates),
            "excluded_frame_total": len(excluded_candidates),
            "eligible_by_camera": eligible_by_camera,
            "excluded_by_camera": excluded_by_camera,
            "frames": sample_records,
        }
        write_json_atomic(manifest_path, manifest_payload)
    sampled_by_camera: dict[str, int] = {}
    for sample_record in sample_records:
        sampled_by_camera[sample_record["camera_id"]] = sampled_by_camera.get(sample_record["camera_id"], 0) + 1
    return sample_records, {
        "manifest_path": manifest_path,
        "manifest_reused": manifest_reused,
        "chunk_frame_total": len(chunk_rows),
        "candidate_frame_total": len(candidates),
        "eligible_by_camera": eligible_by_camera,
        "excluded_by_camera": excluded_by_camera,
        "excluded_total": len(excluded_candidates),
        "excluded_splits": excluded_splits,
        "sampled_by_camera": sampled_by_camera,
        "sampled_total": len(sample_records),
        "random_seed": random_seed,
    }


def backup_store_before_prune(store_path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = store_path.with_name(f"annotations_backup_before_prune_{timestamp}.json")
    shutil.copyfile(store_path, backup_path)
    return backup_path


def load_annotation_store(store_path: Path, sample_records: list[dict]) -> dict:
    if store_path.exists():
        with store_path.open(encoding="utf-8") as store_file:
            store_payload = json.load(store_file)
    else:
        store_payload = {"frames": {}}
    active_stems = {record["frame_stem"] for record in sample_records}
    kept_frames = filter_store_frames(store_payload["frames"], active_stems)
    dropped_frame_count = len(store_payload["frames"]) - len(kept_frames)
    if dropped_frame_count and store_path.exists():
        backup_path = backup_store_before_prune(store_path)
        display(Markdown(
            f"**{dropped_frame_count} entry store di luar manifest dibuang** — backup: `{backup_path.name}`"
        ))
    store_payload["frames"] = kept_frames
    store_payload["split_by_camera"] = dict(SPLIT_NAME_BY_CAMERA)
    for sample_record in sample_records:
        kept_frames.setdefault(sample_record["frame_stem"], initialize_annotation_entry(sample_record))
    return store_payload


sample_records, sampling_summary = build_or_load_sample_manifest(
    CHUNK_DIR, CROP_DIR, SAMPLE_MANIFEST_PATH, SAMPLE_TOTAL_FRAMES, RANDOM_SEED,
    SPLIT_NAME_BY_CAMERA, SAMPLE_EXCLUDED_SPLITS,
)
annotation_store = load_annotation_store(ANNOTATION_STORE_PATH, sample_records)

eligible_table_lines = ["| Kamera | Split | Frame eligible | Disampel |", "|---|---|---|---|"]
for camera_id, eligible_count in sampling_summary["eligible_by_camera"].items():
    sampled_count = sampling_summary["sampled_by_camera"].get(camera_id, 0)
    split_name = resolve_camera_split(camera_id, SPLIT_NAME_BY_CAMERA)
    eligible_table_lines.append(f"| {camera_id} | {split_name} | {eligible_count} | {sampled_count} |")
excluded_table_lines = ["| Kamera | Split | Frame dikecualikan |", "|---|---|---|"]
for camera_id, excluded_count in sampling_summary["excluded_by_camera"].items():
    split_name = resolve_camera_split(camera_id, SPLIT_NAME_BY_CAMERA)
    excluded_table_lines.append(f"| {camera_id} | {split_name} | {excluded_count} |")
display(Markdown(
    f"## Sampling frame anotasi\\nManifest: `{sampling_summary['manifest_path']}` "
    f"({'reuse' if sampling_summary['manifest_reused'] else 'baru dibuat'}) · "
    f"kandidat ber-gambar {sampling_summary['candidate_frame_total']} dari {sampling_summary['chunk_frame_total']} frame chunk · "
    f"disampel **{sampling_summary['sampled_total']}** (seed {sampling_summary['random_seed']}) · "
    f"split dikecualikan: **{', '.join(sampling_summary['excluded_splits'])}** ({sampling_summary['excluded_total']} frame)"
))
display(Markdown("### Frame yang masuk anotasi (bebas kebocoran)\\n" + "\\n".join(eligible_table_lines)))
if excluded_table_lines[2:]:
    display(Markdown("### Frame dikecualikan (kamera split test — tidak boleh ikut anotasi)\\n" + "\\n".join(excluded_table_lines)))
display(Markdown("Lanjut ke sel berikutnya untuk memuat store anotasi."))'''

STORE_SOURCE = '''from IPython.display import Markdown, display

STORE_FRAMES_KEY = "frames"

annotation_store = load_annotation_store(ANNOTATION_STORE_PATH, sample_records)


def save_annotation_store() -> None:
    write_json_atomic(ANNOTATION_STORE_PATH, annotation_store)


def frame_stems_in_order() -> list[str]:
    return [sample_record["frame_stem"] for sample_record in sample_records]


def active_boxes(frame_stem: str) -> list[dict]:
    entry = annotation_store[STORE_FRAMES_KEY][frame_stem]
    return [box_entry for box_entry in entry["boxes"] if not box_entry["deleted"]]


def reviewed_count() -> int:
    return sum(
        1 for frame_stem in frame_stems_in_order()
        if annotation_store[STORE_FRAMES_KEY].get(frame_stem, {}).get("reviewed")
    )


def box_source_counts() -> dict[str, int]:
    source_counts = {BOX_SOURCE_PSEUDO: 0, BOX_SOURCE_MANUAL: 0, BOX_SOURCE_EDITED: 0}
    for entry in annotation_store[STORE_FRAMES_KEY].values():
        for box_entry in entry["boxes"]:
            if box_entry["deleted"]:
                continue
            source_counts[box_entry["source"]] = source_counts.get(box_entry["source"], 0) + 1
    return source_counts


def class_box_counts() -> dict[str, int]:
    class_counts = {class_name: 0 for class_name in STORE_CLASS_NAMES}
    for entry in annotation_store[STORE_FRAMES_KEY].values():
        for box_entry in entry["boxes"]:
            if box_entry["deleted"]:
                continue
            class_counts[box_entry["class_name"]] = class_counts.get(box_entry["class_name"], 0) + 1
    return class_counts


def pending_boxes_in_reviewed_frames() -> list[str]:
    pending_stems = []
    for frame_stem, entry in annotation_store[STORE_FRAMES_KEY].items():
        if not entry["reviewed"]:
            continue
        if any(box_entry["class_name"] == PENDING_CLASS_NAME for box_entry in entry["boxes"] if not box_entry["deleted"]):
            pending_stems.append(frame_stem)
    return sorted(pending_stems)


def pending_box_frames() -> list[str]:
    return sorted(
        frame_stem
        for frame_stem, entry in annotation_store[STORE_FRAMES_KEY].items()
        if any(
            not box_entry["deleted"] and box_entry["class_name"] == PENDING_CLASS_NAME
            for box_entry in entry["boxes"]
        )
    )


def validate_annotation_store() -> list[str]:
    frame_stems = set(frame_stems_in_order())
    validation_errors = []
    unknown_frames = sorted(set(annotation_store[STORE_FRAMES_KEY]) - frame_stems)
    if unknown_frames:
        validation_errors.append(f"{len(unknown_frames)} frame di store tidak ada di manifest (contoh: {unknown_frames[0]})")
    for frame_stem, entry in annotation_store[STORE_FRAMES_KEY].items():
        for box_index, box_entry in enumerate(entry["boxes"]):
            if box_entry["class_name"] not in STORE_CLASS_NAMES:
                validation_errors.append(f"{frame_stem} box {box_index}: kelas tidak dikenal {box_entry['class_name']}")
            box = box_entry["box"]
            if box[2] - box[0] < MIN_BOX_SPAN_PIXELS or box[3] - box[1] < MIN_BOX_SPAN_PIXELS:
                validation_errors.append(f"{frame_stem} box {box_index}: span < {MIN_BOX_SPAN_PIXELS}px")
    return validation_errors


if ANNOTATION_STORE_PATH.exists():
    save_annotation_store()

store_errors = validate_annotation_store()
class_counts = class_box_counts()
source_counts = box_source_counts()
pending_stems = pending_boxes_in_reviewed_frames()
class_table_lines = ["| Kelas | Box |", "|---|---|"]
for class_name in STORE_CLASS_NAMES:
    class_table_lines.append(f"| `{class_name}` | {class_counts[class_name]} |")
display(Markdown(
    f"Store: `{ANNOTATION_STORE_PATH}`\\n\\n"
    f"| Metrik | Nilai |\\n|---|---|\\n"
    f"| Frame di manifest | {len(frame_stems_in_order())} |\\n"
    f"| Frame di store | {len(annotation_store[STORE_FRAMES_KEY])} |\\n"
    f"| Frame reviewed | {reviewed_count()} |\\n"
    f"| Box aktif (pseudo/manual/edited) | {source_counts[BOX_SOURCE_PSEUDO]} / {source_counts[BOX_SOURCE_MANUAL]} / {source_counts[BOX_SOURCE_EDITED]} |"
))
display(Markdown("### Distribusi box per kelas\\n" + "\\n".join(class_table_lines)))
if pending_stems:
    display(Markdown(
        f"**{len(pending_stems)} frame reviewed masih punya box `{PENDING_CLASS_NAME}`** — "
        f"export akan diblokir sampai semuanya diberi kelas. Contoh: `{pending_stems[0]}`."
    ))
if store_errors:
    display(Markdown("**Peringatan validasi:**\\n\\n" + "\\n".join(f"- {error}" for error in store_errors)))
else:
    display(Markdown("Validasi store: **OK** (kelas valid, span box ≥ batas minimum, tidak ada frame asing)."))'''

PREVIEW_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import numpy as np
from IPython.display import Markdown, display


def active_boxes_for_entry(entry: dict) -> list[dict]:
    return [box_entry for box_entry in entry["boxes"] if not box_entry["deleted"]]


def box_caption(box_entry: dict) -> str:
    if box_entry["class_name"] != PENDING_CLASS_NAME:
        return box_entry["class_name"]
    hint_vehicle_type = box_entry.get("hint_vehicle_type")
    return f"? ({hint_vehicle_type})" if hint_vehicle_type else "?"


def preview_entry(image_path: Path, entry: dict) -> np.ndarray:
    preview_image = Image.open(image_path).convert("RGB")
    drawer = ImageDraw.Draw(preview_image)
    box_offset = max(2, int(entry["image_width"] * 0.005))
    for box_entry in active_boxes_for_entry(entry):
        box = box_entry["box"]
        outline_color = CLASS_COLORS[box_entry["class_name"]]
        drawer.rectangle([box[0], box[1], box[2], box[3]], outline=outline_color, width=2)
        drawer.text((box[0] + box_offset, max(0, box[1] - 2 * box_offset)), box_caption(box_entry), fill=outline_color)
    return np.array(preview_image)


preview_stems = [sample_record["frame_stem"] for sample_record in sample_records[:PREVIEW_FRAME_COUNT]]
figure, axes = plt.subplots(1, len(preview_stems), figsize=(6 * len(preview_stems), 6))
if len(preview_stems) == 1:
    axes = [axes]
for axis, frame_stem in zip(axes, preview_stems):
    sample_record = next(record for record in sample_records if record["frame_stem"] == frame_stem)
    axis.imshow(preview_entry(Path(sample_record["image_path"]), annotation_store["frames"][frame_stem]))
    axis.set_title(f"{sample_record['camera_id']}\\n{len(active_boxes(frame_stem))} box")
    axis.axis("off")
figure.suptitle("Cek awal: box pseudo (putih putus-putus = belum ditentukan)")
figure.tight_layout()
figure.savefig(ANNOTATION_ROOT / "annotation_preview_before.png", dpi=110)
display(figure)
plt.close(figure)
display(Markdown(
    f"Preview `{PREVIEW_FRAME_COUNT}` frame pertama disimpan ke `{ANNOTATION_ROOT / 'annotation_preview_before.png'}`."
))'''

UI_SOURCE = '''%matplotlib widget
import matplotlib.pyplot as plt
from matplotlib import colors as matplotlib_colors
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from ipywidgets import Button, HBox, ToggleButtons, VBox, HTML
from IPython.display import Markdown, display

BOX_HANDLE_PIXELS = 3.0
STATUS_PADDING_PIXELS = 3.0
UNDO_LIMIT = 40

frame_stem_order = frame_stems_in_order()
session_state = {"index": 0, "selected": None, "undo": [], "status_counter": 0, "show_box_labels": True}

label_toggle_button = None
class_toggle = None

figure, axis = plt.subplots(figsize=FIGURE_SIZE)
plt.subplots_adjust(bottom=0.30)


def current_frame_stem() -> str:
    return frame_stem_order[session_state["index"]]


def current_sample_record() -> dict:
    frame_stem = current_frame_stem()
    return next(record for record in sample_records if record["frame_stem"] == frame_stem)


def image_extent() -> tuple[float, float, float, float]:
    sample_record = current_sample_record()
    return (-0.5, sample_record["image_width"] - 0.5, sample_record["image_height"] - 0.5, -0.5)


def push_undo() -> None:
    frame_stem = current_frame_stem()
    snapshot = [dict(box_entry) for box_entry in annotation_store["frames"][frame_stem]["boxes"]]
    session_state["undo"].append((frame_stem, snapshot))
    session_state["undo"] = session_state["undo"][-UNDO_LIMIT:]


def normalise_box(raw_box: tuple[float, float, float, float]) -> list[float] | None:
    x_low, x_high = sorted((raw_box[0], raw_box[2]))
    y_low, y_high = sorted((raw_box[1], raw_box[3]))
    sample_record = current_sample_record()
    x_low = min(max(x_low, 0.0), sample_record["image_width"])
    x_high = min(max(x_high, 0.0), sample_record["image_width"])
    y_low = min(max(y_low, 0.0), sample_record["image_height"])
    y_high = min(max(y_high, 0.0), sample_record["image_height"])
    if x_high - x_low < MIN_BOX_SPAN_PIXELS or y_high - y_low < MIN_BOX_SPAN_PIXELS:
        return None
    return [round(x_low, 2), round(y_low, 2), round(x_high, 2), round(y_high, 2)]


def drop_orphan_selection() -> None:
    entry = annotation_store["frames"][current_frame_stem()]
    available = [index for index, box_entry in enumerate(entry["boxes"]) if not box_entry["deleted"]]
    if session_state["selected"] is not None and session_state["selected"] not in available:
        session_state["selected"] = available[-1] if available else None


def status_html() -> str:
    frame_stem = current_frame_stem()
    entry = annotation_store["frames"][frame_stem]
    source_counts = box_source_counts()
    reviewed_flag = "reviewed" if entry["reviewed"] else "belum reviewed"
    pending_count = sum(
        1 for box_entry in entry["boxes"]
        if not box_entry["deleted"] and box_entry["class_name"] == PENDING_CLASS_NAME
    )
    return (
        f"<b>Frame {session_state['index'] + 1}/{len(frame_stem_order)}</b> · {frame_stem} "
        f"({entry['image_width']}x{entry['image_height']}) · {reviewed_flag}<br>"
        f"Review progress: <b>{reviewed_count()}</b>/{len(frame_stem_order)} frame · "
        f"{len(active_boxes(frame_stem))} box · belum ditentukan: <b>{pending_count}</b> · "
        f"terpilih: {session_state['selected']}<br>"
        f"Box aktif — pseudo: {source_counts[BOX_SOURCE_PSEUDO]} · manual: {source_counts[BOX_SOURCE_MANUAL]} · "
        f"edited: {source_counts[BOX_SOURCE_EDITED]}"
    )


def draw_frame() -> None:
    axis.clear()
    sample_record = current_sample_record()
    axis.imshow(plt.imread(sample_record["image_path"]))
    axis.set_xlim(image_extent()[0], image_extent()[1])
    axis.set_ylim(image_extent()[2], image_extent()[3])
    entry = annotation_store["frames"][current_frame_stem()]
    for box_entry in entry["boxes"]:
        if box_entry["deleted"]:
            continue
        box = box_entry["box"]
        selected = session_state["selected"] is not None and entry["boxes"].index(box_entry) == session_state["selected"]
        outline_color = CLASS_COLORS[box_entry["class_name"]]
        line_style = "--" if box_entry["source"] == BOX_SOURCE_PSEUDO else "-"
        axis.add_patch(Rectangle(
            (box[0], box[1]), box[2] - box[0], box[3] - box[1],
            fill=False, edgecolor=outline_color, linestyle=line_style,
            linewidth=3.0 if selected else 1.4,
        ))
        marker_color = "white" if selected else outline_color
        axis.plot(
            [box[0] + BOX_HANDLE_PIXELS, box[2] - BOX_HANDLE_PIXELS],
            [box[1] + BOX_HANDLE_PIXELS, box[3] - BOX_HANDLE_PIXELS],
            marker="+", markersize=7, color=marker_color, linewidth=0,
        )
        if session_state["show_box_labels"]:
            axis.text(
                box[0] + STATUS_PADDING_PIXELS, max(0.0, box[1] - STATUS_PADDING_PIXELS),
                box_caption(box_entry),
                color=outline_color, fontsize=9, bbox=dict(fc="black", alpha=0.6), va="bottom",
            )
            box_number = entry["boxes"].index(box_entry)
            axis.text(
                box[2] - STATUS_PADDING_PIXELS, box[3] - STATUS_PADDING_PIXELS, str(box_number),
                color=marker_color, fontsize=9, ha="right", va="bottom", bbox=dict(fc="black", alpha=0.6),
            )
    axis.set_title(f"{sample_record['camera_id']} · hotkey 1-9 = kelas box terpilih")
    figure.canvas.draw_idle()


def refresh_status() -> None:
    session_state["status_counter"] += 1
    if session_state["status_counter"] % STATUS_UPDATE_EVERY_FRAMES == 0 and status_widget is not None:
        status_widget.value = status_html()


def commit_store() -> None:
    save_annotation_store()
    refresh_status()


def on_rectangle_selected(first_click, second_click) -> None:
    normalised_box = normalise_box((first_click.xdata, first_click.ydata, second_click.xdata, second_click.ydata))
    if normalised_box is None:
        return
    push_undo()
    entry = annotation_store["frames"][current_frame_stem()]
    selected_class = CLASS_NAME_BY_TOGGLE[class_toggle.value]
    entry["boxes"].append({
        "box": normalised_box,
        "class_name": selected_class,
        "source": BOX_SOURCE_MANUAL,
        "deleted": False,
        "hint_vehicle_type": None,
    })
    session_state["selected"] = len(entry["boxes"]) - 1
    entry["reviewed"] = True
    commit_store()
    draw_frame()
    status_widget.value = status_html()


def on_box_click(event) -> None:
    if event.inaxes is not axis or event.xdata is None or event.ydata is None:
        return
    entry = annotation_store["frames"][current_frame_stem()]
    containing_boxes = []
    for box_index, box_entry in enumerate(entry["boxes"]):
        if box_entry["deleted"]:
            continue
        box = box_entry["box"]
        if box[0] <= event.xdata <= box[2] and box[1] <= event.ydata <= box[3]:
            containing_boxes.append(box_index)
    if containing_boxes:
        session_state["selected"] = containing_boxes[-1]
        show_selected_class_in_toggle()
        draw_frame()
    elif session_state["selected"] is not None:
        session_state["selected"] = None
        draw_frame()


def keep_within_store() -> None:
    session_state["index"] = min(max(session_state["index"], 0), len(frame_stem_order) - 1)


def go_to_index(new_index: int) -> None:
    session_state["index"] = new_index
    keep_within_store()
    session_state["selected"] = None
    drop_orphan_selection()
    entry = annotation_store["frames"][current_frame_stem()]
    available = [index for index, box_entry in enumerate(entry["boxes"]) if not box_entry["deleted"]]
    if available:
        session_state["selected"] = available[-1]
    show_selected_class_in_toggle()
    draw_frame()
    status_widget.value = status_html()


def go_next(unused_button=None) -> None:
    go_to_index(session_state["index"] + 1)


def go_previous(unused_button=None) -> None:
    go_to_index(session_state["index"] - 1)


def go_to_next_unlabeled(unused_button=None) -> None:
    pending_stems = pending_boxes_in_reviewed_frames()
    available_stems = [stem for stem in pending_stems if stem in frame_stem_order]
    if not available_stems:
        status_widget.value = "Tidak ada frame unlabeled yang tersedia di manifest aktif."
        return
    current_stem = current_frame_stem()
    later_stems = [stem for stem in available_stems if stem > current_stem]
    target_stem = later_stems[0] if later_stems else available_stems[0]
    go_to_index(frame_stem_order.index(target_stem))


def mark_reviewed(unused_button=None) -> None:
    annotation_store["frames"][current_frame_stem()]["reviewed"] = True
    commit_store()
    go_next()


def skip_frame(unused_button=None) -> None:
    go_next()


def toggle_box_labels(unused_button=None) -> None:
    session_state["show_box_labels"] = not session_state["show_box_labels"]
    if label_toggle_button is not None:
        label_toggle_button.description = label_toggle_description()
    draw_frame()
    status_widget.value = status_html()


def label_toggle_description() -> str:
    return "Label: ON (T)" if session_state["show_box_labels"] else "Label: OFF (T)"


def remove_selected(unused_button=None) -> None:
    selected_index = session_state["selected"]
    if selected_index is None:
        return
    entry = annotation_store["frames"][current_frame_stem()]
    if selected_index >= len(entry["boxes"]) or entry["boxes"][selected_index]["deleted"]:
        return
    push_undo()
    entry["boxes"][selected_index]["deleted"] = True
    entry["reviewed"] = True
    session_state["selected"] = None
    drop_orphan_selection()
    commit_store()
    draw_frame()
    status_widget.value = status_html()


def undo_last(unused_button=None) -> None:
    if not session_state["undo"]:
        return
    frame_stem, snapshot = session_state["undo"].pop()
    annotation_store["frames"][frame_stem]["boxes"] = [dict(box_entry) for box_entry in snapshot]
    session_state["index"] = frame_stem_order.index(frame_stem)
    session_state["selected"] = None
    drop_orphan_selection()
    commit_store()
    draw_frame()
    status_widget.value = status_html()


def apply_toggle_to_selection(unused_change=None) -> None:
    selected_index = session_state["selected"]
    if selected_index is None:
        return
    entry = annotation_store["frames"][current_frame_stem()]
    if selected_index >= len(entry["boxes"]) or entry["boxes"][selected_index]["deleted"]:
        return
    selected_class = CLASS_NAME_BY_TOGGLE[class_toggle.value]
    if entry["boxes"][selected_index]["class_name"] == selected_class:
        return
    push_undo()
    entry["boxes"][selected_index]["class_name"] = selected_class
    entry["boxes"][selected_index]["source"] = BOX_SOURCE_EDITED
    entry["reviewed"] = True
    commit_store()
    draw_frame()
    status_widget.value = status_html()


def show_selected_class_in_toggle() -> None:
    selected_index = session_state["selected"]
    if selected_index is None:
        return
    entry = annotation_store["frames"][current_frame_stem()]
    if selected_index >= len(entry["boxes"]) or entry["boxes"][selected_index]["deleted"]:
        return
    selected_label = CLASS_TOGGLE_BY_NAME.get(entry["boxes"][selected_index]["class_name"])
    if selected_label is None or class_toggle.value == selected_label:
        return
    class_toggle.value = selected_label


def on_key_press(event) -> None:
    key_actions = {
        "n": go_next, "right": go_next,
        "p": go_previous, "left": go_previous,
        "r": mark_reviewed, "s": skip_frame,
        "x": remove_selected, "delete": remove_selected,
        "u": undo_last,
        "t": toggle_box_labels, "f": go_to_next_unlabeled,
    }
    key_name = event.key.lower() if event.key else ""
    if key_name.isdigit() and 1 <= int(key_name) <= len(CLASS_NAMES):
        class_toggle.value = CLASS_TOGGLE_LABELS[int(key_name) - 1]
        return
    action = key_actions.get(key_name)
    if action is not None:
        action()


remove_selector = RectangleSelector(
    axis, on_rectangle_selected, useblit=False, button=[1],
    minspanx=MIN_BOX_SPAN_PIXELS, minspany=MIN_BOX_SPAN_PIXELS, interactive=False,
    props=dict(facecolor=matplotlib_colors.to_rgba("white", 0.12), edgecolor="white", linewidth=1.0, linestyle=":"),
)
for selector_attribute, fallback_value in {"_corner_order": [], "_active_handle": None}.items():
    if not hasattr(remove_selector, selector_attribute):
        setattr(remove_selector, selector_attribute, fallback_value)
figure.canvas.mpl_connect("button_press_event", on_box_click)
figure.canvas.mpl_connect("key_press_event", on_key_press)

class_toggle = ToggleButtons(options=CLASS_TOGGLE_LABELS, value=CLASS_TOGGLE_LABELS[0], description="Kelas:")
class_toggle.observe(apply_toggle_to_selection, names="value")
label_toggle_button = Button(description=label_toggle_description())
previous_button = Button(description="Prev (p)")
next_button = Button(description="Next (n)")
reviewed_button = Button(description="Reviewed (r)")
delete_button = Button(description="Hapus (x)")
undo_button = Button(description="Undo (u)")
unlabeled_button = Button(description="Next unlabeled (f)")
status_widget = HTML(value=status_html())
controls_box = VBox([
    HTML("<b>Kelas box terpilih — hotkey 1-9</b>"), class_toggle,
    HBox([previous_button, next_button, reviewed_button, unlabeled_button, delete_button, undo_button, label_toggle_button]),
    status_widget,
])
next_button.on_click(go_next)
previous_button.on_click(go_previous)
reviewed_button.on_click(mark_reviewed)
delete_button.on_click(remove_selected)
undo_button.on_click(undo_last)
unlabeled_button.on_click(go_to_next_unlabeled)
label_toggle_button.on_click(toggle_box_labels)

display(controls_box)
go_to_index(0)
display(Markdown(
    "Drag = box baru · klik box = pilih · **1-9** = set kelas · **x** = hapus · **u** = undo · "
    "**n/p** = navigasi · **r** = reviewed + next · **T** = label ON/OFF. "
    "Autosave tiap perubahan ke `annotations.json`.\\n\\n"
    "**Catatan teknis.** `RectangleSelector` dibuat dengan `interactive=False`, sehingga matplotlib "
    "3.9+ tidak membuat atribut handle internalnya (`_corner_order`, `_active_handle`). Atribut itu "
    "tetap dibaca saat tombol **r** (modifier rotate bawaan) ditahan sambil mouse bergerak, dan "
    "menghasilkan `AttributeError: 'RectangleSelector' object has no attribute '_corner_order'`. "
    "Sel ini mengisi atribut yang hilang agar drag tetap jalan; handle rotate/resize tetap nonaktif."
))'''

EXPORT_SOURCE = '''import shutil
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from IPython.display import Markdown, display

COCO_CATEGORY_START_ID = 1
COCO_IMAGE_ID_START = 1
COCO_ANNOTATION_ID_START = 1


def normalize_box_to_yolo(box: list[float], image_width: int, image_height: int) -> list[float]:
    centre_x = (box[0] + box[2]) / 2.0 / image_width
    centre_y = (box[1] + box[3]) / 2.0 / image_height
    box_width = (box[2] - box[0]) / image_width
    box_height = (box[3] - box[1]) / image_height
    return [centre_x, centre_y, box_width, box_height]


def reviewed_sample_records() -> list[dict]:
    return [
        sample_record
        for sample_record in sample_records
        if annotation_store["frames"][sample_record["frame_stem"]]["reviewed"]
    ]


def assert_no_excluded_split(records: list[dict]) -> None:
    leaked_cameras = sorted({
        sample_record["camera_id"]
        for sample_record in records
        if resolve_camera_split(sample_record["camera_id"], SPLIT_NAME_BY_CAMERA) in SAMPLE_EXCLUDED_SPLITS
    })
    if leaked_cameras:
        raise ValueError(
            f"Kebocoran split: kamera {leaked_cameras} termasuk split {SAMPLE_EXCLUDED_SPLITS} tetapi ikut export."
        )


def assert_no_pending_class_in_reviewed_frames() -> None:
    pending_stems = pending_boxes_in_reviewed_frames()
    if pending_stems:
        raise ValueError(
            f"{len(pending_stems)} frame reviewed masih punya box {PENDING_CLASS_NAME} "
            f"(contoh: {pending_stems[:5]}). Setiap box wajib diberi kelas sebelum export."
        )


def build_coco_payload(records: list[dict]) -> dict:
    coco_images = []
    coco_annotations = []
    annotation_id = COCO_ANNOTATION_ID_START
    for image_id, sample_record in enumerate(records, start=COCO_IMAGE_ID_START):
        entry = annotation_store["frames"][sample_record["frame_stem"]]
        coco_images.append({
            "id": image_id,
            "file_name": f"{sample_record['frame_stem']}.png",
            "width": entry["image_width"],
            "height": entry["image_height"],
            "camera_id": sample_record["camera_id"],
            "split": resolve_camera_split(sample_record["camera_id"], SPLIT_NAME_BY_CAMERA),
        })
        for box_entry in entry["boxes"]:
            if box_entry["deleted"]:
                continue
            box = box_entry["box"]
            coco_annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": CLASS_NAMES.index(box_entry["class_name"]) + COCO_CATEGORY_START_ID,
                "bbox": [box[0], box[1], box[2] - box[0], box[3] - box[1]],
                "area": (box[2] - box[0]) * (box[3] - box[1]),
                "iscrowd": 0,
                "source": box_entry["source"],
            })
            annotation_id += 1
    return {
        "info": {
            "description": "NB02b anotasi kondisi lampu kendaraan (frame ROI-crop + fit 640px)",
            "sample_total_frames": SAMPLE_TOTAL_FRAMES,
            "random_seed": RANDOM_SEED,
            "excluded_splits": SAMPLE_EXCLUDED_SPLITS,
            "reviewed_frames": len(records),
        },
        "licenses": [],
        "images": coco_images,
        "annotations": coco_annotations,
        "categories": [
            {"id": class_index + COCO_CATEGORY_START_ID, "name": class_name, "supercategory": "kondisi-lampu"}
            for class_index, class_name in enumerate(CLASS_NAMES)
        ],
    }


def write_yolo_gold_set(records: list[dict]) -> tuple[dict, list[dict]]:
    missing_image_paths = [record["image_path"] for record in records if not Path(record["image_path"]).exists()]
    if missing_image_paths:
        raise FileNotFoundError(
            f"{len(missing_image_paths)} gambar crop hilang (contoh: {missing_image_paths[0]}); "
            "dataset gold lama tidak disentuh."
        )
    for existing_path in ANNOTATION_YOLO_IMAGE_DIR.glob("*.png"):
        existing_path.unlink()
    for existing_path in ANNOTATION_YOLO_LABEL_DIR.glob("*.txt"):
        existing_path.unlink()
    label_counts_by_class = {class_name: 0 for class_name in TRAINING_CLASS_NAMES}
    label_file_count = 0
    exported_records = []
    for sample_record in records:
        entry = annotation_store["frames"][sample_record["frame_stem"]]
        label_lines = []
        for box_entry in entry["boxes"]:
            if box_entry["deleted"]:
                continue
            yolo_box = normalize_box_to_yolo(box_entry["box"], entry["image_width"], entry["image_height"])
            training_class = training_class_name(box_entry["class_name"])
            class_index = TRAINING_CLASS_NAMES.index(training_class)
            label_lines.append(f"{class_index} " + " ".join(f"{coordinate:.6f}" for coordinate in yolo_box))
            label_counts_by_class[training_class] += 1
        if not label_lines:
            continue
        target_stem = sample_record["frame_stem"]
        shutil.copyfile(sample_record["image_path"], ANNOTATION_YOLO_IMAGE_DIR / f"{target_stem}.png")
        (ANNOTATION_YOLO_LABEL_DIR / f"{target_stem}.txt").write_text(
            "\\n".join(label_lines) + "\\n", encoding="utf-8"
        )
        label_file_count += 1
        exported_records.append(sample_record)
    return {
        "gold_images": len(list(ANNOTATION_YOLO_IMAGE_DIR.glob("*.png"))),
        "gold_labels": label_file_count,
        "gold_boxes": sum(label_counts_by_class.values()),
        "gold_boxes_by_class": label_counts_by_class,
    }, exported_records


def write_split_map(records: list[dict]) -> None:
    split_lines = ["frame_stem,camera_id,split"]
    for sample_record in records:
        split_name = resolve_camera_split(sample_record["camera_id"], SPLIT_NAME_BY_CAMERA)
        split_lines.append(f"{sample_record['frame_stem']},{sample_record['camera_id']},{split_name}")
    SPLIT_MAP_PATH.write_text("\\n".join(split_lines) + "\\n", encoding="utf-8")


def build_annotation_preview(records: list[dict], preview_path: Path, preview_count: int) -> None:
    preview_records = records[:preview_count] if records else sample_records[:preview_count]
    figure, axes = plt.subplots(1, max(1, len(preview_records)), figsize=(6 * max(1, len(preview_records)), 6))
    if len(preview_records) == 1:
        axes = [axes]
    for axis, sample_record in zip(axes, preview_records):
        entry = annotation_store["frames"][sample_record["frame_stem"]]
        preview_image = Image.open(sample_record["image_path"]).convert("RGB")
        drawer = ImageDraw.Draw(preview_image)
        for box_entry in entry["boxes"]:
            if box_entry["deleted"]:
                continue
            box = box_entry["box"]
            outline_color = CLASS_COLORS[box_entry["class_name"]]
            drawer.rectangle([box[0], box[1], box[2], box[3]], outline=outline_color, width=2)
        review_state = "reviewed" if entry["reviewed"] else "belum reviewed"
        axis.imshow(preview_image)
        axis.set_title(f"{sample_record['camera_id']}\\n{review_state} · {len(active_boxes(sample_record['frame_stem']))} box")
        axis.axis("off")
    figure.suptitle("Hasil anotasi NB02b")
    figure.tight_layout()
    figure.savefig(preview_path, dpi=110)
    plt.close(figure)


reviewed_records = reviewed_sample_records()
manifest_by_stem = {record["frame_stem"]: record for record in sample_records}
all_pending_frame_stems = pending_box_frames()
if all_pending_frame_stems:
    display(Markdown("### Semua frame yang masih punya anotasi unlabeled"))
    pending_lines = []
    for frame_stem in all_pending_frame_stems:
        sample_record = manifest_by_stem.get(frame_stem)
        image_exists = bool(sample_record and Path(sample_record["image_path"]).exists())
        state = "tersedia di UI" if image_exists else "tidak ada di manifest/gambar crop hilang"
        pending_lines.append(f"- `{frame_stem}` — {state}")
    display(Markdown("\\n".join(pending_lines)))
else:
    display(Markdown("Tidak ada frame dengan anotasi unlabeled."))
assert_no_excluded_split(reviewed_records)
assert_no_pending_class_in_reviewed_frames()
store_validation_errors = validate_annotation_store()
if store_validation_errors:
    raise ValueError("Store gagal validasi sebelum export:\\n- " + "\\n- ".join(store_validation_errors))
coco_payload = build_coco_payload(reviewed_records)
with COCO_EXPORT_PATH.open("w", encoding="utf-8") as coco_file:
    json.dump(coco_payload, coco_file, ensure_ascii=False, indent=2)
gold_summary, exported_records = write_yolo_gold_set(reviewed_records)
write_split_map(exported_records)
build_annotation_preview(exported_records, ANNOTATION_ROOT / "annotation_preview_after.png", PREVIEW_FRAME_COUNT)

source_counts = box_source_counts()
class_counts = class_box_counts()
annotation_summary = {
    "sample_total_frames": len(frame_stems_in_order()),
    "reviewed_frames": len(reviewed_records),
    "random_seed": RANDOM_SEED,
    "excluded_splits": SAMPLE_EXCLUDED_SPLITS,
    "split_by_camera": SPLIT_NAME_BY_CAMERA,
    "class_names": CLASS_NAMES,
    "box_source_counts": source_counts,
    "class_box_counts": class_counts,
    "coco_annotations": len(coco_payload["annotations"]),
    "coco_path": str(COCO_EXPORT_PATH),
    "split_map_path": str(SPLIT_MAP_PATH),
    **gold_summary,
}
with SUMMARY_PATH.open("w", encoding="utf-8") as summary_file:
    json.dump(annotation_summary, summary_file, ensure_ascii=False, indent=2)

summary_table_lines = [
    "| Metrik | Nilai |", "|---|---|",
    f"| Frame disampel | {annotation_summary['sample_total_frames']} |",
    f"| Split dikecualikan (anti-bocor) | {', '.join(annotation_summary['excluded_splits'])} |",
    f"| Frame reviewed | {annotation_summary['reviewed_frames']} |",
    f"| Anotasi COCO (box) | {annotation_summary['coco_annotations']} |",
    f"| Box pseudo / manual / edited | {source_counts[BOX_SOURCE_PSEUDO]} / {source_counts[BOX_SOURCE_MANUAL]} / {source_counts[BOX_SOURCE_EDITED]} |",
    f"| Gambar gold-set | {annotation_summary['gold_images']} |",
    f"| Label gold-set (file) | {annotation_summary['gold_labels']} |",
    f"| Box gold-set | {annotation_summary['gold_boxes']} |",
]
class_table_lines = ["| Kelas | Box |", "|---|---|"]
for class_name in CLASS_NAMES:
    class_table_lines.append(f"| `{class_name}` | {class_counts[class_name]} |")
training_table_lines = ["| Kelas latih | Box |", "|---|---|"]
for class_name in TRAINING_CLASS_NAMES:
    training_table_lines.append(f"| `{class_name}` | {gold_summary['gold_boxes_by_class'][class_name]} |")
display(Markdown("## Export hasil anotasi\\n" + "\\n".join(summary_table_lines)))
display(Markdown("### Box per kelas anotasi\\n" + "\\n".join(class_table_lines)))
display(Markdown(
    "### Box per kelas latih\\nKelas `*Berwarna` runtuh ke `*Normal`/`*Silau` padanannya \\u2192 "
    "label YOLO hanya memakai 5 kelas.\\n\\n" + "\\n".join(training_table_lines)
))
display(Markdown(
    f"COCO: `{COCO_EXPORT_PATH}`\\n\\nGold-set YOLO: `{ANNOTATION_YOLO_ROOT}` "
    f"(`images/`, `labels/`, `split_map.csv`)\\n\\nSummary: `{SUMMARY_PATH}`\\n\\n"
    f"Preview: `{ANNOTATION_ROOT / 'annotation_preview_after.png'}`"
))
display(Markdown(
    "`data.yaml` sengaja **tidak** ditulis di sini. **NB04** yang mematerialisasi "
    "`images/{train,val,test}` dari `split_map.csv`, sehingga tidak ada `train == val`."
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(SAMPLING_SOURCE),
        nbformat.v4.new_code_cell(STORE_SOURCE),
        nbformat.v4.new_code_cell(PREVIEW_SOURCE),
        nbformat.v4.new_code_cell(UI_SOURCE),
        nbformat.v4.new_code_cell(EXPORT_SOURCE),
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
