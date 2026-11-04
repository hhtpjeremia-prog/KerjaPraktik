#!/usr/bin/env python3
"""Build NB01_acquisition.ipynb for Riset-HampirMatang (multi-camera Bogor Kab scraping)."""

import json
from pathlib import Path

import nbformat

from notebook_gate import verify_code_cells

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB01_acquisition.ipynb"

NAVBAR_MARKDOWN = """# Akuisisi Data — CCTV Multi-Kamera Kab. Bogor (atcsindonesia.com)

Pipeline arsitektur V2, tahap NB01: katalog kamera, rekaman video, sampling frame, deduplikasi, split, dan template ROI.

- Domain: `atcsindonesia.com` aggregator; stream asli dari server Dishub Kab. Bogor (`itscctv-dishub.bogorkab.go.id`, HLS publik).
- 5 kamera target (dari Domain-link.md): `bogor-kab-bappenda1`, `bogor-kab-bappenda2`, `bogor-kab-bappenda3`, `bogor-kab-pdam2`, `bogor-kab-pemda2`.
- Setiap re-run menghasilkan video bernama unik (timestamp) sehingga tidak ada yang tertimpa; manifest mencatat semua sesi.
- Output tersimpan di `output/` dalam folder ini; semua sel sengaja tidak dieksekusi — jalankan manual per sel di VSCode.

Alur: **1)** konfigurasi, **2)** katalog kamera (cache 24 jam), **3)** rekaman video, **4)** sampling frame, **5)** deduplikasi dHash, **6)** group split per kamera + template ROI.
"""

CONFIG_SOURCE = '''from pathlib import Path
import random
import os
import subprocess
import shutil

from IPython.display import Markdown, display

from pipeline_common import SPLIT_NAME_BY_CAMERA

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
KATALOG_DIR = OUTPUT_ROOT / "katalog"
VIDEO_DIR = OUTPUT_ROOT / "videos"
FRAMES_DIR = OUTPUT_ROOT / "frames"
MANIFEST_DIR = OUTPUT_ROOT / "manifests"
SPLIT_DIR = OUTPUT_ROOT / "split"
ROI_DIR = OUTPUT_ROOT / "roi"

for output_dir in (KATALOG_DIR, VIDEO_DIR, FRAMES_DIR, MANIFEST_DIR, SPLIT_DIR, ROI_DIR):
    output_dir.mkdir(parents=True, exist_ok=True)

CATALOG_API_URL = "https://atcsindonesia.com/api/cctv?limit=all"
CATALOG_CACHE_PATH = KATALOG_DIR / "cctv_all.json"
CATALOG_MAX_AGE_HOURS = 24
CATALOG_REQUEST_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Referer": "https://atcsindonesia.com/",
}

TARGET_CAMERA_IDS = [
    "bogor-kab-bappenda1",
    "bogor-kab-bappenda2",
    "bogor-kab-bappenda3",
    "bogor-kab-pdam2",
    "bogor-kab-pemda2",
]

RECORD_DURATION_SECONDS = 600
INTER_CAMERA_DELAY_RANGE_SECONDS = (8, 20)
FFMPEG_EXECUTABLE = shutil.which("ffmpeg")
FFMPEG_NETWORK_TIMEOUT_MICROSECONDS = 10000000
FFMPEG_RECONNECT_DELAY_MAX_SECONDS = 10

SAMPLE_INTERVAL_FRAMES = 15
FRAME_RETENTION_MIN_SECONDS = 5

MISSING_SPLIT_CAMERAS = [camera_id for camera_id in TARGET_CAMERA_IDS if camera_id not in SPLIT_NAME_BY_CAMERA]

ROI_TEMPLATE_RESIZE = (640, 640)

if FFMPEG_EXECUTABLE is None:
    raise RuntimeError("ffmpeg tidak ditemukan di PATH. Install ffmpeg dan tambahkan ke PATH sebelum menjalankan sel ini.")
if MISSING_SPLIT_CAMERAS:
    raise RuntimeError(f"kamera target belum punya split di pipeline_common: {MISSING_SPLIT_CAMERAS}")

display(Markdown("Konfigurasi siap. Root: " + str(RISET_ROOT)))'''

CATALOG_SOURCE = '''import json
import time
from pathlib import Path
from urllib.request import Request, urlopen

from IPython.display import Markdown, display


def load_target_cameras(catalog_path: Path, target_ids: list[str]) -> list[dict]:
    with catalog_path.open(encoding="utf-8-sig") as catalog_file:
        catalog_payload = json.load(catalog_file)
    cameras_by_id = {camera["id"]: camera for camera in catalog_payload["data"]}
    return [cameras_by_id[camera_id] for camera_id in target_ids if camera_id in cameras_by_id]


def fetch_or_load_catalog(cache_path: Path, api_url: str, request_headers: dict, max_age_hours: int) -> dict:
    if cache_path.exists():
        cache_age_hours = (time.time() - cache_path.stat().st_mtime) / 3600
        if cache_age_hours < max_age_hours:
            with cache_path.open(encoding="utf-8-sig") as catalog_file:
                return json.load(catalog_file)
    request = Request(api_url, headers=request_headers)
    with urlopen(request, timeout=60) as response:
        catalog_payload = json.load(response)
    with cache_path.open("w", encoding="utf-8") as catalog_file:
        json.dump(catalog_payload, catalog_file, ensure_ascii=False)
    return catalog_payload


catalog_payload = fetch_or_load_catalog(CATALOG_CACHE_PATH, CATALOG_API_URL, CATALOG_REQUEST_HEADERS, CATALOG_MAX_AGE_HOURS)
target_cameras = load_target_cameras(CATALOG_CACHE_PATH, TARGET_CAMERA_IDS)
target_snapshot_path = KATALOG_DIR / "bogor_kab_targets.json"
with target_snapshot_path.open("w", encoding="utf-8") as snapshot_file:
    json.dump(target_cameras, snapshot_file, ensure_ascii=False, indent=1)

display(Markdown(f"Katalog: {len(catalog_payload['data'])} kamera (sumber: cache/API). Target ditemukan: {len(target_cameras)}/{len(TARGET_CAMERA_IDS)}."))
for camera in target_cameras:
    display(Markdown(f"- `{camera['id']}` | {camera.get('title','')} | status: `{camera['status']}` | {camera['streamingUrl']}"))'''

RECORD_SOURCE = '''import csv
import random
import re
import subprocess
import time
from datetime import datetime
from pathlib import Path

from IPython.display import Markdown, display

VIDEO_EXTENSION = "mp4"
MANIFEST_FIELD_NAMES = [
    "camera_id", "camera_title", "streaming_url", "video_path",
    "started_at", "finished_at", "duration_seconds", "file_size_bytes", "ffmpeg_returncode",
]


def safe_camera_token(camera_id: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", camera_id)


def timestamped_video_path(camera_id: str, output_dir: Path) -> Path:
    timestamp_token = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"atcs_{safe_camera_token(camera_id)}_{timestamp_token}.{VIDEO_EXTENSION}"
    return output_dir / filename


def record_stream_with_ffmpeg(streaming_url: str, output_path: Path, duration_seconds: int) -> subprocess.CompletedProcess:
    ffmpeg_arguments = [
        FFMPEG_EXECUTABLE,
        "-y",
        "-hide_banner",
        "-loglevel", "error",
        "-rw_timeout", str(FFMPEG_NETWORK_TIMEOUT_MICROSECONDS),
        "-reconnect", "1",
        "-reconnect_streamed", "1",
        "-reconnect_delay_max", str(FFMPEG_RECONNECT_DELAY_MAX_SECONDS),
        "-i", streaming_url,
        "-t", str(duration_seconds),
        "-c", "copy",
        "-an",
        "-movflags", "+faststart",
        str(output_path),
    ]
    return subprocess.run(ffmpeg_arguments, capture_output=True, text=True, timeout=duration_seconds * 2 + 120)


def append_manifest_row(manifest_path: Path, field_names: list[str], record: dict) -> None:
    manifest_exists = manifest_path.exists()
    with manifest_path.open("a", newline="", encoding="utf-8") as manifest_file:
        manifest_writer = csv.DictWriter(manifest_file, fieldnames=field_names)
        if not manifest_exists:
            manifest_writer.writeheader()
        manifest_writer.writerow(record)


def record_target_camera(camera: dict) -> dict:
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    output_path = timestamped_video_path(camera["id"], VIDEO_DIR)
    process = record_stream_with_ffmpeg(camera["streamingUrl"], output_path, RECORD_DURATION_SECONDS)
    file_size = output_path.stat().st_size if output_path.exists() else 0
    return {
        "camera_id": camera["id"],
        "camera_title": camera.get("title", ""),
        "streaming_url": camera["streamingUrl"],
        "video_path": str(output_path.relative_to(RISET_ROOT)),
        "started_at": started_at,
        "finished_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "duration_seconds": RECORD_DURATION_SECONDS,
        "file_size_bytes": file_size,
        "ffmpeg_returncode": process.returncode,
    }


def run_acquisition(target_cameras: list[dict], manifest_path: Path) -> None:
    display(Markdown(f"Memulai rekaman {len(target_cameras)} kamera ({RECORD_DURATION_SECONDS} detik/kamera)."))
    for camera_index, camera in enumerate(target_cameras):
        display(Markdown(f"Rekam `{camera['id']}` dari {camera['streamingUrl']} ..."))
        record = record_target_camera(camera)
        append_manifest_row(manifest_path, MANIFEST_FIELD_NAMES, record)
        status_text = "OK" if record["ffmpeg_returncode"] == 0 else f"GAGAL rc={record['ffmpeg_returncode']}"
        display(Markdown(f"  {status_text} | {record['video_path']} | {record['file_size_bytes']} bytes"))
        if camera_index < len(target_cameras) - 1:
            delay_seconds = random.uniform(*INTER_CAMERA_DELAY_RANGE_SECONDS)
            display(Markdown(f"  jeda {delay_seconds:.0f} detik sebelum kamera berikutnya (anti IP block)."))
            time.sleep(delay_seconds)


manifest_path = MANIFEST_DIR / "scraping_manifest.csv"
run_acquisition(target_cameras, manifest_path)'''

SAMPLE_SOURCE = '''import csv
from pathlib import Path

import cv2


def sample_frames_from_video(video_path: Path, output_dir: Path, interval_frames: int) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    video_capture = cv2.VideoCapture(str(video_path))
    sampled_frame_paths: list[Path] = []
    frame_index = 0
    while True:
        frame_ok, frame = video_capture.read()
        if not frame_ok:
            break
        if frame_index % interval_frames == 0:
            frame_path = output_dir / f"frame_{frame_index:06d}.png"
            cv2.imwrite(str(frame_path), frame)
            sampled_frame_paths.append(frame_path)
        frame_index += 1
    video_capture.release()
    return sampled_frame_paths


def build_frame_manifest(manifest_path: Path, frame_paths_by_video: dict[str, list[Path]]) -> None:
    with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
        manifest_writer = csv.writer(manifest_file)
        manifest_writer.writerow(["video_path", "frame_path"])
        for video_path, frame_paths in frame_paths_by_video.items():
            for frame_path in frame_paths:
                manifest_writer.writerow([video_path, str(frame_path.relative_to(RISET_ROOT))])


def run_sampling(video_manifest_path: Path, frames_root: Path) -> tuple[dict[str, list[Path]], list[str]]:
    frames_by_video: dict[str, list[Path]] = {}
    empty_video_rows: list[str] = []
    with video_manifest_path.open(encoding="utf-8") as manifest_file:
        for manifest_row in csv.DictReader(manifest_file):
            video_path = Path(manifest_row["video_path"])
            sampled_frames = sample_frames_from_video(video_path, frames_root / video_path.stem, SAMPLE_INTERVAL_FRAMES)
            frames_by_video[manifest_row["video_path"]] = sampled_frames
            if not sampled_frames:
                empty_video_rows.append(f"{manifest_row['camera_id']} ({video_path.name})")
    return frames_by_video, empty_video_rows


frame_manifest_path = MANIFEST_DIR / "frames_manifest.csv"
frames_by_video, empty_video_rows = run_sampling(manifest_path, FRAMES_DIR)
if not any(frames_by_video.values()):
    raise RuntimeError(f"tidak ada frame yang berhasil disampling dari {len(frames_by_video)} video; periksa isi output/videos")
build_frame_manifest(frame_manifest_path, frames_by_video)
display(Markdown(
    f"Sampling selesai: **{sum(len(frames) for frames in frames_by_video.values())}** frame "
    f"dari **{len(frames_by_video)}** video. Manifest: `{frame_manifest_path}`"
))
if empty_video_rows:
    display(Markdown(
        "**Video tanpa satu pun frame** (rekaman gagal atau berkas hilang): "
        + ", ".join(f"`{empty_video_row}`" for empty_video_row in empty_video_rows)
    ))'''

DEDUP_SOURCE = '''import csv
from pathlib import Path

import cv2
import numpy as np

from IPython.display import Markdown, display

HASH_IMAGE_SIZE = (9, 8)
DUPLICATE_WINDOW_FRAMES = 1
DUPLICATE_HASH_DISTANCE = 4


def frame_dhash(frame, hash_image_size: tuple[int, int]) -> int:
    resized_frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), hash_image_size, interpolation=cv2.INTER_AREA)
    pixel_row = resized_frame.astype(int)
    row_diff = pixel_row[:, 1:] >= pixel_row[:, :-1]
    return sum(1 << bit_index for bit_index, bit_value in enumerate(row_diff.flatten()) if bit_value)


def duplicate_flags(frame_hashes: list[int]) -> list[bool]:
    flags = []
    for index, current_hash in enumerate(frame_hashes):
        previous_hashes = frame_hashes[max(0, index - DUPLICATE_WINDOW_FRAMES):index]
        flags.append(any((current_hash ^ previous_hash).bit_count() <= DUPLICATE_HASH_DISTANCE for previous_hash in previous_hashes if previous_hash >= 0))
    return flags


def build_dedup_manifest(frame_manifest_path: Path, dedup_manifest_path: Path) -> None:
    frame_entries = []
    with frame_manifest_path.open(encoding="utf-8") as manifest_file:
        for manifest_row in csv.DictReader(manifest_file):
            frame_entries.append((manifest_row["video_path"], manifest_row["frame_path"]))
    hashes_by_video = {}
    for video_path, frame_path_text in frame_entries:
        sampled_frame = cv2.imread(frame_path_text)
        hashes_by_video.setdefault(video_path, []).append(
            frame_dhash(sampled_frame, HASH_IMAGE_SIZE) if sampled_frame is not None else -1
        )
    duplicate_by_video = {video_path: duplicate_flags(frame_hashes) for video_path, frame_hashes in hashes_by_video.items()}
    positions_by_video = {video_path: 0 for video_path in hashes_by_video}
    with dedup_manifest_path.open("w", newline="", encoding="utf-8") as dedup_file:
        dedup_writer = csv.writer(dedup_file)
        dedup_writer.writerow(["video_path", "frame_path", "frame_hash", "is_duplicate"])
        for video_path, frame_path_text in frame_entries:
            position = positions_by_video[video_path]
            frame_hash = hashes_by_video[video_path][position]
            dedup_writer.writerow([video_path, frame_path_text, frame_hash, duplicate_by_video[video_path][position]])
            positions_by_video[video_path] += 1


build_dedup_manifest(frame_manifest_path, MANIFEST_DIR / "dedup_manifest.csv")
display(Markdown(f"Deduplikasi: dHash + Hamming <= {DUPLICATE_HASH_DISTANCE} pada jendela {DUPLICATE_WINDOW_FRAMES} frame. NB02 tetap memvalidasi ulang sebelum pemrosesan."))
'''

SPLIT_SOURCE = '''import csv
import json
import os
from pathlib import Path

from IPython.display import Markdown, display


def build_group_split_assignment(target_cameras: list[dict], split_assignment: dict[str, str]) -> list[dict]:
    return [
        {
            "camera_id": camera["id"],
            "camera_title": camera.get("title", ""),
            "lat": camera.get("lat", ""),
            "lng": camera.get("lng", ""),
            "split": split_assignment.get(camera["id"], "unassigned"),
        }
        for camera in target_cameras
    ]


group_split_path = SPLIT_DIR / "group_split.csv"
group_split_rows = build_group_split_assignment(target_cameras, SPLIT_NAME_BY_CAMERA)
with group_split_path.open("w", newline="", encoding="utf-8") as split_file:
    split_writer = csv.DictWriter(split_file, fieldnames=list(group_split_rows[0].keys()))
    split_writer.writeheader()
    split_writer.writerows(group_split_rows)

roi_config_path = ROI_DIR / "roi_config.json"
if not roi_config_path.exists():
    roi_template = {
        camera["id"]: {
            "label": camera.get("title", camera["id"]),
            "roi": {"x": 0, "y": 0, "width": 1280, "height": 720},
            "resize_to": list(ROI_TEMPLATE_RESIZE),
        }
        for camera in target_cameras
    }
    with roi_config_path.with_name(roi_config_path.name + ".tmp").open("w", encoding="utf-8") as roi_file:
        json.dump(roi_template, roi_file, ensure_ascii=False, indent=2)
    os.replace(roi_config_path.with_name(roi_config_path.name + ".tmp"), roi_config_path)

display(Markdown(f"Group split: {group_split_path} | Template ROI: {roi_config_path} (isi koordinat crop setelah melihat preview frame di `output/frames/`)."))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(CATALOG_SOURCE),
        nbformat.v4.new_code_cell(RECORD_SOURCE),
        nbformat.v4.new_code_cell(SAMPLE_SOURCE),
        nbformat.v4.new_code_cell(DEDUP_SOURCE),
        nbformat.v4.new_code_cell(SPLIT_SOURCE),
    ]
    return nbformat.v4.new_notebook(cells=notebook_cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}})


def main() -> None:
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    verify_code_cells(OUTPUT_NOTEBOOK, 6, 1)


if __name__ == "__main__":
    main()
