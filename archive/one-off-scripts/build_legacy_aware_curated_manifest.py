#!/usr/bin/env python3
"""Buat input curated NB02 sambil mempertahankan frame gold lama yang sudah direview."""

import csv
import json
import random
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline_common import SPLIT_NAME_BY_CAMERA

ROOT = Path(__file__).resolve().parent.parent
MANIFESTS = ROOT / "output" / "manifests"
SAMPLE_MANIFEST = ROOT / "output" / "annotation" / "sample_manifest.json"
ANNOTATIONS = ROOT / "output" / "annotation" / "annotations.json"
TARGET_FRAMES = 600
MIN_GAP_FRAMES = 30
HASH_DISTANCE = 4
SEED = 20261102


def frame_stem(row: dict) -> str:
    return Path(row["video_path"]).stem + "_" + Path(row["frame_path"]).stem


def main() -> None:
    rows = list(csv.DictReader((MANIFESTS / "dedup_manifest.csv").open(encoding="utf-8")))
    old_manifest = json.loads(SAMPLE_MANIFEST.read_text(encoding="utf-8"))
    annotations = json.loads(ANNOTATIONS.read_text(encoding="utf-8"))
    protected = {
        record["frame_stem"]
        for record in old_manifest["frames"]
        if annotations["frames"].get(record["frame_stem"], {}).get("reviewed")
    }
    rows_by_video = defaultdict(list)
    for row in rows:
        row = dict(row)
        row["video_path"] = str((ROOT / row["video_path"]).resolve()) if not Path(row["video_path"]).is_absolute() else row["video_path"]
        row["frame_path"] = str((ROOT / row["frame_path"]).resolve()) if not Path(row["frame_path"]).is_absolute() else row["frame_path"]
        row["camera_id"] = Path(row["video_path"]).stem.split("_")[1]
        row["frame_number"] = int(Path(row["frame_path"]).stem.split("_")[-1])
        rows_by_video[row["video_path"]].append(row)
    candidates = []
    for video_rows in rows_by_video.values():
        video_rows.sort(key=lambda row: row["frame_number"])
        previous_hashes = []
        for row in video_rows:
            current_hash = int(row["frame_hash"])
            duplicate = any((current_hash ^ previous_hash).bit_count() <= HASH_DISTANCE for previous_hash in previous_hashes[-1:])
            previous_hashes.append(current_hash)
            if not duplicate and SPLIT_NAME_BY_CAMERA.get(row["camera_id"]) in {"train", "val"}:
                row["frame_stem"] = frame_stem(row)
                candidates.append(row)
    by_camera = defaultdict(list)
    for row in candidates:
        by_camera[row["camera_id"]].append(row)
    quotas = {camera: max(1, TARGET_FRAMES * len(camera_rows) // len(candidates)) for camera, camera_rows in by_camera.items()}
    while sum(quotas.values()) < TARGET_FRAMES:
        camera = max(by_camera, key=lambda name: len(by_camera[name]) - quotas[name])
        quotas[camera] += 1
    random_generator = random.Random(SEED)
    selected = []
    for camera, camera_rows in sorted(by_camera.items()):
        protected_rows = [row for row in camera_rows if row["frame_stem"] in protected]
        chosen = protected_rows.copy()
        shuffled = camera_rows.copy()
        random_generator.shuffle(shuffled)
        for row in shuffled:
            if row in chosen or len(chosen) >= quotas[camera]:
                continue
            if all(abs(row["frame_number"] - chosen_row["frame_number"]) >= MIN_GAP_FRAMES for chosen_row in chosen if chosen_row not in protected_rows):
                if all(abs(row["frame_number"] - protected_row["frame_number"]) >= MIN_GAP_FRAMES for protected_row in protected_rows):
                    chosen.append(row)
        selected.extend(chosen)
    selected.sort(key=lambda row: (row["camera_id"], row["frame_number"]))
    backup = MANIFESTS / f"curated_backup_{datetime.now():%Y%m%d_%H%M%S}"
    for name in ("curated_frames_manifest.csv", "curated_dedup_manifest.csv"):
        existing = MANIFESTS / name
        if existing.exists():
            backup.mkdir(parents=True, exist_ok=True)
            shutil.copy2(existing, backup / name)
    field_rows = []
    for row in selected:
        field_rows.append({
            "video_path": str(Path(row["video_path"]).relative_to(ROOT)),
            "frame_path": str(Path(row["frame_path"]).relative_to(ROOT)),
            "frame_hash": row["frame_hash"],
            "is_duplicate": "False",
        })
    with (MANIFESTS / "curated_frames_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "frame_path"])
        writer.writeheader()
        writer.writerows({key: row[key] for key in ("video_path", "frame_path")} for row in field_rows)
    with (MANIFESTS / "curated_dedup_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "frame_path", "frame_hash", "is_duplicate"])
        writer.writeheader()
        writer.writerows(field_rows)
    summary = {"selected": len(selected), "protected_reviewed": len(protected), "by_camera": {camera: sum(row["camera_id"] == camera for row in selected) for camera in by_camera}, "backup": str(backup.relative_to(ROOT)) if backup.exists() else None}
    (MANIFESTS / "curated_sampling_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
