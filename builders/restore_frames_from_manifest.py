#!/usr/bin/env python3
"""Pulihkan frame yang hilang dari video lokal tanpa menghitung ulang hash manifest."""

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2
from pipeline_common import difference_hash


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
VIDEOS = OUTPUT / "videos"
FRAMES = OUTPUT / "frames"
MANIFESTS = OUTPUT / "manifests"
BACKUP_STEM = f"manifest_backup_{datetime.now():%Y%m%d_%H%M%S}"
FRAME_NUMBER = re.compile(r"frame_(\d+)\.(?:png|jpe?g)$", re.IGNORECASE)


def project_relative(value: str) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else value
    return str(path)


def local_video_path(value: str) -> Path:
    candidate = ROOT / value if not Path(value).is_absolute() else Path(value)
    if candidate.exists():
        return candidate
    matches = list(VIDEOS.glob(Path(value).name))
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(f"Video tidak ditemukan/ambigu: {value}")


def write_csv_atomic(path: Path, header: list[str], rows: list[list[str]] | list[dict]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        if rows and isinstance(rows[0], dict):
            writer = csv.DictWriter(handle, fieldnames=header)
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
    temporary.replace(path)


def restore_video_frames(video_path: Path, video_rows: list[dict]) -> None:
    targets = {
        int(FRAME_NUMBER.fullmatch(Path(row["frame_path"]).name).group(1)): row
        for row in video_rows
    }
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Video tidak bisa dibuka: {video_path}")
    frame_number = 0
    try:
        while targets:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Frame {min(targets)} gagal dibaca dari {video_path.name}")
            row = targets.pop(frame_number, None)
            if row is not None:
                frame_path = FRAMES / video_path.stem / Path(row["frame_path"]).name
                frame_path.parent.mkdir(parents=True, exist_ok=True)
                if not cv2.imwrite(str(frame_path), frame):
                    raise OSError(f"Gagal menulis {frame_path}")
            frame_number += 1
    finally:
        capture.release()


def main() -> None:
    dedup_path = MANIFESTS / "dedup_manifest.csv"
    if not dedup_path.exists():
        raise FileNotFoundError(dedup_path)
    rows = list(csv.DictReader(dedup_path.open(encoding="utf-8")))
    if not rows:
        raise RuntimeError("dedup_manifest.csv kosong")
    backup_name = BACKUP_STEM
    backup_index = 1
    while (MANIFESTS / backup_name).exists():
        backup_name = f"{BACKUP_STEM}_{backup_index}"
        backup_index += 1
    backup_path = MANIFESTS / backup_name
    backup_path.mkdir(parents=True, exist_ok=False)
    for name in ("frames_manifest.csv", "dedup_manifest.csv"):
        source = MANIFESTS / name
        if source.exists():
            shutil.copy2(source, backup_path / name)

    rows_by_video: dict[Path, list[dict]] = {}
    for row in rows:
        video_path = local_video_path(row["video_path"])
        if not FRAME_NUMBER.fullmatch(Path(row["frame_path"]).name):
            raise ValueError(f"Nama frame tidak dikenali: {row['frame_path']}")
        rows_by_video.setdefault(video_path, []).append(row)
    for video_path, video_rows in rows_by_video.items():
        restore_video_frames(video_path, video_rows)
    restored = len(rows)
    verified = 0
    for row in rows:
        video_path = local_video_path(row["video_path"])
        frame_path = FRAMES / video_path.stem / Path(row["frame_path"]).name
        row["video_path"] = str(video_path.relative_to(ROOT))
        row["frame_path"] = str(frame_path.relative_to(ROOT))
        row["frame_hash"] = str(int(row["frame_hash"]))
        # Hash manifest tetap sumber kebenaran; ini hanya cek bahwa file pulih cocok.
        image = cv2.imread(str(frame_path))
        if image is None:
            raise RuntimeError(f"Frame pulih tidak dapat dibaca: {frame_path}")
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        actual_hash = difference_hash(gray)
        if actual_hash == int(row["frame_hash"]):
            verified += 1

    if verified != len(rows):
        report = MANIFESTS / "frame_restore_report.txt"
        report.write_text(
            f"rows={len(rows)}\nrestored={restored}\nexisting={len(rows)-restored}\n"
            f"hash_matches={verified}\nhash_mismatches={len(rows)-verified}\n"
            f"backup={backup_path.relative_to(ROOT)}\nmanifest_rewritten=false\n",
            encoding="utf-8",
        )
        raise RuntimeError(
            f"Hash mismatch: {len(rows)-verified} frame; manifest CSV tetap tidak diubah. Laporan: {report}"
        )

    write_csv_atomic(
        dedup_path,
        ["video_path", "frame_path", "frame_hash", "is_duplicate"],
        rows,
    )
    write_csv_atomic(
        MANIFESTS / "frames_manifest.csv",
        ["video_path", "frame_path"],
        [[row["video_path"], row["frame_path"]] for row in rows],
    )
    report = MANIFESTS / "frame_restore_report.txt"
    report.write_text(
        f"rows={len(rows)}\nrestored={restored}\nexisting={len(rows)-restored}\nhash_matches={verified}\n"
        f"hash_mismatches={len(rows)-verified}\nbackup={backup_path.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    print(report)


if __name__ == "__main__":
    main()
