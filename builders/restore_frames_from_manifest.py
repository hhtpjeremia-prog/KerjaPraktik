#!/usr/bin/env python3
"""Pulihkan frame yang hilang dari video lokal tanpa menghitung ulang hash manifest."""

import csv
import re
import shutil
from datetime import datetime
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "output"
VIDEOS = OUTPUT / "videos"
FRAMES = OUTPUT / "frames"
MANIFESTS = OUTPUT / "manifests"
BACKUP = MANIFESTS / f"manifest_backup_{datetime.now():%Y%m%d_%H%M%S}"
FRAME_NUMBER = re.compile(r"frame_(\d+)\.png$")


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
    BACKUP.mkdir(parents=True, exist_ok=False)
    for name in ("frames_manifest.csv", "dedup_manifest.csv"):
        source = MANIFESTS / name
        if source.exists():
            shutil.copy2(source, BACKUP / name)

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
        resized = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(int)
        actual_hash = sum(1 << i for i, bit in enumerate((resized[:, 1:] >= resized[:, :-1]).flatten()) if bit)
        if actual_hash == int(row["frame_hash"]):
            verified += 1

    with dedup_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["video_path", "frame_path", "frame_hash", "is_duplicate"])
        writer.writeheader()
        writer.writerows(rows)
    with (MANIFESTS / "frames_manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_path", "frame_path"])
        writer.writerows([[row["video_path"], row["frame_path"]] for row in rows])
    report = MANIFESTS / "frame_restore_report.txt"
    report.write_text(
        f"rows={len(rows)}\nrestored={restored}\nexisting={len(rows)-restored}\nhash_matches={verified}\n"
        f"hash_mismatches={len(rows)-verified}\nbackup={BACKUP.relative_to(ROOT)}\n",
        encoding="utf-8",
    )
    if verified != len(rows):
        raise RuntimeError(f"Hash mismatch: {len(rows)-verified} frame; manifest tidak diubah hash-nya.")
    print(report)


if __name__ == "__main__":
    main()
