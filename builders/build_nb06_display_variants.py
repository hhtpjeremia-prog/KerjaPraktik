#!/usr/bin/env python3
"""Create dashboard display-only variants without rerunning YOLO/ByteTrack."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = ROOT / "output"
DATA_ROOT = OUTPUT_ROOT / "smoke" if os.environ.get("PIPELINE_SMOKE") == "1" else OUTPUT_ROOT
NB06 = DATA_ROOT / "nb06"
VIDEO_DIR = OUTPUT_ROOT / "videos"
TRACKING_PATH = NB06 / "tracking_detections.csv"
OVERLAY_MANIFEST_PATH = NB06 / "overlay_manifest.json"
EVENT_MANIFEST_PATH = NB06 / "event_clip_manifest.json"
RAW_NO_MASK_DIR = NB06 / "overlays_without_mask_raw"
BROWSER_ROOT = NB06 / "overlays_browser"
NO_MASK_BROWSER_DIR = BROWSER_ROOT / "without_mask"
MASKED_CROP_DIR = NB06 / "event_crops_with_mask"
FFMPEG = shutil.which("ffmpeg")
LABELS_ENABLED = os.environ.get("NB06_OVERLAY_LABELS", "1") == "1"


def resolve_manifest_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def write_json_atomic(path: Path, payload: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def draw_compact_label(frame: np.ndarray, text: str, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = min(0.38, max(0.25, 0.38 * max(x2 - x1, 1) / 120))
    thickness, pad = 1, 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    left = max(0, min(x1 + pad, frame.shape[1] - width - 2 * pad))
    top = max(0, min(y1 + pad, frame.shape[0] - height - baseline - 2 * pad))
    right, bottom = left + width + 2 * pad, top + height + baseline + 2 * pad
    cv2.rectangle(frame, (left, top), (right, bottom), (15, 15, 15), cv2.FILLED)
    cv2.putText(frame, text, (left + pad, bottom - baseline - pad), font, scale,
                (0, 0, 255), thickness, cv2.LINE_AA)


def build_no_mask_video(item: dict, tracking: pd.DataFrame) -> tuple[Path, Path]:
    video_stem = str(item["video_stem"])
    source = VIDEO_DIR / f"{video_stem}.mp4"
    if not source.exists():
        raise FileNotFoundError(f"Video sumber tidak ditemukan: {source}")
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Video sumber tidak bisa dibuka: {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS) or item.get("fps") or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_path = RAW_NO_MASK_DIR / f"{video_stem}_overlay_without_mask.mp4"
    writer = cv2.VideoWriter(str(raw_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"VideoWriter mp4v tidak bisa dibuat: {raw_path}")

    line = item.get("counting_line", {})
    y = int(line.get("y_pixel", round(height * 0.72)))
    x1 = int(line.get("x_start", 0))
    x2 = int(line.get("x_end", width - 1))
    rows_by_frame = {
        int(frame_number): rows.to_dict(orient="records")
        for frame_number, rows in tracking.groupby("frame_index", sort=False)
    }
    frame_index = written = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        cv2.line(frame, (x1, y), (x2, y), (0, 220, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, "GARIS HITUNG", (x1, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 220, 255), 1, cv2.LINE_AA)
        for row in rows_by_frame.get(frame_index, ()):
            if row["class_name"] not in {"MobilTidakStandar", "MotorTidakStandar"}:
                continue
            box = tuple(int(round(row[name])) for name in ("frame_x1", "frame_y1", "frame_x2", "frame_y2"))
            cv2.rectangle(frame, box[:2], box[2:], (0, 0, 255), 2)
            if LABELS_ENABLED:
                draw_compact_label(frame, f"ID {int(row['track_id'])} · {float(row['score']) * 100:.0f}%", box)
        writer.write(frame)
        frame_index += 1
        written += 1
        if written % 1000 == 0:
            print(f"{video_stem}: {written} frame no-mask dibuat", flush=True)
    capture.release()
    writer.release()
    if written == 0 or not raw_path.exists() or raw_path.stat().st_size == 0:
        raise RuntimeError(f"Overlay tanpa mask kosong: {raw_path}")

    if not FFMPEG:
        raise RuntimeError("ffmpeg tidak ditemukan di PATH untuk membuat video H.264.")
    NO_MASK_BROWSER_DIR.mkdir(parents=True, exist_ok=True)
    browser_path = NO_MASK_BROWSER_DIR / raw_path.name
    temporary = browser_path.with_name(f"{browser_path.stem}.tmp{browser_path.suffix}")
    result = subprocess.run([
        FFMPEG, "-y", "-i", str(raw_path), "-an", "-c:v", "libx264", "-preset", "fast",
        "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(temporary),
    ], capture_output=True, text=True)
    if result.returncode or not temporary.exists() or temporary.stat().st_size == 0:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"Konversi no-mask ke H.264 gagal: {result.stderr[-1500:]}")
    temporary.replace(browser_path)
    return raw_path, browser_path


def add_masked_gallery_crops(event_rows: list[dict]) -> None:
    vehicle_rows = [
        row for row in event_rows
        if row.get("class_name") in {"MobilTidakStandar", "MotorTidakStandar"}
    ]
    if not vehicle_rows:
        return
    import torch
    from PIL import Image
    from transformers import Sam2Model, Sam2Processor

    model_id = os.environ.get("NB06_SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = Sam2Processor.from_pretrained(model_id)
    model = Sam2Model.from_pretrained(model_id).to(device).eval()
    MASKED_CROP_DIR.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(vehicle_rows, start=1):
        source_crop = resolve_manifest_path(str(row.get("crop_path", "")))
        if not source_crop.exists():
            continue
        with Image.open(source_crop) as image_handle:
            image = image_handle.convert("RGB")
        width, height = image.size
        inputs = processor(
            images=image,
            input_boxes=[[[0.0, 0.0, float(width - 1), float(height - 1)]]],
            return_tensors="pt",
        ).to(device)
        with torch.inference_mode():
            outputs = model(**inputs)
        masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
        quality = outputs.iou_scores[0].detach().cpu()
        mask = masks[0, int(quality.argmax())].numpy().astype(bool)
        crop_array = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        color = np.array([0, 0, 255], dtype=np.float32)
        crop_array[mask] = np.clip(crop_array[mask].astype(np.float32) * 0.7 + color * 0.3, 0, 255).astype(np.uint8)
        target_dir = MASKED_CROP_DIR / str(row["class_name"])
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / f"{source_crop.stem}_with_mask.jpg"
        if not cv2.imwrite(str(target), crop_array):
            raise RuntimeError(f"Gagal menyimpan crop bermask: {target}")
        row["crop_path_without_mask"] = str(source_crop)
        row["crop_path_with_mask"] = str(target)
        if index % 50 == 0:
            print(f"Crop bermask {index}/{len(vehicle_rows)}", flush=True)
    del model


def main() -> None:
    if not TRACKING_PATH.exists() or not OVERLAY_MANIFEST_PATH.exists() or not EVENT_MANIFEST_PATH.exists():
        raise FileNotFoundError("NB06 harus selesai sebelum membuat varian tampilan dashboard.")
    overlay_rows = json.loads(OVERLAY_MANIFEST_PATH.read_text(encoding="utf-8"))
    event_rows = json.loads(EVENT_MANIFEST_PATH.read_text(encoding="utf-8"))
    tracking_frame = pd.read_csv(TRACKING_PATH)
    RAW_NO_MASK_DIR.mkdir(parents=True, exist_ok=True)
    for item in overlay_rows:
        subset = tracking_frame[tracking_frame["video_stem"] == item["video_stem"]]
        raw_path, browser_path = build_no_mask_video(item, subset)
        masked_path = resolve_manifest_path(str(item.get("overlay_path", "")))
        item["overlay_path_with_mask"] = str(masked_path)
        item["overlay_path_without_mask"] = str(browser_path)
        item["overlay_path_without_mask_raw"] = str(raw_path)
        item["overlay_variants"] = {"with_mask": str(masked_path), "without_mask": str(browser_path)}
    add_masked_gallery_crops(event_rows)
    write_json_atomic(OVERLAY_MANIFEST_PATH, overlay_rows)
    write_json_atomic(EVENT_MANIFEST_PATH, event_rows)
    print(f"Varian tampilan dashboard selesai: {len(overlay_rows)} video; {len(event_rows)} event.")


if __name__ == "__main__":
    main()
