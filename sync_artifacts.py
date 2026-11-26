"""Salin artefak ringan (~13 MB) secara read-only dari output/ ke deploy_streamlit/artifacts/
dan ubah semua path absolut Windows menjadi path relatif POSIX untuk Linux Streamlit Cloud."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DEPLOY_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = DEPLOY_ROOT.parent
SRC_OUTPUT = PROJECT_ROOT / "output"
DST_ARTIFACTS = DEPLOY_ROOT / "artifacts"


def _copy_file(src: Path, dst: Path) -> None:
    if not src.is_file():
        print(f"[SKIP] tidak ditemukan: {src}")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _copy_tree(src: Path, dst: Path) -> int:
    if not src.is_dir():
        print(f"[SKIP] folder tidak ditemukan: {src}")
        return 0
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst)
    return sum(1 for p in dst.rglob("*") if p.is_file())


def _extract_overlay_relpath(raw: str | None) -> str | None:
    if not raw:
        return None
    norm = str(raw).replace("\\", "/")
    marker = "overlays_browser/"
    idx = norm.lower().find(marker)
    if idx != -1:
        return norm[idx + len(marker) :]
    return Path(norm).name


def _extract_crop_relpath(raw: str | None) -> str | None:
    if not raw:
        return None
    norm = str(raw).replace("\\", "/")
    for marker in ("event_crops_with_mask/", "event_crops/"):
        idx = norm.lower().find(marker)
        if idx != -1:
            return f"artifacts/nb06/{norm[idx:]}"
    return None


def main() -> None:
    nb06_src = SRC_OUTPUT / "nb06"
    nb06_dst = DST_ARTIFACTS / "nb06"
    nb06_dst.mkdir(parents=True, exist_ok=True)

    # 1. File ringan NB06
    for fname in (
        "counting_summary.json",
        "counting_by_video.csv",
        "counting_timeline.csv",
    ):
        _copy_file(nb06_src / fname, nb06_dst / fname)

    # 2. Normalisasi overlay_manifest.json -> POSIX relpath untuk HF Dataset & fallback lokal
    overlay_src = nb06_src / "overlay_manifest.json"
    if overlay_src.is_file():
        items = json.loads(overlay_src.read_text(encoding="utf-8"))
        cleaned = []
        for item in items:
            with_rel = _extract_overlay_relpath(
                item.get("overlay_path_with_mask") or item.get("overlay_path")
            )
            without_rel = _extract_overlay_relpath(item.get("overlay_path_without_mask"))
            cleaned.append(
                {
                    "video_stem": item["video_stem"],
                    "camera_id": item["camera_id"],
                    "split": item["split"],
                    "fps": item.get("fps", 25.0),
                    "frames_processed": item.get("frames_processed", 0),
                    "duration_seconds": item.get("duration_seconds", 0.0),
                    "counting_line": item.get("counting_line", {}),
                    "segmentation_overlay": item.get("segmentation_overlay", {}),
                    "codec": item.get("codec", "H.264 (libx264)"),
                    "video_relpath_with_mask": with_rel,
                    "video_relpath_without_mask": without_rel,
                    "overlay_path_with_mask": f"overlays_browser/{with_rel}" if with_rel else None,
                    "overlay_path_without_mask": f"overlays_browser/{without_rel}" if without_rel else None,
                }
            )
        (nb06_dst / "overlay_manifest.json").write_text(
            json.dumps(cleaned, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 3. Normalisasi event_clip_manifest.json -> hapus path clip 12GB, simpan path crop relatif POSIX
    event_src = nb06_src / "event_clip_manifest.json"
    if event_src.is_file():
        events = json.loads(event_src.read_text(encoding="utf-8"))
        cleaned_events = []
        for ev in events:
            if ev.get("class_name") not in {"MobilTidakStandar", "MotorTidakStandar"}:
                continue
            crop_with = _extract_crop_relpath(ev.get("crop_path_with_mask"))
            crop_without = _extract_crop_relpath(
                ev.get("crop_path_without_mask") or ev.get("crop_path")
            )
            cleaned_events.append(
                {
                    "video_stem": ev.get("video_stem"),
                    "camera_id": ev.get("camera_id"),
                    "vehicle_id": ev.get("vehicle_id"),
                    "class_name": ev.get("class_name"),
                    "timestamp_seconds": ev.get("timestamp_seconds", 0.0),
                    "first_frame": ev.get("first_frame", 0),
                    "last_frame": ev.get("last_frame", 0),
                    "mean_score": ev.get("mean_score", 0.0),
                    "crop_path": crop_without,
                    "crop_path_with_mask": crop_with,
                    "crop_path_without_mask": crop_without,
                }
            )
        (nb06_dst / "event_clip_manifest.json").write_text(
            json.dumps(cleaned_events, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    # 4. Salin folder event_crops & event_crops_with_mask (~5.8 MB)
    n1 = _copy_tree(nb06_src / "event_crops", nb06_dst / "event_crops")
    n2 = _copy_tree(nb06_src / "event_crops_with_mask", nb06_dst / "event_crops_with_mask")

    # 5. Salin metrik validasi & holdout test (< 1 MB)
    _copy_file(SRC_OUTPUT / "nb04" / "val_metrics.json", DST_ARTIFACTS / "nb04" / "val_metrics.json")
    runs_dir = SRC_OUTPUT / "nb04" / "runs"
    if runs_dir.is_dir():
        results_csvs = sorted(runs_dir.glob("*/results.csv"))
        if results_csvs:
            latest = results_csvs[-1]
            _copy_file(latest, DST_ARTIFACTS / "nb04" / "runs" / latest.parent.name / "results.csv")
    _copy_file(SRC_OUTPUT / "nb05" / "test_metrics.json", DST_ARTIFACTS / "nb05" / "test_metrics.json")
    _copy_file(
        SRC_OUTPUT / "ground_truth" / "annotation_summary.json",
        DST_ARTIFACTS / "ground_truth" / "annotation_summary.json",
    )
    _copy_file(
        SRC_OUTPUT / "attribute_classifier" / "training_history.csv",
        DST_ARTIFACTS / "attribute_classifier" / "training_history.csv",
    )
    _copy_file(
        SRC_OUTPUT / "attribute_classifier" / "metrics.json",
        DST_ARTIFACTS / "attribute_classifier" / "metrics.json",
    )

    total_bytes = sum(p.stat().st_size for p in DST_ARTIFACTS.rglob("*") if p.is_file())
    print(f"Copied crops: {n1} + {n2}")
    print(f"Total deploy_streamlit/artifacts size: {total_bytes / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    main()
