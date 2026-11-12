#!/usr/bin/env python3
"""Repair holdout GT dimensions and rebuild its three-class detection export."""

from __future__ import annotations

import csv
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_common import TRAINING_CLASS_NAMES, training_class_name


GT_ROOT = ROOT / "output" / "ground_truth"
YOLO_ROOT = GT_ROOT / "yolo"
STORE_PATH = GT_ROOT / "annotations.json"
MANIFEST_PATH = GT_ROOT / "sample_manifest.json"
COCO_PATH = GT_ROOT / "annotations_coco.json"
SUMMARY_PATH = GT_ROOT / "annotation_summary.json"
STAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_ROOT = GT_ROOT / f"_backup_yolo_repair_{STAMP}"
STAGING_ROOT = GT_ROOT / f".yolo_repair_staging_{STAMP}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    store = read_json(STORE_PATH)
    coco = read_json(COCO_PATH)
    summary = read_json(SUMMARY_PATH)
    records = manifest["frames"]
    if not records:
        raise ValueError("manifest GT kosong")
    if TRAINING_CLASS_NAMES != ["Mobil", "Motor", "PantulanCahaya"]:
        raise ValueError(f"kontrak kelas detector berubah: {TRAINING_CLASS_NAMES}")

    coco_images = {Path(item["file_name"]).stem: item for item in coco.get("images", [])}
    seen: set[str] = set()
    dimensions_updated = 0
    box_counts = {name: 0 for name in TRAINING_CLASS_NAMES}
    reflection_boxes = 0
    split_rows = ["frame_stem,camera_id,split"]
    image_dir = STAGING_ROOT / "images"
    label_dir = STAGING_ROOT / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)

    # Validate every source and every active box before taking a backup or writing changes.
    prepared: list[tuple[dict, dict, Path, int, int]] = []
    for record in records:
        stem = record["frame_stem"]
        if stem in seen:
            raise ValueError(f"frame duplikat di manifest: {stem}")
        seen.add(stem)
        camera = record["camera_id"]
        split = store.get("split_by_camera", {}).get(camera, record.get("split", "unknown"))
        if split != "test" or record.get("split") != "test":
            raise ValueError(f"GT holdout memuat split selain test: {stem}: {split}")
        image_source = Path(record["image_path"])
        if not image_source.is_absolute():
            image_source = ROOT / image_source
        if not image_source.exists():
            raise FileNotFoundError(f"gambar sumber hilang: {stem}: {image_source}")
        with Image.open(image_source) as image:
            width, height = image.size
        manifest_size = (int(record["image_width"]), int(record["image_height"]))
        if manifest_size != (width, height):
            raise ValueError(f"dimensi manifest/gambar berbeda untuk {stem}: {manifest_size} != {(width, height)}")
        entry = store["frames"].get(stem)
        if entry is None or not entry.get("reviewed"):
            raise ValueError(f"frame tidak memiliki anotasi GT yang direview: {stem}")
        for index, box in enumerate(entry.get("boxes", [])):
            if box.get("deleted"):
                continue
            x1, y1, x2, y2 = map(float, box["box"])
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > width or y2 > height:
                raise ValueError(f"bbox di luar gambar aktif: {stem} box {index}: {box['box']} vs {(width, height)}")
            mapped_class = training_class_name(box["class_name"])
            if mapped_class is None:
                raise ValueError(f"kelas anotasi test tidak dikenal/belum selesai: {stem} box {index}: {box['class_name']}")
            box_counts[mapped_class] += 1
            if mapped_class == "PantulanCahaya":
                reflection_boxes += 1
        prepared.append((record, entry, image_source, width, height))

    if len(coco_images) != len(records):
        raise ValueError(f"jumlah gambar COCO {len(coco_images)} tidak cocok dengan manifest {len(records)}")

    for record, entry, source, width, height in prepared:
        stem = record["frame_stem"]
        if (entry.get("image_width"), entry.get("image_height")) != (width, height):
            dimensions_updated += 1
        entry["image_width"], entry["image_height"] = width, height
        coco_images[stem]["width"], coco_images[stem]["height"] = width, height
        lines = []
        for box in entry.get("boxes", []):
            if box.get("deleted"):
                continue
            class_name = training_class_name(box["class_name"])
            if class_name is None:
                raise ValueError(f"kelas anotasi test tidak dikenal/belum selesai: {stem}: {box['class_name']}")
            x1, y1, x2, y2 = map(float, box["box"])
            values = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
            if not (0 <= values[0] <= 1 and 0 <= values[1] <= 1 and 0 < values[2] <= 1 and 0 < values[3] <= 1
                    and values[0] - values[2] / 2 >= -1e-9 and values[1] - values[3] / 2 >= -1e-9
                    and values[0] + values[2] / 2 <= 1 + 1e-9 and values[1] + values[3] / 2 <= 1 + 1e-9):
                raise ValueError(f"koordinat YOLO invalid setelah normalisasi: {stem}: {values}")
            class_id = TRAINING_CLASS_NAMES.index(class_name)
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in values))
        shutil.copy2(source, image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_rows.append(f"{stem},{record['camera_id']},test")

    image_stems = {p.stem for p in image_dir.glob("*.png")}
    label_stems = {p.stem for p in label_dir.glob("*.txt")}
    if image_stems != label_stems or len(image_stems) != len(records):
        raise ValueError("staging export GT tidak lengkap")
    (STAGING_ROOT / "split_map.csv").write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    export_manifest = {
        "status": "complete",
        "completed_at": datetime.now().isoformat(),
        "split": "test",
        "images": len(image_stems),
        "labels": len(label_stems),
        "object_boxes": sum(box_counts.values()),
        "object_boxes_by_class": box_counts,
        "reflection_boxes_as_explicit_detector_target": reflection_boxes,
        "class_names": TRAINING_CLASS_NAMES,
    }
    (STAGING_ROOT / "export_manifest.json").write_text(json.dumps(export_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    BACKUP_ROOT.mkdir(parents=True, exist_ok=False)
    for path in (STORE_PATH, COCO_PATH, SUMMARY_PATH):
        shutil.copy2(path, BACKUP_ROOT / path.name)
    if YOLO_ROOT.exists():
        YOLO_ROOT.rename(BACKUP_ROOT / "yolo")
    try:
        STAGING_ROOT.rename(YOLO_ROOT)
    except Exception:
        if (BACKUP_ROOT / "yolo").exists() and not YOLO_ROOT.exists():
            (BACKUP_ROOT / "yolo").rename(YOLO_ROOT)
        raise

    summary["gold_images"] = len(image_stems)
    summary["gold_labels"] = len(label_stems)
    summary["gold_boxes"] = sum(box_counts.values())
    summary["gold_boxes_by_class"] = box_counts
    summary["detector_gt_reflection_target_boxes"] = reflection_boxes
    atomic_json(STORE_PATH, store)
    atomic_json(COCO_PATH, coco)
    atomic_json(SUMMARY_PATH, summary)
    print(json.dumps({
        **export_manifest,
        "dimension_metadata_updated": dimensions_updated,
        "backup": str(BACKUP_ROOT),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
