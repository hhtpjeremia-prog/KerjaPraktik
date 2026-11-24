#!/usr/bin/env python3
"""Safely resync stale annotation image dimensions and rebuild YOLO/COCO exports."""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path

from PIL import Image

ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT))
from pipeline_common import TRAINING_CLASS_NAMES, training_class_name


ANNOTATION_ROOT = ROOT / "output" / "annotation"
YOLO_ROOT = ANNOTATION_ROOT / "yolo"
STORE_PATH = ANNOTATION_ROOT / "annotations.json"
MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"
COCO_PATH = ANNOTATION_ROOT / "annotations_coco.json"
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
BACKUP_ROOT = ANNOTATION_ROOT / f"_backup_dimension_repair_{TIMESTAMP}"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_json(path: Path, payload: dict) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    manifest = read_json(MANIFEST_PATH)
    store = read_json(STORE_PATH)
    coco = read_json(COCO_PATH)
    records = {record["frame_stem"]: record for record in manifest["frames"]}
    store_frames = store["frames"]

    updates: list[tuple[str, tuple[int, int], tuple[int, int]]] = []
    image_sources: dict[str, Path] = {}
    for stem, record in records.items():
        source = Path(record["image_path"])
        if not source.is_absolute():
            source = ROOT / source
        if not source.exists():
            raise FileNotFoundError(f"gambar manifest hilang: {stem}: {source}")
        with Image.open(source) as image:
            actual_size = image.size
        manifest_size = (int(record["image_width"]), int(record["image_height"]))
        if actual_size != manifest_size:
            raise ValueError(f"dimensi manifest/gambar tidak cocok: {stem}: {manifest_size} != {actual_size}")
        image_sources[stem] = source
        entry = store_frames.get(stem)
        if entry is None:
            raise ValueError(f"entry anotasi hilang untuk {stem}")
        old_size = (int(entry["image_width"]), int(entry["image_height"]))
        for index, box_entry in enumerate(entry.get("boxes", [])):
            if box_entry.get("deleted"):
                continue
            x1, y1, x2, y2 = map(float, box_entry["box"])
            if x1 < 0 or y1 < 0 or x2 <= x1 or y2 <= y1 or x2 > actual_size[0] or y2 > actual_size[1]:
                raise ValueError(
                    f"box {index} di luar gambar aktif untuk {stem}: "
                    f"box={(x1, y1, x2, y2)}, image_size={actual_size}; tidak ada koordinat yang diubah."
                )
        if old_size != actual_size:
            updates.append((stem, old_size, actual_size))

    if not updates:
        export_manifest_path = YOLO_ROOT / "export_manifest.json"
        images = list((YOLO_ROOT / "images").glob("*.png"))
        labels = list((YOLO_ROOT / "labels").glob("*.txt"))
        if len(images) != len(records) or len(labels) != len(records):
            raise ValueError("Export YOLO saat ini tidak lengkap; jalankan NB02b untuk membuat ulang export.")
        export_manifest_path.write_text(json.dumps({
            "status": "complete",
            "completed_at": datetime.now().isoformat(),
            "gold_images": len(images),
            "gold_labels": len(labels),
            "gold_boxes": sum(
                1 for label_path in labels for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()
            ),
            "gold_boxes_by_class": {
                name: sum(
                    1 for label_path in labels for line in label_path.read_text(encoding="utf-8").splitlines()
                    if line.strip() and int(line.split()[0]) == index
                )
                for index, name in enumerate(TRAINING_CLASS_NAMES)
            },
        }, indent=2), encoding="utf-8")
        print("Tidak ada metadata dimensi yang perlu diperbaiki.")
        return

    BACKUP_ROOT.mkdir(parents=True, exist_ok=False)
    shutil.copy2(STORE_PATH, BACKUP_ROOT / STORE_PATH.name)
    shutil.copy2(COCO_PATH, BACKUP_ROOT / COCO_PATH.name)
    shutil.copytree(YOLO_ROOT, BACKUP_ROOT / "yolo")

    for stem, _, (width, height) in updates:
        store_frames[stem]["image_width"] = width
        store_frames[stem]["image_height"] = height

    coco_image_by_stem = {Path(image["file_name"]).stem: image for image in coco.get("images", [])}
    for stem, record in records.items():
        image = coco_image_by_stem.get(stem)
        if image is None:
            raise ValueError(f"gambar COCO tidak ditemukan untuk {stem}")
        image["width"] = int(record["image_width"])
        image["height"] = int(record["image_height"])

    label_dir = YOLO_ROOT / "labels"
    image_dir = YOLO_ROOT / "images"
    label_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)
    box_total = 0
    class_counts = {name: 0 for name in TRAINING_CLASS_NAMES}
    split_rows = ["frame_stem,camera_id,split"]
    for stem, record in records.items():
        entry = store_frames[stem]
        width, height = int(record["image_width"]), int(record["image_height"])
        lines: list[str] = []
        for box_entry in entry.get("boxes", []):
            if box_entry.get("deleted"):
                continue
            training_name = training_class_name(box_entry["class_name"])
            if training_name is None:
                continue
            class_id = TRAINING_CLASS_NAMES.index(training_name)
            x1, y1, x2, y2 = map(float, box_entry["box"])
            normalized = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), (x2 - x1) / width, (y2 - y1) / height)
            if any(value < 0 or value > 1 for value in normalized):
                raise ValueError(f"hasil normalisasi di luar [0,1] untuk {stem}: {normalized}")
            lines.append(f"{class_id} " + " ".join(f"{value:.6f}" for value in normalized))
            class_counts[training_name] += 1
            box_total += 1
        shutil.copy2(image_sources[stem], image_dir / f"{stem}.png")
        (label_dir / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        split_rows.append(f"{stem},{record['camera_id']},{record['split']}")

    atomic_write_json(STORE_PATH, store)
    atomic_write_json(COCO_PATH, coco)
    (YOLO_ROOT / "split_map.csv").write_text("\n".join(split_rows) + "\n", encoding="utf-8")
    print(json.dumps({
        "updated_frames": len(updates),
        "dimension_changes_sample": [{"frame_stem": stem, "from": old, "to": new} for stem, old, new in updates[:5]],
        "dimension_change_groups": {
            f"{old[0]}x{old[1]} -> {new[0]}x{new[1]}": sum(1 for _, previous, current in updates if previous == old and current == new)
            for _, old, new in updates
        },
        "yolo_images": len(records),
        "yolo_boxes": box_total,
        "yolo_boxes_by_class": class_counts,
        "backup": str(BACKUP_ROOT),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
