"""Refresh COCO and five-class YOLO exports from the current NB02b annotation store."""

from __future__ import annotations

import csv
import json
import shutil
from datetime import datetime
from pathlib import Path
import sys

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from pipeline_common import CLASS_NAMES, PENDING_CLASS_NAME, SPLIT_NAME_BY_CAMERA


ANNOTATION_ROOT = ROOT / "output" / "annotation"
STORE_PATH = ANNOTATION_ROOT / "annotations.json"
SAMPLE_MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"
YOLO_ROOT = ANNOTATION_ROOT / "yolo"
COCO_PATH = ANNOTATION_ROOT / "annotations_coco.json"


def main() -> None:
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(SAMPLE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if store.get("class_names") != CLASS_NAMES:
        raise ValueError(f"Taksonomi store tidak cocok: {store.get('class_names')}")
    records = [
        record
        for record in manifest.get("frames", [])
        if store.get("frames", {}).get(record["frame_stem"], {}).get("reviewed")
    ]
    if not records:
        raise ValueError("Tidak ada frame reviewed untuk diekspor.")

    class_counts = {name: 0 for name in CLASS_NAMES}
    coco_images = []
    coco_annotations = []
    split_rows = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    staging = ANNOTATION_ROOT / f".yolo_export_staging_{timestamp}"
    image_dir, label_dir = staging / "images", staging / "labels"
    image_dir.mkdir(parents=True)
    label_dir.mkdir(parents=True)
    annotation_id = 1

    try:
        for image_id, record in enumerate(records, start=1):
            stem = record["frame_stem"]
            camera_id = record["camera_id"]
            expected_split = SPLIT_NAME_BY_CAMERA.get(camera_id)
            if expected_split is None or record.get("split") != expected_split:
                raise ValueError(f"Split manifest tidak konsisten untuk {stem}: {record.get('split')}")
            entry = store["frames"][stem]
            source_image = Path(record["image_path"])
            if not source_image.is_absolute():
                source_image = ROOT / source_image
            if not source_image.is_file():
                raise FileNotFoundError(f"Citra reviewed tidak ditemukan: {source_image}")
            with Image.open(source_image) as image:
                width, height = image.size
            if (width, height) != (int(entry["image_width"]), int(entry["image_height"])):
                raise ValueError(f"Dimensi citra/store berbeda untuk {stem}: {(width, height)}")

            yolo_lines = []
            coco_images.append({
                "id": image_id,
                "file_name": f"{stem}.png",
                "width": width,
                "height": height,
                "camera_id": camera_id,
                "split": expected_split,
            })
            for box_entry in entry.get("boxes", []):
                if box_entry.get("deleted", False):
                    continue
                class_name = box_entry.get("class_name")
                if class_name == PENDING_CLASS_NAME:
                    raise ValueError(f"Frame reviewed masih memiliki kelas pending: {stem}")
                if class_name not in CLASS_NAMES:
                    raise ValueError(f"Kelas tidak dikenal di {stem}: {class_name}")
                x1, y1, x2, y2 = map(float, box_entry["box"])
                if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                    raise ValueError(f"BBox di luar citra pada {stem}: {(x1, y1, x2, y2)}")
                box_w, box_h = x2 - x1, y2 - y1
                class_index = CLASS_NAMES.index(class_name)
                yolo_box = ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height), box_w / width, box_h / height)
                yolo_lines.append(str(class_index) + " " + " ".join(f"{value:.6f}" for value in yolo_box))
                class_counts[class_name] += 1
                coco_annotations.append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_index + 1,
                    "bbox": [x1, y1, box_w, box_h],
                    "area": box_w * box_h,
                    "iscrowd": 0,
                    "source": box_entry.get("source", "unknown"),
                })
                annotation_id += 1

            shutil.copyfile(source_image, image_dir / f"{stem}.png")
            (label_dir / f"{stem}.txt").write_text("\n".join(yolo_lines) + ("\n" if yolo_lines else ""), encoding="utf-8")
            split_rows.append({"frame_stem": stem, "camera_id": camera_id, "split": expected_split})

        coco_payload = {
            "info": {
                "description": "NB02b reviewed annotations; five-class taxonomy",
                "reviewed_frames": len(records),
                "class_names": CLASS_NAMES,
                "generated_at": datetime.now().isoformat(),
            },
            "licenses": [],
            "images": coco_images,
            "annotations": coco_annotations,
            "categories": [
                {"id": index + 1, "name": name, "supercategory": "kondisi-lampu"}
                for index, name in enumerate(CLASS_NAMES)
            ],
        }
        (staging / "split_map.csv").write_text(
            "frame_stem,camera_id,split\n" + "".join(
                f"{row['frame_stem']},{row['camera_id']},{row['split']}\n" for row in split_rows
            ),
            encoding="utf-8",
        )
        export_summary = {
            "status": "complete",
            "completed_at": datetime.now().isoformat(),
            "class_names": CLASS_NAMES,
            "gold_images": len(coco_images),
            "gold_labels": len(coco_images),
            "gold_boxes": len(coco_annotations),
            "gold_boxes_by_class": class_counts,
        }
        (staging / "export_manifest.json").write_text(json.dumps(export_summary, indent=2), encoding="utf-8")

        coco_staging = ANNOTATION_ROOT / f".annotations_coco_staging_{timestamp}.json"
        coco_staging.write_text(json.dumps(coco_payload, ensure_ascii=False, indent=2), encoding="utf-8")
        backup = ANNOTATION_ROOT / f"_backup_refresh_exports_{timestamp}"
        backup.mkdir()
        if YOLO_ROOT.exists():
            YOLO_ROOT.replace(backup / "yolo")
        if COCO_PATH.exists():
            shutil.copy2(COCO_PATH, backup / "annotations_coco.json")
        try:
            staging.replace(YOLO_ROOT)
            coco_staging.replace(COCO_PATH)
        except Exception:
            if not YOLO_ROOT.exists() and (backup / "yolo").exists():
                (backup / "yolo").replace(YOLO_ROOT)
            raise
        print(json.dumps({**export_summary, "backup": str(backup)}, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()
