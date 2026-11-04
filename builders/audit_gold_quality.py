#!/usr/bin/env python3
"""Audit kualitas gold set: box duplikat, box terlalu kecil, refleksi tumpang tindih, dan proxy recall proposal DINO."""

import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = PROJECT_ROOT / "output" / "annotation" / "annotations.json"
MANIFEST_PATH = PROJECT_ROOT / "output" / "annotation" / "sample_manifest.json"
AUDIT_CSV_PATH = PROJECT_ROOT / "output" / "annotation" / "gold_quality_audit.csv"
AUDIT_SUMMARY_PATH = PROJECT_ROOT / "output" / "annotation" / "gold_quality_summary.json"

DUPLICATE_IOU_THRESHOLD = 0.85
REFLECTION_OVERLAP_IOU = 0.30
MIN_LEARNABLE_BOX_SIDE_PIXELS = 8.0
BORDER_TOLERANCE_PIXELS = 1.0
PSEUDO_SOURCES = ("pseudo", "edited")
MANUAL_SOURCE = "manual"
REFLECTION_CLASS_NAME = "PantulanCahaya"
PENDING_CLASS_NAME = "BelumDitentukan"


def box_iou(first_box: list[float], second_box: list[float]) -> float:
    intersection_width = min(first_box[2], second_box[2]) - max(first_box[0], second_box[0])
    intersection_height = min(first_box[3], second_box[3]) - max(first_box[1], second_box[1])
    if intersection_width <= 0 or intersection_height <= 0:
        return 0.0
    intersection_area = intersection_width * intersection_height
    first_area = (first_box[2] - first_box[0]) * (first_box[3] - first_box[1])
    second_area = (second_box[2] - second_box[0]) * (second_box[3] - second_box[1])
    union_area = first_area + second_area - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def load_payloads() -> tuple[dict, dict]:
    store_payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    manifest_payload = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return store_payload, manifest_payload


def reviewed_frame_entries(store_payload: dict) -> dict[str, dict]:
    return {
        frame_stem: frame_entry
        for frame_stem, frame_entry in store_payload["frames"].items()
        if frame_entry["reviewed"]
    }


def active_boxes(frame_entry: dict) -> list[dict]:
    return [box_entry for box_entry in frame_entry["boxes"] if not box_entry["deleted"]]


def count_duplicate_pairs(boxes: list[dict]) -> int:
    duplicate_count = 0
    for first_index, first_box in enumerate(boxes):
        for second_box in boxes[first_index + 1:]:
            if box_iou(first_box["box"], second_box["box"]) >= DUPLICATE_IOU_THRESHOLD:
                duplicate_count += 1
    return duplicate_count


def count_tiny_boxes(boxes: list[dict]) -> int:
    return sum(
        1 for box_entry in boxes
        if min(box_entry["box"][2] - box_entry["box"][0], box_entry["box"][3] - box_entry["box"][1]) < MIN_LEARNABLE_BOX_SIDE_PIXELS
    )


def count_border_boxes(boxes: list[dict], frame_entry: dict) -> int:
    image_width = frame_entry["image_width"]
    image_height = frame_entry["image_height"]
    border_count = 0
    for box_entry in boxes:
        box_x1, box_y1, box_x2, box_y2 = box_entry["box"]
        touches_border = (
            box_x1 <= BORDER_TOLERANCE_PIXELS
            or box_y1 <= BORDER_TOLERANCE_PIXELS
            or box_x2 >= image_width - BORDER_TOLERANCE_PIXELS
            or box_y2 >= image_height - BORDER_TOLERANCE_PIXELS
        )
        border_count += 1 if touches_border else 0
    return border_count


def count_reflection_overlaps(boxes: list[dict]) -> int:
    reflection_boxes = [box_entry for box_entry in boxes if box_entry["class_name"] == REFLECTION_CLASS_NAME]
    vehicle_boxes = [box_entry for box_entry in boxes if box_entry["class_name"] not in (REFLECTION_CLASS_NAME, PENDING_CLASS_NAME)]
    overlap_count = 0
    for reflection_box in reflection_boxes:
        if any(box_iou(reflection_box["box"], vehicle_box["box"]) >= REFLECTION_OVERLAP_IOU for vehicle_box in vehicle_boxes):
            overlap_count += 1
    return overlap_count


def build_frame_rows(reviewed_frames: dict[str, dict], manifest_payload: dict) -> list[dict]:
    manifest_split_by_frame = {record["frame_stem"]: record["split"] for record in manifest_payload["frames"]}
    manifest_camera_by_frame = {record["frame_stem"]: record["camera_id"] for record in manifest_payload["frames"]}
    frame_rows = []
    for frame_stem in sorted(reviewed_frames):
        frame_entry = reviewed_frames[frame_stem]
        boxes = active_boxes(frame_entry)
        frame_rows.append({
            "frame_stem": frame_stem,
            "camera_id": manifest_camera_by_frame.get(frame_stem, ""),
            "split": manifest_split_by_frame.get(frame_stem, ""),
            "active_boxes": len(boxes),
            "pending_boxes": sum(1 for box_entry in boxes if box_entry["class_name"] == PENDING_CLASS_NAME),
            "duplicate_pairs": count_duplicate_pairs(boxes),
            "tiny_boxes": count_tiny_boxes(boxes),
            "border_boxes": count_border_boxes(boxes, frame_entry),
            "reflection_overlaps": count_reflection_overlaps(boxes),
        })
    return frame_rows


def compute_proposal_recall(reviewed_frames: dict[str, dict]) -> dict:
    pseudo_derived = 0
    manual_vehicle = 0
    deleted_pseudo = 0
    for frame_entry in reviewed_frames.values():
        for box_entry in frame_entry["boxes"]:
            is_reflection = box_entry["class_name"] == REFLECTION_CLASS_NAME
            if box_entry["deleted"]:
                deleted_pseudo += 1 if box_entry["source"] in PSEUDO_SOURCES else 0
                continue
            if is_reflection:
                continue
            if box_entry["source"] in PSEUDO_SOURCES:
                pseudo_derived += 1
            elif box_entry["source"] == MANUAL_SOURCE:
                manual_vehicle += 1
    proposal_total = pseudo_derived + manual_vehicle
    return {
        "pseudo_derived_vehicle_boxes": pseudo_derived,
        "manual_vehicle_boxes": manual_vehicle,
        "deleted_pseudo_boxes": deleted_pseudo,
        "proposal_recall_proxy": round(pseudo_derived / proposal_total, 4) if proposal_total else 0.0,
        "note": (
            "manual_vehicle_boxes dihitung sebagai kendaraan yang tidak dipropose DINO. "
            "Box manual yang sebenarnya menggambar ulang box pseudo yang dihapus ikut terhitung miss, "
            "jadi angka ini cenderung sedikit lebih rendah dari recall sebenarnya."
        ),
    }


def write_csv(frame_rows: list[dict]) -> None:
    field_names = list(frame_rows[0]) if frame_rows else []
    with AUDIT_CSV_PATH.open("w", encoding="utf-8", newline="") as audit_file:
        audit_file.write(",".join(field_names) + "\n")
        for frame_row in frame_rows:
            audit_file.write(",".join(str(frame_row[field_name]) for field_name in field_names) + "\n")


def summarise(frame_rows: list[dict], reviewed_frames: dict[str, dict]) -> dict:
    class_totals: dict[str, int] = {}
    for frame_entry in reviewed_frames.values():
        for box_entry in active_boxes(frame_entry):
            class_totals[box_entry["class_name"]] = class_totals.get(box_entry["class_name"], 0) + 1
    split_totals: dict[str, int] = {}
    for frame_row in frame_rows:
        split_name = frame_row["split"] or "tidak diketahui"
        split_totals[split_name] = split_totals.get(split_name, 0) + 1
    return {
        "reviewed_frames": len(frame_rows),
        "reviewed_frames_by_split": split_totals,
        "active_boxes": sum(frame_row["active_boxes"] for frame_row in frame_rows),
        "frames_with_duplicate_pairs": sum(1 for frame_row in frame_rows if frame_row["duplicate_pairs"] > 0),
        "duplicate_pairs": sum(frame_row["duplicate_pairs"] for frame_row in frame_rows),
        "tiny_boxes": sum(frame_row["tiny_boxes"] for frame_row in frame_rows),
        "border_boxes": sum(frame_row["border_boxes"] for frame_row in frame_rows),
        "reflection_overlaps": sum(frame_row["reflection_overlaps"] for frame_row in frame_rows),
        "class_totals": dict(sorted(class_totals.items(), key=lambda item: -item[1])),
        "thresholds": {
            "duplicate_iou": DUPLICATE_IOU_THRESHOLD,
            "reflection_overlap_iou": REFLECTION_OVERLAP_IOU,
            "min_learnable_box_side_pixels": MIN_LEARNABLE_BOX_SIDE_PIXELS,
            "border_tolerance_pixels": BORDER_TOLERANCE_PIXELS,
        },
        **compute_proposal_recall(reviewed_frames),
    }


def main() -> None:
    store_payload, manifest_payload = load_payloads()
    reviewed_frames = reviewed_frame_entries(store_payload)
    if not reviewed_frames:
        raise ValueError("belum ada frame reviewed; audit tidak bisa dijalankan")
    frame_rows = build_frame_rows(reviewed_frames, manifest_payload)
    summary_payload = summarise(frame_rows, reviewed_frames)
    write_csv(frame_rows)
    AUDIT_SUMMARY_PATH.write_text(json.dumps(summary_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"frame reviewed      : {summary_payload['reviewed_frames']} ({summary_payload['reviewed_frames_by_split']})")
    print(f"box aktif           : {summary_payload['active_boxes']}")
    print(f"pasangan duplikat   : {summary_payload['duplicate_pairs']} pada {summary_payload['frames_with_duplicate_pairs']} frame")
    print(f"box < {MIN_LEARNABLE_BOX_SIDE_PIXELS:.0f} px          : {summary_payload['tiny_boxes']}")
    print(f"box menyentuh tepi  : {summary_payload['border_boxes']}")
    print(f"refleksi tumpang tindih kendaraan: {summary_payload['reflection_overlaps']}")
    print(f"proxy recall DINO   : {summary_payload['proposal_recall_proxy']:.3f} "
          f"({summary_payload['pseudo_derived_vehicle_boxes']} dipropose vs {summary_payload['manual_vehicle_boxes']} manual)")
    print(f"CSV                 : {AUDIT_CSV_PATH}")
    print(f"summary             : {AUDIT_SUMMARY_PATH}")


if __name__ == "__main__":
    main()
