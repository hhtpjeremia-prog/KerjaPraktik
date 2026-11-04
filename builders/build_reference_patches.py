#!/usr/bin/env python3
"""Bangun set contoh patch silau/normal dari gold set yang sudah dianotasi."""

import json
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
STORE_PATH = PROJECT_ROOT / "output" / "annotation" / "annotations.json"
CROP_ROOT = PROJECT_ROOT / "output" / "nb02" / "crops"
REFERENCE_DIR = PROJECT_ROOT / "output" / "annotation" / "reference_patches"
PATCHES_PER_CLASS = 30
CLASS_GROUPS = (
    ("silau", "MobilSilau"),
    ("silau", "MotorSilau"),
    ("normal", "MobilNormal"),
    ("normal", "MotorNormal"),
)


def pick_boxes(class_name: str) -> list[dict]:
    store_payload = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    selected_by_frame: dict[str, dict] = {}
    for frame_stem, frame_entry in store_payload["frames"].items():
        if not frame_entry["reviewed"]:
            continue
        for box_entry in frame_entry["boxes"]:
            if box_entry["deleted"] or box_entry["class_name"] != class_name:
                continue
            if frame_stem in selected_by_frame:
                continue
            selected_by_frame[frame_stem] = {
                "frame_stem": frame_stem,
                "camera_id": frame_stem.split("_")[1],
                "class_name": class_name,
                "box": box_entry["box"],
                "source": box_entry["source"],
            }
    return sorted(selected_by_frame.values(), key=lambda row: row["frame_stem"])[:PATCHES_PER_CLASS]


def save_patch(candidate: dict) -> Path:
    image = cv2.imread(str(CROP_ROOT / f"{candidate['frame_stem']}.png"))
    if image is None:
        raise FileNotFoundError(str(CROP_ROOT / f"{candidate['frame_stem']}.png"))
    box = candidate["box"]
    patch = image[
        int(max(0, round(box[1]))):int(max(0, round(box[3]))),
        int(max(0, round(box[0]))):int(max(0, round(box[2]))),
    ]
    output_path = REFERENCE_DIR / f"{candidate['frame_stem']}_{candidate['class_name']}.png"
    cv2.imwrite(str(output_path), patch)
    return output_path


def main() -> None:
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    patches = []
    for group, class_name in CLASS_GROUPS:
        for candidate in pick_boxes(class_name):
            patches.append({
                "group": group,
                "frame_stem": candidate["frame_stem"],
                "camera_id": candidate["camera_id"],
                "class_name": candidate["class_name"],
                "source": candidate["source"],
                "patch_path": save_patch(candidate).as_posix(),
            })
    index_path = PROJECT_ROOT / "output" / "annotation" / "reference_patches_index.json"
    index_path.write_text(json.dumps(patches, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{len(patches)} patch -> {REFERENCE_DIR}")


if __name__ == "__main__":
    main()
