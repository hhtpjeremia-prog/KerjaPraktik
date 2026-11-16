"""Migrate the active NB02b store from the 9-class taxonomy to 4 vehicle events."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
ANNOTATION_ROOT = ROOT / "output" / "annotation"
STORE_PATH = ANNOTATION_ROOT / "annotations.json"
MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"

CLASS_MAP = {
    "MobilNormal": "MobilStandar",
    "MobilSilau": "MobilTidakStandar",
    "MobilBerwarna": "MobilTidakStandar",
    "MobilSilauBerwarna": "MobilTidakStandar",
    "MotorNormal": "MotorStandar",
    "MotorSilau": "MotorTidakStandar",
    "MotorBerwarna": "MotorTidakStandar",
    "MotorSilauBerwarna": "MotorTidakStandar",
    "PantulanCahaya": "PantulanCahaya",
}
ACTIVE_CLASS_NAMES = [
    "MobilStandar",
    "MobilTidakStandar",
    "MotorStandar",
    "MotorTidakStandar",
    "PantulanCahaya",
]


def backup(path: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    target = path.with_name(f"{path.stem}_before_taxonomy_{stamp}{path.suffix}")
    target.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    return target


def main() -> None:
    if not STORE_PATH.exists():
        raise FileNotFoundError(STORE_PATH)
    store = json.loads(STORE_PATH.read_text(encoding="utf-8"))
    if store.get("class_names") == ACTIVE_CLASS_NAMES:
        print("annotations.json sudah memakai taksonomi baru; tidak ada perubahan.")
        return

    store_backup = backup(STORE_PATH)
    changed = 0
    for frame in store.get("frames", {}).values():
        for box in frame.get("boxes", []):
            if box.get("deleted"):
                continue
            old_name = box.get("class_name")
            if old_name == "BelumDitentukan":
                # Pending boxes harus tetap pending agar NB02b tetap memaksa anotasi selesai.
                continue
            if old_name not in CLASS_MAP:
                raise ValueError(f"kelas lama tidak dikenal: {old_name}")
            new_name = CLASS_MAP[old_name]
            if old_name != new_name:
                box["class_name"] = new_name
                changed += 1
    store["class_names"] = ACTIVE_CLASS_NAMES
    store["taxonomy"] = "vehicle_type_x_lamp_condition_v2"
    STORE_PATH.write_text(json.dumps(store, indent=2, ensure_ascii=False), encoding="utf-8")

    manifest_backup = None
    if MANIFEST_PATH.exists():
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest.get("class_names") != ACTIVE_CLASS_NAMES:
            manifest_backup = backup(MANIFEST_PATH)
            manifest["class_names"] = ACTIVE_CLASS_NAMES
            manifest["taxonomy"] = "vehicle_type_x_lamp_condition_v2"
            MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Migrated boxes: {changed}")
    print(f"Store backup: {store_backup}")
    if manifest_backup:
        print(f"Manifest backup: {manifest_backup}")


if __name__ == "__main__":
    main()
