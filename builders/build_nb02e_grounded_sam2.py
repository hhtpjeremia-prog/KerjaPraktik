#!/usr/bin/env python3
"""Build NB02e: SAM2 masks for manual, gold-test and FM pseudo-train boxes."""

from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "NB02e_grounded_sam2_masks.ipynb"

MARKDOWN = """# NB02e — Mask SAM2 untuk fitur classifier

NB02e membuat mask untuk semua bbox yang dipakai CNN: anotasi manual train/validation, gold test,
dan pseudo-label FM train saja. Box manual pada frame yang sudah direview menjadi sumber utama;
pseudo box dari frame itu tidak ditambahkan. SAM2 hanya membuat mask dan tidak mengubah label.

Mask disimpan di `output/nb02e_feature_masks/`. NB04b menghitung photometry di dalam mask,
fitur bentuk mask, lalu melatih e-means hanya dari fitur split train. Label cluster one-hot menjadi
fitur tambahan CNN. NB06 membuat mask serupa langsung dari setiap bbox inferensi video, sehingga
train dan inferensi memakai kontrak fitur sama. Manifest bisa dilanjutkan setelah interupsi.
`NB02E_MAX_FRAMES` hanya untuk preview; NB04b menolak manifest preview atau incomplete.
"""

CONFIG = r'''import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from IPython.display import Markdown, display
from transformers import Sam2Model, Sam2Processor

ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
MANUAL_MANIFEST = ROOT / "output" / "annotation" / "sample_manifest.json"
MANUAL_STORE = ROOT / "output" / "annotation" / "annotations.json"
GT_MANIFEST = ROOT / "output" / "ground_truth" / "sample_manifest.json"
GT_STORE = ROOT / "output" / "ground_truth" / "annotations.json"
PROPOSALS = ROOT / "output" / "nb02d_fiveclass" / "base" / "proposal_manifest.json"
OUT = ROOT / "output" / "nb02e_feature_masks"
MASK_DIR = OUT / "masks"
MANIFEST = OUT / "mask_manifest.json"
MODEL_ID = os.environ.get("NB02E_SAM2_MODEL", "facebook/sam2.1-hiera-tiny")
MAX_FRAMES = int(os.environ.get("NB02E_MAX_FRAMES", "0"))  # 0 = full run
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
ACTIVE_CLASSES = {
    "MobilStandar", "MobilTidakStandar", "MotorStandar", "MotorTidakStandar", "PantulanCahaya"
}
OUT.mkdir(parents=True, exist_ok=True)
MASK_DIR.mkdir(parents=True, exist_ok=True)

required_sources = [MANUAL_MANIFEST, MANUAL_STORE, GT_MANIFEST, GT_STORE, PROPOSALS]
missing_sources = [str(path) for path in required_sources if not path.exists()]
if missing_sources:
    raise FileNotFoundError("Sumber mask belum lengkap: " + "; ".join(missing_sources))
proposal_payload = json.loads(PROPOSALS.read_text(encoding="utf-8"))
if proposal_payload.get("status") != "complete":
    raise ValueError("Proposal NB02d belum complete; NB02e menolak output parsial.")

def resolve_project_path(value):
    path = Path(value)
    resolved = path.resolve() if path.is_absolute() else (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"Path manifest harus berada di dalam root proyek: {value}") from error
    if not resolved.exists():
        raise FileNotFoundError(f"Berkas untuk mask tidak ditemukan: {resolved}")
    return resolved

def store_project_path(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()

from attribute_classifier import mask_cache_key

requests = {}
human_reviewed_stems = set()

def add_request(frame_stem, image_value, box, class_name, source_name, split_name):
    if class_name not in ACTIVE_CLASSES:
        raise ValueError(f"Kelas sumber di luar taxonomy aktif: {frame_stem}: {class_name}")
    image_path = resolve_project_path(image_value)
    mask_id = mask_cache_key(frame_stem, box)
    request = {
        "mask_id": mask_id,
        "frame_stem": frame_stem,
        "image_path": store_project_path(image_path),
        "box": [float(value) for value in box],
        "class_name": class_name,
        "source": source_name,
        "split": split_name,
    }
    existing = requests.get(mask_id)
    if existing and (existing["image_path"] != request["image_path"] or existing["box"] != request["box"]):
        raise ValueError(f"Collision mask_id {mask_id}")
    requests.setdefault(mask_id, request)

for manifest_path, store_path, split_override, source_name in [
    (MANUAL_MANIFEST, MANUAL_STORE, None, "human_annotation"),
    (GT_MANIFEST, GT_STORE, "test", "gold_test"),
]:
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    store_payload = json.loads(store_path.read_text(encoding="utf-8"))
    for record in manifest_payload.get("frames", []):
        frame_stem = record["frame_stem"]
        entry = store_payload.get("frames", {}).get(frame_stem)
        if not entry or not entry.get("reviewed"):
            continue
        human_reviewed_stems.add(frame_stem)
        split_name = split_override or record.get(
            "split", store_payload.get("split_by_camera", {}).get(record.get("camera_id"), "unknown")
        )
        for box_entry in entry.get("boxes", []):
            if box_entry.get("deleted") or box_entry.get("class_name") not in ACTIVE_CLASSES:
                continue
            add_request(
                frame_stem, record["image_path"], box_entry["box"], box_entry["class_name"],
                source_name, split_name,
            )

# FM proposals supplement train only; any reviewed human frame suppresses its pseudo boxes,
# including reviewed-empty frames.
for record in proposal_payload.get("frames", []):
    if record.get("split") != "train" or record["frame_stem"] in human_reviewed_stems:
        continue
    for detection in record.get("detections", []):
        add_request(
            record["frame_stem"], record["image_path"], detection["box"],
            detection.get("class_name"), "grounding_dino_pseudo", "train",
        )

all_requests = list(requests.values())
grouped_frames = {}
for request in all_requests:
    grouped_frames.setdefault(request["frame_stem"], []).append(request)
frame_stems = sorted(grouped_frames)
selected_stems = frame_stems[:MAX_FRAMES] if MAX_FRAMES else frame_stems
selected_requests = [request for stem in selected_stems for request in grouped_frames[stem]]
source_hasher = hashlib.sha256()
for source_path in required_sources:
    source_hasher.update(source_path.name.encode("utf-8"))
    source_hasher.update(source_path.read_bytes())
source_signature = source_hasher.hexdigest()
prior = json.loads(MANIFEST.read_text(encoding="utf-8")) if MANIFEST.exists() else {
    "schema_version": 2, "items": {},
}
if prior.get("schema_version") != 2:
    raise ValueError("Manifest mask lama tidak kompatibel; arsipkan lalu mulai di output/nb02e_feature_masks.")
if (
    prior.get("source_signature") not in (None, source_signature)
    or prior.get("sam2_model") not in (None, MODEL_ID)
):
    # Preserve old PNGs on disk, but drop their cache references and rebuild the
    # current manifest. This makes annotation/model changes safely rerunnable.
    prior = {"schema_version": 2, "items": {}}
done = prior.setdefault("items", {})
for cached_item in done.values():
    for path_field in ("image_path", "mask_path"):
        if cached_item.get(path_field):
            cached_item[path_field] = store_project_path(resolve_project_path(cached_item[path_field]))
print(f"SAM2 {MODEL_ID} · device={DEVICE} · frames={len(selected_stems)} · boxes={len(selected_requests)} · cache={len(done)}")
'''

RUN = r'''processor = Sam2Processor.from_pretrained(MODEL_ID)
model = Sam2Model.from_pretrained(MODEL_ID).to(DEVICE).eval()

for frame_stem in selected_stems:
    frame_requests = grouped_frames[frame_stem]
    if all(
        request["mask_id"] in done
        and resolve_project_path(done[request["mask_id"]]["mask_path"]).exists()
        for request in frame_requests
    ):
        continue
    image_path = resolve_project_path(frame_requests[0]["image_path"])
    image = Image.open(image_path).convert("RGB")
    boxes = [[request["box"] for request in frame_requests]]
    inputs = processor(images=image, input_boxes=boxes, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model(**inputs)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu())[0]
    quality = outputs.iou_scores[0].detach().cpu()
    for index, request in enumerate(frame_requests):
        if request["mask_id"] in done and resolve_project_path(done[request["mask_id"]]["mask_path"]).exists():
            continue
        best_index = int(quality[index].argmax())
        mask = masks[index, best_index].numpy().astype(np.uint8) * 255
        mask_path = MASK_DIR / f"{request['mask_id']}.png"
        Image.fromarray(mask, mode="L").save(mask_path)
        done[request["mask_id"]] = {
            **request,
            "sam2_iou_score": float(quality[index, best_index]),
            "mask_path": store_project_path(mask_path),
        }
    prior.update({
        "schema_version": 2,
        "status": "in_progress",
        "source_signature": source_signature,
        "source_files": [store_project_path(path) for path in required_sources],
        "sam2_model": MODEL_ID,
        "class_names": sorted(ACTIVE_CLASSES),
        "device": DEVICE,
    })
    MANIFEST.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")

required_mask_ids = {request["mask_id"] for request in selected_requests}
all_mask_ids = {request["mask_id"] for request in all_requests}
completed_mask_ids = {
    mask_id for mask_id in all_mask_ids
    if mask_id in done and resolve_project_path(done[mask_id]["mask_path"]).exists()
}
prior.update({
    "schema_version": 2,
    "source_signature": source_signature,
    "source_files": [store_project_path(path) for path in required_sources],
    "source_frame_total": len(frame_stems),
    "source_box_total": len(all_requests),
    "processed_frame_total": len(set(item["frame_stem"] for item in selected_requests)),
    "processed_box_total": len(completed_mask_ids),
    "status": "complete" if not MAX_FRAMES and required_mask_ids == completed_mask_ids else (
        "preview_complete" if MAX_FRAMES and required_mask_ids == completed_mask_ids else "in_progress"
    ),
})
MANIFEST.write_text(json.dumps(prior, ensure_ascii=False, indent=2), encoding="utf-8")
display(Markdown(
    f"Mask tersimpan: **{prior['processed_box_total']} / {prior['source_box_total']} box** "
    f"· manifest `{MANIFEST}` · status `{prior['status']}`"
))
'''


def main() -> None:
    notebook = nbformat.v4.new_notebook(
        cells=[
            nbformat.v4.new_markdown_cell(MARKDOWN),
            nbformat.v4.new_code_cell(CONFIG),
            nbformat.v4.new_code_cell(RUN),
        ],
        metadata={"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"}},
    )
    nbformat.write(notebook, OUTPUT)
    loaded = nbformat.read(OUTPUT, as_version=4)
    nbformat.validate(loaded)
    for index, cell in enumerate(loaded.cells):
        if cell.cell_type == "code":
            compile(cell.source, f"NB02e cell {index}", "exec")
    print(f"Generated {OUTPUT}")


if __name__ == "__main__":
    main()
