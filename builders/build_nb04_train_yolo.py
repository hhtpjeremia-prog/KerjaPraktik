#!/usr/bin/env python3
"""Build NB04_train_yolo.ipynb: materialisasi split resmi dari gold set dan training YOLO 9 kelas."""

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_gate import verify_code_cells


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB04_train_yolo.ipynb"

EXPECTED_CODE_CELL_COUNT = 7
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB04 — Materialisasi Split & Training YOLO 9 Kelas

NB02b sudah mengekspor gold set beserta `split_map.csv`. NB04 **tidak membuat label baru**: ia hanya
menyusun ulang gambar+label ke struktur `{train,val,test}` sesuai peta split per kamera, menulis
`data.yaml`, lalu melatih detektor.

**Posisi:** `NB02 → NB02b → NB03 → NB04 (latih) → NB05 (hitung)`

## Aturan yang dipegang

- **Split per kamera**, dibaca dari `split_map.csv` hasil NB02b. Satu kamera mustahil muncul di dua
  split, sehingga tidak ada kebocoran lewat frame yang mirip dari video yang sama.
- **Semua label adalah gold.** Tidak ada percampuran label pseudo: NB02 hanya memberi lokasi, dan
  lokasi itu sudah dikoreksi manusia di NB02b. Frame yang tidak direview tidak ikut dataset.
- **9 kelas**, `nc: 9`. Indeks kelas mengikuti urutan `TRAINING_CLASS_NAMES` di NB02b — jangan diubah di sini,
  karena file label sudah memakai indeks tersebut.
- **`PantulanCahaya` = hard-negative.** Ikut dilatih supaya model tidak menghitung pantulan aspal
  sebagai lampu, tetapi tidak pernah masuk angka kendaraan di NB05.
- **Gambar di-hardlink** (fallback copy). Menghapus folder dataset tidak menghapus crop NB02.
- **Guard**: training ditolak bila split `val` terlalu kecil, karena metrik dari val yang nyaris
  kosong hanya menyesatkan.

## Keterbatasan yang harus dicatat di laporan

Gold set **tidak** memuat kamera split `test` (dikecualikan di NB02b). Konsekuensinya:
- metrik akurasi hanya sah untuk split `val`;
- NB05 menghitung jumlah kendaraan di kamera test (**tanpa** ground truth), jadi angka hitung tidak
  punya pasangan akurasi di split itu.

## Mode smoke

`PIPELINE_SMOKE=1` → root `output/smoke/`, `EPOCHS = 1`, guard minimum dilonggarkan.
Tujuannya memverifikasi alur, bukan mengukur performa. Device tetap `cuda` bila tersedia —
smoke tidak menurunkan jalur komputasi, hanya ukuran datanya.

Output: `output/nb04/{dataset/, split_audit.csv, class_balance.csv, training_config.json,
val_metrics.json}`, bobot di `output/nb04/runs/<RUN_NAME>/weights/best.pt`.
"""

CONFIG_SOURCE = '''from pathlib import Path
import csv
import json
import os
import shutil

import pandas as pd
import torch
from IPython.display import Markdown, display

from pipeline_common import TRAINING_CLASS_NAMES

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT

ANNOTATION_ROOT = DATA_ROOT / "annotation"
SOURCE_IMAGE_DIR = ANNOTATION_ROOT / "yolo" / "images"
SOURCE_LABEL_DIR = ANNOTATION_ROOT / "yolo" / "labels"
SPLIT_MAP_PATH = ANNOTATION_ROOT / "yolo" / "split_map.csv"

NB04_ROOT = DATA_ROOT / "nb04"
DATASET_ROOT = NB04_ROOT / "dataset"
DATASET_IMAGE_DIR = DATASET_ROOT / "images"
DATASET_LABEL_DIR = DATASET_ROOT / "labels"
DATA_YAML_PATH = DATASET_ROOT / "data.yaml"
RUNS_DIR = NB04_ROOT / "runs"
SPLIT_AUDIT_PATH = NB04_ROOT / "split_audit.csv"
CLASS_BALANCE_PATH = NB04_ROOT / "class_balance.csv"
TRAINING_CONFIG_PATH = NB04_ROOT / "training_config.json"
VALIDATION_METRICS_PATH = NB04_ROOT / "val_metrics.json"

for output_directory in (NB04_ROOT, DATASET_ROOT, DATASET_IMAGE_DIR, DATASET_LABEL_DIR, RUNS_DIR):
    output_directory.mkdir(parents=True, exist_ok=True)

CLASS_INDEX_BY_NAME = {class_name: class_index for class_index, class_name in enumerate(TRAINING_CLASS_NAMES)}
SPLIT_ORDER = ["train", "val", "test"]
SPLIT_MAP_FIELDS = ("frame_stem", "camera_id", "split")

MODEL_WEIGHTS = "yolo11n.pt"
RUN_NAME = "nb04_yolo11n_smoke" if SMOKE_MODE else "nb04_yolo11n"
EPOCHS = 1 if SMOKE_MODE else 60
IMAGE_SIZE = 640
BATCH_SIZE = 8
DATALOADER_WORKERS = 0
RANDOM_SEED = 20261026
EARLY_STOP_PATIENCE = 20
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MINIMUM_VAL_FRAMES = 1 if SMOKE_MODE else 30
MINIMUM_CLASS_TRAIN_BOXES = 1 if SMOKE_MODE else 20
MINIMUM_CLASS_VAL_BOXES = 1 if SMOKE_MODE else 5
LABEL_DECIMALS = 6

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

display(Markdown(
    f"NB04 siap. Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · device: **{DEVICE}** · "
    f"model `{MODEL_WEIGHTS}` · epoch {EPOCHS} · imgsz {IMAGE_SIZE} · batch {BATCH_SIZE} · nc {len(TRAINING_CLASS_NAMES)}"
))'''

SPLIT_SOURCE = '''def read_split_map(split_map_path: Path) -> pd.DataFrame:
    if not split_map_path.exists():
        raise FileNotFoundError(f"{split_map_path} tidak ada. Jalankan export NB02b lebih dulu.")
    split_frame = pd.read_csv(split_map_path, dtype=str)
    if tuple(split_frame.columns) != SPLIT_MAP_FIELDS:
        raise ValueError(f"kolom split_map.csv {tuple(split_frame.columns)} != {SPLIT_MAP_FIELDS}")
    split_frame = split_frame.fillna("").apply(lambda column: column.str.strip())
    duplicated_frames = split_frame[split_frame.duplicated("frame_stem", keep=False)]
    if not duplicated_frames.empty:
        raise ValueError(f"frame muncul lebih dari sekali di split_map.csv: {duplicated_frames['frame_stem'].head().tolist()}")
    unexpected_splits = sorted(set(split_frame["split"]) - set(SPLIT_ORDER))
    if unexpected_splits:
        raise ValueError(f"split tidak dikenal di split_map.csv: {unexpected_splits}")
    cameras_in_multiple_splits = split_frame.groupby("camera_id")["split"].nunique()
    leaking_cameras = cameras_in_multiple_splits[cameras_in_multiple_splits > 1].index.tolist()
    if leaking_cameras:
        raise ValueError(f"kamera muncul di lebih dari satu split: {leaking_cameras}")
    missing_images = [frame_stem for frame_stem in split_frame["frame_stem"] if not (SOURCE_IMAGE_DIR / f"{frame_stem}.png").exists()]
    if missing_images:
        raise ValueError(f"gambar hilang untuk {len(missing_images)} frame (contoh: {missing_images[:5]})")
    missing_labels = [frame_stem for frame_stem in split_frame["frame_stem"] if not (SOURCE_LABEL_DIR / f"{frame_stem}.txt").exists()]
    if missing_labels:
        raise ValueError(f"label hilang untuk {len(missing_labels)} frame (contoh: {missing_labels[:5]})")
    return split_frame


split_map_frame = read_split_map(SPLIT_MAP_PATH)

camera_audit_frame = (
    split_map_frame.groupby(["camera_id", "split"]).size().rename("frame_count").reset_index()
)
camera_audit_frame.to_csv(SPLIT_AUDIT_PATH, index=False)

split_frame_summary = pd.DataFrame([
    {
        "split": split_name,
        "frames": int((split_map_frame["split"] == split_name).sum()),
        "cameras": int(split_map_frame.loc[split_map_frame["split"] == split_name, "camera_id"].nunique()),
    }
    for split_name in SPLIT_ORDER
])

display(Markdown("### Peta split per kamera (`split_map.csv` dari NB02b)"))
display(camera_audit_frame)
display(Markdown("### Jumlah frame per split"))
display(split_frame_summary)'''

DATASET_SOURCE = '''def link_or_copy(source_path: Path, destination_path: Path) -> str:
    if destination_path.exists():
        destination_path.unlink()
    try:
        os.link(source_path, destination_path)
        return "hardlink"
    except OSError:
        shutil.copy2(source_path, destination_path)
        return "copy"


def write_data_yaml(dataset_root: Path, data_yaml_path: Path) -> None:
    yaml_lines = [
        f"path: {dataset_root.as_posix()}",
        f"train: images/{SPLIT_ORDER[0]}",
        f"val: images/{SPLIT_ORDER[1]}",
        f"test: images/{SPLIT_ORDER[2]}",
        f"nc: {len(TRAINING_CLASS_NAMES)}",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {class_name}" for index, class_name in enumerate(TRAINING_CLASS_NAMES))
    data_yaml_path.write_text("\\n".join(yaml_lines) + "\\n", encoding="utf-8")


for split_name in SPLIT_ORDER:
    (DATASET_IMAGE_DIR / split_name).mkdir(parents=True, exist_ok=True)
    (DATASET_LABEL_DIR / split_name).mkdir(parents=True, exist_ok=True)
    for stale_path in (DATASET_IMAGE_DIR / split_name).glob("*.png"):
        stale_path.unlink()
    for stale_path in (DATASET_LABEL_DIR / split_name).glob("*.txt"):
        stale_path.unlink()

dataset_rows = []
for split_row in split_map_frame.itertuples(index=False):
    destination_image_path = DATASET_IMAGE_DIR / split_row.split / f"{split_row.frame_stem}.png"
    destination_label_path = DATASET_LABEL_DIR / split_row.split / f"{split_row.frame_stem}.txt"
    link_mode = link_or_copy(SOURCE_IMAGE_DIR / f"{split_row.frame_stem}.png", destination_image_path)
    link_or_copy(SOURCE_LABEL_DIR / f"{split_row.frame_stem}.txt", destination_label_path)
    dataset_rows.append({
        "frame_stem": split_row.frame_stem,
        "camera_id": split_row.camera_id,
        "split": split_row.split,
        "link_mode": link_mode,
    })

dataset_frame = pd.DataFrame(dataset_rows)
write_data_yaml(DATASET_ROOT, DATA_YAML_PATH)

display(Markdown(
    f"Dataset dimaterialisasi: **{len(dataset_frame)} frame** · "
    f"link mode: {dataset_frame['link_mode'].value_counts().to_dict()} · `data.yaml` → `{DATA_YAML_PATH}`"
))'''

SANITY_SOURCE = '''def count_split_boxes(split_name: str) -> dict:
    class_counts = {class_name: 0 for class_name in TRAINING_CLASS_NAMES}
    label_paths = sorted((DATASET_LABEL_DIR / split_name).glob("*.txt"))
    for label_path in label_paths:
        for label_line in label_path.read_text(encoding="utf-8").splitlines():
            if not label_line.strip():
                continue
            class_index = int(label_line.split()[0])
            if class_index < 0 or class_index >= len(TRAINING_CLASS_NAMES):
                raise ValueError(f"indeks kelas {class_index} di luar rentang pada {label_path.name}")
            class_counts[TRAINING_CLASS_NAMES[class_index]] += 1
    return {"split": split_name, "images": len(label_paths), "boxes": sum(class_counts.values()), **class_counts}


class_balance_frame = pd.DataFrame([count_split_boxes(split_name) for split_name in SPLIT_ORDER])
class_balance_frame.to_csv(CLASS_BALANCE_PATH, index=False)

pairing_issues = []
for split_name in SPLIT_ORDER:
    split_image_stems = {path.stem for path in (DATASET_IMAGE_DIR / split_name).glob("*.png")}
    split_label_stems = {path.stem for path in (DATASET_LABEL_DIR / split_name).glob("*.txt")}
    if split_image_stems != split_label_stems:
        pairing_issues.append({
            "split": split_name,
            "image_without_label": len(split_image_stems - split_label_stems),
            "label_without_image": len(split_label_stems - split_image_stems),
        })

val_frame_count = int(class_balance_frame.loc[class_balance_frame["split"] == "val", "images"].iloc[0])
val_box_count = int(class_balance_frame.loc[class_balance_frame["split"] == "val", "boxes"].iloc[0])
train_frame_count = int(class_balance_frame.loc[class_balance_frame["split"] == "train", "images"].iloc[0])

display(Markdown("### Keseimbangan kelas per split"))
display(class_balance_frame)
if pairing_issues:
    display(Markdown("### Pasangan gambar-label bermasalah"))
    display(pd.DataFrame(pairing_issues))
    raise ValueError(f"pasangan gambar-label tidak lengkap: {pairing_issues}")

if train_frame_count == 0:
    raise ValueError("split train kosong; periksa peta split kamera.")
if val_frame_count < MINIMUM_VAL_FRAMES or val_box_count == 0:
    raise ValueError(
        f"split val hanya {val_frame_count} frame / {val_box_count} box (minimum {MINIMUM_VAL_FRAMES} frame). "
        "Tambah anotasi frame kamera val di NB02b."
    )
if int(class_balance_frame.loc[class_balance_frame["split"] == "test", "images"].iloc[0]) == 0:
    display(Markdown(
        "ℹ️ Split `test` kosong. Ini konsekuensi gold set tidak memuat kamera test: "
        "akurasi hanya diukur di `val`, sedangkan kamera test dipakai NB05 untuk menghitung."
    ))

train_split_row = class_balance_frame.loc[class_balance_frame["split"] == "train"].iloc[0]
val_split_row = class_balance_frame.loc[class_balance_frame["split"] == "val"].iloc[0]
short_train_classes = [
    class_name for class_name in TRAINING_CLASS_NAMES
    if int(train_split_row[class_name]) < MINIMUM_CLASS_TRAIN_BOXES
]
short_val_classes = [
    class_name for class_name in TRAINING_CLASS_NAMES
    if int(val_split_row[class_name]) < MINIMUM_CLASS_VAL_BOXES
]
if short_train_classes:
    raise ValueError(
        f"kelas tanpa cukup box di train: {short_train_classes} (minimum {MINIMUM_CLASS_TRAIN_BOXES}). "
        "Tambah anotasi di NB02b, atau persempit TRAINING_CLASS_NAMES di pipeline_common. "
        "Melatih dengan kelas kosong hanya menghasilkan mAP yang tidak bisa ditafsirkan."
    )
if short_val_classes:
    display(Markdown(
        f"⚠️ Kelas dengan < {MINIMUM_CLASS_VAL_BOXES} box di `val`: "
        + ", ".join(f"`{class_name}` ({int(val_split_row[class_name])})" for class_name in short_val_classes)
        + ". Training tetap jalan, tetapi AP kelas itu akan berisiko tinggi dan harus dilaporkan sebagai estimasi lemah."
    ))
display(Markdown(f"Guard lolos: train {train_frame_count} frame · val {val_frame_count} frame / {val_box_count} box."))'''

TRAIN_SOURCE = '''from ultralytics import YOLO

model = YOLO(MODEL_WEIGHTS)
training_arguments = {
    "data": str(DATA_YAML_PATH),
    "epochs": EPOCHS,
    "imgsz": IMAGE_SIZE,
    "batch": BATCH_SIZE,
    "device": DEVICE,
    "project": str(RUNS_DIR),
    "name": RUN_NAME,
    "seed": RANDOM_SEED,
    "workers": DATALOADER_WORKERS,
    "patience": EARLY_STOP_PATIENCE,
    "exist_ok": True,
    "plots": True,
}

display(Markdown(f"Mulai training `{MODEL_WEIGHTS}` → `{RUNS_DIR / RUN_NAME}`"))
training_result = model.train(**training_arguments)

best_weights_path = RUNS_DIR / RUN_NAME / "weights" / "best.pt"
training_payload = {
    "smoke_mode": SMOKE_MODE,
    "model_weights": MODEL_WEIGHTS,
    "run_name": RUN_NAME,
    "class_names": TRAINING_CLASS_NAMES,
    "split_source": str(SPLIT_MAP_PATH),
    "data_yaml": str(DATA_YAML_PATH),
    "dataset_frames": int(len(dataset_frame)),
    "arguments": training_arguments,
    "best_weights": str(best_weights_path),
    "best_weights_exists": best_weights_path.exists(),
}
TRAINING_CONFIG_PATH.write_text(json.dumps(training_payload, indent=2, ensure_ascii=False), encoding="utf-8")

display(Markdown(f"Training selesai. Bobot terbaik: `{best_weights_path}` (ada: **{best_weights_path.exists()}**)"))'''

VALIDATE_SOURCE = '''best_model = YOLO(str(best_weights_path))
validation_metrics = best_model.val(
    data=str(DATA_YAML_PATH),
    split="val",
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    device=DEVICE,
    workers=DATALOADER_WORKERS,
    project=str(RUNS_DIR),
    name=f"{RUN_NAME}_val",
    exist_ok=True,
    plots=True,
)

val_class_frame = pd.DataFrame(validation_metrics.summary())
val_overall = {
    "precision": float(validation_metrics.box.mp),
    "recall": float(validation_metrics.box.mr),
    "mAP50": float(validation_metrics.box.map50),
    "mAP50-95": float(validation_metrics.box.map),
}

display(Markdown("### Metrik per kelas pada split val (label gold)"))
display(val_class_frame)
display(Markdown(
    f"**Gabungan val** — precision {val_overall['precision']:.3f} | recall {val_overall['recall']:.3f} | "
    f"mAP50 {val_overall['mAP50']:.3f} | mAP50-95 {val_overall['mAP50-95']:.3f}"
))

VALIDATION_METRICS_PATH.write_text(
    json.dumps(
        {
            "smoke_mode": SMOKE_MODE,
            "split": "val",
            "label_source": "gold",
            "val_frames": val_frame_count,
            "overall": val_overall,
            "per_class": val_class_frame.to_dict(orient="records"),
            "best_weights": str(best_weights_path),
        },
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)'''

SUMMARY_SOURCE = '''balance_records = class_balance_frame.to_dict(orient="records")
header_line = "| split | gambar | box | " + " | ".join(TRAINING_CLASS_NAMES) + " |"
divider_line = "|---" * (3 + len(TRAINING_CLASS_NAMES)) + "|"
balance_lines = "\\n".join(
    f"| {row['split']} | {row['images']} | {row['boxes']} | "
    + " | ".join(str(row[class_name]) for class_name in TRAINING_CLASS_NAMES)
    + " |"
    for row in balance_records
)

display(Markdown(
    f"""### Ringkasan NB04

Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · dataset: `{DATASET_ROOT}` · bobot: `{best_weights_path}`

{header_line}
{divider_line}
{balance_lines}

**Metrik val** — mAP50 {val_overall["mAP50"]:.3f} | mAP50-95 {val_overall["mAP50-95"]:.3f} | precision {val_overall["precision"]:.3f} | recall {val_overall["recall"]:.3f}

**Yang harus ditulis di laporan**
- Semua label adalah gold hasil NB02b; tidak ada label pseudo yang ikut dilatih.
- Split `test` tidak punya gold, jadi tidak ada metrik akurasi di sana.
  NB05 memakai kamera test hanya untuk **menghitung**, bukan mengukur akurasi.
- Kelas `PantulanCahaya` adalah hard-negative dan tidak pernah dihitung sebagai kendaraan.
"""
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(SPLIT_SOURCE),
        nbformat.v4.new_code_cell(DATASET_SOURCE),
        nbformat.v4.new_code_cell(SANITY_SOURCE),
        nbformat.v4.new_code_cell(TRAIN_SOURCE),
        nbformat.v4.new_code_cell(VALIDATE_SOURCE),
        nbformat.v4.new_code_cell(SUMMARY_SOURCE),
    ]
    return nbformat.v4.new_notebook(
        cells=notebook_cells,
        metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3"},
            "language_info": {"name": "python"},
        },
    )


def main() -> None:
    notebook = build_notebook()
    nbformat.write(notebook, OUTPUT_NOTEBOOK)
    verify_code_cells(OUTPUT_NOTEBOOK, EXPECTED_CODE_CELL_COUNT, EXPECTED_MARKDOWN_CELL_COUNT)


if __name__ == "__main__":
    main()
