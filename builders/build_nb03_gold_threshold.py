#!/usr/bin/env python3
"""Build NB03_gold_threshold.ipynb: audit gold set dan karakterisasi fotometrik per kelas."""

import sys
from pathlib import Path

import nbformat

sys.path.insert(0, str(Path(__file__).resolve().parent))

from notebook_gate import verify_code_cells


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_NOTEBOOK = PROJECT_ROOT / "NB03_gold_threshold.ipynb"

EXPECTED_CODE_CELL_COUNT = 5
EXPECTED_MARKDOWN_CELL_COUNT = 1

NAVBAR_MARKDOWN = """# NB03 — Audit Gold Set & Karakterisasi Fotometrik

Tahap ini **tidak melatih apa pun** dan tidak mengubah anotasi. Tugasnya menjawab satu pertanyaan
sebelum waktu GPU dipakai: *apakah label kondisi lampu di NB02b konsisten dengan ciri fotometrik
yang bisa diukur?*

**Posisi:** `NB02 (crop + pseudo-lokasi) → NB02b (anotasi kondisi) → NB03 (audit) → NB04 (training) → NB05 (hitung)`

## Yang diperiksa

1. **Kelengkapan & kebocoran.** Semua frame di store harus `reviewed`, tidak boleh ada box
   `BelumDitentukan`, dan tidak boleh ada frame dari kamera split `test`.
2. **Keseimbangan kelas.** Jumlah box per kelas per split. Kelas dengan box terlalu sedikit
   ditandai, bukan digagalkan, karena jumlahnya keputusan pemilik riset.
3. **Karakterisasi fotometrik.** Untuk setiap box dihitung di ruang HSV:
   - `bright_ratio` = fraksi piksel dengan `V >= BRIGHT_VALUE_THRESHOLD` (indikasi bloom/silau)
   - `mean_saturation` = rata-rata kanal `S` (indikasi warna)
   - `dominant_hue` = hue dominan pada piksel bersaturasi cukup, dibucket `HUE_BUCKET_SIZE`
     (satuan OpenCV 0-180, bukan derajat)
   - `width_px`, `area_ratio` = ukuran box relatif frame

   Statistik per kelas dipakai untuk **mendokumentasikan** pemisahan `Normal` / `Silau` /
   `Berwarna`, bukan untuk menetapkan label. Label tetap milik manusia; angkanya jadi dasar
   argumen di laporan bahwa definisi operasional bisa diukur.

## Kenapa tidak ada kalibrasi threshold DINO lagi

Versi lama NB03 mengkalibrasi ambang DINO untuk menebak kelas. Di V2 kondisi lampu **tidak pernah**
ditebak model, melainkan ditetapkan manusia di NB02b. Ambang yang tersisa dan masih relevan adalah
ambang *confidence* model hasil training, dan itu dikalibrasi di **NB05** pada split `val`.

## Mode smoke

Set environment variable `PIPELINE_SMOKE=1` sebelum menjalankan notebook: semua path pindah ke
`output/smoke/` sehingga run uji tidak menyentuh artefak asli.

Output: `output/nb03/{gold_audit.json, gold_boxes.csv, gold_summary_by_class.csv,
photometry_by_class.png}`.
"""

CONFIG_SOURCE = '''from pathlib import Path
import json
import os

import cv2
import numpy as np
import pandas as pd
from IPython.display import Markdown, display

from pipeline_common import CLASS_NAMES, PENDING_CLASS_NAME, TRAINING_CLASS_NAMES, training_class_name

RISET_ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path.cwd())).resolve()
if RISET_ROOT.name not in {"Riset-HampirMatang", "NewStyle"}:
    raise RuntimeError("Set RISET_PROJECT_ROOT ke root proyek atau jalankan notebook dari root proyek.")
OUTPUT_ROOT = RISET_ROOT / "output"
SMOKE_MODE = os.environ.get("PIPELINE_SMOKE", "0") == "1"
DATA_ROOT = OUTPUT_ROOT / "smoke" if SMOKE_MODE else OUTPUT_ROOT
NB02_ROOT = DATA_ROOT / "nb02"
CROP_DIR = NB02_ROOT / "crops"
ANNOTATION_ROOT = DATA_ROOT / "annotation"
ANNOTATION_STORE_PATH = ANNOTATION_ROOT / "annotations.json"
SAMPLE_MANIFEST_PATH = ANNOTATION_ROOT / "sample_manifest.json"
NB03_ROOT = DATA_ROOT / "nb03"
GOLD_BOXES_PATH = NB03_ROOT / "gold_boxes.csv"
CLASS_SUMMARY_PATH = NB03_ROOT / "gold_summary_by_class.csv"
AUDIT_PATH = NB03_ROOT / "gold_audit.json"
FIGURE_PATH = NB03_ROOT / "photometry_by_class.png"

NB03_ROOT.mkdir(parents=True, exist_ok=True)

SPLIT_ORDER = ["train", "val", "test"]
EXCLUDED_SPLITS = ["test"]

BRIGHT_VALUE_THRESHOLD = 240
COLORED_SATURATION_MIN = 80
COLORED_VALUE_MIN = 80
HUE_BUCKET_SIZE = 10
MIN_BOX_SIDE_PIXELS = 4
MIN_BOXES_PER_CLASS_WARNING = 15
TABLE_DECIMALS = 3

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 40)

display(Markdown(
    f"NB03 siap. Mode: **{'SMOKE' if SMOKE_MODE else 'PENUH'}** · data root: `{DATA_ROOT}` · "
    f"kelas latih: {len(TRAINING_CLASS_NAMES)} (dari {len(CLASS_NAMES)} kelas anotasi) · ambang bloom `V >= {BRIGHT_VALUE_THRESHOLD}`"
))'''

LOAD_SOURCE = '''def load_json_document(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"berkas wajib tidak ada: {path}. Jalankan tahap sebelumnya lebih dulu.")
    with path.open(encoding="utf-8") as source_file:
        return json.load(source_file)


def resolve_camera_split(manifest_payload: dict, store_payload: dict) -> dict:
    camera_split = dict(store_payload.get("split_by_camera", {}))
    if camera_split:
        return camera_split
    return {
        frame_record["camera_id"]: frame_record.get("split", "unknown")
        for frame_record in manifest_payload["frames"]
    }


def collect_gold_box_rows(manifest_payload: dict, store_payload: dict) -> list[dict]:
    frame_records = {frame_record["frame_stem"]: frame_record for frame_record in manifest_payload["frames"]}
    camera_split = resolve_camera_split(manifest_payload, store_payload)
    box_rows = []
    for frame_stem, frame_entry in store_payload["frames"].items():
        if frame_stem not in frame_records:
            raise ValueError(f"frame {frame_stem} ada di store tetapi tidak ada di sample_manifest.json")
        if not frame_entry["reviewed"]:
            continue
        frame_record = frame_records[frame_stem]
        camera_id = frame_record["camera_id"]
        split_name = camera_split.get(camera_id, "unknown")
        for box_index, box_entry in enumerate(frame_entry["boxes"]):
            if box_entry["deleted"]:
                continue
            box = [float(coordinate) for coordinate in box_entry["box"]]
            box_rows.append({
                "frame_stem": frame_stem,
                "box_index": box_index,
                "camera_id": camera_id,
                "split": split_name,
                "class_name": box_entry["class_name"],
                "source": box_entry["source"],
                "image_width": frame_entry["image_width"],
                "image_height": frame_entry["image_height"],
                "box_x1": box[0],
                "box_y1": box[1],
                "box_x2": box[2],
                "box_y2": box[3],
            })
    return box_rows


def assert_gold_set_is_usable(store_payload: dict, box_rows: list[dict]) -> int:
    reviewed_total = sum(1 for frame_entry in store_payload["frames"].values() if frame_entry["reviewed"])
    if reviewed_total == 0:
        raise ValueError("belum ada frame reviewed di store. Anotasi dulu di NB02b.")
    pending_frames = sorted(
        frame_stem
        for frame_stem, frame_entry in store_payload["frames"].items()
        if frame_entry["reviewed"]
        and any(
            not box_entry["deleted"] and box_entry["class_name"] == PENDING_CLASS_NAME
            for box_entry in frame_entry["boxes"]
        )
    )
    if pending_frames:
        raise ValueError(
            f"{len(pending_frames)} frame reviewed masih punya box {PENDING_CLASS_NAME} "
            f"(contoh: {pending_frames[:5]}). Selesaikan di NB02b sebelum audit."
        )
    unknown_classes = sorted({row["class_name"] for row in box_rows} - set(CLASS_NAMES))
    if unknown_classes:
        raise ValueError(f"kelas tidak dikenal di store: {unknown_classes}")
    leaked_splits = sorted({row["split"] for row in box_rows} & set(EXCLUDED_SPLITS))
    if leaked_splits:
        leaked_frames = [row["frame_stem"] for row in box_rows if row["split"] in leaked_splits][:5]
        raise ValueError(
            f"kebocoran: gold set memuat split {leaked_splits} (contoh: {leaked_frames}). "
            "Kamera test tidak boleh masuk gold set latih/validasi."
        )
    if not box_rows:
        raise ValueError("tidak ada box aktif di frame reviewed; tidak ada yang bisa diaudit.")
    min_side = min(min(row["box_x2"] - row["box_x1"], row["box_y2"] - row["box_y1"]) for row in box_rows)
    if min_side < MIN_BOX_SIDE_PIXELS:
        raise ValueError(f"ada box dengan sisi {min_side:.1f} px < {MIN_BOX_SIDE_PIXELS} px; perbaiki di NB02b.")
    return reviewed_total


sample_manifest_payload = load_json_document(SAMPLE_MANIFEST_PATH)
annotation_store_payload = load_json_document(ANNOTATION_STORE_PATH)
gold_box_rows = collect_gold_box_rows(sample_manifest_payload, annotation_store_payload)
reviewed_frame_total = assert_gold_set_is_usable(annotation_store_payload, gold_box_rows)
for box_row in gold_box_rows:
    box_row["class_name"] = training_class_name(box_row["class_name"])

gold_box_frame = pd.DataFrame(gold_box_rows)
gold_box_frame["width_px"] = gold_box_frame["box_x2"] - gold_box_frame["box_x1"]
gold_box_frame["height_px"] = gold_box_frame["box_y2"] - gold_box_frame["box_y1"]
gold_box_frame["area_ratio"] = (
    gold_box_frame["width_px"] * gold_box_frame["height_px"]
    / (gold_box_frame["image_width"] * gold_box_frame["image_height"])
)
gold_box_frame.to_csv(GOLD_BOXES_PATH, index=False)

display(Markdown(
    f"Gold set dimuat: **{reviewed_frame_total} frame reviewed** · **{len(gold_box_frame)} box** · "
    f"tanpa box pending, tanpa frame split test."
))'''

AUDIT_SOURCE = '''def build_split_summary(gold_frame: pd.DataFrame) -> pd.DataFrame:
    split_rows = []
    for split_name in SPLIT_ORDER:
        split_slice = gold_frame[gold_frame["split"] == split_name]
        split_rows.append({
            "split": split_name,
            "frames": int(split_slice["frame_stem"].nunique()),
            "boxes": int(len(split_slice)),
            "cameras": int(split_slice["camera_id"].nunique()),
        })
    return pd.DataFrame(split_rows)


def build_class_summary(gold_frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for class_name in TRAINING_CLASS_NAMES:
        class_slice = gold_frame[gold_frame["class_name"] == class_name]
        summary_rows.append({
            "class_name": class_name,
            "boxes": int(len(class_slice)),
            "train_boxes": int((class_slice["split"] == "train").sum()),
            "val_boxes": int((class_slice["split"] == "val").sum()),
            "test_boxes": int((class_slice["split"] == "test").sum()),
        })
    return pd.DataFrame(summary_rows)


def frames_without_boxes(store_payload: dict) -> list[str]:
    return sorted(
        frame_stem
        for frame_stem, frame_entry in store_payload["frames"].items()
        if frame_entry["reviewed"]
        and not any(not box_entry["deleted"] for box_entry in frame_entry["boxes"])
    )


split_summary_frame = build_split_summary(gold_box_frame)
class_summary_frame = build_class_summary(gold_box_frame)
underrepresented_classes = class_summary_frame[
    class_summary_frame["boxes"] < MIN_BOXES_PER_CLASS_WARNING
]["class_name"].tolist()
empty_frames = frames_without_boxes(annotation_store_payload)
source_counts = gold_box_frame["source"].value_counts().to_dict()

display(Markdown("### Frame & box per split"))
display(split_summary_frame)
display(Markdown("### Box per kelas"))
display(class_summary_frame)
display(Markdown(
    f"Sumber box aktif: {source_counts}. Frame negatif (reviewed tanpa box): **{len(empty_frames)}**."
))
if underrepresented_classes:
    display(Markdown(
        f"Kelas dengan < {MIN_BOXES_PER_CLASS_WARNING} box: "
        + ", ".join(f"`{class_name}`" for class_name in underrepresented_classes)
        + ". Metrik per kelas untuk kelas ini tidak stabil, pertimbangkan menambah anotasi."
    ))

gold_audit_payload = {
    "smoke_mode": SMOKE_MODE,
    "reviewed_frames": int(reviewed_frame_total),
    "negative_frames": len(empty_frames),
    "total_boxes": int(len(gold_box_frame)),
    "box_source_counts": {str(source_name): int(source_count) for source_name, source_count in source_counts.items()},
    "underrepresented_classes": underrepresented_classes,
    "min_boxes_per_class_warning": MIN_BOXES_PER_CLASS_WARNING,
    "split_summary": split_summary_frame.to_dict(orient="records"),
    "class_summary": class_summary_frame.to_dict(orient="records"),
}
AUDIT_PATH.write_text(json.dumps(gold_audit_payload, indent=2, ensure_ascii=False), encoding="utf-8")
display(Markdown(f"Audit kelengkapan ditulis ke `{AUDIT_PATH}`."))'''

PHOTOMETRY_SOURCE = '''def compute_box_photometry(hsv_image: np.ndarray, box_row: dict) -> dict:
    image_height, image_width = hsv_image.shape[:2]
    x_low = int(max(0, min(image_width - 1, round(box_row["box_x1"]))))
    x_high = int(max(x_low + 1, min(image_width, round(box_row["box_x2"]))))
    y_low = int(max(0, min(image_height - 1, round(box_row["box_y1"]))))
    y_high = int(max(y_low + 1, min(image_height, round(box_row["box_y2"]))))
    patch = hsv_image[y_low:y_high, x_low:x_high]
    if patch.size == 0:
        return {"bright_ratio": float("nan"), "mean_saturation": float("nan"), "dominant_hue": -1}
    hue = patch[:, :, 0]
    saturation = patch[:, :, 1]
    value = patch[:, :, 2]
    colored_mask = (saturation >= COLORED_SATURATION_MIN) & (value >= COLORED_VALUE_MIN)
    if colored_mask.any():
        hue_histogram = np.bincount(hue[colored_mask] // HUE_BUCKET_SIZE)
        dominant_hue = int(hue_histogram.argmax()) * HUE_BUCKET_SIZE
    else:
        dominant_hue = -1
    return {
        "bright_ratio": float((value >= BRIGHT_VALUE_THRESHOLD).mean()),
        "mean_saturation": float(saturation.mean()),
        "dominant_hue": dominant_hue,
    }


def annotate_photometry(gold_frame: pd.DataFrame, crop_directory: Path) -> pd.DataFrame:
    metric_rows = []
    metric_indices = []
    for frame_stem, frame_index in gold_frame.groupby("frame_stem", sort=True).groups.items():
        image_path = crop_directory / f"{frame_stem}.png"
        if not image_path.exists():
            raise FileNotFoundError(f"crop tidak ditemukan untuk {frame_stem}: {image_path}")
        hsv_image = cv2.cvtColor(cv2.imread(str(image_path)), cv2.COLOR_BGR2HSV)
        for row_index in frame_index:
            metric_rows.append(compute_box_photometry(hsv_image, gold_frame.loc[row_index].to_dict()))
            metric_indices.append(row_index)
    return pd.DataFrame(metric_rows, index=metric_indices)


photometry_frame = annotate_photometry(gold_box_frame, CROP_DIR)
gold_box_frame = pd.concat([gold_box_frame, photometry_frame], axis=1)
gold_box_frame.to_csv(GOLD_BOXES_PATH, index=False)


def build_photometry_summary(gold_frame: pd.DataFrame) -> pd.DataFrame:
    summary_rows = []
    for class_name in TRAINING_CLASS_NAMES:
        class_slice = gold_frame[gold_frame["class_name"] == class_name]
        if class_slice.empty:
            summary_rows.append({"class_name": class_name, "boxes": 0})
            continue
        hue_values = class_slice.loc[class_slice["dominant_hue"] >= 0, "dominant_hue"]
        summary_rows.append({
            "class_name": class_name,
            "boxes": int(len(class_slice)),
            "bright_ratio_median": round(float(class_slice["bright_ratio"].median()), TABLE_DECIMALS),
            "bright_ratio_p90": round(float(class_slice["bright_ratio"].quantile(0.90)), TABLE_DECIMALS),
            "saturation_median": round(float(class_slice["mean_saturation"].median()), TABLE_DECIMALS),
            "dominant_hue_mode": int(hue_values.mode().iloc[0]) if not hue_values.empty else -1,
            "width_px_median": round(float(class_slice["width_px"].median()), TABLE_DECIMALS),
            "area_ratio_median": round(float(class_slice["area_ratio"].median()), TABLE_DECIMALS),
        })
    return pd.DataFrame(summary_rows)


photometry_summary_frame = build_photometry_summary(gold_box_frame)
photometry_summary_frame.to_csv(CLASS_SUMMARY_PATH, index=False)

display(Markdown(
    f"### Karakterisasi fotometrik per kelas\\n"
    f"`bright_ratio` = fraksi piksel `V >= {BRIGHT_VALUE_THRESHOLD}` di dalam box. "
    f"`dominant_hue` = hue dominan (OpenCV 0-180, hanya piksel `S >= {COLORED_SATURATION_MIN}`, "
    f"`V >= {COLORED_VALUE_MIN}`), bucket {HUE_BUCKET_SIZE}."
))
display(photometry_summary_frame)'''

FIGURE_SOURCE = '''import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plotted_classes = [class_name for class_name in TRAINING_CLASS_NAMES if (gold_box_frame["class_name"] == class_name).any()]
bright_values = [
    gold_box_frame.loc[gold_box_frame["class_name"] == class_name, "bright_ratio"].to_numpy()
    for class_name in plotted_classes
]
saturation_values = [
    gold_box_frame.loc[gold_box_frame["class_name"] == class_name, "mean_saturation"].to_numpy()
    for class_name in plotted_classes
]

figure, axes = plt.subplots(1, 2, figsize=(15, 6))
axes[0].boxplot(bright_values, tick_labels=plotted_classes, orientation="horizontal")
axes[0].set_title(f"Bright ratio (fraksi piksel V >= {BRIGHT_VALUE_THRESHOLD})")
axes[0].set_xlabel("bright_ratio")
axes[1].boxplot(saturation_values, tick_labels=plotted_classes, orientation="horizontal")
axes[1].set_title("Rata-rata saturasi di dalam box")
axes[1].set_xlabel("mean_saturation")
for axis in axes:
    axis.grid(alpha=0.3)
    axis.tick_params(axis="y", labelsize=8)
figure.suptitle(f"Karakterisasi gold set NB03, {len(gold_box_frame)} box / {reviewed_frame_total} frame")
figure.tight_layout()
figure.savefig(FIGURE_PATH, dpi=120)
plt.close(figure)

bright_by_class = gold_box_frame.groupby("class_name")["bright_ratio"].median().sort_values(ascending=False)
separation_line = " · ".join(
    f"`{class_name}` {median_value:.3f}" for class_name, median_value in bright_by_class.items()
)
val_frames = int(gold_box_frame.loc[gold_box_frame["split"] == "val", "frame_stem"].nunique())
train_frames = int(gold_box_frame.loc[gold_box_frame["split"] == "train", "frame_stem"].nunique())

display(Markdown(
    f"### Ringkasan NB03\\n\\n"
    f"Gambar: `{FIGURE_PATH}`\\n\\n"
    f"Median `bright_ratio` per kelas (urut menurun): {separation_line}\\n\\n"
    f"| Metrik | Nilai |\\n|---|---|\\n"
    f"| Frame reviewed | {reviewed_frame_total} |\\n"
    f"| Frame negatif (tanpa box) | {len(empty_frames)} |\\n"
    f"| Box total | {len(gold_box_frame)} |\\n"
    f"| Frame train / val | {train_frames} / {val_frames} |\\n"
    f"| Kelas kurang sampel | {', '.join(underrepresented_classes) if underrepresented_classes else 'tidak ada'} |\\n"
))
display(Markdown(
    "Angka di atas **tidak** menetapkan label. Label tetap hasil anotasi NB02b; tabel ini bukti bahwa "
    "pemisahan `Normal`/`Silau`/`Berwarna` punya jejak terukur. Kalau dua kelas bertumpuk total, "
    "definisi operasionalnya perlu dipertegas di dokumen riset."
))'''


def build_notebook() -> nbformat.NotebookNode:
    notebook_cells = [
        nbformat.v4.new_markdown_cell(NAVBAR_MARKDOWN),
        nbformat.v4.new_code_cell(CONFIG_SOURCE),
        nbformat.v4.new_code_cell(LOAD_SOURCE),
        nbformat.v4.new_code_cell(AUDIT_SOURCE),
        nbformat.v4.new_code_cell(PHOTOMETRY_SOURCE),
        nbformat.v4.new_code_cell(FIGURE_SOURCE),
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
