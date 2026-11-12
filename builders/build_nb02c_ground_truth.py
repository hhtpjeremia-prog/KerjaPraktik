#!/usr/bin/env python3
"""Build a separate annotation notebook for held-out test ground truth."""
from pathlib import Path
import sys
import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "builders"))
from build_nb02b_annotation import build_notebook

notebook = build_notebook()
notebook.cells[0].source = notebook.cells[0].source.replace(
    "# NB02b — Anotasi Manual Kondisi Lampu",
    "# NB02c — Anotasi Ground Truth Holdout",
).replace(
    "Tahap ini mengisi satu-satunya hal yang **tidak boleh** ditebak model: **kondisi lampu**.",
    "Notebook ini menyimpan label di `output/ground_truth/`, terpisah dari training. Frame curated test yang ada adalah challenge set; tambahkan sampel sistematis dari video mentah dan frame kosong sebelum memakai hasilnya sebagai estimasi representatif.",
)
for cell in notebook.cells:
    if cell.cell_type != "code":
        continue
    cell.source = cell.source.replace(
        'ANNOTATION_ROOT = DATA_ROOT / "annotation"',
        'ANNOTATION_ROOT = DATA_ROOT / "ground_truth"',
    ).replace(
        'SAMPLE_EXCLUDED_SPLITS = ["test"]',
        'SAMPLE_EXCLUDED_SPLITS = ["train", "val"]',
    )
output = ROOT / "NB02c_ground_truth.ipynb"
nbformat.write(notebook, output)
print(f"Generated {output}")
