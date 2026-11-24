#!/usr/bin/env python3
"""Build a separate annotation notebook for held-out test ground truth."""

from pathlib import Path
import sys

import nbformat

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "builders"))
from build_nb02b_annotation import (
    EXPECTED_CODE_CELL_COUNT,
    EXPECTED_MARKDOWN_CELL_COUNT,
    build_notebook,
)
from notebook_gate import verify_code_cells


def build_ground_truth_notebook() -> nbformat.NotebookNode:
    notebook = build_notebook()
    first_lines = notebook.cells[0].source.splitlines()
    if first_lines and first_lines[0].startswith("# NB02b"):
        first_lines[0] = first_lines[0].replace("NB02b", "NB02c", 1)
        notebook.cells[0].source = "\n".join(first_lines)
    notebook.cells[0].source = notebook.cells[0].source.replace(
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
    return notebook


def main() -> None:
    output = ROOT / "NB02c_ground_truth.ipynb"
    nbformat.write(build_ground_truth_notebook(), output)
    verify_code_cells(output, EXPECTED_CODE_CELL_COUNT, EXPECTED_MARKDOWN_CELL_COUNT)
    print(f"Generated {output}")


if __name__ == "__main__":
    main()
