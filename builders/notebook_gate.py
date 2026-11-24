#!/usr/bin/env python3
"""Gerbang verifikasi notebook: struktur dan sintaks sel konsisten, tanpa output tertanam."""

import argparse
import runpy
import sys
import tempfile
from pathlib import Path

import nbformat


FORBIDDEN_SNIPPETS = ("__file__", "Tugas_Manda")

BUILDER_NOTEBOOK_PAIRS = (
    ("build_nb01_acquisition.py", "NB01_acquisition.ipynb", "build_notebook"),
    ("build_nb02_pseudo_label.py", "NB02_pseudo_label.ipynb", "build_notebook"),
    ("build_nb02b_annotation.py", "NB02b_annotation.ipynb", "build_notebook"),
    ("build_nb02c_ground_truth.py", "NB02c_ground_truth.ipynb", "build_ground_truth_notebook"),
    ("build_nb02d_finetune.py", "NB02d_finetune_grounding_dino_base.ipynb", "build_notebook"),
    ("build_nb02e_grounded_sam2.py", "NB02e_grounded_sam2_masks.ipynb", "build_notebook"),
    ("build_nb03_gold_threshold.py", "NB03_gold_threshold.ipynb", "build_notebook"),
    ("build_nb04_train_yolo.py", "NB04_train_yolo.ipynb", "build_notebook"),
    ("build_nb04b_attribute_classifier.py", "NB04b_attribute_classifier.ipynb", "build_notebook"),
    ("build_nb05_evaluation.py", "NB05_evaluation.ipynb", "build_notebook"),
    ("build_nb06_tracking_count.py", "NB06_tracking_count.ipynb", "build_notebook"),
)


def build_from_source(root: Path, builder_name: str, function_name: str):
    builders_dir = root / "builders"
    for entry in (str(root), str(builders_dir)):
        if entry not in sys.path:
            sys.path.insert(0, entry)
    namespace = runpy.run_path(
        str(builders_dir / builder_name),
        run_name=f"_notebook_gate_{Path(builder_name).stem}",
    )
    if function_name not in namespace:
        raise RuntimeError(f"{builder_name} tidak menyediakan {function_name}()")
    return namespace[function_name]()


def verify_all_notebooks(root: Path, *, sync_sources: bool = False) -> None:
    for builder_name, notebook_name, function_name in BUILDER_NOTEBOOK_PAIRS:
        notebook_path = root / notebook_name
        generated = build_from_source(root, builder_name, function_name)
        with tempfile.TemporaryDirectory(prefix="notebook_gate_") as temporary_directory:
            temporary_notebook = Path(temporary_directory) / notebook_name
            nbformat.write(generated, temporary_notebook)
            generated = nbformat.read(temporary_notebook, as_version=4)
        committed = nbformat.read(notebook_path, as_version=4)
        nbformat.validate(generated)
        if len(generated.cells) != len(committed.cells):
            raise SystemExit(
                f"{notebook_name}: jumlah sel builder ({len(generated.cells)}) berbeda "
                f"dengan notebook ({len(committed.cells)}); sinkronkan struktur secara manual."
            )
        changed_cells = []
        for index, (source_cell, committed_cell) in enumerate(zip(generated.cells, committed.cells)):
            if source_cell.cell_type != committed_cell.cell_type:
                raise SystemExit(f"{notebook_name}: tipe sel berbeda pada indeks {index}.")
            if source_cell.cell_type == "code":
                python_source = "\n".join(
                    line for line in source_cell.source.splitlines()
                    if not line.lstrip().startswith(("%", "!"))
                )
                compile(python_source, f"{notebook_name}:cell-{index}", "exec")
            if source_cell.source == committed_cell.source:
                continue
            changed_cells.append(index)
            if sync_sources:
                committed_cell.source = source_cell.source
                if committed_cell.cell_type == "code":
                    committed_cell.outputs = []
                    committed_cell.execution_count = None
        if changed_cells and not sync_sources:
            raise SystemExit(
                f"{notebook_name}: source sel {changed_cells} berbeda dari builder. "
                "Jalankan python builders/notebook_gate.py --sync-sources lalu cek ulang."
            )
        if sync_sources and changed_cells:
            nbformat.write(committed, notebook_path)
            print(f"SYNC {notebook_name}: sel berubah {changed_cells}; output sel kode terkait dibersihkan.")
        elif sync_sources:
            print(f"OK   {notebook_name}: source sudah cocok; tidak ada file ditulis.")
        else:
            print(f"OK   {notebook_name}: semua source sel cocok dengan builder.")


def verify_code_cells(
    notebook_path: Path,
    expected_code_cells: int,
    expected_markdown_cells: int,
) -> None:
    """Pastikan notebook hasil builder bersih dan selaras dengan jumlah sel yang diharapkan."""
    loaded_notebook = nbformat.read(notebook_path, as_version=4)
    code_cells = [cell for cell in loaded_notebook.cells if cell.cell_type == "code"]
    markdown_cells = [cell for cell in loaded_notebook.cells if cell.cell_type == "markdown"]
    errors: list[str] = []
    if len(code_cells) != expected_code_cells:
        errors.append(f"jumlah sel kode {len(code_cells)} != {expected_code_cells}")
    if len(markdown_cells) != expected_markdown_cells:
        errors.append(f"jumlah sel markdown {len(markdown_cells)} != {expected_markdown_cells}")
    for cell_index, cell in enumerate(code_cells):
        if cell.outputs:
            errors.append(f"sel kode {cell_index} memiliki output ter-eksekusi")
        for forbidden_snippet in FORBIDDEN_SNIPPETS:
            if forbidden_snippet in cell.source:
                errors.append(f"sel kode {cell_index} memuat {forbidden_snippet}")
        try:
            python_source = "\n".join(
                line for line in cell.source.splitlines()
                if not line.lstrip().startswith(("%", "!"))
            )
            compile(python_source, f"{notebook_path.name}:cell-{cell_index}", "exec")
        except SyntaxError as error:
            errors.append(f"sel kode {cell_index} sintaks tidak valid: {error}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(
        f"Verifikasi OK: {notebook_path} "
        f"({len(code_cells)} sel kode + {len(markdown_cells)} markdown, sintaks valid, output kosong)"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("notebook", type=Path, nargs="?")
    parser.add_argument("--code-cells", type=int)
    parser.add_argument("--markdown-cells", type=int)
    parser.add_argument("--check-all", action="store_true")
    parser.add_argument("--sync-sources", action="store_true")
    arguments = parser.parse_args()
    if arguments.check_all or arguments.sync_sources:
        verify_all_notebooks(Path(__file__).resolve().parent.parent, sync_sources=arguments.sync_sources)
    elif arguments.notebook is not None and arguments.code_cells is not None and arguments.markdown_cells is not None:
        verify_code_cells(arguments.notebook, arguments.code_cells, arguments.markdown_cells)
    else:
        parser.error("gunakan notebook + --code-cells/--markdown-cells atau --check-all/--sync-sources")


if __name__ == "__main__":
    main()
