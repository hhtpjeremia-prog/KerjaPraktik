#!/usr/bin/env python3
"""Gerbang verifikasi notebook: struktur dan sintaks sel konsisten, tanpa output tertanam."""

from pathlib import Path

import nbformat


FORBIDDEN_SNIPPETS = ("__file__", "Tugas_Manda")


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
