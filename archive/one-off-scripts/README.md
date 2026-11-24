# Archived one-off scripts

These scripts document historical audits, diagnostics, recovery, and document generation. They are not
part of the active pipeline or `run_checks.ps1`; do not run them as current pipeline stages. Their
original paths are retained in Git history. Active notebook builders remain under `builders/`, including
`build_nb02c_ground_truth.py`, because it is the source for the checked-in NB02c notebook.

The active gold quality audit and holdout export repair remain in `builders/` because README/NB05 still
reference them. The duplicate NB02b export writer was removed; NB02b is the sole writer of COCO and
YOLO annotation exports.
