# Ponytail audit remediation

Status of the 18 findings in `kritik-saran/ponytail-audit.md`, checked against the active five-class
pipeline. Source changes are complete; finding 1 is complete after the scoped commit for this work.

| # | Finding | Status | Remediation |
|---:|---|---|---|
| 1 | Pipeline changes uncommitted | Fixed | Commit the active five-class pipeline, SAM2/e-means integration, dashboard fixes, runner changes, and this remediation record together. User-generated `Laporan/`, `Nanya/`, `Bimbingan.txt`, and unrelated flowchart artifacts are excluded. |
| 2 | Dashboard duplicate measured columns | Fixed | Deduplicate selected columns before numeric conversion in `dashboard/data.py`. |
| 3 | Stale two-class logic tests | Fixed | Assertions and fixtures now use the five-class vocabulary and current crossing/box contracts. |
| 4 | Smoke check uses stale NB05 box-total keys | Fixed | NB05 now emits keys from `BOX_COUNT_CONTRACT`; smoke verifies key-set equality before validating totals. |
| 5 | Dashboard verifier expects old labels | Fixed | Verifier checks the rendered crossing metrics, camera table, and five-class breakdown. |
| 6 | Mask feature index overwritten | Fixed | Removed the second write; feature index 6 retains the valid-mask indicator in every environment. |
| 7 | Closing ROI picker could save partial polygon | Fixed | Only Enter confirms a polygon; closing/cancelling preserves the existing ROI. |
| 8 | Notebook logs passed to auto-editing agent | Fixed | Runner no longer invokes Codex. It saves diagnostics and stops; an optional single retry is explicit and makes no source changes. |
| 9 | Repository folder-name guard | Fixed | Active notebooks resolve the root from marker files (`pipeline_common.py`, `requirements.txt`) rather than folder names. |
| 10 | Annotation store loaded twice | Fixed | NB02b and generated NB02c initialize the store once per notebook pass. |
| 11 | Dense full-resolution PNG frame sampling | Fixed in source | NB01 samples every five seconds based on video FPS and writes JPEG quality 90. Existing outputs are unchanged; this affects the next NB01 acquisition. |
| 12 | Duplicate HSV thresholds | Fixed | NB03 imports the shared thresholds from `attribute_classifier.py`. |
| 13 | Duplicate dHash implementations | Fixed | NB01 and NB02 share dHash and Hamming-window helpers from `pipeline_common.py`. |
| 14 | Dead constants/helpers | Fixed | Removed the listed unused ratio, retention, and NB02b class/path constants. |
| 15 | One-time taxonomy migration in builders | Fixed | Moved to `archive/do_not_run/` with a warning README. |
| 16 | No synthetic count-crossing coverage | Fixed in source | Added a test for stable crossings plus a fragmented track and extracted the production crossing predicate to `pipeline_common.py`. The test has not been run. |
| 17 | Identity annotation-to-training map | Fixed | Removed the identity mapping; NB06 uses the class name directly and the shared validator accepts only active classes. |
| 18 | e-means vendoring/version unclear | Fixed | Documented the FAISS `v1.16.0` pin, why native PyTorch assignment is used, and the installer/source provenance. |

## Validation and rerun notes

- Python AST, PowerShell parser, and notebook JSON/schema checks passed. The suite and training notebooks
  were not run during this remediation.
- NB04 was stopped before completing. The current five-class NB02b export was refreshed and its prior
  export preserved under `output/annotation/_backup_refresh_exports_20261116_173512_174734`.
- To resume with the current dataset, rerun from NB04 after this audit work is committed. NB01's new
  sampling/compression policy applies only if acquiring frames again; applying it requires reviewing
  the newly sampled frames because the frame set changes.
