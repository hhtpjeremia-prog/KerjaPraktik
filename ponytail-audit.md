
> **Status tindak lanjut 12 November 2026:** temuan audit ditangani atau diberi batasan dalam [PONYTAIL_REMEDIATION_2026-11-12.md](PONYTAIL_REMEDIATION_2026-11-12.md). File ini dipertahankan sebagai rekaman audit asli; bila ada perbedaan, gunakan status remediasi terbaru.

A one-person research pipeline that turns 5 public CCTV streams (Kab. Bogor) into a lamp-condition vehicle count. Notebook builders (builders/build_nb01..nb06) generate notebooks that sample frames, collect human annotation, train YOLO11n + a ResNet18 attribute classifier, then count non-standard lamp bboxes per frame; dashboard/app.py (Streamlit) reads NB06 artifacts and shows the numbers.
Load I assumed: one researcher, one Windows machine with one CUDA GPU, notebooks run by hand in VS Code, dashboard viewed by a handful of people on localhost. No auth, no money, no concurrent writers. That's why "one process" bugs below are bugs, not scale choices.
Must fix
1. class_counts reports 0 for every standard vehicle — builders/build_nb06_tracking_count.py:659-684
- What this is: official_bbox_frame is built from nonstandard_frame, then class_counts is computed from that subset for all 5 class names.
- Problem: MobilStandar, MotorStandar, PantulanCahaya are structurally always 0. The real artifact on disk proves it: output/nb06/counting_summary.json:26-32 says MobilStandar: 0 while 55M raw detections exist, and counting_by_video.csv publishes those zero columns.
- Fix: compute class_counts from qualified_frame (all classes), and keep the non-standard subset only for counting_totals.
- If we skip it: every downstream chart, audit, or report that reads class balance from NB06 states that no standard vehicle was ever seen. That number is already in a CSV someone may paste into a slide.
2. A camera that ran but counted nothing vanishes — builders/build_nb06_tracking_count.py:692-712
- What this is: per_video_rows is built with groupby("video_stem") over rows that already exist; a video with zero counted bboxes produces no row.
- Problem: the dashboard's camera status only checks "is this camera in the table" (dashboard/data.py:136-144), so a processed-but-empty camera is shown as tidak ada video. Live evidence: output/smoke/nb06/counting_by_video.csv is 2 bytes while the summary says videos_counted: 1.
- Fix: build rows from the counted video list and fill 0 for missing videos.
- If we skip it: you will debug a "camera not downloaded" problem that is actually a "camera found nothing" result.
3. An empty counting_by_video.csv crashes the dashboard — dashboard/data.py:181-187 + dashboard/data.py:91-105
- What this is: pd.read_csv on a header-less/empty file raises EmptyDataError (a ValueError), which is caught and turned into STATE_ERROR.
- Problem: the user sees "Artefak NB06 ada tetapi tidak bisa dibaca" instead of 0. The 2-byte smoke file is exactly that case, today. If the file read somehow succeeds, build_camera_table then does groupby("camera_id") on a frame with no columns and raises KeyError.
- Fix: treat an empty file as an empty frame with the expected columns, and return zeros.
- If we skip it: the honest-zero case looks like a broken pipeline every time a run finds nothing.
4. The end-to-end smoke check cannot pass — builders/smoke_pipeline.py:128,171-187
- What this is: the only whole-pipeline check still uses the pre-rename contract.
- Problem: it passes "jumlah_box_mobil" / "jumlah_box_motor", but the real keys are jumlah_box_mobil_tidak_standar / ..._motor_... → KeyError at totals[mobil_key]. Line 128 also indexes class_counts["PantulanCahaya"] on the NB05 payload, which only contains Mobil/Motor. The partition math is stale too.
- Fix: for NB05 partition over TRAINING_CLASS_NAMES; for NB06 over CLASS_NAMES; use the real contract keys.
- If we skip it: nobody can prove the numbers are consistent; every regression is found by staring at the dashboard.
5. The classifier trains on augmented features but normalizes with clean-crop statistics — builders/train_attribute_classifier.py:81-84 vs :198-204
- What this is: CropDataset applies TRAIN_AUGMENT (brightness ±0.20, contrast ±0.20, saturation ±0.08) before extract_pixel_features, while feature_mean/feature_std are computed from un-augmented crops.
- Problem: all 35 numeric features that feed the fusion head come from a jittered distribution that is then divided by clean-image statistics, and at eval/serve time the input is clean. Train and serve see different feature scales — exactly the failure that never throws an error.
- Fix: move feature extraction above the augmentation call (or compute stats from augmented crops).
- If we skip it: val F1 looks acceptable while the served model is quietly mis-normalized; you will chase this as "the CNN is just weak".
Should fix
6. build_reference_patches.py selects class names that no longer exist — builders/build_reference_patches.py:15-18
- What this is: a one-off script still asking for MobilSilau, MotorSilau, MobilNormal, MotorNormal.
- Problem: no annotation can match, so it exits cleanly with an empty index — a silent no-op. Grep confirms nothing imports it.
- Fix: delete the file (76 lines).
- If we skip it: someone runs it for a paper figure and concludes no reference patches qualify.
7. NB05 publishes a counting table that is n/a by construction — builders/build_nb05_evaluation.py:454-476
- What this is: box_totals maps BOX_COUNT_CONTRACT (annotation classes MobilTidakStandar) against class_counts built from TRAINING_CLASS_NAMES (Mobil).
- Problem: both contract keys are always "n/a", yet counting_by_camera.csv is written and shown as a result. test_pipeline_logic.py:174-180 asserts this all-n/a state, locking it in. It also duplicates NB06's counting loop.
- Fix: delete the contract block and the CSV, or map contract classes through training_class_name.
- If we skip it: two "counting" tables exist, one of them permanently empty, and readers cannot tell which is authoritative.
8. NB06 writes the timeline twice; the first write is dead — builders/build_nb06_tracking_count.py:617-628 vs :663-676
- What this is: timeline_frame is written to COUNTING_TIMELINE_PATH, then reassigned to the bbox-level frame and written to the same path.
- Problem: 81 MB (counting_timeline.csv) written and discarded. The scope/fps/timestamp columns are also computed twice.
- Fix: delete 620-628.
- If we skip it: double the I/O on the slowest artifact for zero output.
9. Three dead blocks after return in the dashboard — dashboard/app.py:342-359, :482-499, :526-535
- What this is: ~110 lines of the old 9-class UI sitting after unconditional returns.
- Problem: the old silau/berwarna contract and the 9-class bar chart are still in the file, with CONTRACT_LABELS entries that no longer have keys. A reader edits dead code and wonders why nothing changes.
- Fix: delete the three blocks and the orphan labels.
- If we skip it: the file says two contradictory things about what the dashboard shows.
10. The dashboard theme is gitignored — .gitignore:30 + .streamlit/config.toml
- What this is: .streamlit/ is ignored, but config.toml holds the project palette (#1F7A4D, light/dark blocks) that the app's styling assumes.
- Problem: a fresh clone or a deploy gets Streamlit defaults; the STYLE_BLOCK in app.py:55-63 was tuned against that palette.
- Fix: un-ignore .streamlit/config.toml only.
- If we skip it: the deliverable looks different on every machine and nobody knows why.
11. Caption states a rule that no longer applies — dashboard/app.py:102-104
- What this is: "minimal N kemunculan per kendaraan" is printed from summary["min_track_hits"].
- Problem: the official count no longer filters on MIN_TRACK_HITS (build_nb06_tracking_count.py:678-680 takes bbox_counted_frame with no hits filter). In the real artifact, min_track_hits: 3 is shown next to a number produced without it.
- Fix: drop the clause, or word it as applying to the audit number only.
- If we skip it: the dashboard makes a methodological claim the data does not support — the worst kind of bug in a research deliverable.
12. Tests still use removed class names — test_pipeline_logic.py:129,137,146,153-156
- What this is: check_adjacent_fragments_stitched uses MobilNormal, check_chain_of_fragments_stitched uses MotorBerwarna, check_tracker_threshold_guard uses MobilSilau/MotorNormal.
- Problem: they pass only because class_family does startswith("Mobil"). The reflection test no longer proves reflection separation for classes that exist.
- Fix: rename to the 5 live classes.
- If we skip it: the safety net for taxonomy changes is made of strings that cannot occur in production.
13. The checkpoint contract is checked in the wrong place — builders/build_nb06_tracking_count.py:369,375-376
- What this is: load_classifier runs at 369, the model_type != "resnet18_hsv_fusion" guard runs inside the per-box function at 375.
- Problem: the on-disk output/attribute_classifier/metrics.json says model_type: resnet18 (old revision). A run with zero vehicle boxes never triggers the guard and finishes with an empty result; a run with boxes loads the wrong model first, then dies minutes in.
- Fix: check attribute_metadata["model_type"] immediately after load_classifier, and name the retrain command in the message.
- If we skip it: the failure appears mid-video, or not at all.
14. The gold set is deleted before it is re-exported — builders/build_nb02b_annotation.py:1192-1195
- What this is: every *.png and *.txt in the annotation YOLO dir is unlinked, then re-copied one frame at a time (1213-1216).
- Problem: the missing-source guard at 1186-1191 covers one failure mode only. A crash mid-loop (disk full, locked file) leaves a half-written dataset with no backup — while annotations.json pruning at 506-510 does take a backup.
- Fix: export to a temp dir and swap at the end, or back up the old gold root first.
- If we skip it: one failed export trains NB04 on a partial dataset that looks complete.
15. The same transform exists twice — attribute_classifier.py:161-166 and builders/train_attribute_classifier.py:55-58
- What this is: IMAGE_TRANSFORM and IMAGE_TENSOR_TRANSFORM are identical Compose chains.
- Problem: they must stay byte-identical (same resize/pad/normalize) or training and serving disagree. Nothing enforces that.
- Fix: import IMAGE_TRANSFORM in the trainer.
- If we skip it: one mean= edit in one file silently splits train from inference.
16. The same number is published in three columns — builders/build_nb06_tracking_count.py:701-702,870-871 + dashboard/data.py:99-114
- What this is: vehicles_counted and bbox_counted are both written with len(official_bbox_frame), and both are summed into the camera table alongside the two contract keys.
- Problem: three identical columns in a public table; a reader cannot tell which one is authoritative. app.py:138 already falls back bbox_counted → vehicles_counted, which only exists because of this duplication.
- Fix: keep bbox_counted; drop vehicles_counted (or vice-versa) and the fallback.
- If we skip it: a presentation bug becomes permanent once someone sums two of the three columns.
Nice to have
17. Dead model classes and dead labels — attribute_classifier.py:82-98,184-196; dashboard/app.py:31-40
- What this is: AttributeMLP, AttributeCNN, and the model_type == "resnet18" / default branches are unreachable from current callers; CONTRACT_LABELS still maps 6 silau/berwarna keys.
- Problem: grep shows AttributeCNN is only referenced by load_classifier and create_project_brief.py:260 prose. It reads as a supported format.
- Fix: delete the classes and the legacy branches; keep one clear error for unknown model_type.
- If we skip it: a future reader wires up the legacy path and gets a confusing shape error.
18. Two writers, one file — builders/build_nb03_gold_threshold.py:214,356
- What this is: gold_box_frame.to_csv(GOLD_BOXES_PATH) appears twice in the same build script.
- Problem: the second write masks a failure in the first, and the file is written twice.
- Fix: keep one call.
- If we skip it: nothing breaks today; it is just one more place to keep in sync.
19. Tests need a human to remember them — requirements.txt, builders/verify_dashboard_states.py:98-102
- What this is: no pytest, no CI, no runner. test_pipeline_logic.py and verify_dashboard_states.py are scripts, and the latter asserts exact element counts (== 5 dataframes, == 4 metrics).
- Problem: any legitimate UI tweak breaks the check, and nothing runs either script automatically, so the check is only as good as memory.
- Fix: pytest + one pytest.ini that collects both; assert on content, not widget counts.
- If we skip it: the suite stays green-by-absence, and finding 12 never gets caught.
20. Fragment stitching is quadratic — pipeline_common.py:203-269
- What this is: for each track row, loop over all open groups of the same family.
- Problem: today it runs on 2,242 tracks (vehicles_total: 2063) and is only used for presentation/tests, so it is fine. It is O(tracks × open groups) and grows with longer videos.
- Fix: leave it; if a run ever stalls here, group by family and frame window first.
- If we skip it: nothing. This is a note, not a task.
Verdict: Not shippable as a research result yet, but the core idea is sound. Three concrete correctness bugs sit between you and an honest number: standard-class counts are always 0 (#1), a zero-result camera disappears or errors the dashboard (#2, #3), and the classifier is normalized against the wrong distribution (#5). The smoke check that should have caught the first three is itself broken (#4). Fix Must + #6/#7 (deletions only) and you have a pipeline whose output means what the README says.
Lean: -250 lines possible, -0 dependencies possible. (≈110 from app.py dead blocks, 76 from build_reference_patches.py, 30 from NB05's n/a contract, 26 from AttributeCNN/AttributeMLP, 10 from NB06's dead timeline, 6 from CONTRACT_LABELS, 4 from the duplicated transform, 2 from the double NB03 write; the two duplicate count columns save 2 more. All 17 pins are used — ipympl/ipywidgets by the annotation UI.)
Not checked: NB01..NB06 notebook cells themselves (only the builders that generate them), build_nb02c_ground_truth.py, build_legacy_aware_curated_manifest.py, build_roi_config.py, and the one-off diagnostics (audit_gold_quality.py, bench_nb02_verifier.py, diag_blob_stats_nb02.py, manual_check_nb02.py, mine_blue_candidates.py, probe_label_definition.py, extract_cv_knowledge.py, restore_frames_from_manifest.py) — I read their headers and greps only, not their bodies. I also did not open the 81 MB counting_timeline.csv or tracking_detections.csv, and ran nothing.
Skipped: everything under output/ beyond the two NB06 summaries and one CSV header, and the runs/ plots. Risk: findings #1/#2 are proven from artifacts on disk; #5 is reasoning from code only, with no training log in hand to confirm the magnitude.