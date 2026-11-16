
> **Status tindak lanjut 12 November 2026:** temuan review ditangani atau diberi batasan dalam [PONYTAIL_REMEDIATION_2026-11-12.md](PONYTAIL_REMEDIATION_2026-11-12.md). File ini dipertahankan sebagai rekaman review asli; bila ada perbedaan, gunakan status remediasi terbaru.

What this change does:
It moves the research pipeline from 9 lamp classes to 5 (Mobil/Motor × Standar/TidakStandar + PantulanCahaya), shrinks the YOLO detector to 2 classes, and changes the official counting unit from "unique vehicles (ByteTrack)" to "non-standard bbox per frame". The attribute classifier is replaced: a 35-float HSV MLP becomes a frozen-then-finetuned ResNet18 fused with the same numeric features, and NB06 now runs that CNN on every detection. The dashboard is rebuilt on the new artifacts. Load I assumed: one researcher on one laptop; one Python process per notebook, 5 CCTV files (2 test cameras ≈ 2–3 min each at 25 fps); the Streamlit dashboard is a single local session.
Must fix
1. Smoke runner calls contract keys and a class that no longer exist — builders/smoke_pipeline.py:114-131, 171-178, 186
- What this is: check_counting_contract() still reads class_counts["PantulanCahaya"] (line 128) and the NB05 call passes "jumlah_box_mobil" / "jumlah_box_motor" (lines 175-176), keys this change renamed to jumlah_box_*_tidak_standar.
- Problem: after you create the classifier checkpoint, python builders/smoke_pipeline.py dies with KeyError: 'PantulanCahaya' (NB05's class_counts now holds only Mobil/Motor), and after fixing that it dies on KeyError: 'jumlah_box_mobil'. The documented smoke path (README.md:156) is dead at the verification step.
- Fix: pass the two real box keys, and skip the partition assert when the totals are n/a; make the PantulanCahaya term .get(..., 0).
- If we skip it: your next smoke run fails at the last step, right after the slow YOLO+CNN part, and looks like a pipeline bug.
2. A counted camera with zero non-standard boxes disappears from counting_by_video.csv — builders/build_nb06_tracking_count.py:692-712
- What this is: per_video_rows loops over official_bbox_frame.groupby("video_stem"), so videos with no non-standard bbox produce no row.
- Problem: two failures follow. With zero rows, pd.DataFrame([]) has no columns, so the CSV is 2 bytes and the dashboard jumps to STATE_ERROR ("tidak bisa dibaca") although NB06 finished; the current output/smoke/nb06/counting_by_video.csv is exactly this. With only some rows missing, dashboard/data.py:136-144 reports that camera as tidak ada video even though its video was processed and counted.
- Fix: build the rows from the counted scan_jobs (split test) and sum official_bbox_frame per video_stem with a zero default, so every processed camera gets a row.
- If we skip it: the headline table can silently hide a CCTV point or crash the dashboard on a clean run.
Should fix
3. Published key says "vehicles", value is bbox-per-frame — pipeline_common.py:38-45, builders/build_nb06_tracking_count.py:870-873, dashboard/app.py:31-40, 441-442
- What this is: COUNT_CONTRACT keys are still jumlah_kendaraan_mobil/jumlah_kendaraan_motor and are written into counting_summary.json / counting_by_video.csv, while vehicles_counted and bbox_counted now hold the same bbox number.
- Problem: anyone reading the JSON (or a thesis table) quotes "4 312 kendaraan mobil" for what is 4 312 box-frames — one car stopped in traffic can supply 200 of them. vehicles_counted duplicating bbox_counted keeps that misreading alive, and the camera table also shows two columns (Silau mobil, Silau motor) that are permanently n/a because those keys are gone.
- Fix: rename the two keys to jumlah_bbox_mobil_tidak_standar / jumlah_bbox_motor_tidak_standar, drop vehicles_counted from the per-video rows (keep unique_vehicles_audit in the summary), and delete the two n/a legend columns plus the six dead CONTRACT_LABELS entries.
- If we skip it: the number is correct in code and wrong in every document that quotes it.
4. NB05 now publishes a contract table that is n/a by construction — builders/build_nb05_evaluation.py:462-476, pipeline_common.py:46-52, test_pipeline_logic.py:174-180
- What this is: UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS is computed as "contract class ∩ TRAINING_CLASS_NAMES"; contract classes are annotation names, training classes are Mobil/Motor, so it always contains all keys.
- Problem: counting_by_camera.csv is left with camera_id, frames, and two n/a columns — the only per-camera table of NB05 carries no counts. The test asserts that all-n/a result, so it locks the emptiness instead of catching a regression.
- Fix: delete the box-contract block from NB05, delete UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS, and delete its assertion; NB05 already reports class_counts per class, which is what its title promises.
- If we skip it: a reader opens the NB05 per-camera artifact, sees nothing, and cannot tell whether the model found nothing or the code gave up.
5. The CNN runs once per bbox per frame, so the same vehicle is re-classified every frame — builders/build_nb06_tracking_count.py:372-383, 456
- What this is: classify_vehicle_condition() runs a full ResNet18 forward, batch size 1, for every Mobil/Motor box on every frame; nothing is cached per track_id.
- Problem: the official number is per-frame, so a classifier that flips on one frame adds a box and on another removes one — the total depends on classifier noise, not traffic. It also adds one extra forward per box on top of YOLO's per-frame forward, with no batching.
- Fix: cache the result in a dict keyed by (video_stem, track_id) inside process_video_tracking and reuse it; one vehicle is classified once and the count stops jittering.
- If we skip it: you get a slower NB06 and a headline number you cannot defend, because re-running it can change the total without any code change.
6. builders/build_reference_patches.py:14-19, 57-72 still selects removed classes and reports success
- What this is: CLASS_GROUPS asks for MobilSilau, MotorSilau, MobilNormal, MotorNormal; after the migration no box carries those names.
- Problem: main() writes an empty index and prints 0 patch -> ... with exit code 0, so the R17 reference-patch step (MASTER_DOKUMENTASI_HAMPIR.md:1428) quietly produces nothing.
- Fix: delete the script and its doc line — the class definitions it was building references for (silau/berwarna) no longer exist in the taxonomy.
- If we skip it: a future run "succeeds" and leaves an empty reference set that nothing warns about.
7. Dashboard cannot tell old artifacts from new ones — dashboard/data.py:169-209
- What this is: the loader checks only that two files exist and parse; it never checks the taxonomy.
- Problem: output/smoke/nb06/counting_summary.json on this machine still holds MobilNormal…PantulanCahaya. Because of finding 2 it errors today, but as soon as a stale summary has rows the page renders 0 for every class with no warning, which DESIGN.md:44 explicitly forbids ("n/a must not become 0", numbers must not lie).
- Fix: one check in load_counting_snapshot — if set(summary["class_counts"]) != set(CLASS_NAMES), return STATE_ERROR with "artefak dari versi taksonomi lama; jalankan ulang NB06".
- If we skip it: you will eventually screenshot zeros and believe them.
8. set_index("epoch") crashes when the column is missing — dashboard/app.py:274, 296
- What this is: both guards filter the column list for presence but then index on "epoch", which is not part of the filtered list.
- Problem: a results.csv or training_history.csv without an epoch column (trimmed or older export) raises KeyError and kills the whole page, even though all other chart columns are available.
- Fix: add "epoch" to the required set — only chart when "epoch" in frame.columns and at least one value column exists.
- If we skip it: one unexpected metrics file takes the entire dashboard down at the demo.
Nice to have
9. Dead code after early returns and a stale 9-class caption — dashboard/app.py:342-359, 483-511, 527-536, imports at 15/26, builders/build_nb06_tracking_count.py:617-628
- What this is: three unreachable blocks in app.py (vehicle selectbox, contract columns, 5-class chart — its caption still describes "sembilan event akhir"), four imports only used there, and in NB06 a unique-vehicle timeline that is computed and then overwritten at line 663.
- Problem: the dead NB06 block also writes counting_timeline.csv before the real write, so two writers touch one file; the dead caption tells the next reader the taxonomy is still 9 classes.
- Fix: delete all of it.
- If we skip it: the next person edits the wrong block and wonders why nothing changes.
10. Legend red is not the overlay red, and manual radius fails the design pre-flight — dashboard/app.py:50-52, 244-250
- What this is: EVENT_COLORS["TidakStandar"] = "#E53935" while NB06 draws (0,0,255) BGR (= #FF0000); the legend HTML hardcodes border-radius:3px and #777.
- Problem: dashboard/DESIGN.md:86 counts manual radii and mismatched event colors as bugs, not taste, and the legend swatch does not match the pixels in the video it explains.
- Fix: set #FF0000 and drop the two inline border-radius/border styles (plain text rows).
- If we skip it: the design bar file you wrote is already false, so it stops being enforced.
11. Stitching tests use class names that no longer exist — test_pipeline_logic.py:129, 137, 146, 153-156
- What this is: MobilNormal, MotorBerwarna, MobilSilau, MotorNormal in the fragment-stitching and tracker-guard checks.
- Problem: they pass only because class_family() matches by prefix, so they no longer test a class the pipeline can produce; nothing covers the new case where a fragment flips MobilStandar → MobilTidakStandar (both resolve to Mobil and get stitched into one vehicle).
- Fix: use current names and add one assert that MobilStandar + MobilTidakStandar fragments of the same vehicle merge.
- If we skip it: the suite stays green while testing a vocabulary that was deleted.
Verdict: not mergeable yet — finding 1 breaks the smoke run you are about to launch, and finding 2 can hide a CCTV point or fake an artifact error on a successful run. Fix those two first, then 3–6 in the same pass.
Lean: -200 lines possible. (≈70 from app.py dead blocks + labels + imports, 76 by deleting build_reference_patches.py, 26 from the NB05 box-contract and its assertion, 15 from the unused AttributeCNN class and its load_classifier branch, 12 from the overwritten NB06 timeline, 6 from MEASURABLE_CONTRACT_KEYS.)
Not checked: the actual end-to-end run (NB06 needs an output/attribute_classifier/best.pt that does not exist yet), CNN training quality/hyperparameters beyond reading them, NB02c_ground_truth and migrate_annotation_taxonomy.py behavior on real data, and whether the dashboard's numbers are correct — only whether they are honest.