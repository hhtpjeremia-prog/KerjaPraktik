# Protokol anotasi dan ground truth

> **Kontrak pipeline aktif (2026-11-16):** lima kelas seragam di FM/YOLO/CNN; continued fine-tune dibatalkan. FM pseudo-label hanya tambahan train. NB02e menghasilkan cache mask SAM2 untuk train/val/GT test; NB04b menggabungkan embedding RGB, 77 fitur bbox+mask, dan cluster e-means yang fit pada train saja. Counting resmi adalah satu crossing per track MobilTidakStandar/MotorTidakStandar; garis awal di 72% tinggi frame dan dipotong mengikuti ROI.

> **Cakupan angka:** jumlah curated/ROI di bawah adalah snapshot audit 7 November 2026, bukan jaminan jumlah terbaru. Periksa manifest dan store aktif sebelum mengambil keputusan jumlah sampel. Status koreksi metadata anotasi 12 November dijelaskan di [PONYTAIL_REMEDIATION_2026-11-12.md](PONYTAIL_REMEDIATION_2026-11-12.md).

Gunakan `output/curated_frames/` untuk data anotasi terarah melalui daftar resmi
`output/manifests/curated_frames_manifest.csv`; jangan mengambil semua PNG tanpa manifest. Audit
menemukan 981 PNG di folder dan 528 baris manifest. Dari 528, 194 sudah dianotasi (139 train, 55 val).
Sisa curated adalah 56 train, 24 val, dan 254 test. Curated dipilih manual sehingga cocok sebagai data
train terarah/challenge set, bukan satu-satunya sumber evaluasi representatif.

Simpan ground truth terpisah di `output/ground_truth/`. Ambil sampel evaluasi utama secara sistematis
dari video mentah pada kamera holdout, termasuk frame kosong; jangan gunakan test untuk train atau
memilih threshold. Simpan `video_stem`, `camera_id`, `frame_index`, `timestamp_seconds`, `split`,
`review_status`, dan `annotation_source`, serta daftar bbox `class_name` + `xyxy`. Frame kosong tetap
disimpan dengan `boxes: []`. Bekukan label test setelah review dan laporkan metrik per kamera/kelas.

Anotasi manual memberi lima kelas yang sama untuk Grounding DINO, YOLO, dan CNN. Proposal pseudo FM
ditambahkan hanya ke train; val/test yang dipakai sebagai bukti metrik berasal dari gold labels. GT test
tidak masuk training atau pemilihan threshold. `PantulanCahaya` tetap kelas deteksi tetapi dikeluarkan
dari hitungan kendaraan dan overlay event NB06. Counting utama NB06 adalah satu crossing per vehicle
track tidak standar; bbox-per-frame disimpan terpisah sebagai metrik pendamping.

NB02b/NB02c membangun crop curated yang belum ada di `output/nb02/crops/` dari PNG mentah di `output/curated_frames/` memakai ROI aktif di `output/roi/roi_config.json`, lalu menyimpannya di `output/curated_crops_current_roi/`.


**Perbaikan ROI ground truth (2026-11-07).** Audit awal menemukan 92/254 crop NB02c memakai ROI lama; 162 crop NB02 cocok dengan ROI aktif. Seluruh 92 crop dibuat ulang dari frame sumber dan manifest NB02c diperbarui agar menunjuk ke `output/curated_crops_current_roi/`. Pemeriksaan ulang memastikan semua 254 crop NB02c cocok pixel-per-pixel dengan ROI aktif. Empat frame lama yang sempat berstatus reviewed semuanya kosong dan kini perlu direview; 88 frame lainnya memang belum direview. Seluruh 981 PNG dalam `output/curated_crops/` juga telah dibuat ulang dengan ROI aktif; versi sebelumnya dipertahankan di `output/_backup_curated_crops_old_roi_20261107/`.
NB02 menyimpan fingerprint ROI di `nb02_summary.json`; NB02b/NB02c memverifikasi fingerprint tersebut dan signature chunk sebelum memakai crop/deteksi. Jika konfigurasi ROI berubah, notebook anotasi berhenti sampai NB02 selesai dijalankan ulang dengan ROI aktif.

\n
