# Riset-HampirMatang

> **Kontrak aktif 2026-11-18 (menggantikan kontrak lama di bawah).** Anotasi, Grounding DINO base,
> YOLO, dan CNN memakai lima kelas sama: `MobilStandar`, `MobilTidakStandar`, `MotorStandar`,
> `MotorTidakStandar`, `PantulanCahaya`. `BelumDitentukan` bukan kelas model. Continued fine-tune
> dibatalkan; notebook continued dihapus dan output/checkpoint lama hanya arsip. Pseudo-label FM
> dapat menambah data train, tetapi bukan gold untuk validation/test. Grounded-SAM 2 wajib membuat
> mask untuk overlay video NB06; fitur turunannya (HSV/luminance relatif, saturation/clipping,
> luas/bentuk mask, komponen dan distribusi spasial cahaya) serta e-means menjadi input CNN.
> Pantulan tetap dilatih,
> tetapi tidak dihitung sebagai kendaraan atau digambar sebagai event.

Pipeline riset untuk mendeteksi kondisi lampu kendaraan, menampilkan mask SAM2, dan menghitung
track kendaraan tidak standar saat melintasi garis virtual. Bbox-per-frame disimpan sebagai metrik
pendamping. Notebook pipeline utama yang memiliki builder Python di root **dibangun** dari
`builders/build_nb*.py`; jangan edit `.ipynb` langsung. Setelah mengubah builder, jalankan
`python builders/notebook_gate.py --sync-sources` untuk menyamakan source sel. Output sel kode yang
source-nya berubah dibersihkan; output sel lain dipertahankan. `run_checks.ps1` memverifikasi
kesesuaian notebook dan builder.

## Kontrak label dan model

Kelima kelas yang sama dipakai oleh Grounding DINO, YOLO, dan CNN. Proposal FM diterima sebagai
tambahan pseudo-label train sesuai kebijakan proyek; hasil pseudo tidak dipakai untuk mengukur
akurasi validation/test. NB02e membangkitkan mask SAM2 dari bbox. NB04b menggabungkan embedding RGB,
77 fitur numerik bbox+mask (fitur geometri ke-7 bernama `valid_mask`, untuk membedakan mask valid
dari mask kosong), dan cluster e-means yang dipelajari hanya dari train. NB06 menghitung
ulang fitur dan cluster dengan mask SAM2 per bbox.

## Menyiapkan environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

`torch`/`torchvision` di pin memakai build CUDA 12.4. Untuk mesin tanpa CUDA, ganti dengan build CPU
dari indeks PyTorch resmi; sisa paket tetap sama.

## Urutan menjalankan

### Rerun terarah setelah Audit 3 (2026-11-18)

Perubahan Audit 3 menyentuh logic NB05/NB06 dan dashboard, bukan data pseudo-label, anotasi manual, Ground
Truth, mask, atau bobot model. Jika seluruh input upstream sudah lengkap dan konsisten, jalankan:

```powershell
.\builders\run_pipeline.ps1 -StartAt NB05
```

Perintah itu menjalankan NB05 kemudian NB06. Pastikan tidak ada proses aktif yang sedang menjalankan
NB05/NB06 dari VS Code atau runner. NB02b/NB02c/NB02d/NB02e/NB03/NB04/NB04b tidak perlu diulang untuk
perbaikan Audit 3. Rerun penuh hanya diperlukan jika data, mask, konfigurasi, atau bobot upstream berubah.
Detail dan batas checkpoint ada di [`builders/PIPELINE_RUNNER.md`](builders/PIPELINE_RUNNER.md).

Runner menjalankan pemeriksaan logika pipeline dan dashboard terlebih dahulu melalui
`builders/run_checks.ps1`. Jalankan pemeriksaan yang sama secara mandiri dengan
`.\builders\run_checks.ps1` dari PowerShell.

Jalankan per sel, berurutan, di VS Code dengan kernel `.venv` dan **cwd = root proyek** (notebook
mengandalkan `from pipeline_common import ...` tanpa mengubah `sys.path`).

1. `NB01_acquisition.ipynb` â€” akuisisi video, sampling frame, dan split per kamera.
2. `NB02_pseudo_label.ipynb` â€” proposal lokasi objek dan crop ROI; proposal bukan ground truth.
3. `NB02b_annotation.ipynb` — anotasi manual train/validation yang tersedia (butuh `ipympl`).
4. Jalankan hanya `NB02d_finetune_grounding_dino_base.ipynb`; continued fine-tune dibatalkan.
5. Proposal base ditambahkan sebagai pseudo-label train di NB04/NB04b; pseudo val/test tidak dipakai.
6. `NB02c_ground_truth.ipynb` — anotasi holdout test; hanya untuk evaluasi.
7. `NB02e_grounded_sam2_masks.ipynb` — membuat mask untuk box manual train/val, GT test, serta
   pseudo-label train; lanjutkan sampai manifest berstatus `complete`.
8. `NB03_gold_threshold.ipynb` — audit/karakterisasi gold set.
9. `NB04_train_yolo.ipynb` — latih detector lima kelas, pseudo FM ditambahkan ke train saja.
10. Sebelum NB04b, pasang source e-means satu kali:
    `powershell -ExecutionPolicy Bypass -File builders/install_emeans.ps1`.
11. `NB04b_attribute_classifier.ipynb` — latih ResNet18 lima kelas dengan fitur mask/e-means;
    pseudo FM hanya menambah train.
12. `NB05_evaluation.ipynb` — kalibrasi/evaluasi dengan gold validation/test bila GT tersedia.
13. `NB06_tracking_count.ipynb` — PantulanCahaya dikeluarkan dari hitungan kendaraan dan overlay event.

### Jalankan berurutan dan tangani error

Runner PowerShell dan penanganan manual saat ada notebook gagal dijelaskan di
[`builders/PIPELINE_RUNNER.md`](builders/PIPELINE_RUNNER.md). Karena NB02d sedang berjalan saat runner
disiapkan, setelah training itu selesai gunakan:

```powershell
.\builders\run_pipeline.ps1 -StartAt NB02e
```

Saat notebook gagal, runner menyimpan log dan prompt diagnosis untuk ditinjau manusia. Runner tidak
menjalankan agent yang dapat mengubah source berdasarkan teks output notebook. Ringkasan konteks proyek
untuk diagnosis ada di [`builders/CODEX_RUNNER_CONTEXT.md`](builders/CODEX_RUNNER_CONTEXT.md).
Dokumen audit lama yang telah digantikan dipertahankan di [`archive/superseded-project-notes/`](archive/superseded-project-notes/README.md); kontrak aktif ada di master dokumentasi dan kode sumber.

Untuk menjalankan alur komputasi dari NB02d pada kesempatan lain, gunakan
`-StartAt NB02d` hanya jika NB02d tidak sedang berjalan di proses lain. NB02b dan NB02c tetap perlu
diselesaikan secara interaktif terlebih dahulu.

NB02b sekarang mensyaratkan jarak minimum 30 frame antar frame sampel dalam kamera yang sama
(sekitar 1,2 detik pada 25 fps). Manifest sampling lama akan ditolak dan harus dibuat ulang agar
frame berurutan yang identik tidak masuk gold set. Tombol/hotkey `r` hanya menandai frame reviewed jika tidak ada bbox aktif berlabel `BelumDitentukan`; perubahan bbox sendiri tidak menyelesaikan review satu frame.

Kamera `bappenda2` dan `pdam2` adalah split `test`: tidak pernah masuk train/val. Label evaluasi
holdout disimpan terpisah di `output/ground_truth/` melalui NB02c dan tidak boleh dipakai untuk fit.

## Classifier atribut cahaya

YOLO dan CNN memakai lima kelas identik. NB04b melatih ResNet18 supervised dengan embedding RGB, 77 angka (35 fitur HSV/statistik bbox, 35 fitur foreground mask SAM2, 7 geometri mask), dan one-hot cluster e-means. Jumlah cluster dipilih dengan BIC yang membandingkan objective linear dengan model segmented-linear ber-elbow interior, dari kandidat `k=2..12` dengan tiga restart per kandidat. Jika tidak ada elbow yang didukung BIC, training berhenti setelah menyimpan seluruh objective kandidat ke `emeans_k_selection.csv`; tidak ada fallback ke jumlah cluster tetap. Kurva yang berhasil dipilih juga disimpan sebagai `emeans_k_selection.png`. Mask kosong menghasilkan vektor foreground nol, dicatat pada `mask_quality.json`, dan menghentikan training jika melebihi 1%. Scaler dan centroid hanya di-fit dari train; validation memilih checkpoint, test dievaluasi terhadap GT yang tersedia. E-means tidak memakai label kelas dan tidak menjamin peningkatan akurasi. Weighted CrossEntropy menangani ketimpangan kelas. HSV/value dan mask adalah proksi citra, bukan pengukuran lux/candela.

e-means memakai implementasi resmi Meta FAIR dari FAISS `v1.16.0`, vendored karena jalur assignment
FAISS tidak dibutuhkan dan tidak praktis pada Windows; provenance dan pemasangannya dijelaskan di
[`vendor/faiss_e_means/UPSTREAM.md`](vendor/faiss_e_means/UPSTREAM.md).

Training sengaja berhenti jika split `val` kosong atau dukungan atribut terlalu kecil, supaya model
tidak sekadar menghafal kamera/frame. Jalankan setelah gold set memiliki data train dan val cukup:

```powershell
.\.venv\Scripts\jupyter nbconvert --execute --to notebook --inplace NB04b_attribute_classifier.ipynb
```

Artefak classifier ditulis ke `output/attribute_classifier/`. Metrik test classifier (precision,
recall, F1, MSE/Brier, RMSE) hanya muncul dari frame NB02c yang sudah direview. MSE/RMSE mengukur
error probabilitas label kondisi, bukan geometri bbox. Training menghasilkan kurva pemilihan k di
`emeans_k_selection.png` dan scatter PCA 2D di `emeans_pca_scatter.png`; PCA hanya untuk inspeksi,
di-fit memakai fitur train, dan tidak mengubah assignment e-means 77 dimensi.

NB02e membuat cache mask SAM2 untuk box manual train/validation, GT test, serta pseudo FM train. NB06 menjalankan SAM2 per frame untuk overlay dan input classifier dengan kontrak fitur sama. Mask untuk kelas kondisi kendaraan ditampilkan (merah untuk TidakStandar, biru untuk Standar); keluaran `PantulanCahaya` menjadi background. Bbox merah dan counting line hanya untuk kelas TidakStandar. Jika label, proposal, atau model SAM2 berubah, jalankan NB02e sampai `complete`, lalu NB04b dan NB06. Source e-means resmi dipasang oleh script PowerShell ke `vendor/faiss_e_means`; package Faiss/GPU tidak diperlukan karena assignment memakai backend native PyTorch. E-means adalah fitur eksperimen tanpa jaminan kenaikan akurasi; bandingkan hasil validation pada split yang sama.

## Angka resmi

NB06 menghitung **track mobil/motor tidak standar satu kali saat center bbox melintasi garis**.
Arah (`up`/`down`) dan frame crossing tersimpan di `counting_timeline.csv`. Bbox-per-frame tetap
tersedia di `bbox_per_frame_timeline.csv` sebagai metrik aktivitas deteksi, bukan hitungan kendaraan.
Garis default dipasang di y=72% frame lalu dipotong mengikuti polygon ROI; posisinya tercatat di
`overlay_manifest.json` dan terlihat pada overlay video.

## Regenerasi notebook

```powershell
python builders/build_nb01_acquisition.py
python builders/build_nb02_pseudo_label.py
python builders/build_nb02b_annotation.py
python builders/build_nb03_gold_threshold.py
python builders/build_nb02d_finetune.py
python builders/build_nb02e_grounded_sam2.py
python builders/build_nb04_train_yolo.py
python builders/build_nb04b_attribute_classifier.py
python builders/build_nb05_evaluation.py
python builders/build_nb06_tracking_count.py
```

## Pinpoint ROI

Gunakan [NB01b_roi_config.ipynb](NB01b_roi_config.ipynb) setelah NB01 selesai untuk menggambar ulang polygon ROI pada
frame resolusi asli tiap kamera. Notebook menyimpan `frame_size`, memperbarui rectangle resolusi
asli, memvalidasi batas polygon, dan mempertahankan ROI lama jika kamera dilewati. Jalankan dengan
kernel `.venv`; notebook memakai backend `%matplotlib tk` agar jendela picker benar-benar memblokir.
`Esc` mempertahankan ROI lama, `Enter` menyimpan polygon dengan minimal 3 titik.

Builder notebook pipeline menjaga source notebook konsisten dengan kode di `builders/`. Gate notebook memeriksa struktur/source yang diizinkan; hasil cell lama dipertahankan saat source notebook disinkronkan.

`NB01b_roi_config.ipynb` dan `NB01c_curated_sampling.ipynb` diedit langsung. `NB04b_attribute_classifier.ipynb`
dibangun dari `builders/build_nb04b_attribute_classifier.py`. Notebook builder lainnya harus dibangun ulang
setiap kali builder berubah; jangan edit `.ipynb`-nya langsung. Regenerasi menulis ulang notebook, jadi gunakan hanya saat memang ingin memperbarui source notebook; sinkronisasi audit 12 November mempertahankan output yang sudah ada.

## Pemeriksaan cepat tanpa GPU

```powershell
python test_pipeline_logic.py
```

Menguji kontrak hitung, penyambungan fragmen track, guard ambang tracker, fitur atribut (anti-NaN),
filter store anotasi, serta loader dashboard (state kosong/parsial/rusak, agregasi per kamera,
dan format angka `n/a`).

## Memulihkan frame yang hilang

Jika `output/frames/` kosong tetapi `output/manifests/dedup_manifest.csv` masih ada, jalankan:

```powershell
.venv\Scripts\python builders\restore_frames_from_manifest.py
```

Skrip ini mengambil frame dari video lokal berdasarkan nomor frame pada manifest, mempertahankan
`frame_hash` lama, menormalkan path manifest menjadi relatif terhadap root proyek, dan membuat
backup manifest sebelum perubahan. Hash hanya dipakai untuk verifikasi hasil pemulihan; tidak dibuat
ulang sebagai sumber data baru.

Setelah dependency terpasang, regenerasi notebook dari builder agar notebook lama tidak membawa
path mesin lain:

```powershell
$env:RISET_PROJECT_ROOT = (Get-Location).Path
python builders\build_nb01_acquisition.py
python builders\build_nb02_pseudo_label.py
python builders\build_nb02b_annotation.py
python builders\build_nb02d_finetune.py
python builders\build_nb02e_grounded_sam2.py
python builders\build_nb03_gold_threshold.py
python builders\build_nb04_train_yolo.py
python builders\build_nb04b_attribute_classifier.py
python builders\build_nb05_evaluation.py
python builders\build_nb06_tracking_count.py
```

## Audit mutu gold set

```powershell
python builders/audit_gold_quality.py
```

Membaca `output/annotation/annotations.json` dan menulis `gold_quality_summary.json` +
`gold_quality_audit.csv`: pasangan box duplikat, box di bawah 8 px, box yang menyentuh tepi crop,
refleksi yang menumpuk di atas kendaraan, dan **proxy recall proposal DINO** (box hasil pseudo
dibanding box yang digambar manual). Tidak mengubah anotasi apa pun.

## Smoke end-to-end

```powershell
$env:PIPELINE_SMOKE=1
python builders/smoke_pipeline.py
```

Menjalankan NB02â€“NB06 di `output/smoke/` (bukan artefak riset), memakai model dan GPU.

## Dashboard

```powershell
streamlit run dashboard/app.py
```

Dashboard membaca `output/nb06/counting_summary.json`, `counting_by_video.csv`, dan
`counting_timeline.csv`: crossing track tidak standar, rincian mobil/motor, playback dan detail
crossing pada frame yang dipilih, serta tabel semua titik CCTV.
Titik CCTV yang tidak masuk split hitung tetap tampil dengan statusnya, supaya cakupan dashboard
tidak terlihat lebih luas daripada yang benar-benar dihitung NB06.

Untuk melihat hasil smoke sebelum NB06 asli selesai, arahkan dashboard ke artefak smoke:

```powershell
$env:NB06_ROOT = "$PWD\output\smoke\nb06"
streamlit run dashboard/app.py
```

Aturan visual dan pre-flight anti-slop ada di [dashboard/DESIGN.md](dashboard/DESIGN.md). Tema
warna dan tipografi hanya berasal dari [.streamlit/config.toml](.streamlit/config.toml).

`output/nb06/counting_by_video.csv` memuat satu baris untuk setiap video test yang diproses, termasuk video dengan nol crossing. Kolom kelas menunjukkan bbox deteksi per frame; `crossings_counted` dan kunci kontrak menghitung track kendaraan tidak standar yang melintasi garis.

Overlay NB06 menampilkan mask SAM2 dan bbox `TidakStandar` dengan warna merah. Counting line tampil
kuning. Label overlay aktif secara
default dan hanya menampilkan `ID <track_id> Â· <confidence>%` dalam ukuran kecil di pojok bbox. Set
`NB06_OVERLAY_LABELS=0` bila label ingin dimatikan.

NB06 juga menulis `counting_timeline.csv` (satu baris per vehicle track yang melintasi garis) dan
`overlay_manifest.json` (video, FPS, durasi, split, dan path overlay). Dashboard memakai keduanya untuk
playback serta detail crossing pada frame yang dipilih melalui slider. Pemutar video Streamlit tidak
menyinkronkan callback frame ke angka; overlay/video dan tabel crossing tetap dapat diperiksa bersamaan.
Dashboard menampilkan video dan counting
secara side-by-side, dengan legend permanen yang menjelaskan warna dan tipe kendaraan.

NB06 membuat dua bentuk artefak event: video konteks di `event_clips/<kelas>/` dan crop bbox pada
`event_crops/<kelas>/`. Galeri dashboard memakai crop bbox, bukan full-frame clip, dan menyediakan
filter kelas/CCTV, detail event, navigasi sebelumnya/berikutnya, serta grid 8Ã—8. Video overlay yang
ditulis sementara oleh OpenCV dengan codec `mp4v`, lalu cell terakhir NB06 mengonversi overlay ke
H.264 (`libx264`) melalui FFmpeg di `overlays_browser/` agar kompatibel dengan browser. Video sumber
tetap disimpan di `overlays/`. NB06 juga menyimpan preview inferensi berupa dua frame per CCTV di
`output/nb06/inference_preview_grid.png` dan file PNG per kamera di `output/nb06/inference_previews/`.
Caption menunjukkan prediksi CNN dan jumlah assignment e-means per frame; rincian per track ada di
`inference_preview_manifest.json` dan `tracking_detections.csv`.
Pemutar video dan galeri menyediakan toggle **Mask tampil / Mask disembunyikan**. Mode tanpa mask
tetap menampilkan bbox merah, label, dan garis hitung; kedua mode memakai hasil klasifikasi yang sama.
Mask ditampilkan secara default. Toggle hanya memilih varian presentasi; ia tidak mengubah inferensi,
kelas, counting, atau metrik. Pasangan video/crop dibuat setelah NB06 selesai tanpa mengulang YOLO
atau ByteTrack. Proses pasca-NB06 memerlukan video sumber, manifest deteksi, dan model SAM2 untuk
membuat crop galeri bermask.
NB06 juga menyimpan checkpoint atomik per video di `output/nb06/video_checkpoints/`. Setelah
interupsi, video yang checkpoint-nya valid akan dilewati; video yang terputus di tengah tetap diproses
ulang dari awal karena state ByteTrack tidak disimpan per frame. Checkpoint divalidasi terhadap video,
bobot YOLO/CNN, konfigurasi SAM2, threshold, ROI, dan konfigurasi tracker. File overlay parsial bukan
checkpoint dan tidak dapat dipakai untuk memulihkan video yang terputus. Setelah semua video selesai,
NB06 menyusun ulang tabel global, manifest, preview, dan varian presentasi dari checkpoint yang valid.
Untuk melanjutkan run yang terputus, jalankan kembali NB06; jangan hapus `video_checkpoints/` yang
valid. Jika checkpoint tidak ada atau tidak lolos validasi, video terkait akan diproses ulang.

## Catatan historis dan prosedur rerun lama

Bagian berikut adalah catatan dari kontrak pipeline sebelum 16 November 2026. Untuk taksonomi dan
urutan aktif, gunakan blok **Kontrak aktif** di bagian atas dokumen. Angka tiga kelas, mapping YOLO
ke Mobil/Motor, varian continued, dan review wajib atas seluruh pseudo-label sudah tidak berlaku.

NB06 menjalankan YOLO pada crop ROI yang sama dengan pipeline training. Pantulan disaring sebelum
ByteTrack. Bbox kendaraan dipetakan kembali ke koordinat frame penuh, lalu tipe dari YOLO digabung
dengan kondisi dari CNN pada track ByteTrack. Overlay hanya menampilkan bbox `TidakStandar` merah.
Hitungan resmi
adalah **bbox per frame**: satu kendaraan yang terlihat di banyak frame dapat dihitung berulang.
ByteTrack ID dipakai untuk tracking/audit, bukan untuk mengubah hitungan utama menjadi jumlah
kendaraan unik. `class_counts` pada summary adalah deteksi lolos ambang menurut kelas anotasi;
`event_class_counts` dan `counting_totals` hanya menghitung bbox TidakStandar. NB05 melaporkan
deteksi bbox `Mobil`/`Motor`, bukan kondisi lampu.

### Dampak perbaikan metadata ROI

Audit 12 November menemukan metadata dimensi lama pada 81 anotasi. Koordinat bbox cocok dengan
gambar ROI aktif, sehingga bbox tidak digeser. Metadata store/COCO dan export YOLO lokal sudah
diperbaiki dengan backup. NB06 juga diperbaiki agar inferensinya memakai crop ROI seperti training.
Jangan hapus sumber video, frame, ROI, crop, `sample_manifest.json`, atau `annotations.json`.
Jangan jalankan ulang migrasi taxonomy.

Untuk menyegarkan hasil pipeline setelah koreksi ini, jalankan:

1. **NB03** untuk memperbarui audit/karakterisasi gold set.
2. **NB04** untuk melatih ulang YOLO memakai normalisasi bbox yang sudah benar.
3. **NB04b** untuk melatih ulang classifier dan membuat checkpoint yang cocok dengan NB06.
4. **NB05** untuk evaluasi dan kalibrasi detector dari bobot NB04 terbaru.
5. **NB06** untuk tracking, overlay, gallery, dan hitungan memakai bobot/threshold dari run yang sama.

Export anotasi lokal sudah dibangun ulang oleh repair metadata. Tidak perlu mengulang anotasi manual,
sampling NB02b, atau NB02. Jika store anotasi berubah setelah repair, jalankan hanya sel export NB02b
terlebih dahulu; builder sekarang menyiapkan export staging, memvalidasi, lalu mengganti export aktif
sambil menyimpan backup. Hindari penghapusan massal output: bila artefak turunan perlu diganti,
backup dulu dan biarkan notebook menulis hasil baru.

Train/validation/test tetap dipisah menurut kamera. Test GT dari NB02c hanya untuk evaluasi setelah
review; test tidak dipakai untuk training atau pemilihan threshold. Jika test GT belum lengkap,
metrik test harus dilaporkan unavailable. Kurva train/val per epoch bukan metrik test. Ekspor GT
YOLO untuk NB05 memakai ID `0=Mobil`, `1=Motor`, `2=PantulanCahaya`; box standar dan tidak standar digabung per jenis
kendaraan, sedangkan pantulan dievaluasi sebagai kelas terpisah. Script `builders/repair_ground_truth_export.py` menyelaraskan metadata dimensi GT dengan
gambar ROI aktif, memvalidasi batas bbox, dan menyimpan ekspor lama sebagai backup.

### Remediasi Ponytail Audit 3

Temuan dan status penanganan tiap poin dicatat di
[`PONYTAIL_AUDIT_3_REMEDIATION.md`](PONYTAIL_AUDIT_3_REMEDIATION.md). Pemeriksaan source dan unit
logic bukan bukti akurasi model; training atau inferensi riset tidak dijalankan sebagai bagian
remediasi ini.

Skrip diagnosis/sekali pakai yang tidak dipanggil pipeline aktif disimpan di
[`archive/one-off-scripts/`](archive/one-off-scripts/README.md). Generator NB02c tetap berada di
`builders/` karena menjadi sumber notebook GT.

## Artefak

Semua keluaran di `output/`: `videos/`, `frames/`, `manifests/`, `split/`, `roi/`, `katalog/`,
`nb02/`, `annotation/`, `nb03/`, `nb04/`, `nb05/`, `nb06/`. Berkas yang dipindahkan keluar dari
pipeline ada di `trash_2026-10-26/` (lihat `RISK_REGISTER.md`).


## Catatan audit terbaru

Temuan audit lama dan tindak lanjutnya disimpan sebagai arsip; arsip tersebut bukan spesifikasi kode aktif.


## Dokumentasi proyek

- [Master dokumentasi](MASTER_DOKUMENTASI_HAMPIR.md) â€” arsitektur lengkap dan catatan sejarah; bagian aktif ditandai di awal dokumen.
- [Risk register](RISK_REGISTER.md) â€” snapshot risiko aktif dan log historis.
- [Protokol ground truth](GROUND_TRUTH_GUIDE.md) â€” pemisahan anotasi train/validation dan holdout test.
- [Arsip audit dan catatan superseded](archive/superseded-project-notes/README.md) â€” disimpan untuk jejak historis, bukan panduan menjalankan pipeline.
- [Rujukan ilmiah](RUJUKAN_INFRA_CV.md) dan [pengetahuan pipeline CV](KNOWLEDGE_PIPELINE_CV.md) â€” sumber akademik serta pola yang benar-benar relevan.
- [Bar desain dashboard](dashboard/DESIGN.md) â€” aturan UI dan arti angka yang ditampilkan.
- [Audit 28 Oktober](AUDIT_ACTION_2026-10-28.md) dan [handoff 4 November](AUDIT_HANDOFF_2026-11-04.md) adalah rekaman historis, bukan kontrak implementasi terkini. Review 18 Oktober dipindahkan ke arsip superseded.






\n
