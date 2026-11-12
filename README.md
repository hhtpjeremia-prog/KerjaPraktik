# Riset-HampirMatang

Pipeline riset untuk mendeteksi kendaraan dengan kondisi lampu standar/tidak standar dan menghitung
bbox tidak standar per frame dari rekaman CCTV. Notebook pipeline utama di root **dibangun** dari
`builders/build_nb*.py`; jangan edit `.ipynb` langsung, edit builder lalu regenerasi.

## Kontrak label dan model

Anotasi aktif memakai empat label kendaraan/kondisi (`MobilStandar`, `MobilTidakStandar`,
`MotorStandar`, `MotorTidakStandar`) dan `PantulanCahaya`. YOLO dan Grounding DINO (FM) memiliki
tiga target: `Mobil`, `Motor`, `PantulanCahaya`. CNN supervised memprediksi `Standar`,
`TidakStandar`, atau `PantulanCahaya`. Pantulan dipelajari dan dievaluasi, lalu disaring sebelum
ByteTrack, overlay, dan counting kendaraan.

## Menyiapkan environment

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

`torch`/`torchvision` di pin memakai build CUDA 12.4. Untuk mesin tanpa CUDA, ganti dengan build CPU
dari indeks PyTorch resmi; sisa paket tetap sama.

## Urutan menjalankan

Jalankan per sel, berurutan, di VS Code dengan kernel `.venv` dan **cwd = root proyek** (notebook
mengandalkan `from pipeline_common import ...` tanpa mengubah `sys.path`).

1. `NB01_acquisition.ipynb` — akuisisi video, sampling frame, dan split per kamera.
2. `NB02_pseudo_label.ipynb` — proposal lokasi objek dan crop ROI; proposal bukan ground truth.
3. `NB02b_annotation.ipynb` — anotasi awal train/validation yang direview manusia (butuh `ipympl`).
4. `NB02d_finetune_grounding_dino.ipynb` — fine-tune FM dan siapkan proposal tambahan train/val; semua wajib direview lagi di NB02b.
5. Export ulang NB02b sampai target 826 train/190 val tercapai; pertahankan 254 frame test.
6. `NB02c_ground_truth.ipynb` — anotasi holdout test opsional; hanya untuk evaluasi, bukan training atau pemilihan threshold.
7. `NB03_gold_threshold.ipynb` — audit/karakterisasi gold set.
8. `NB04_train_yolo.ipynb` — latih detector tiga kelas Mobil/Motor/PantulanCahaya.
9. `NB04b_attribute_classifier.ipynb` — latih classifier Standar/TidakStandar/PantulanCahaya.
10. `NB05_evaluation.ipynb` — kalibrasi detector di validation dan evaluasi test hanya bila GT tersedia.
11. `NB06_tracking_count.ipynb` — filter PantulanCahaya sebelum ByteTrack, overlay, dan counting.

NB02b sekarang mensyaratkan jarak minimum 30 frame antar frame sampel dalam kamera yang sama
(sekitar 1,2 detik pada 25 fps). Manifest sampling lama akan ditolak dan harus dibuat ulang agar
frame berurutan yang identik tidak masuk gold set.

Kamera `bappenda2` dan `pdam2` adalah split `test`: tidak pernah masuk train/val. Label evaluasi
holdout disimpan terpisah di `output/ground_truth/` melalui NB02c dan tidak boleh dipakai untuk fit.

## Classifier atribut cahaya

Detektor YOLO bertugas mendeteksi `Mobil`, `Motor`, dan `PantulanCahaya`. Notebook
`NB04b_attribute_classifier.ipynb` melatih classifier supervised tiga kelas (`Standar`, `TidakStandar`, `PantulanCahaya`)
berbasis crop bbox dengan ResNet18 pretrained yang menggabungkan embedding CNN dan 35 fitur
numerik HSV/statistik piksel. Weighted CrossEntropy menangani ketimpangan kelas; normalisasi fitur dipelajari
hanya dari train. Fitur HSV adalah proksi citra, bukan pengukuran lux/candela.

Training sengaja berhenti jika split `val` kosong atau dukungan atribut terlalu kecil, supaya model
tidak sekadar menghafal kamera/frame. Jalankan setelah gold set memiliki data train dan val cukup:

```powershell
.\.venv\Scripts\jupyter nbconvert --execute --to notebook --inplace NB04b_attribute_classifier.ipynb
```

Artefak classifier ditulis ke `output/attribute_classifier/`. Metrik test classifier (precision,
recall, F1, MSE/Brier, RMSE) hanya muncul dari frame NB02c yang sudah direview. MSE/RMSE mengukur
error probabilitas label kondisi, bukan geometri bbox.

## Angka resmi

NB06 menerbitkan counting utama dalam satuan **bbox per frame**. Satu kendaraan dapat muncul
berulang pada frame berbeda. ByteTrack tetap menyediakan `track_id` untuk audit dan stabilitas
visual, tetapi tidak dipakai untuk mengagregasi angka utama menjadi kendaraan unik.

## Regenerasi notebook

```powershell
python builders/build_nb01_acquisition.py
python builders/build_nb02_pseudo_label.py
python builders/build_nb02b_annotation.py
python builders/build_nb03_gold_threshold.py
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

Menjalankan NB02–NB06 di `output/smoke/` (bukan artefak riset), memakai model dan GPU.

## Dashboard

```powershell
streamlit run dashboard/app.py
```

Dashboard membaca `output/nb06/counting_summary.json`, `counting_by_video.csv`, dan
`counting_timeline.csv`: bbox tidak standar aktif, rincian mobil/motor, playback per frame, dan
tabel semua titik CCTV.
Titik CCTV yang tidak masuk split hitung tetap tampil dengan statusnya, supaya cakupan dashboard
tidak terlihat lebih luas daripada yang benar-benar dihitung NB06.

Untuk melihat hasil smoke sebelum NB06 asli selesai, arahkan dashboard ke artefak smoke:

```powershell
$env:NB06_ROOT = "$PWD\output\smoke\nb06"
streamlit run dashboard/app.py
```

Aturan visual dan pre-flight anti-slop ada di [dashboard/DESIGN.md](dashboard/DESIGN.md). Tema
warna dan tipografi hanya berasal dari [.streamlit/config.toml](.streamlit/config.toml).

`output/nb06/counting_by_video.csv` memuat satu baris untuk setiap video test yang diproses, termasuk video dengan nol event. Kolom kelas menunjukkan deteksi per kelas; `bbox_counted` dan dua kunci kontrak hanya menghitung bbox TidakStandar.

Overlay NB06 hanya menampilkan bbox `TidakStandar` dengan warna merah. Label overlay aktif secara
default dan hanya menampilkan `ID <track_id> · <confidence>%` dalam ukuran kecil di pojok bbox. Set
`NB06_OVERLAY_LABELS=0` bila label ingin dimatikan.

NB06 juga menulis `counting_timeline.csv` (satu baris per bbox tidak standar per frame) dan
`overlay_manifest.json` (video, FPS, durasi, split, dan path overlay). Dashboard memakai keduanya untuk
playback serta counting aktif berdasarkan frame yang sedang dipilih. Dashboard menampilkan video dan counting
secara side-by-side, dengan legend permanen yang menjelaskan warna dan tipe kendaraan.

NB06 membuat dua bentuk artefak event: video konteks di `event_clips/<kelas>/` dan crop bbox pada
`event_crops/<kelas>/`. Galeri dashboard memakai crop bbox, bukan full-frame clip, dan menyediakan
filter kelas/CCTV, detail event, navigasi sebelumnya/berikutnya, serta grid 8×8. Video overlay yang
tidak dapat diputar browser dapat dikonversi melalui cell terakhir NB06 ke `overlays_browser/` dengan
codec AVC1; video sumber tidak dihapus.

## Kontrak aktif dan prosedur rerun (13 November 2026)

Annotation store menyimpan empat kelas kendaraan (`MobilStandar`, `MobilTidakStandar`,
`MotorStandar`, `MotorTidakStandar`) serta `PantulanCahaya`; COCO memuat lima kategori anotasi.
Export YOLO memetakan empat label kendaraan menjadi `Mobil`/`Motor` dan mengekspor
`PantulanCahaya` sebagai kelas detector ketiga. NB04b memakai
ResNet18 pretrained dengan weighted CrossEntropy dan 35 fitur numerik HSV/statistik piksel untuk memprediksi
`Standar`/`TidakStandar`/`PantulanCahaya`. Fitur numerik dihitung dari crop bersih; ColorJitter
hanya mengubah input citra CNN saat training. Normalisasi numerik memakai statistik train saja.
Fitur piksel adalah proksi visual, bukan ukuran lux/candela.

`NB02d_finetune_grounding_dino.ipynb` adalah tahap fine-tune FM sebelum menambah anotasi. Notebook
memakai train untuk update bobot, val untuk pemilihan checkpoint, lalu membuat proposal hanya dari
kamera train/val. Target akhir saat ini 826 train + 190 val + 254 test (65/15/20 berdasarkan frame).
Semua proposal FM tetap wajib dikoreksi manusia di NB02b sebelum menjadi label supervised.

### Rerun setelah perubahan taksonomi

1. Jalankan NB02d dan review seluruh proposal tambahan melalui NB02b. Target proposal 631 train +
   111 val dihitung dari baseline 195/79; jika jumlah reviewed saat ini berubah, perbarui baseline.
2. Jalankan export NB02b agar YOLO train/val memiliki `Mobil`, `Motor`, `PantulanCahaya`.
3. Jika test GT sudah direview, jalankan `python builders/repair_ground_truth_export.py` untuk
   memperbarui export YOLO test menjadi tiga kelas. Skrip membuat backup export lama otomatis.
4. Jalankan NB03, NB04, NB04b, NB05, lalu NB06. NB04/04b harus dilatih ulang karena jumlah kelas
   dan kepala classifier berubah; bobot lama tidak kompatibel.
5. Jangan menghapus annotation store, test GT, video, frame, ROI, atau crop sumber. Hasil run lama
   sebelum perubahan ini sudah disalin ke `output/_backup_pre_nb02d_20261113_021457/`.

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

### Verifikasi yang telah dilakukan

Pada 12 November 2026 source Python yang disentuh lolos `py_compile`, source notebook NB02, NB02b,
NB03, NB04, NB04b, NB05, dan NB06 cocok dengan builder, dan `git diff --check` bersih. Training,
inferensi video, smoke end-to-end, dan test suite tidak dijalankan pada sesi audit ini. Jadi hasil
performa model setelah retraining tetap harus dievaluasi; pemeriksaan source tidak membuktikan
akurasi model.

## Artefak

Semua keluaran di `output/`: `videos/`, `frames/`, `manifests/`, `split/`, `roi/`, `katalog/`,
`nb02/`, `annotation/`, `nb03/`, `nb04/`, `nb05/`, `nb06/`. Berkas yang dipindahkan keluar dari
pipeline ada di `trash_2026-10-26/` (lihat `RISK_REGISTER.md`).


## Catatan audit terbaru

Tindak lanjut audit Ponytail dan koreksi metadata ROI (12 November 2026) dicatat di [`PONYTAIL_REMEDIATION_2026-11-12.md`](PONYTAIL_REMEDIATION_2026-11-12.md). Setelah perubahan dimensi anotasi, jalankan NB03 lalu NB04, NB04b, NB05, dan NB06 agar semua artefak turunan konsisten.


## Dokumentasi proyek

- [Master dokumentasi](MASTER_DOKUMENTASI_HAMPIR.md) — arsitektur lengkap dan catatan sejarah; bagian aktif ditandai di awal dokumen.
- [Risk register](RISK_REGISTER.md) — snapshot risiko aktif dan log historis.
- [Protokol ground truth](GROUND_TRUTH_GUIDE.md) — pemisahan anotasi train/validation dan holdout test.
- [Audit Ponytail dan tindak lanjut](PONYTAIL_REMEDIATION_2026-11-12.md) — hasil review kode, koreksi ROI, dan batas verifikasi.
- [Rujukan ilmiah](RUJUKAN_INFRA_CV.md) dan [pengetahuan pipeline CV](KNOWLEDGE_PIPELINE_CV.md) — sumber akademik serta pola yang benar-benar relevan.
- [Bar desain dashboard](dashboard/DESIGN.md) — aturan UI dan arti angka yang ditampilkan.
- [Audit 28 Oktober](AUDIT_ACTION_2026-10-28.md), [handoff 4 November](AUDIT_HANDOFF_2026-11-04.md), dan [review 18 Oktober](REVIEW_2026-10-18.md) adalah rekaman historis, bukan kontrak implementasi terkini.
