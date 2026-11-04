# Riset-HampirMatang

Pipeline riset untuk menghitung kondisi lampu kendaraan (silau, berwarna, silau+berwarna) dan
jumlah kendaraan unik mobil/motor dari rekaman CCTV. Tujuh notebook di root **dibangun** dari
`builders/build_nb*.py`; jangan edit `.ipynb` langsung, edit builder lalu regenerasi.

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

1. `NB01_acquisition.ipynb` — rekam video + sampling frame + split per kamera.
2. `NB02_pseudo_label.ipynb` — pseudo-label lampu (Grounding DINO) + crop ROI.
3. `NB02b_annotation.ipynb` — anotasi manual gold set (butuh `ipympl`).
4. `NB03_gold_threshold.ipynb` — audit fotometrik gold set.
5. `NB04_train_yolo.ipynb` — latih detector kendaraan dengan split per kamera.
6. `NB04b_attribute_classifier.ipynb` — latih classifier piksel multi-label untuk silau/berwarna.
7. `NB05_evaluation.ipynb` — kalibrasi cakupan detector di split `val`.
8. `NB06_tracking_count.ipynb` — ROI-masked full-frame + classifier atribut + ByteTrack di split `test`.

NB02b sekarang mensyaratkan jarak minimum 30 frame antar frame sampel dalam kamera yang sama
(sekitar 1,2 detik pada 25 fps). Manifest sampling lama akan ditolak dan harus dibuat ulang agar
frame berurutan yang identik tidak masuk gold set.

Kamera `bappenda2` dan `pdam2` adalah split `test`: tidak pernah masuk train/val dan tidak pernah
dianotasi manual.

## Classifier atribut cahaya

Detektor YOLO tidak boleh sekaligus menjadi penentu `silau` dan `berwarna`: atribut tersebut bisa
muncul bersamaan dan perlu dibaca dari piksel crop kendaraan. Notebook
`NB04b_attribute_classifier.ipynb` melatih classifier multi-label (`silau`, `berwarna`) berbasis
fitur RGB/HSV piksel dari anotasi gold; `attribute_classifier.py` hanya berisi komponen model.

Training sengaja berhenti jika split `val` kosong atau dukungan atribut terlalu kecil, supaya model
tidak sekadar menghafal kamera/frame. Jalankan setelah gold set memiliki data train dan val cukup:

```powershell
.\venv\Scripts\jupyter nbconvert --execute --to notebook --inplace NB04b_attribute_classifier.ipynb
```

Artefak classifier ditulis ke `output/attribute_classifier/`. Model ini belum boleh dipakai untuk
angka resmi sebelum evaluasi per atribut di kamera test juga memiliki label gold.

## Angka resmi

Hanya **NB06** yang menerbitkan 8 angka kondisi lampu + jumlah kendaraan unik. NB05 menerbitkan
**box per frame** dengan kunci `jumlah_box_*`, dipakai sebagai pemeriksaan cakupan, bukan angka
populasi.

## Regenerasi notebook

```powershell
python builders/build_nb01_acquisition.py
python builders/build_nb02_pseudo_label.py
python builders/build_nb02b_annotation.py
python builders/build_nb03_gold_threshold.py
python builders/build_nb04_train_yolo.py
python builders/build_nb05_evaluation.py
python builders/build_nb06_tracking_count.py
```

## Pinpoint ROI

Gunakan [NB01b_roi_config.ipynb](NB01b_roi_config.ipynb) setelah NB01 selesai untuk menggambar ulang polygon ROI pada
frame resolusi asli tiap kamera. Notebook menyimpan `frame_size`, memperbarui rectangle resolusi
asli, memvalidasi batas polygon, dan mempertahankan ROI lama jika kamera dilewati. Jalankan dengan
kernel `.venv`; notebook memakai backend `%matplotlib tk` agar jendela picker benar-benar memblokir.
`Esc` mempertahankan ROI lama, `Enter` menyimpan polygon dengan minimal 3 titik.

Setiap builder menjalankan gate: jumlah sel tetap, tanpa komentar `#`, tanpa `print(`, tanpa
`__file__`, output sel kosong.

`NB01b_roi_config.ipynb`, `NB01c_curated_sampling.ipynb`, dan `NB04b_attribute_classifier.ipynb`
tidak punya builder — ketiganya diedit langsung dan tidak ikut diregenerasi. Notebook builder
lainnya harus dibangun ulang setiap kali builder berubah; jangan edit `.ipynb`-nya langsung.

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

Dashboard membaca `output/nb06/counting_summary.json` dan `output/nb06/counting_by_video.csv`:
total kendaraan unik, delapan angka kontrak, rincian per kelas, dan tabel semua titik CCTV.
Titik CCTV yang tidak masuk split hitung tetap tampil dengan statusnya, supaya cakupan dashboard
tidak terlihat lebih luas daripada yang benar-benar dihitung NB06.

Untuk melihat hasil smoke sebelum NB06 asli selesai, arahkan dashboard ke artefak smoke:

```powershell
$env:NB06_ROOT = "$PWD\output\smoke\nb06"
streamlit run dashboard/app.py
```

Aturan visual dan pre-flight anti-slop ada di [dashboard/DESIGN.md](dashboard/DESIGN.md). Tema
warna dan tipografi hanya berasal dari [.streamlit/config.toml](.streamlit/config.toml).

`output/nb06/counting_by_video.csv` memuat kolom `camera_id`, `video_stem`, `vehicles_counted`,
sembilan kolom kelas, dan delapan kolom kontrak; dashboard menjumlahkannya per titik CCTV.

## Artefak

Semua keluaran di `output/`: `videos/`, `frames/`, `manifests/`, `split/`, `roi/`, `katalog/`,
`nb02/`, `annotation/`, `nb03/`, `nb04/`, `nb05/`, `nb06/`. Berkas yang dipindahkan keluar dari
pipeline ada di `trash_2026-10-26/` (lihat `RISK_REGISTER.md`).
