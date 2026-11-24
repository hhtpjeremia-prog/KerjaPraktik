# Master Dokumentasi â€” Riset HampirMatang

> **Kontrak aktif (2026-11-18; mengesampingkan bagian historis yang bertentangan di bawah).** Anotasi, Grounding DINO base, YOLO, dan CNN memakai lima kelas sama: `MobilStandar`, `MobilTidakStandar`, `MotorStandar`, `MotorTidakStandar`, `PantulanCahaya`. `BelumDitentukan` bukan kelas model. Continued fine-tune dibatalkan dan notebook-nya dihapus; output lama arsip. Pseudo-label FM boleh masuk train, tidak menggantikan gold validation/test. NB02e membangun mask SAM2 untuk fitur classifier; NB04b memadukan RGB, 77 fitur numerik bbox+mask, dan one-hot e-means yang fit hanya pada train. NB06 menghitung ulang fitur dari mask SAM2 untuk inferensi dan overlay. Mask untuk kelas Standar/TidakStandar tampil; PantulanCahaya menjadi background; bbox/counting hanya kelas TidakStandar. Preview NB06 menyimpan dua frame inferensi per CCTV dengan ringkasan kelas CNN dan cluster e-means di `output/nb06/inference_preview_grid.png`; rincian per track disimpan dalam manifest preview dan tracking CSV. E-means adalah fitur eksperimen, belum terbukti meningkatkan akurasi. Counting utama adalah satu crossing per vehicle track TidakStandar; bbox-per-frame disimpan sebagai metrik pendamping. Garis awal diset pada y=72% tinggi frame, dipotong ke ROI, dan arah crossing dicatat. NB06 membuat checkpoint atomik per video; saat dilanjutkan, video lengkap dengan checkpoint valid dilewati, sedangkan video terputus diproses ulang dari awal. Pasangan tampilan mask/tanpa-mask dibuat pasca-inferensi dan toggle dashboard hanya presentasi. MQTT belum termasuk implementasi aktif. Dokumen audit superseded dipertahankan di `archive/superseded-project-notes/README.md`.


**Proyek:** Deteksi Lampu Kendaraan Malam Hari di Jalan Raya Kabupaten Bogor
**Dataset:** `CCTV Kab Bogor_ Raya Lintang Kita (Rawamurang) (2026-08-01)`
**Repo lokal:** `TugasJere/KerjaPraktik/Riset-HampirMatang/`
**Revisi terakhir:** 2026-11-18 (Ponytail Audit 3; notebook-source gate; pemulihan dan isolasi smoke; refactor logic counting; prosedur rerun NB05→NB06)

> **Catatan historis 2026-11-12.** Dokumen Ponytail terkait dipertahankan di `archive/superseded-project-notes/`; implementasi aktif dan panduan rerun tercantum di README dan builders. Temuan ROI pada waktu itu memperbaiki metadata store/COCO dan export YOLO lokal; rincian historis tidak menjadi petunjuk rerun untuk perubahan terkini.

> **Kontrak aktif sebelumnya (2026-11-12; superseded 2026-11-13).** Annotation store sudah dimigrasikan ke 4 kelas kendaraan/kondisi:
> `MobilStandar`, `MobilTidakStandar`, `MotorStandar`, `MotorTidakStandar`. `PantulanCahaya`
> tetap disimpan untuk audit, tetapi saat ini tidak menjadi target YOLO atau hard-negative berlabel khusus. YOLO mendeteksi
> `Mobil`/`Motor`, classifier supervised memprediksi `Standar`/`TidakStandar`, dan NB06 menghitung
> bbox `TidakStandar` per frame. Semua uraian 9-kelas, enam metrik kondisi, dan jumlah kendaraan unik pada bagian historis bukan kontrak aktif.

> **Catatan historis (2026-11-13; kontrak sudah berubah).** YOLO/FM memakai tiga kelas deteksi `Mobil`, `Motor`,
> `PantulanCahaya`; NB04b memakai ResNet18+fitur HSV dengan weighted CrossEntropy untuk tiga label
> `Standar`, `TidakStandar`, `PantulanCahaya`. Pantulan dievaluasi lalu disaring sebelum ByteTrack,
> overlay, dan counting. Train/val memilih model; test GT hanya untuk evaluasi akhir.

> **Implementasi dashboard historis (6 November; kontrak hitung diganti 16 November).** NB06 memproses video terbaru setiap kamera dan
> membedakan scope `official` untuk split `test` dari `presentation_only` untuk `train`/`val`.
> Artefak lama `counting_timeline.csv` saat itu satu baris per bbox tidak standar per frame. Overlay hanya
> menggambar bbox tidak standar merah dengan label kecil `ID <track_id> Â· <confidence>%`. Dashboard
> menampilkan video/counting side-by-side dan galeri crop bbox pada `event_crops/`.

> **Status verifikasi source per 2026-11-06 (snapshot historis; lihat pembaruan 2026-11-12 di atas).** Builder NB02b, NB03, NB04, NB05, NB04b, dan NB06
> berhasil membangun notebook setelah perbaikan taxonomy/gate. NB05 menghasilkan baseline detector
> dua kelas. Source compile, 17 logic checks, dan dashboard state checks lolos. Belum ada training
> atau inferensi penuh pada artefak refactor ini; hasil model baru belum tersedia. Sebelum rerun,
> pertahankan `annotations.json` hasil migrasi dan seluruh sumber video/frame/crop. Jalankan export
> NB02b (bukan sampling/anotasi ulang), kemudian NB03 â†’ NB04 â†’ NB04b â†’ NB05 â†’ NB06.

---

## 0. Ringkasan Revisi

### 0.1 Revisi 2026-10-18

1. **ROI final & disetujui user** â€” 5 polygon selesai digambar, diverifikasi
   tidak bersilangan dan seluruh verteks di dalam batas frame. Overlay visual
   di `output/roi/preview/` dicek user â†’ benar.
2. **Model DINO ditetapkan: `grounding-dino-tiny`.** Ditemukan mismatch â€”
   builder sebelumnya menunjuk `grounding-dino-base` (swin-b), padahal
   dokumentasi riset konsisten menyebut tiny (swin-t). Sudah diperbaiki dan
   notebook di-regenerate. Jalur upgrade ke base tercatat di Â§5.5.
3. **Koreksi dokumentasi (P1)** â€” banyak angka/fakta yang salah:
   - Nama kamera: `B01`â€“`B05` â†’ **`bogor-kab-bappenda1/2/3`, `pdam2`, `pemda2`**.
   - Resolusi: `768Ã—576`/`1280Ã—720` â†’ **berbeda per kamera** (`704Ã—576`,
     `640Ã—480`, `640Ã—360`); fps seragam 25.
   - Struktur berkas: banyak file yang didokumentasikan **tidak pernah ada**
     (seluruh `src/*.py`, folder `notebook/`, builder NB03/04/05/FM).
   - **NB02b (CLAHE) tidak ada** â€” NB02 mandiri: ROI + dedup di dalam notebook,
     tanpa CLAHE. **Catatan 2026-10-20:** nama `NB02b` kini dipakai untuk
     **tahap anotasi manual** (koreksi pseudo-label NB02) â€” lihat Â§10.0. Yang
     tidak pernah ada adalah NB02b **versi CLAHE**, bukan slot NB02b itu sendiri.
4. **Catatan:** `output/roi/roi_config.json.broken` adalah sisa file rusak lama,
   bisa dihapus bila tidak diperlukan.

### 0.2 Revisi 2026-10-15

Revisi besar setelah review menyeluruh (8 reviewer paralel) dan pembacaan
literatur Grounded-SAM. Isi utama:

1. **Redesain NB02 (pseudo-labelling)** â€” dari SAM+DINO+geometry (Mode A lama)
   menjadi **DINO-only multi-pass (Mode B)** + **Mode A (lampu-sepasang)** +
   **grouping geometri dengan vote DINO sebagai tiebreak**. Alasan: dokumen
   resmi Grounded-SAM menegaskan SAM hanya segmentasi, bukan klasifikasi, dan
   tidak memberi improvement signifikan pada fine-grained detection; biaya
   VRAM/deteksi 2x tidak sepadan.
2. **ROI direkonstruksi menjadi polygon 5 kamera** (sebelumnya 2 rectangle).
3. **Register risiko dibuat** (`RISK_REGISTER.md`, v1.5, 19 risiko) dan semua
   isu review dipetakan sebagai risiko terkontrol.
4. **Backup dibuat** sebelum perubahan file (`backup_sebelum_fix_20261015/`).
5. **Output NB02 lama dipindah** ke `_trash_riset_hampirmatang_20261015/`
   supaya tidak bercampur dengan hasil run baru.

**Status eksekusi saat ini:** ROI final âœ… Â· model tiny âœ… Â· builder NB02
diperbaiki bertahap (threshold 0.15 â†’ **0.30**, filter luas 40%, prompt bertitik,
verifier kendaraan FRCNN) Â· notebook NB02 (8 sel kode, lolos self-check) Â·
**perlu HAPUS chunks & RE-RUN NB02** â€” lihat Â§0.3â€“Â§0.4.

### 0.3 Temuan & Perbaikan Kualitas Label NB02 (Run 2026-10-18)

**Status: akar masalah sudah ditemukan, perbaikan SUDAH diimplementasikan di
builder, tinggal RE-RUN NB02.**

**Masalah pada run pertama (1.548 frame unik â†’ 3.336 box):**

1. **Box raksasa (kritis).** 1.537 dari 3.336 box (**46.1%**) menutupi **>80% luas
   crop** â€” tidak wajar untuk satu lampu. Score box raksasa justru **lebih tinggi**
   (median 0.372) daripada box normal (0.303), jadi menaikkan threshold tidak
   menyelesaikan.
2. **Grouping Mode A praktis mati.** `LampuMobil` dengan `lamp_count=2` (union
   pasangan) hanya **19 dari 1.101 (1.7%)** â€” 98% `LampuMobil` ditentukan vote DINO,
   bukan geometri â€” kebalikan dari desain Â§3.5.
3. **Deteksi sedikit.** Median ~2 box/frame; banyak kendaraan di citra tidak
   tertangkap. Rasio kelas terbalik (`LampuMotor` 2.235 > `LampuMobil` 1.101).

**Akar masalah (lewat eksperimen GPU 20 frame â€” sweep threshold Ã— prompt Ã—
tiny-vs-base):**

- **`DETECTION_THRESHOLD` terlalu tinggi (0.25).** Di crop malam hari, 0.25 hanya
  menangkap ~2 box/frame. Turun ke **0.15** menaikkan box wajar ~4Ã— (â†’ ~7â€“9/frame)
  tanpa menambah box raksasa secara proporsional.
- **`TEXT_THRESHOLD` ternyata NO-OP** di `transformers` 5.14 â€” `score` dihitung dari
  max probabilitas teks atas seluruh 256 dimensi kosakata, jadi `text_threshold`
  hanya dipakai ekstraksi label, bukan filter box. Tidak berpengaruh ke jumlah box.
- **Box raksasa itu "region" DINO-tiny yang lolos** karena pipeline tidak punya
  filter luas sama sekali (hanya filter box < 5 px). 92% box normal berada **di
  dalam** box besar â€” DINO memberi 1 box region + 1 box lampu.
- **Crop ROI TIDAK merusak** deteksi (crop justru â‰¥ full frame di 4/5 frame uji).
- Model alternatif `grounding-dino-base` menghasilkan ~0 box raksasa, tapi 2Ã—
  lebih berat; jalur upgrade dicatat di Â§5.5.

**Perbaikan yang sudah diimplementasikan di builder (2026-10-18):**

- `DETECTION_THRESHOLD` 0.25 â†’ **0.15**.
- Filter luas box baru: `MAX_BOX_AREA_RATIO = 0.40` â€” box menutupi **>40% luas
  crop** otomatis dibuang (dipakai di `clip_and_filter_boxes`).
- `CONFIG_SIGNATURE` ikut berubah (threshold + filter) â†’ **chunk lama otomatis
  ditolak**; Anda harus **hapus isi `output/nb02/chunks/` lalu re-run NB02**.

**Verifikasi (5 frame, 1 per kamera, konfigurasi baru):** box wajar naik jadi
**~7.8 box/frame** (bappenda1 14, bappenda2 8, bappenda3 4, pdam2 6, pemda2 7),
box raksasa terbuang otomatis (2â€“3 per frame). Sebelumnya median 2 box/frame.

**Bukti visual proses:** sel visualisasi NB02 (sel 4) menampilkan overlay box per
kamera + histogram ukuran box/score langsung di notebook; file PNG di
`output/nb02/previews/`.

---

### 0.4 Paket 3-Sinyal & Threshold 0.30 (Implementasi 2026-10-19)

**Status: implementasi SELESAI di builder + notebook, tinggal hapus chunk & re-run.**

Konteks: run 2 (threshold 0.15) diverifikasi bersih â€” **1.529 frame unik â†’ 4.967
box**, 0 box > 40% crop, grouping union naik ke **22,9%**, 19 kontaminasi run 1
terhapus. Setelah itu riset foundation models (disetujui 2026-10-19)
merekomendasikan perubahan berikut; seluruhnya sudah diimplementasikan:

1. **Threshold 0.15 â†’ 0.30** (`DETECTION_THRESHOLD`) â€” 0.15 memicu pasangan
   lampu palsu (penyebab "2 motor jadi 1 mobil"); 0.30 mengikuti rekomendasi
   riset untuk menekan grouping keliru.
2. **Prompt bertitik + lowercase** â€” `"car headlight."` / `"motorcycle headlight."`
   (pola resmi Grounding DINO pada transformers 5.x).
3. **Verifier kendaraan torchvision** â€” `fasterrcnn_resnet50_fpn_v2` bobot COCO;
   kelas COCO di-remap (`car/bus/truck` â†’ `LampuMobil`, `motorcycle` â†’
   `LampuMotor`), `box_score_thresh` 0.4, vote diambil dari deteksi dengan
   **coverage tertinggi â‰¥ 0.3** terhadap box lampu (IoU tidak dipakai â€” box
   lampu vs kendaraan selalu overlap kecil). Berjalan di **CUDA**: terukur
   **1,15 d/frame** (DINO+verifier gabungan), **peak VRAM 1,82 GiB** dari
   ~4,3 GiB. Bobot 167 MB diunduh sekali saat run pertama (butuh internet);
   bila bobot gagal dimuat, verifier dimatikan otomatis dan deteksi tetap jalan
   tanpa vote.
4. **Hierarki kelas baru**: (a) grup â‰¥ 2 lampu â†’ `LampuMobil`; (b) vote FRCNN â†’
   kelas kendaraannya; (c) fallback argmax skor DINO multi-pass. Field baru per
   deteksi: `vehicle_vote` dan `signals_agree` (false = geometri/vote
   bertentangan â†’ ditandai **oranye `[konflik]`** di overlay dan masuk statistik
   sel histogram â€” bahan review NB03).
5. **Verifier blob terang (connected components cv2) DITOLAK berdasarkan data**:
   diuji pada 400 frame nyata, aturan "â‰¥2 blob â†’ mobil" nyaris tidak memisahkan
   (dua blob â‰¥60 px: **52%** grup â‰¥2 lampu vs **43%** grup 1 lampu) dan justru
   membalik **61%** deteksi 1-lampu menjadi `LampuMobil`. Kode & konstanta blob
   dibuang; dipertimbangkan ulang di NB03 setelah gold-set ada.
6. **`CONFIG_SIGNATURE` berubah lagi** (threshold + prompt + `detection_model_id`
   + `chunk_frame_count` + parameter verifier; sekaligus perbaikan review P0-2)
   â†’ **chunk run 2 otomatis ditolak. WAJIB hapus isi `output/nb02/chunks/`
   lalu re-run NB02 penuh.**

Referensi: benchmark `bench_nb02_verifier.py` (2026-10-19), statistik blob
`diag_blob_stats_nb02.py` (400 frame), arsip `archive/superseded-project-notes/REVIEW_2026-10-18.md` (P0-2 ikut
diperbaiki).

---

## 1. Tujuan dan kontrak pipeline aktif

Pipeline ini menganalisis rekaman CCTV malam untuk mendeteksi kendaraan dan mengidentifikasi
kondisi lampu sebagai `Standar` atau `TidakStandar`, terpisah untuk mobil dan motor. Ini adalah
klasifikasi visual berbasis data kamera; tanpa pengukuran fotometrik langsung, hasilnya tidak
membuktikan kepatuhan terhadap ambang candela/lux dalam regulasi.

### Label dan peran model

| Tahap | Keluaran aktif | Penjelasan |
|---|---|---|
| Store anotasi / COCO | `MobilStandar`, `MobilTidakStandar`, `MotorStandar`, `MotorTidakStandar`, `PantulanCahaya` | Lima kelas anotasi manual yang menjadi sumber target supervised. |
| Grounding DINO base | Lima kelas anotasi | Proposal pseudo-label menggunakan kelas yang sama dengan store anotasi; masuk sebagai tambahan train. |
| YOLO | Lima kelas anotasi | Melokalisasi kelas kendaraan dan PantulanCahaya; PantulanCahaya disaring sebelum ByteTrack. |
| NB04b classifier | Lima kelas anotasi | ResNet18 menggabungkan crop RGB, 77 fitur numerik (35 bbox photometry + 35 foreground-mask photometry + 7 geometri mask), dan one-hot e-means dengan k hasil seleksi BIC; scaler dan centroid hanya fit train. Mask kosong memakai vektor foreground nol dan dicatat; proporsi di atas 1% menghentikan training. |
| NB06 | kelas hasil classifier per track | YOLO memberi bbox/track, CNN menentukan kelas; crossing line menghitung track MobilTidakStandar dan MotorTidakStandar satu kali per video. |

### Satuan angka dan evaluasi

Hitungan resmi NB06 adalah **satu crossing per track tidak standar** pada video split `test`.
Counting line horizontal awal berada di 72% tinggi frame dan dipotong mengikuti ROI tiap kamera;
arah dan frame crossing dicatat. `bbox_per_frame_timeline.csv` menyimpan deteksi per frame sebagai
metrik pendamping. Overlay wajib menampilkan mask SAM2 dan bbox TidakStandar merah serta garis hitung kuning.

NB05 mengukur deteksi tipe kendaraan Mobil/Motor dan mengkalibrasi threshold dari validation.
NB04/04b melatih model; validation membantu pemilihan model/threshold; test hanya untuk pengukuran
akhir setelah ground truth NB02c direview. Metrik test tidak tersedia bila ground truth belum cukup.
Kurva train/validation per epoch bukan akurasi test. MSE/RMSE untuk classifier mengukur error
probabilitas label multiclass (Brier score/root Brier); error koordinat/ukuran bbox dilaporkan terpisah.

### ROI, preprocessing, dan batas interpretasi

Training detector dan inferensi NB06 memakai crop ROI yang sama. NB06 memetakan bbox pada crop
kembali ke koordinat frame penuh untuk overlay. Audit 12 November 2026 menemukan dimensi metadata
store lama pada 81 frame; koordinat bbox sesuai gambar ROI aktif sehingga bbox tidak digeser.
Metadata store/COCO dan export YOLO lokal sudah diperbaiki dengan backup. Karena normalisasi label
berubah, model detector dan classifier perlu dilatih ulang sebelum hasil baru dipakai.

Crop CNN menjaga rasio aspek: sisi terpanjang diubah ke 224 px lalu diberi padding, bukan
stretch/center-crop. Pixel RGB dinormalisasi dengan mean/std ImageNet. Fitur numerik dihitung dari
crop bersih dan dinormalisasi memakai statistik train; ColorJitter hanya diterapkan pada cabang
citra CNN saat training. Fitur HSV adalah proksi dari video yang dapat dipengaruhi exposure, white
balance, blur, dan kompresi; ia bukan alat ukur intensitas cahaya fisik.

Kelas Silau/Berwarna dan rencana reference patches pada bagian historis tidak menjadi target model
aktif. Rujuk README untuk urutan rerun, arsip audit untuk konteks historis,
dan `RISK_REGISTER.md` untuk status risiko yang sudah disaring.

---

## 2. Dataset

Sumber: `CCTV Kab Bogor_ Raya Lintang Kita (Rawamurang) (2026-08-01)` â€” folder
berisi 5 kamera:

- **B01** â€” Gerbang Tol Ciawi
- **B02** â€” Gerbang Tol Ciawi 2
- **B03** â€” Ciawi
- **B04** â€” Jalan Raya Lintang Kita, Gunung Sindurato
- **B05** â€” Jalan Raya Lintang Kita, Ciparigi

**Kunci nama file (`baca_footage.py`):**

- `atcs_<camera_id>_<YYYYMMDD>_<HHMMSS>.mp4` â†’ kamera sesuai `<camera_id>`

**Aturan pemetaan nama file (penting â€” jangan diubah):**

- Huruf `O` pada nama file CCTV **selalu dianggap angka `0`**.
- Setelah substitusi, hanya **6 digit pertama** yang dipakai sebagai
  timestamp: `YYMMDDHHMMSS`.

### 2.1 Konfigurasi Video (Terverifikasi 2026-10-18)

Ukuran dan fps berbeda per kamera â€” **bukan satu resolusi seragam**:

| Kamera | Resolusi | FPS | Frames/video | Durasi |
|---|---|---|---|---|
| bogor-kab-bappenda1 | 704Ã—576 | 25.00 | 15.000 | 600 s |
| bogor-kab-bappenda2 | 640Ã—480 | 25.00 | 15.000 | 600 s |
| bogor-kab-bappenda3 | 640Ã—480 | 25.00 | 15.000 | 600 s |
| bogor-kab-pdam2 | 640Ã—360 | 24.92 | 14.950 | 600 s |
| bogor-kab-pemda2 | 640Ã—480 | 25.00 | 7.840 | 314 s |

Konsekuensi: **jangan pernah mengasumsikan 1280Ã—720 atau 768Ã—576.** Semua
koordinat (ROI, box) harus di-clamp ke ukuran frame aktual per kamera â€” sudah
ditangani `crop_and_fit_frame` di NB02 via `Image.open(...).size`.

### 2.2 Ringkasan Dataset

| Item | Nilai | Sumber |
|---|---|---|
| Total file (dataset penuh) | 439 | Perencanaan awal |
| Durasi (dataset penuh) | 34.27 jam | Perencanaan awal |
| Kamera | 5 (`bogor-kab-bappenda1/2/3`, `pdam2`, `pemda2`) | Terverifikasi |
| Video di folder kerja saat ini | 5 (1 per kamera) | Terverifikasi 2026-10-18 |
| Periode | 1â€“31 Agustus 2026 | Perencanaan awal |
| Frame di `frames_manifest.csv` | 4.520 | Terverifikasi 2026-10-18 |

> **Catatan:** angka 439 file / 34.27 jam adalah rencana dataset penuh.
> Folder kerja saat ini (`output/videos/`) berisi **5 video contoh** (satu per
> kamera). Sesuaikan klaim laporan dengan cakupan yang benar-benar diproses.

---

## 3. Grounding untuk Model

Model yang digunakan: **GroundingDINO-tiny (swin-t)**.

### 3.1 Kategori dan Prompt

Taksonomi yang dipakai (2 kelas â€” diputuskan, lihat Â§3.3):

| Prompt DINO | Nama kelas akhir | Definisi |
|---|---|---|
| `car headlight.` | **LampuMobil** | Seluruh area terang lampu depan mobil (headlight), termasuk gabungan pasangan |
| `motorcycle headlight.` | **LampuMotor** | Seluruh area terang lampu depan sepeda motor (headlight) |

Catatan penamaan: nama kelas memakai awalan `Lampu` agar konsisten dan tidak
bertabrakan dengan nama kelas COCO lain (`car`, `motorcycle`,
`motorcycle headlight road`) saat eksperimen prompt. Prompt ditulis
**lowercase + diakhiri titik** â€” pola resmi Grounding DINO pada transformers 5.x
(berbentuk kalimat pendek; tanpa titik, skoring teksnya kurang stabil).

### 3.2 Prompt Engineering untuk Lampu

GroundingDINO menangani **satu frasa per kelas**. Untuk `text_threshold` bekerja
benar, setiap frasa hanya boleh merujuk **satu kelas**. Karena itu pemrosesan
dilakukan **multi-pass (satu frasa per kelas)**:

| Pass | Frasa tunggal | Hasil lalu digabung |
|---|---|---|
| 1 | `"car headlight."` | kelas `LampuMobil` |
| 2 | `"motorcycle headlight."` | kelas `LampuMotor` |

Duplikat dari dua pass digabung berdasarkan IoU â‰¥ 0.5 (`LAMP_MERGE_IOU_THRESHOLD`);
kelas grup ditentukan hierarki geometri â†’ vote verifier â†’ vote DINO (lihat Â§3.5).

### 3.3 Alasan Pemilihan Taksonomi 2 Kelas (Diputuskan)

Keputusan: **2 kelas `LampuMobil`/`LampuMotor`**. Pertimbangan:

- Dataset CCTV jarak jauh, malam hari, lampu hanya titik terang â†’ membedakan
  mobil vs motor hanya dari bentuk/ukuran lampu.
- Menambah kelas `LampuMobilJauh`/`LampuMobilDekat` memecah proporsi sangat
  tidak seimbang (mobil â‰ˆ 91% vs motor â‰ˆ 9%) menjadi kelas-kelas kecil.
- Berisiko menambah noise, bukan mengurangi.

### 3.4 Mitigasi Bias Box â€” Mode A (Lampu-Sepasang)

Mode A yang dipakai (terpilih dari 3 opsi yang dibahas):

| Kelas | Definisi box |
|---|---|
| **LampuMobil** | **Satu box** hasil **union** dari pasangan lampu mobil (bbox gabungan) |
| **LampuMotor** | **Satu box** untuk lampu tunggal sepeda motor |

Rationale: pada mobil, DINO cenderung memberi 1â€“2 box per lampu atau 1 box
gabungan â†’ berubah-ubah â†’ noise saat training. Dengan union pasangan, output
lebih konsisten. Pada motor, lampu depannya satu â†’ tidak ada merging.

### 3.5 Strategi Pemberian Kelas (Grouping Geometri + Vote DINO)

Keputusan: **grouping berbasis GEOMETRI, vote DINO sebagai tiebreak.**

Alasan menolak **Opsi 1 (vote DINO murni)**: dokumen resmi Grounded-SAM
menegaskan DINO hanya bisa mengenali **objek makro** (`car`, `motorcycle`)
dengan cukup akurasi, sedangkan `taillight`/`headlight` hanya diberi label
**"lumayan"** â€” justru kasus yang kita hadapi. Menjadikan vote sebagai
mekanisme utama berarti mewariskan kelemahan itu ke pseudo-label.

Alasan memilih geometri: pada citra malam jarak jauh, **jumlah lampu mobil
(2) vs motor (1) adalah pembeda paling andal** â€” lebih stabil daripada
keyakinan DINO yang lemah pada sub-kategori.

Aturan grouping (untuk kandidat LampuMotor â€” yaitu kandidat yang **tidak**
tergabung):

1. Kandidat lampu yang **tidak tergabung** dengan kandidat lain â†’ **LampuMotor**
   (kecuali hierarki kelas di bawah menentukan lain).
2. Kandidat lampu yang tergabung menjadi **â‰¥ 2** (pasangan) â†’ union â†’ **LampuMobil**.
3. Syarat digabung: jarak tengah â‰¤ `80 px`, selisih tinggi tengah â‰¤ `25 px`,
   rasio lebar â‰¤ `2.5x`. (Duplikat antar-pass digabung lebih dulu bila
   IoU â‰¥ 0.5 â€” lihat Â§3.2.)

**Hierarki penentuan kelas per grup (implementasi saat ini, 2026-10-19):**

1. Grup â‰¥ 2 lampu â†’ **`LampuMobil`** (geometri â€” pembeda paling andal).
2. Grup 1 lampu â†’ **vote verifier torchvision FasterRCNN** â€” kelas kendaraan
   COCO (`car/bus/truck` â†’ `LampuMobil`, `motorcycle` â†’ `LampuMotor`) yang
   box-nya menutupi â‰¥ 30% area box lampu (`VEHICLE_VOTE_MIN_COVERAGE`).
3. Tanpa vote â†’ **argmax skor DINO multi-pass** (tiebreak terakhir).

Catatan implementasi (penting): skor DINO dinilai dari **nilai logit/score
mentah** yang diambil saat inferensi multi-pass, **bukan** dari `text_threshold`
(`text_threshold` hanya pemilih box biner, bukan pengukur keyakinan). Pemisahan
pasangan berdasarkan vote DINO (aturan lama) **tidak** diimplementasikan â€”
pasangan selalu `LampuMobil`. Semua sinyal yang tersedia disimpan per deteksi
(`vehicle_vote`, `signals_agree`) untuk review di NB03.

---

## 4. Alur Riset (Pipeline)

```
Scraping & Ekstraksi Frame (NB01) âœ…
        â”‚
        â–¼
Pseudo-labelling (NB02) âœ… selesai (2.018 box, 1.482 gambar YOLO)
  â”œâ”€ ROI polygon per kamera (di dalam NB02)
  â”œâ”€ Dedup frame (dedup_manifest.csv)
  â”œâ”€ GroundingDINO multi-pass (Mode B)
  â””â”€ Grouping geometri + vote DINO (Mode A)
        â”‚
        â–¼
Anotasi Manual (NB02b) âœ… dibuat 2026-10-20
  â””â”€ Koreksi pseudo-label (400 frame) â†’ COCO (fine-tune DINO) + gold-set YOLO
        â”‚
        â–¼
Validasi & Kalibrasi (NB03) âŒ belum dibuat
  â””â”€ Gold-set + kalibrasi threshold
        â”‚
        â–¼
Training YOLO (NB04) âŒ belum dibuat
        â”‚
        â–¼
Evaluasi & Analisis Error (NB05) âŒ belum dibuat
```

> **Koreksi 2026-10-18:** tidak ada tahap NB02b **versi CLAHE** maupun NB02a
> (frame filtering terpisah). NB02 bersifat mandiri: ROI polygon dan dedup
> frame ditangani di dalam notebook yang sama. CLAHE tidak dipakai.
>
> **Koreksi 2026-10-20:** slot **NB02b** kini terisi sebagai **tahap anotasi
> manual** (koreksi pseudo-label NB02) â€” lihat Â§10.0. Urutan eksekusi nyata:
> `NB01 â†’ NB02 (pseudo) â†’ NB02b (koreksi) â†’ fine-tune DINO â†’ re-NB02 â†’ NB03`.

### 4.1 Posisi FM pada Pipeline

FM (fitur *Fine-grained Matching*) adalah **tahap lanjutan** setelah YOLO
dilatih dan dievaluasi.

```
YOLOv11 Detection â†’ Deep Feature Extraction â†’ Fine-Grained Matching â†’ Tracking
```

Fokus FM berada di **tahap matching**, yaitu menentukan apakah dua crop dari
frame berbeda berasal dari **kendaraan yang sama**.

### 4.2 Peran Pseudo-labelling

Pseudo-labelling adalah **tahap kritis** yang menentukan kualitas label untuk
training YOLO. Output NB02 akan dilatih sebagai **gold-set** untuk
mengkalibrasi threshold DINO di NB03.

---

## 5. Keputusan Desain dari Review 2026-10-15

### 5.1 Mode Pseudo-labelling: Mode B (DINO-only, SAM dihapus)

Keputusan: **hapus SAM/segformer**, pakai DINO saja.

Riset (sumber resmi):

- Dokumen resmi Grounded-SAM: SAM hanya untuk **segmentasi**, **bukan**
  klasifikasi.
- Pada deteksi *fine-grained* (mis. 2 jenis lampu), SAM **tidak memberi
  improvement signifikan**.
- GroundingDINO sudah memberi **box + confidence** â†’ cukup untuk
  pseudo-labelling.
- SAM hanya berguna jika butuh **mask presisi** untuk crop / analisis shape.

Konsekuensi: **semua parameter `SAM_*` tidak dipakai** (dipertahankan di config
hanya sebagai referensi). Pipeline jadi lebih sederhana dan hemat VRAM.

### 5.2 Multi-pass per frasa kelas

Karena GroundingDINO menangani satu frasa per kelas, pemrosesan dilakukan dua
pass (lihat Â§3.2). Hasil dua pass digabung: duplikat IoU â‰¥ 0.9 disatukan,
kelas ditentukan vote DINO.

### 5.3 Konstanta Grouping (Mode A)

| Konstanta | Nilai | Fungsi |
|---|---|---|
| `GROUP_MAX_CENTER_DISTANCE_PIXELS` | `80` | Jarak maks tengah 2 lampu agar dianggap sepasang |
| `GROUP_MAX_VERTICAL_OFFSET_PIXELS` | `25` | Selisih maks tinggi tengah |
| `GROUP_SIZE_RATIO_LIMIT` | `2.5` | Rasio maks ukuran 2 lampu |
| `LAMP_MERGE_IOU_THRESHOLD` | `0.5` | IoU minimum untuk merging |
| `MULTI_LAMP_MIN_COUNT` | `2` | Minimum jumlah lampu untuk jadi LampuMobil |

Nilai ini bersifat **awal** dan akan dikalibrasi ulang di NB03 berdasarkan
gold-set.

### 5.4 Threshold (diubah 2026-10-19, kalibrasi di NB03)

| Parameter | Nilai Saat Ini | Riwayat & Status |
|---|---|---|
| `DETECTION_THRESHOLD` | **0.30** | 0.25 (awal) â†’ 0.15 (fix run 1) â†’ **0.30** (paket 3-sinyal â€” 0.15 memicu pasangan lampu palsu). Kalibrasi ulang di NB03 |
| `TEXT_THRESHOLD` | 0.20 | No-op di transformers 5.x (hanya pemilih label, bukan filter box) |
| `MAX_BOX_AREA_RATIO` | 0.40 | Filter box raksasa (> 40% luas crop dibuang) |
| `VEHICLE_SCORE_THRESHOLD` | 0.4 | Ambang skor deteksi kendaraan verifier (lihat Â§5.6) |
| `VEHICLE_VOTE_MIN_COVERAGE` | 0.3 | Vote kendaraan hanya sah bila menutupi â‰¥ 30% area box lampu |
| Kalibrasi di NB03 | Optimal | Rencana â€” target F1 maksimum pada gold-set |

### 5.5 Model DINO (diputuskan 2026-10-18)

| Item | Nilai | Keterangan |
|---|---|---|
| `DETECTION_MODEL_ID` | `IDEA-Research/grounding-dino-tiny` | Dipakai untuk run pertama NB02 |
| Varian | swin-t | Sesuai dokumentasi riset, ringan di RTX 3050 4GB |

**Jalur upgrade:** bila hasil anotasi NB02 (dicek di NB03 lewat gold-set)
kurang bagus â€” miss detection tinggi pada lampu kecil atau banyak false
positive â€” ganti `DETECTION_MODEL_ID` di `builders/build_nb02_pseudo_label.py`
menjadi `IDEA-Research/grounding-dino-base` (swin-b), lalu regenerate notebook
dan re-run NB02. Pertimbangan biaya: base butuh download ~900 MB tambahan dan
VRAM lebih besar, tetapi akurasi sedikit lebih baik pada objek kecil.

### 5.6 Verifier Kendaraan (diputuskan 2026-10-19)

| Item | Nilai | Keterangan |
|---|---|---|
| Model verifier | `fasterrcnn_resnet50_fpn_v2` (torchvision, bobot `COCO_V1`) | Vote sekunder untuk grup 1 lampu. OWLv2 / Florence-2 / YOLO-World di-skip (lebih berat, tanpa keuntungan terukur) |
| `VEHICLE_CLASS_REMAP` | `car/bus/truck â†’ LampuMobil`, `motorcycle â†’ LampuMotor` | Hanya kelas kendaraan COCO yang dipakai; kelas lain diabaikan |
| `VEHICLE_SCORE_THRESHOLD` | 0.4 | Ambang skor deteksi kendaraan |
| `VEHICLE_VOTE_MIN_COVERAGE` | 0.3 | IoU box lampu vs kendaraan selalu kecil â†’ dipakai **coverage** (luas irisan / luas box lampu); vote sah bila â‰¥ 0.3 |
| `VERIFIER_DEVICE` | `cuda` | Benchmark 2026-10-19: **1,15 d/frame** (DINO+verifier), **peak VRAM 1,82 GiB** dari ~4,3 GiB; bobot 167 MB diunduh sekali (butuh internet) |
| Gagal dimuat | Verifier dimatikan otomatis | `load_vehicle_verifier` menangkap exception â†’ vote `None`, deteksi DINO tetap jalan |

Vote **sekonder** â€” hanya menentukan kelas bila geometri grup < 2 lampu, tidak
pernah menghapus box. Kelemahan domain malam (vote lemah/salah) dipantau lewat
flag `signals_agree` dan dikalibrasi di NB03 pada gold-set.

---

## 6. ROI (Region of Interest) â€” Polygon 5 Kamera

### 6.1 Keputusan Desain

ROI direkonstruksi menjadi **polygon per kamera** (sebelumnya 2 rectangle untuk
B01/B02). Setiap kamera mendapat **satu polygon** yang membatasi area jalan
yang relevan.

Alasan: CCTV kabupaten punya area noise besar (pohon, gedung, lampu jalan,
langit). ROI memotong jumlah frame/kandidat yang diproses dan mengurangi
false positive.

### 6.2 Posisi Saat Ini â€” FINAL âœ…

- **Final dan disetujui user (2026-10-18).** `output/roi/roi_config.json`
  berisi polygon untuk semua 5 kamera:

  | Kamera | Frame aktual | Verteks | Catatan |
  |---|---|---|---|
  | bogor-kab-bappenda1 | 704Ã—576 | 4 | trapesium |
  | bogor-kab-bappenda2 | 640Ã—480 | 4 | trapesium |
  | bogor-kab-bappenda3 | 640Ã—480 | 4 | trapesium |
  | bogor-kab-pdam2 | 640Ã—360 | 8 | 3 verteks sengaja dibuat rapat agar watermark tidak ikut |
  | bogor-kab-pemda2 | 640Ã—480 | 24 | mengikuti kurva jalan, disengaja |

- **Ukuran frame berbeda per kamera** â€” bukan 1280Ã—720. Koordinat polygon
  mengikuti PNG asli tiap kamera; builder NB02 meng-clamp ke ukuran frame
  aktual (`Image.open` â†’ `size`), jadi koordinat selalu valid.
- Hasil verifikasi: tidak ada polygon bersilangan, tidak ada verteks di luar
  batas frame. Overlay preview tersimpan di `output/roi/preview/` (5 file PNG).
- Field `roi` (rectangle) di config masih menyimpan nilai template 1280Ã—720 â€”
  **usang dan tidak dipakai**, karena `polygon` diprioritaskan oleh
  `crop_and_fit_frame`. Aman dibiarkan, tapi jangan dijadikan acuan resolusi.

### 6.3 Kontrol Picker (Sudah Tersedia)

| Aksi | Kontrol |
|---|---|
| Tambah verteks | Klik kiri |
| **Undo (hapus verteks terakhir)** | Klik kanan / tombol `u` / `backspace` |
| Simpan polygon | `Enter` |
| Lewati kamera (pertahankan ROI lama) | `Esc` |

Peringatan otomatis juga sudah ada:
- Kurang dari 3 verteks â†’ `Enter` ditolak dengan pesan `butuh minimal 3 verteks`.
- Polygon bersilangan â†’ `Enter` ditolak dengan pesan `poligon bersilangan`, minta undo lalu perbaiki.
- Verteks duplikat (jarak â‰¤ 2 px) dibersihkan otomatis sebelum disimpan.

### 6.4 Validasi ROI â€” SELESAI âœ…

- **Geometri:** semua 5 polygon diverifikasi otomatis â€” tidak bersilangan,
  seluruh verteks di dalam batas frame masing-masing.
- **Visual:** overlay polygon di atas frame contoh per kamera dibuat di
  `output/roi/preview/*.png` dan **dicek langsung oleh user â†’ benar**.
- **Persetujuan user:** diberikan 2026-10-18. ROI siap dipakai di NB02.

### 6.5 File Terkait

| File | Peran |
|---|---|
| `builders/build_roi_config.py` | Tool interaktif pemilihan polygon (matplotlib) |
| `output/roi/roi_config.json` | Konfigurasi ROI per kamera (5 polygon, final) |
| `output/roi/preview/*.png` | Overlay visual per kamera untuk verifikasi |

> Catatan: konstanta ROI didefinisikan di dalam `builders/build_roi_config.py`
> (konstanta grup dan nilai batas). Tidak ada modul `src/config_roi.py`
> terpisah.

---

## 7. Rencana Risiko (Register Risiko)

Dokumen terpisah: `RISK_REGISTER.md` (v1.5). Isi ringkas:

- **19 risiko** terdaftar (B1â€“B4, H1â€“H10, I1â€“I5).
- Semua isu hasil review 8 reviewer dipetakan sebagai risiko terkontrol.
- Kontrol pre-fix **C1â€“C4** (backup, self-check, config signature, gold-set)
  sudah dijalankan sebelum perubahan kode.
- **Gate verifikasi** ditetapkan sebelum run NB02 (lihat register).

---

## 8. Backup & Hasil Lama

### 8.1 Backup Sebelum Perubahan

`backup_sebelum_fix_20261015/` sudah dipindahkan ke
`trash_2026-10-26/audit_2026-10-28/` pada 2026-10-28 (audit temuan 14): isinya
duplikat byte-per-byte dari builder aktif, sehingga berisiko diedit orang lain.
Tidak ada rollback point terpisah lagi â€” riwayat perubahan mengandalkan dokumen
audit + `RISK_REGISTER.md` (R16).

Backup ini adalah **rollback point** sebelum perubahan NB02.

### 8.2 Output NB02 Lama (Dipindah, Bukan Dihapus)

- Lokasi lama: `output/nb02/` (15 file JSON + 837 crop jpg)
- Lokasi sekarang: `_trash_riset_hampirmatang_20261015/b02/`
- Alasan: agar tidak bercampur saat run ulang NB02.

Output NB01 (`output/nb01/`) **tidak dihapus** karena scraping tidak berubah.

---

## 9. Struktur Berkas (Terverifikasi 2026-10-18)

```
Riset-HampirMatang/
â”œâ”€â”€ MASTER_DOKUMENTASI_HAMPIR.md      â† Dokumen ini
â”œâ”€â”€ RISK_REGISTER.md                  â† Register risiko (v1.5)
â”œâ”€â”€ NB01_acquisition.ipynb            â†’ Scraping & ekstraksi frame
â”œâ”€â”€ NB02_pseudo_label.ipynb           â†’ Pseudo-label DINO (dihasilkan builder)
â”œâ”€â”€ builders/
â”‚   â”œâ”€â”€ build_nb01_acquisition.py     â†’ Builder NB01 (+ template ROI, group split)
â”‚   â”œâ”€â”€ build_nb02_pseudo_label.py    â†’ Builder NB02 (DINO multi-pass, Mode B + A)
â”‚   â”œâ”€â”€ build_roi_config.py           â†’ Tool interaktif polygon ROI
â”‚   â””â”€â”€ manual_check_nb02.py          â†’ Pemeriksaan manual logika crop + grouping (bukan test otomatis)
â”œâ”€â”€ test_pipeline_logic.py            â†’ Self-check tanpa GPU (kontrak hitung + fragment stitching)
â”œâ”€â”€ requirements.txt                  â†’ Pin dependency yang dipakai
â”œâ”€â”€ README.md                         â†’ Cara setup & urutan menjalankan
â”œâ”€â”€ output/
â”‚   â”œâ”€â”€ videos/                       â†’ file .mp4 per rekaman (nama bertimestamp)
â”‚   â”œâ”€â”€ frames/                       â†’ folder frame PNG per video
â”‚   â”œâ”€â”€ katalog/                      â†’ bogor_kab_targets.json, cctv_all.json
â”‚   â”œâ”€â”€ manifests/                    â†’ scraping_manifest, frames_manifest, dedup_manifest
â”‚   â”œâ”€â”€ split/group_split.csv         â†’ Split train/val/test per kamera
â”‚   â”œâ”€â”€ roi/                          â†’ roi_config.json + preview/ (5 PNG)
â”‚   â””â”€â”€ nb02/                         â†’ chunks/, previews/, yolo/
â””â”€â”€ trash_2026-10-26/                 â†’ Berkas yang dipindahkan keluar pipeline (bukan dihapus)
```

> **Penting:** modul `src/*.py` **kosong** â€” seluruh logika hidup sebagai string
> di dalam `builders/build_*.py` lalu dirender menjadi sel notebook. Modul
> NB03/NB04/NB05/FM **belum pernah dibuat** â€” hanya ada sebagai rencana.

### 9.1 Fungsi Berkas yang Ada

| File | Fungsi |
|---|---|
| `builders/build_nb02_pseudo_label.py` | Merender `NB02_pseudo_label.ipynb` (5 sel kode + 1 markdown). Memuat logika DINO-only multi-pass (Mode B) + Mode A grouping. SAM dihapus. |
| `archive/one-off-scripts/manual_check_nb02.py` | Pemeriksaan manual historis; bukan bagian pipeline aktif. Test aktif ada di `test_pipeline_logic.py`. |
| `builders/build_roi_config.py` | Tool interaktif pemilihan polygon ROI per kamera + konstanta ROI. |
| `builders/build_nb01_acquisition.py` | Scraping katalog, ekstraksi frame, group split, template ROI. |

### 9.2 Berkas yang Belum Dibuat

| Tahap | Status |
|---|---|
| NB02b (anotasi manual â€” koreksi pseudo-label) | âœ… Ada (2026-10-20) â€” perlu dijalankan manual |
| NB03 (gold-set + kalibrasi threshold) | âŒ Belum ada |
| NB04 (dataset YOLO + training) | âŒ Belum ada |
| NB05 (evaluasi + analisis error) | âŒ Belum ada |
| FM (fine-grained matching) | âŒ Belum ada |

---

## 10. Status Pipeline Saat Ini

| Tahap | Notebook | Status |
|---|---|---|
| Scraping & Ekstraksi Frame | `NB01_acquisition.ipynb` | âœ… Selesai (5 video, 10 folder frame, 4.520 frame) |
| Pseudo-labelling (DINO Mode B + A) | `NB02_pseudo_label.ipynb` | âœ… Selesai (1.548 frame unik â†’ 6.490 box mentah â†’ 2.018 box YOLO, 1.482 gambar). âš ï¸ Output **UNSplit** â€” pemecahan train/val/test menunggu NB04, lihat Â§10.0b |
| Anotasi Manual (koreksi pseudo-label) | `NB02b_annotation.ipynb` | âœ… Dibuat 2026-10-20 (6 sel kode + 1 markdown, lolos self-check) â†’ **siap dijalankan manual** (lihat Â§10.2) |
| Validasi & Kalibrasi Threshold | NB03 | âŒ Belum dibuat |
| Training YOLO | NB04 | âŒ Belum dibuat |
| Evaluasi & Analisis Error | NB05 | âŒ Belum dibuat |
| Fine-Grained Matching | FM | âŒ Belum dibuat |

### 10.0 NB02b â€” Anotasi Manual Pinpoint (Ditambahkan 2026-10-20)

**Tujuan.** Mengoreksi pseudo-label NB02 secara manual pada sampel frame, dan
memakai hasilnya untuk **dua hal sekaligus**: (a) dataset fine-tune Grounding
DINO (continue pre-trained), (b) gold-set validasi/kalibrasi untuk NB03.

**Posisi.** NB02b diletakkan **setelah** NB02 (karena pseudo-label adalah bahan
yang dikoreksi) meskipun nomornya `01b`. NB02 hanya butuh frame dari NB01, jadi
tidak ada siklus dependensi. Urutan eksekusi nyata:

```
NB01 (akuisisi) â†’ NB02 (pseudo-label) â†’ NB02b (koreksi manual)
                                      â†’ fine-tune Grounding DINO
                                      â†’ re-run NB02 (label otomatis naik kualitas)
                                      â†’ NB03 (gold-set + kalibrasi threshold)
```

**Implementasi.**

| Aspek | Nilai |
|---|---|
| Builder | `builders/build_nb02b_annotation.py` |
| Notebook | `NB02b_annotation.ipynb` (6 sel kode + 1 markdown) |
| Sumber gambar | `output/nb02/yolo/images/*.png` (sudah ROI-crop + fit 640px) |
| Sumber koreksi | `output/nb02/chunks/*.jsonl` (box pseudo-label) |
| **Split gold-set** | Hanya `train` + `val` (kamera `test` dikecualikan). âš ï¸ Gold-set **masih mencampur `train`+`val` dalam satu folder** â€” pecah per split sebelum evaluasi NB03 (lihat R8) |
| Mode | Koreksi pseudo-label: box DINO ditampilkan, user tambah/hapus/ubah kelas |
| Sampling | Stratified proporsional per kamera, `SAMPLE_TOTAL_FRAMES = 400`, seed `20261020` |
| **Anti-kebocoran** | **Kamera split `test` dikecualikan** (`SAMPLE_EXCLUDED_SPLITS = ["test"]`) â€” hanya `train` + `val` dianotasi + guard ganda di sel EXPORT |
| Alokasi nyata | `bappenda1` (train) 267 Â· `pemda2` (val) 87 Â· `bappenda3` (train) 46 = **400** |
| Frame dikecualikan | `bappenda2` (test) 195 Â· `pdam2` (test) 445 = **640** tidak tersentuh |
| UI | `ipympl` (`%matplotlib widget`), `RectangleSelector` + klik, `ToggleButtons` kelas |
| Store | `output/annotation/annotations.json` (per frame, tulis atomik, resume-aman) |
| Manifest | `output/annotation/sample_manifest.json` (disimpan sekali â†’ sampling stabil) |
| Export COCO | `output/annotation/annotations_coco.json` (fine-tune DINO) |
| Export gold-set | `output/annotation/yolo/{images,labels,data.yaml}` |
| Summary | `output/annotation/annotation_summary.json` + 2 PNG preview |

**Alur sel.**

1. Konfigurasi + `mkdir` output + peta split per kamera.
2. Sampling stratified per kamera **setelah membuang split `test`** â†’ manifest (reload bila sudah ada;
   manifest lama dengan `excluded_splits` berbeda akan **menolak** dipakai).
3. Muat store anotasi + `save_annotation_store()` + validasi (kelas, span box, frame asing).
4. Preview 3 frame pertama (box pseudo-label) â€” cek cepat sebelum anotasi.
5. UI interaktif (`ipympl`) â€” autosave tiap mutasi.
6. Export COCO + gold-set YOLO + summary + preview hasil.

**Provenance box.** Tiap box menyimpan `source`: `pseudo` (belum disentuh),
`manual` (ditambah user), `edited` (kelas diubah user). Berguna saat audit
kualitas gold-set di NB03 â€” box yang masih `pseudo` berarti belum diverifikasi
manusia.

**Kelas ke-3 `PantulanCahaya` (diputuskan 2026-10-22).** Awalnya (2026-10-21)
pantulan hanya ditandai sebagai flag `kind: distractor` yang **tidak** ikut export.
Keputusan itu **dibatalkan** setelah data nyata menunjukkan pantulan bukan
fenomena langka:

| Metrik pada 126 frame yang sudah dianotasi | Nilai |
|---|---|
| Box aktif | 940 |
| Box pantulan/glare | **313 (33,3%)** |
| Frame yang mengandung pantulan | 42 dari 126 (**33%**) |
| Proyeksi 400 frame | **~994 box** |

Karena itu `PantulanCahaya` kini **kelas resmi ke-3** dan berperan ganda:

1. **Kelas deteksi** â€” ikut export COCO (fine-tune DINO) dan gold-set YOLO
   (`nc: 3`, index `2`). Model belajar membedakannya, bukan sekadar membuangnya.
2. **Hard-negative** â€” saat penghitungan kendaraan di hilir, deteksi kelas
   `PantulanCahaya` **difilter keluar**. Pantulan tidak pernah lagi terhitung
   sebagai kendaraan.
   âš ï¸ Sejak audit 2026-10-26: daftar objek yang dihitung berubah menjadi 8 angka
   kondisi lampu (**Â§10.6**); `PantulanCahaya` tetap tidak dihitung.

**Migrasi otomatis.** Store lama yang masih memakai `kind: distractor`
dikonversi saat `load_annotation_store()` dijalankan: field `kind` dibuang,
`class_name` diganti `PantulanCahaya`. Sudah diuji pada store nyata â€” **319 box
dimigrasi, 940 box aktif tetap utuh** (183 mobil + 444 motor + 313 pantulan).

**Shortcut keyboard.** `n`/`â†’` next Â· `p`/`â†` prev Â· `x`/`Delete` hapus box
terpilih Â· **`1`/`2`/`3`** set kelas box terpilih Â· `r` reviewed + next Â· `s` skip Â·
`u` undo Â· **`d` set box terpilih ke `PantulanCahaya`** (shortcut tercepat) Â·
**`T` sembunyikan/tampilkan label box** (nama kelas + nomor box). Klik kanan
dijeda oleh `RectangleSelector` (drag = tambah box).

**Yang masih terbuka untuk kelas ini.** NB02 **belum** punya prompt
`road reflection` â€” DINO tetap hanya memprediksi `LampuMobil`/`LampuMotor`,
sehingga box pantulan murni dari anotasi manual. NB04 wajib memfilter kelas
index `2` saat menghitung kendaraan.


**Cara pakai di VS Code.** Buka `NB02b_annotation.ipynb`, pilih kernel
`.venv312`, jalankan sel 1â€“4 berurutan (cepat), lalu sel 5 â†’ **panel gambar
keluar di tab "Output"** (silakan pop-out). Anotasi bisa ditinggal kapan saja â€”
tutup notebook, buka lagi, jalankan sel 3 dan 5, posisi reviewed tetap ada.
Setelah selesai, jalankan sel 6.

**Catatan penting.**

- **Anti-kebocoran (B1).** Sampling membaca `group_split.csv` lewat `SPLIT_NAME_BY_CAMERA` dan
  membuang kamera ber-split `test`. Ini menutup risiko B1 untuk gold-set **dan** untuk dataset
  fine-tune DINO (DINO fine-tuned pada kamera test akan merusak klaim generalisasi lintas-kamera).
  Ada dua lapis proteksi: filter saat sampling + `assert_no_excluded_split()` di sel EXPORT yang
  `raise` bila kamera test entah bagaimana ikut ter-export.
- **Konsekuensi:** kamera test tidak pernah muncul di UI anotasi, jadi tidak ada metrik kualitas
  pada kamera test dari NB02b. Evaluasi kamera test tetap diukur lewat NB03/NB04 dari split YOLO.
- Frame tanpa deteksi NB02 tidak ikut disampel (tidak ada di `yolo/images`) â€”
  ini disengaja agar UI tidak pernah menampilkan frame kosong.
- Gold-set (`output/annotation/yolo`) hanya berisi frame yang ber-`reviewed`;
  frame reviewed tanpa box tetap masuk COCO (sebagai contoh negatif) tetapi
  **tidak** masuk struktur YOLO.
- Mengubah `SAMPLE_TOTAL_FRAMES`/`RANDOM_SEED` atau menambah frame NB02 baru
  membuat manifest lama tidak cocok â†’ hapus `output/annotation/sample_manifest.json`
  (store anotasi tetap aman selama `frame_stem` masih ada di manifest baru).

### 10.1a Aturan Visualisasi Wajib (Ditambahkan 2026-10-18)

> **Setiap notebook tahap (`NBxx`) WAJIB menampilkan visualisasi dari proses
> sampai hasil**, bukan hanya angka. Kebutuhan: tugas riset membutuhkan bukti
> visual di laporan.

Cakupan minimal per notebook:

| Tahap | Visualisasi wajib |
|---|---|
| NB01 | Frame hasil ekstraksi per kamera |
| NB02 | Crop ROI â†’ box DINO (kandidat) â†’ grouping â†’ overlay label final + `vote`/`[konflik]` â†’ histogram ukuran box/score + statistik sinyal (vote & konflik) |
| NB03 | EDA frame, distribusi kelas, box vs gold-set |
| NB04 | Grid contoh dataset YOLO (gambar + label overlay) |
| NB05 | Confusion matrix, contoh TP/FP/FN |

Aturan tampilan: gambar **ditampilkan langsung di notebook** (`display(fig)` /
`plt.show()`), bukan hanya disimpan sebagai file dan ditulis sebagai teks path.
File PNG tetap disimpan untuk keperluan laporan.

### 10.0b Status Split Seluruh Artefak (Audit Kebocoran 2026-10-20)

Audit lintas-notebook menemukan bahwa kebocoran split **tidak hanya di NB02b**. Kondisi
seluruh artefak per 2026-10-20:

| Artefak | Split status | Aman untuk apa | Risiko |
|---|---|---|---|
| `output/nb02/chunks/*.jsonl` | UNSplit (5 kamera) | Sumber pseudo-label, audit coverage | Bukan dataset â€” jangan dipakai training langsung |
| `output/nb02/yolo/{images,labels}` | **UNSplit** (1.482 gambar, 640 di antaranya `test`) | Inspeksi visual, sumber gambar NB02b | ðŸ”´ **Tidak boleh** training: `data.yaml` `train==val` |
| `output/nb02/yolo/data.yaml` | Ditandai `split_status: UNSplit` | â€” | ðŸ”´ Jangan dipakai NB04 sebelum dipecah |
| `output/nb02/yolo/split_camera_map.csv` | Pemetaan resmi kameraâ†’split | Dasar pemecahan di NB04 | â€” |
| `output/annotation/*` (NB02b) | `train` + `val` saja (bebas `test`) âœ… | Fine-tune DINO, gold-set | ðŸŸ  `train`+`val` masih satu folder â†’ metrik per split wajib |

**Peta split resmi** (dari `output/split/group_split.csv`, diduplikasi ke `yolo/split_camera_map.csv`):

| Kamera | Split | Frame di `yolo/images` |
|---|---|---|
| `bogor-kab-bappenda1` | train | 562 |
| `bogor-kab-bappenda3` | train | 98 |
| `bogor-kab-pemda2` | val | 182 |
| `bogor-kab-bappenda2` | test | 195 |
| `bogor-kab-pdam2` | test | 445 |

**Yang sudah diperbaiki:** NB02b mengecualikan split `test` saat sampling (+ guard di export).
NB02 tidak lagi menyamar sebagai dataset siap-training â€” `data.yaml` bertanda `UNSplit`,
peta split ditulis eksplisit, dan sel summary mencetak peringatan leakage.

**Diperbarui 2026-10-26 â€” NB04 sudah memecah dan mematerialisasi split:**

| Artefak | Split status | Aman untuk apa | Risiko |
|---|---|---|---|
| `output/nb04/dataset/{images,labels}/{train,val,test}` | **Tersplit** (660/182/640), 1:1 gambar:label | Training + evaluasi | ðŸŸ  544 frame train dan 640 frame test masih berlabel pseudo; hanya 126 frame gold (semua train) |
| `output/nb04/dataset/data.yaml` | `train/val/test` tiga folder berbeda, `nc: 3` | Training YOLO | â€” |
| `output/nb04/split_audit.csv`, `class_balance.csv` | Audit per kamera + histogram kelas | Verifikasi kebocoran | â€” |
| `output/nb03/*` | Turunan langsung gold set per split | Kalibrasi threshold, diagnostik sinyal | ðŸŸ  val 0 frame gold â†’ kalibrasi belum sah |
| `output/nb05/*` | Kosong (menunggu bobot NB04) | â€” | Diisi saat NB05 dijalankan |

**Yang masih terbuka (wajib sebelum training/evaluasi):**

1. âœ… ~~**NB04** memecah `yolo/images` â†’ `images/{train,val,test}` memakai `split_camera_map.csv`,
   lalu tulis `data.yaml` yang benar-benar menunjuk tiga folder berbeda.~~ Selesai 2026-10-26.
2. âœ… ~~**NB03** memecah gold-set NB02b per split dan **melaporkan metrik terpisah** untuk
   `train` vs `val`; kalibrasi threshold DINO hanya dari `val`. Jangan melaporkan mAP
   gabungan train+val sebagai generalisasi.~~ Selesai 2026-10-26 (NB03 selalu memisah split).
3. ðŸŸ  **Anotasi 87 frame `val`** â€” tanpa ini NB03 tidak punya dasar kalibrasi dan NB04
   menolak training (guard `MINIMUM_GOLD_VAL_FRAMES = 30`).
4. ðŸŸ  **Gold-set test terpisah** â€” split test tidak punya anotasi manual; lihat R13.

### 10.1 Catatan: NB02 Mandiri (Tanpa Preprocessing NB02b / CLAHE)

**NB02b versi preprocessing CLAHE tidak diperlukan untuk NB02.** NB02 bersifat
mandiri:

- Membaca frame **asli** dari `output/frames/` (bukan hasil CLAHE).
- Menerapkan **ROI polygon sendiri** (`crop_to_polygon`) di dalam notebook.
- **Tidak** menerapkan CLAHE.

Jadi NB02 tetap dijalankan **langsung setelah NB01** â€” tidak ada tahap
preprocessing yang mendahuluinya. Slot `NB02b` kini dipakai untuk anotasi
manual, yang justru berjalan **setelah** NB02 (Â§10.0). Bila riset ingin memakai
CLAHE, itu keputusan desain baru yang perlu dibahas terpisah (lihat Â§13).

### 10.3 Urutan Pekerjaan Berikutnya

0. â›” **PERBAIKI TAKSONOMI DULU (audit 2026-10-26)** â€” target riset adalah
   **silau / berwarna / silau+berwarna per jenis kendaraan**, sedangkan seluruh
   kode masih memakai `LampuMobil`/`LampuMotor`/`PantulanCahaya`. Jangan latih
   NB04 dan jangan kutip angka NB05 sebelum Â§13.9â€“Â§13.10 diputuskan. Rincian dan
   kontrak hitung: **Â§10.6**.
1. âœ… ~~Finalisasi ROI polygon~~ â€” selesai 2026-10-18.
2. âœ… ~~Validasi ROI + approval user~~ â€” selesai 2026-10-18.
3. âœ… ~~Jalankan NB02 (run 1)~~ â€” selesai 2026-10-18, **hasil bermasalah (R1)**.
4. âœ… ~~**Perbaiki NB02 (R1)**~~ â€” run 2 diverifikasi bersih (2026-10-18);
   paket 3-sinyal + threshold 0.30 + verifier FRCNN ditambahkan 2026-10-19
   (lihat Â§0.4) â†’ **sudah di-run ulang 2026-10-19** (1.482 gambar / 2.018 box
   YOLO). Verifikasi box raksasa & statistik vote/konflik tercatat di
   sel visualisasi NB02.
5. ðŸ”„ **Jalankan NB02b (anotasi manual)** â€” 400 frame stratified dari `train`+`val`
   saja, koreksi box pseudo-label, export COCO + gold-set (lihat Â§10.0). Butuh
   waktu manual Â±3â€“5 jam; bisa dicicil, store resume-aman. **Status 2026-10-26:
   126/400 frame selesai, semuanya kamera `train` (`bappenda1`); 87 frame `val`
   belum tersentuh** â€” inilah penghambat NB03 (kalibrasi) dan NB04 (guard).
6. **Fine-tune Grounding DINO** â€” continue pre-trained dari
   `output/annotation/annotations_coco.json`, lalu **re-run NB02** agar label
   otomatis naik kualitas.
7. âœ… ~~**Pecah split (WAJIB sebelum NB03/NB04)**~~ â€” **selesai di NB04 (2026-10-26)**:
   `builders/build_nb04_train_yolo.py` mematerialisasi
   `output/nb04/dataset/{images,labels}/{train,val,test}` dari
   `output/nb02/yolo/split_camera_map.csv` (parser melewati baris komentar),
   hardlink gambar, label **gold menimpa pseudo** untuk frame teranotasi, dan
   `data.yaml` `nc: 3`. Hasil: **660 train / 182 val / 640 test**, jumlah
   gambar = label di tiap split, kamera `test` tidak pernah masuk train/val.
   Menutup sisa B1/R7; R8 (gold-set satu folder) ditutup karena metrik selalu
   dilaporkan per split.
8. âœ… ~~**Buat NB03**~~ â€” **dibuat 2026-10-26** (`NB03_gold_threshold.ipynb`,
   6 sel kode). EDA gold set + **kalibrasi threshold post-hoc** dari skor yang
   sudah tersimpan di `output/nb02/chunks/*.jsonl` (tanpa GPU), IoU-matching
   greedy per kelas, metrik per split, kalibrasi **hanya dari `val`**.
   Menjalankan pada data saat ini (126 frame gold, semuanya `train`) memberi
   F1 makro train **0,334** di threshold 0,30 dan **turun monoton** sampai
   0,00 di 0,75 â€” indikasi DINO menyimpan banyak TP berbobot rendah, dan
   **val belum bisa mengkalibrasi apa pun** (0 frame gold). `PantulanCahaya`
   dikeluarkan dari F1 makro karena pseudo-label masih 2 kelas.
9. âœ… ~~**Buat NB04**~~ â€” **dibuat & smoke-tested 2026-10-26**
   (`NB04_train_yolo.ipynb`, 7 sel kode). `yolo11n`, imgsz 640, batch 8,
   workers 0. **Guard aktif:** training ditolak selama frame val gold < 30
   (`MINIMUM_GOLD_VAL_FRAMES`), karena metrik val dari label pseudo tidak sah.
   Smoke test 1 epoch (guard dimatikan sementara) membuktikan alur training +
   val + penulisan bobot berjalan di RTX 3050 4 GB.
10. âœ… ~~**Buat NB05**~~ â€” **dibuat & smoke-tested 2026-10-26**
    (`NB05_evaluation.ipynb`, 7 sel kode). Metrik per split + confusion matrix
    dari objek `model.val()` (tanpa `pycocotools`), sapuan confidence
    sekali-jalan, **proxy counting** dengan filter kelas `PantulanCahaya` (R11),
    galeri error terburuk. Inferensi sapuan/counting berjalan di **CPU**
    (`PREDICT_DEVICE`) karena `predict` di GPU 4 GB OOM setelah dua `val`.
    âš ï¸ Kontrak hitung ini **digantikan** oleh 8 angka kondisi lampu di Â§10.6
    setelah audit taksonomi 2026-10-26. **Sejak 2026-10-28** sel hitung NB05
    menerbitkan `jumlah_box_*` (`counting_unit: "box_per_frame"`), bukan
    `jumlah_kendaraan_*`; angka kendaraan unik hanya dari NB06 (Â§10.6.1).
11. **Selesaikan anotasi NB02b â€” prioritas: 87 frame `val`** â€” tanpa frame val
    gold, NB03 tidak bisa merekomendasikan threshold dan NB04 menolak training.
    Setelah 400 frame selesai, lanjut **gold-set test terpisah Â±100 frame**
    dari kamera `test` (`pdam2`, `bappenda2`) agar R13 bisa ditutup.
12. **Fine-tune Grounding DINO** â€” continue pre-trained dari
    `output/annotation/annotations_coco.json`, lalu **re-run NB02** agar label
    otomatis naik kualitas, lalu jalankan ulang NB03 untuk kalibrasi.
13. âœ… ~~**Hitung hasil distractor NB02b**~~ â€” sudah diputuskan 2026-10-22:
    `PantulanCahaya` **naik jadi kelas resmi** (`nc: 3`, 313/940 box = 33,3%),
    dipakai sebagai hard-negative (lihat Â§10.0 dan Â§10.5).

---

### 10.5 Dataset Riset-4 â€” Inventaris & Pemanfaatan (Audit 2026-10-21)

Riset-4 (ATCS Bandung) punya aset anotasi manual. Sumber: `Riset-4/MASTER_DOKUMENTASI_RISET.md`.

**Hasil audit fisik (bukan sekadar dokumen):**

| Aset | Isi nyata | Bisa untuk fine-tune DINO? |
|---|---|---|
| `AnotasiManual/` | 490 crop PNG Â±30Ã—30 px, folder = kelas, **tanpa koordinat**, **tanpa frame asli** | âŒ Tidak |
| `anotasimanual-refined/` | 490 crop PNG, sama, versi diperjelas | âŒ Tidak |
| `golden_set_v3.csv` | 200 baris, **kolom koordinat KORUP** (Excel Indonesia: `19.05` â†’ `19.050.863.647.460.900`) | âŒ Jangan dipakai |
| `golden_set_v3/candidates_index.csv` | koordinat asli sebagai JSON float (`bbox_xyxy`) | âœ… **Sumber koordinat** |
| `Hasil-2/output/frames_sampled/` | 1.000 frame PNG resolusi penuh (640Ã—480) | âœ… **Sumber gambar** |
| `foundation_model/crops/` | 5.844 crop DINO+SAM | âŒ crop, bukan frame |

**Konverter (dibuat 2026-10-21).**

```
Temp/opencode/build_riset4_golden_to_coco.py    (--dry-run untuk cek tanpa tulis)
        â”‚  baca koordinat dari candidates_index.csv (bypass golden_set_v3.csv yang korup)
        â”‚  join ke Hasil-2/output/frames_sampled/frame_XXXXXX.png
        â–¼
Riset-4/output/foundation_model/golden_set_v3_coco.json   (68,5 KB)
```

Hasil nyata: **199 anotasi / 182 gambar** (200 baris âˆ’ 1 `LampuAtap` yang di-drop).
Distribusi: `LampuMobil` 121 Â· `LampuMotor` 77 Â· `PantulanCahaya` **1**.
0 koordinat hilang, 0 frame hilang, resolusi COCO cocok dengan file nyata (640Ã—480).

**Batas pemanfaatan.** 199 box di 182 frame **terlalu sedikit** untuk fine-tune DINO
sendirian (butuh 1.000+). Layak sebagai **tambahan** ke dataset NB02b, bukan pengganti.
`PantulanCahaya` hanya **1 sampel** â€” tidak cukup untuk apa pun.

**Temuan penting untuk keputusan kelas pantulan (Riset-HampirMatang).** Riset-4
menolak kelas `bukan_lampu` berdasarkan data: dari 500 sampel yang diperiksa manual,
hanya **2 yang benar-benar bukan lampu** â†’ FP rate kandidat DINO â‰ˆ **0,4â€“0,8%**.
Implikasi: jumlah pantulan yang lolos deteksi di Riset-HampirMatang mungkin juga kecil,
sehingga kelas YOLO terpisah untuknya berisiko overfit. Karena itu NB02b memakai
mekanisme `kind: distractor` (Â§10.0) untuk **mengukur dulu**, bukan langsung membuat kelas.

---

### 10.6 Audit Keselarasan Taksonomi & Kontrak Hitung (2026-10-26)

**Pemicu.** Pemilik riset mengoreksi tujuan: yang dideteksi adalah **kondisi
lampu** (silau / berwarna / silau **dan** berwarna), dan keluarannya harus berupa
**sistem hitung**, bukan sekadar deteksi.

| Keluaran wajib | Satuan |
|---|---|
| Jumlah kendaraan mobil | kendaraan |
| Jumlah kendaraan motor | kendaraan |
| Jumlah lampu silau â€” mobil | kendaraan/lampu (Â§13.10) |
| Jumlah lampu silau â€” motor | kendaraan/lampu |
| Jumlah lampu berwarna â€” mobil | kendaraan/lampu |
| Jumlah lampu berwarna â€” motor | kendaraan/lampu |
| Jumlah lampu silau + berwarna â€” mobil | kendaraan/lampu |
| Jumlah lampu silau + berwarna â€” motor | kendaraan/lampu |

**Bukti audit per notebook:**

| NB | Bukti di kode/artefak | Selaras? |
|---|---|---|
| NB01 | Akuisisi frame + ROI polygon; tidak menyentuh kelas | âœ… netral |
| NB02b | `builders/build_nb02b_annotation.py:74` â†’ `CLASS_NAMES = ["LampuMobil", "LampuMotor", "PantulanCahaya"]`; hotkey `1/2/3` + `d`; store 126 frame reviewed = **946 box** (182 mobil / 447 motor / 317 pantulan) | âŒ |
| NB02 | `builders/build_nb02_pseudo_label.py:75` â†’ `CLASS_PROMPTS = [("LampuMobil", "car headlight."), ("LampuMotor", "motorcycle headlight.")]` | âŒ |
| NB03 | Kalibrasi threshold terhadap gold mobil/motor; F1 makro train 0,334 @0,30 | âŒ |
| NB04 | `build_nb04_train_yolo.py:94` + `dataset/data.yaml` `nc: 3`; split 660/182/640 | âŒ |
| NB05 | `COUNTED_CLASS_NAMES = ["LampuMobil", "LampuMotor"]` + `count_vehicles()` + filter indeks 2 (R11) | âŒ |

**Kesimpulan.** Ini **salah taksonomi pada level target**, bukan bug kecil. Yang
ada sekarang menjawab "berapa mobil dan motor", bukan "berapa lampu silau /
berwarna / silau+berwarna per jenis kendaraan". Akibatnya:

- Target `berwarna` dan `silau+berwarna` **tidak ada** di data, pseudo-label,
  model, maupun metrik.
- `PantulanCahaya` (317 box) **bukan** salah satu dari tiga target riset;
  posisinya hanya *hard-negative*, bukan objek yang dilaporkan.
- Seluruh angka NB03â€“NB05 saat ini **tidak boleh** dikutip sebagai hasil riset.

**Kontrak hitung yang diusulkan** (mekanis, tanpa asosiasi tambahan). Kelas
menyandikan **jenis Ã— kondisi**, sehingga setiap angka di tabel atas hanya
penjumlahan:

| Kelas | Masuk "kendaraan mobil" | Masuk "kendaraan motor" | Masuk hitung silau | berwarna | silau+berwarna |
|---|---|---|---|---|---|
| `MobilNormal` | âœ… | â€” | â€” | â€” | â€” |
| `MobilSilau` | âœ… | â€” | âœ… | â€” | â€” |
| `MobilBerwarna` | âœ… | â€” | â€” | âœ… | â€” |
| `MobilSilauBerwarna` | âœ… | â€” | âœ… (rollup) | âœ… (rollup) | âœ… |
| `MotorNormal` | â€” | âœ… | â€” | â€” | â€” |
| `MotorSilau` | â€” | âœ… | âœ… | â€” | â€” |
| `MotorBerwarna` | â€” | âœ… | â€” | âœ… | â€” |
| `MotorSilauBerwarna` | â€” | âœ… | âœ… (rollup) | âœ… (rollup) | âœ… |
| `PantulanCahaya` | â€” | â€” | â€” | â€” | â€” |

**Urutan perbaikan (jangan dibalik).**

1. Tetapkan Â§13.9 (label space) dan Â§13.10 (satuan hitung).
2. Tetapkan definisi operasional: ambang "silau", daftar warna "berwarna",
   aturan tie-break bila ragu â€” keputusan pemilik riset.
3. Putuskan nasib 126 frame yang sudah dianotasi (opsi di Â§13.9).
4. Ubah `CLASS_PROMPTS` NB02 â†’ frasa kondisi lampu, lalu **re-run NB02** (GPU).
5. Regenerasi NB02b (kelas + hotkey), NB04 (`data.yaml` + materialisasi),
   NB03 (kalibrasi), NB05 (kontrak hitung di atas).
6. Baru jalankan training penuh dan laporkan angka.

Sampai langkah 1â€“2 selesai: **jangan** latih NB04 dan **jangan** kutip angka NB05.

### 10.6.1 Tindak Lanjut Audit 2026-10-28 (catatan historis; kontrak hitung diganti 2026-11-16)

Audit 19 temuan (`AUDIT_ACTION_2026-10-28.md`) menemukan satu kebocoran makna: NB05
dan NB06 menerbitkan kunci lama (`jumlah_kendaraan_*`) dengan satuan berbeda (catatan sejarah; nama kunci aktif sekarang memakai `jumlah_bbox_*`). Perbaikannya:

- **NB05** melaporkan bbox detector `Mobil` dan `Motor` yang lolos threshold; hasilnya bukan hitungan kondisi lampu.
- **NB06** menerbitkan `counting_totals` dengan `jumlah_bbox_mobil_tidak_standar` dan `jumlah_bbox_motor_tidak_standar`. Unitnya bbox per frame; `unique_vehicles_audit` terpisah untuk audit tracking.

Sumber tunggal yang baru (`pipeline_common.py`): `CLASS_NAMES`,
`SPLIT_NAME_BY_CAMERA`, `COUNT_CONTRACT`, `BOX_COUNT_CONTRACT`,
`stitch_track_fragments`, `validate_tracker_thresholds`, konstanta `FRAGMENT_*`.
Seluruh builder mengimpornya; tidak ada lagi salinan lokal. Logika tersebut dites
tanpa GPU lewat `test_pipeline_logic.py`.

---

### 10.7 NB06 â€” Hitung Kendaraan Unik (catatan historis; definisi diperbarui 2026-11-16)

> Seluruh deskripsi satuan/angka pada bagian ini menjelaskan rancangan lama. Unit aktif adalah satu track TidakStandar saat melintasi counting line; bbox per frame hanya metrik pendamping. Lihat README.


**Pemicu.** Â§13.11. NB05 menjumlahkan box per **frame sampel** (1 dari setiap 15
frame), sehingga satu kendaraan muncul di puluhan frame dan terhitung puluhan
kali. Pemilik riset menetapkan satuan yang benar: **satu kendaraan dihitung satu
kali, sejak muncul di dalam ROI sampai keluar dari ROI.**

**Posisi stage pada catatan historis tersebut.** Kontrak aktif sekarang menghitung track TidakStandar saat melintasi counting line; bbox per frame hanya metrik pendamping. Lihat README untuk urutan dan satuan aktif.

**Kenapa stage terpisah, bukan menumpang NB05.** Tracking butuh frame
**berurutan**. NB01 hanya menyimpan 1 dari setiap 15 frame (0,6 detik pada 25
fps); pada jarak itu asosiasi antar frame sudah rapuh dan ID akan pecah terus.
Karena itu NB06 membaca **video asli** di `output/videos/` (67.790 frame total),
bukan frame sampel. NB05 tetap dipertahankan sebagai **baseline per frame** dan
tidak boleh dilabeli "jumlah kendaraan".

**Alur data.**

| Tahap | Masukan | Keluaran |
|---|---|---|
| 1. Dekode | `output/videos/*.mp4` setiap frame | array BGR + `frame_index` |
| 2. Crop ROI | frame + `output/roi/roi_config.json` | crop polygon + `offset_x`, `offset_y`, `scale` |
| 3. Deteksi + track | crop + `output/nb04/.../best.pt` + `builders/bytetrack_nb06.yaml` | `track_id`, kelas, skor, box (koordinat crop) |
| 4. Pemetaan balik | box crop + offset + scale | `frame_x1..frame_y2` (koordinat frame asli) |
| 5. Filter | skor vs `output/nb05/thresholds.json` per kelas | deteksi yang sah |
| 6. Agregasi | deteksi per `(video, track_id)` | satu baris = satu kendaraan |
| 7. Hitung | kelas per track + kontrak Â§10.6 | 8 angka |

**Rantai koordinat.** Crop memakai bbox polygon yang area luarnya dihitamkan
(persis NB02), lalu `thumbnail(640)`. Pemetaan balik memakai
`frame = offset + crop / scale`. Skala saat ini 1,0 karena semua crop lebih kecil
dari 640 px, tetapi rumusnya tetap umum. Bukti empiris 2026-10-26: resolusi video
= resolusi PNG frame (704Ã—576 / 640Ã—480 / 640Ã—360), dan crop NB06 = crop NB02
(412Ã—377, 425Ã—303, 315Ã—184) dengan offset yang cocok.

**Aturan hitung.**

1. Deteksi hanya terjadi di dalam polygon; area luar ROI dihitamkan sehingga
   model tidak pernah melihatnya.
2. Ambang confidence **per kelas** dari NB05 diterapkan saat agregasi track.
   Ambang tidak dikalibrasi ulang di NB06.
3. Satu `track_id` = satu kendaraan.
4. Track dengan kemunculan `< MIN_TRACK_HITS` dibuang sebagai kedipan detektor.
5. Kelas per track = **suara mayoritas** antar frame yang lolos ambang; seri
   diputus oleh skor rata-rata lebih tinggi. Alasannya: kondisi lampu satu
   kendaraan dapat terbaca berbeda antar frame, sedangkan satu kendaraan hanya
   boleh menyumbang satu angka.
6. Angka dijumlahkan dengan kontrak Â§10.6. `PantulanCahaya` tetap hard-negative:
   tercatat sebagai track, tetapi nol di kedelapan angka.

**Konfigurasi (nilai saat ini).**

| Konstanta | Nilai | Alasan |
|---|---|---|
| `COUNTED_SPLITS` | `["test"]` | keputusan pemilik riset 2026-10-26: hanya kamera yang tidak pernah dilihat model (`bappenda2`, `pdam2`, 29.950 frame) yang dilaporkan sebagai angka resmi |
| `FRAME_STRIDE` | 1 | setiap frame; tracking tidak boleh melompat |
| `IMAGE_SIZE` | 640 | sama dengan training NB04 dan kalibrasi NB05. Terbukti penting: pada `imgsz=320` deteksi di crop malam hilang total |
| `CONF_FLOOR` / `NMS_IOU` | 0,05 / 0,70 | sama dengan NB05, penyaringan akhir tetap per kelas |
| `TRACKER_CONFIG` | `builders/bytetrack_nb06.yaml` | lihat Â§13.12 |
| `MIN_TRACK_HITS` | 3 (smoke 2) | 0,12 detik pada 25 fps; membuang kedipan 1â€“2 frame |
| `FRAGMENT_MAX_GAP_FRAMES` | 50 | 2 detik; batas jarak waktu track pecahan boleh disambung |
| `FRAGMENT_MIN_IOU` | 0,50 | fragmen harus menempati kotak yang sama (tumpang tindih â‰¥50%) |
| `FRAGMENT_DISTANCE_RATIO` | 0,35 | alternatif IoU: pusat box berpindah < 0,35 Ã— diagonal box |
| `DEVICE` | cuda, fallback cpu | keharusan pemilik riset |

**Satu rekaman per kamera.** NB01 menamai video dengan timestamp dan tidak pernah
menimpa, jadi satu kamera bisa punya beberapa rekaman. NB06 memilih **satu rekaman
terbaru per kamera** (stem terbesar) dan menampilkan tabel "video ditemukan vs
video dihitung"; rekaman lain dilewati dengan alasan tertulis. Nama kamera yang
tidak ada di `SPLIT_NAME_BY_CAMERA` membuat sel JOBS berhenti, bukan dilewati
diam-diam.

**Keluaran** (`output/nb06/`, smoke di `output/smoke/nb06/`):

| Berkas | Isi |
|---|---|
| `tracking_detections.csv` | satu baris per deteksi per frame (termasuk koordinat frame asli) |
| `track_summary.csv` | satu baris per track: kelas hasil mayoritas, `hits`, `first_frame`, `last_frame`, `mean_score` |
| `counting_by_video.csv` | Satu baris per video test; deteksi per kelas, `bbox_counted`, serta dua kontrak bbox TidakStandar. Tidak ada kolom `vehicles_counted`. |
| `counting_summary.json` | agregat + `duplication_factor_vs_per_frame` + ambang yang dipakai |
| `fragment_merges.csv` | daftar kendaraan yang terbentuk dari >1 track (bukti penyambungan, untuk diperiksa manual) |
| `roi_mapping_audit.csv` | bukti offset/scale/crop per video |
| `counting_bar.png` | grafik 8 angka |

**Status verifikasi (2026-10-26).**

| Uji | Cakupan | Hasil |
|---|---|---|
| Smoke `builders/smoke_pipeline.py` | NB02 â†’ NB06 di `output/smoke/` | âœ… exit 0, semua artefak terbentuk, kedua kontrak hitung konsisten |
| Uji logika sintetis | mayoritas, tie-break, `MIN_TRACK_HITS`, rollup pantulan, 4 kasus yang **harus ditolak** (motor vs mobil, berjauhan, dua track serentak) | âœ… |
| Uji integrasi sel TRACK+COUNT | 60 frame video asli `bappenda2`, ambang & `MIN_TRACK_HITS` sengaja diturunkan untuk probe | âœ… 151 deteksi â†’ 151 `track_id` â†’ hitung per track jalan, 0 fragmen disambung (semua track serentak, benar tidak disambung) |
| Uji API `model.track` CPU | yolo11n COCO + yaml kustom + source PIL | âœ… `boxes.id` terisi; `imgsz=320` deteksi hilang total |

Angka smoke seluruhnya nol karena bobot smoke hanya 1 epoch pada 7 gambar palsu: model
menghasilkan 877 box pada `conf=0.001` tetapi 0 box pada `conf=0.05`, jadi tidak ada
yang lolos `CONF_FLOOR`. Itu kelemahan bobot, bukan kegagalan pipeline â€” dan pipeline
melaporkannya lewat `untracked_detections`, bukan menghitung nol tanpa suara.

### 10.7.1 Jaminan Identitas â€” Apa yang Dijamin dan Apa yang Tidak

Pertanyaan pemilik riset 2026-10-26: *"kalau mobil A masuk ROI, pasti terhitung
1? Semua kendaraan punya ID berbeda sehingga tidak terhitung 2Ã—?"*

**Yang dijamin.**

- `track_id` dibuat oleh ByteTrack dan **tidak pernah dipakai dua kali dalam satu
  video**. Dua kendaraan yang berbeda tidak akan berbagi satu ID.
- Selama kendaraan terlihat **terus-menerus**, ia memegang satu ID, dan NB06
  menghitung `track_id`, bukan jumlah box. Jadi mobil A yang muncul di 200 frame
  tetap menyumbang **1**, bukan 200.
- Kendaraan yang sudah ada di dalam ROI pada frame pertama video tetap dihitung
  sekali (track baru terbentuk, lalu dihitung).

**Yang tidak dijamin (dan tidak bisa dijamin tanpa ReID).**

| Kejadian | Akibat | Seberapa sering |
|---|---|---|
| Kendaraan hilang dari pandangan lebih lama dari `track_buffer` (30 frame = 1,2 detik) lalu muncul lagi | ID baru â†’ **terhitung 2Ã—** | tergantung occlusion (tiang, kendaraan besar di depan) |
| Dua kendaraan sangat berdekatan/berpapasan dan box-nya bertukar | ID switch â†’ satu kendaraan bisa terhitung 2Ã— atau dua kendaraan menyatu jadi 1 | malam + lampu menyilaukan: risiko nyata |
| Kendaraan keluar ROI lalu masuk lagi | ID baru â†’ terhitung 2Ã— | jarang, tetapi mungkin di simpang |
| Kendaraan hanya terlihat 1â€“2 frame | dibuang `MIN_TRACK_HITS` â†’ **tidak** terhitung | sering untuk kendaraan cepat di tepi ROI |
| Video berbeda | tracker di-reset â†’ kendaraan yang sama terhitung lagi | pasti, karena tidak ada identitas lintas sesi |

**Konsekuensi penulisan laporan.** Angka NB06 adalah **jumlah kejadian kendaraan
terdeteksi di dalam ROI dan dilacak**, bukan jumlah kendaraan fisik di jalan.
Kalimat itu wajib ada, dan `duplication_factor_vs_per_frame` disertakan sebagai
pembanding agar pembaca melihat asumsi apa yang sedang dipakai.

**Mitigasi yang sudah terpasang (2026-10-26).**

1. **Penyambungan fragmen.** Sebelum dihitung, track yang terputus disambung
   kembali bila: jenisnya sama (mobil/motor), fragmen baru mulai **minimal 1 frame
   setelah** fragmen sebelumnya berakhir dan tidak lebih dari
   `FRAGMENT_MAX_GAP_FRAMES` (50 frame = 2 detik), serta box-nya bertumpuk
   (IoU â‰¥ 0,50) atau pusatnya berpindah < 0,35 Ã— diagonal box. Syarat "minimal
   1 frame setelah" penting: tanpa itu, dua kendaraan yang terlihat **bersamaan**
   bisa ikut menyatu (ditemukan pada uji integrasi 2026-10-26). Setiap
   penggabungan dicatat di `fragment_merges.csv` untuk diperiksa manual.
   Penyambungan **tidak** memperbaiki ID switch yang overlap waktunya.
2. **Penyaringan setelah penyambungan.** `MIN_TRACK_HITS` diterapkan pada
   kendaraan hasil sambungan, bukan pada track mentah. Akibatnya kendaraan yang
   terpecah menjadi dua fragmen pendek tetap terhitung **satu** kali, tidak hilang
   dan tidak menjadi dua.

**Yang tetap tidak dijamin** meskipun ada mitigasi: celah antar kemunculan lebih
dari 2 detik, ID switch antara dua kendaraan yang box-nya tidak berdekatan, dan
kendaraan yang keluar lalu masuk lagi. Penyambungan fragmen adalah **heuristik
yang diukur**, bukan jaminan; karena itu setiap penggabungan diekspor dan
`duplication_factor_vs_per_frame` tetap dilaporkan agar asumsinya terlihat.

## 11. Konvensi Penamaan

Kode kamera memakai **nama lengkap** dari sumber, bukan inisial:

```
bogor-kab-bappenda1 / bogor-kab-bappenda2 / bogor-kab-bappenda3
bogor-kab-pdam2 / bogor-kab-pemda2
```

Format nama video: `atcs_<camera_id>_<YYYYMMDD>_<HHMMSS>.mp4`

Format notebook: `NB<studi>_<nama_tahap>.ipynb` â€” contoh `NB02_pseudo_label.ipynb`

---

## 12. Status Eksperimen Saat Ini

| Eksperimen | Model | Dataset | Status |
|---|---|---|---|
| Pseudo-labelling NB02 | GroundingDINO-tiny | `output/frames/` + ROI polygon final | âœ… Selesai 2026-10-26 (1.548 crop unik, 4.309 deteksi, 0 gagal) |
| Training YOLO | YOLOv11n | NB04 (gold 9 kelas) | ðŸ”§ Dibangun, menunggu anotasi NB02b |
| Hitung kendaraan unik NB06 | YOLOv11n + ByteTrack | video asli kamera test (29.950 frame) | ðŸŸ¡ Smoke NB02â†’NB06 lolos 2026-10-26, menunggu run penuh |
| Fine-Grained Matching | ResNet50 | NB04 | âŒ Dipertimbangkan ulang (Â§13.9 sudah dijawab: tidak perlu) |

---

## 13. Keputusan Terbuka

1. ~~Mode pseudo-labelling (A/B)~~ â€” **Sudah diputuskan: Mode B (DINO-only).**
2. ~~ROI~~ â€” **Sudah final & disetujui user 2026-10-18** (5 polygon, geometri
   tervalidasi, cek visual lolos).
3. ~~Taksonomi 2 vs 3+ kelas~~ â€” **Sudah diputuskan: 2 kelas.**
4. ~~Backup~~ â€” **Sudah dilakukan.**
5. ~~Definisi target deteksi~~ â€” **Sudah diputuskan: Mode A (lampu-sepasang).**
6. ~~Model DINO (tiny vs base)~~ â€” **Sudah diputuskan: `grounding-dino-tiny`
   untuk run pertama.** Jalur upgrade ke `grounding-dino-base` bila hasil
   anotasi di NB03 kurang bagus (lihat Â§5.5).
7. **Nilai konstanta grouping** â€” masih nilai awal, dikalibrasi di NB03.
8. **Apakah CLAHE perlu dipakai** â€” saat ini NB02 berjalan tanpa CLAHE
   (frame asli + ROI). Bila dianggap perlu, itu perubahan desain baru.
9. ~~Label space untuk target silau/berwarna~~ â€” **Sudah diputuskan user:
   9 kelas** (`{Mobil,Motor} Ã— {Normal,Silau,Berwarna,SilauBerwarna}` +
   `PantulanCahaya` sebagai hard-negative). 946 box lama tidak dipakai apa
   adanya; NB02b menganotasi ulang pada label space ini.
10. ~~Satuan hitung "jumlah lampu"~~ â€” **Sudah diputuskan: (a) per kendaraan**
    (1 box = 1 kendaraan hasil union sepasang lampu). Kontrak penjumlahan
    angka ada di Â§10.6.
11. ~~Satuan "jumlah kendaraan" pada frame tersampling~~ â€” **Sudah diputuskan:
    (b) tracker pada frame berurutan.** Konsekuensinya NB05 tetap melaporkan
    *jumlah deteksi per frame* sebagai baseline dan **tidak boleh** dilabeli
    "jumlah kendaraan". Angka resmi kendaraan unik ada di **NB06**
    (`output/nb06/`), yang membaca video asli di `output/videos/` dan
    menjalankan ByteTrack di dalam crop ROI yang sama dengan NB02.
    Aturan hitung: satu `track_id` = satu kendaraan, dihitung sekali sejak
    muncul di dalam ROI; kelas per track = suara mayoritas; track di-reset per
    video. Batas: kendaraan yang keluar lalu masuk lagi terhitung dua kali, dan
    tidak ada identitas lintas sesi (lihat NB06).
12. **Ambang internal tracker** â€” ByteTrack bawaan memakai
    `new_track_thresh: 0.25`, sehingga box ber-confidence di bawah 0,25 tidak
    pernah bisa membentuk track baru dan kelas langka akan undercount
    sistematis. NB06 memakai `builders/bytetrack_nb06.yaml`
    (`new_track_thresh 0.05`, `track_high_thresh 0.05`, `track_low_thresh 0.02`,
    `track_buffer 30`), yaitu lantai yang sama dengan `CONF_FLOOR` NB06.
    Aturan yang mengikat: `new_track_thresh â‰¤ min(ambang per kelas)`; sel JOBS
    NB06 menolak jalan bila aturan ini dilanggar. Penyaringan akhir tetap
    dilakukan ambang per kelas hasil NB05 saat agregasi track, jadi tracker
    hanya mengurus asosiasi. Bila seluruh deteksi gagal mendapat `track_id`,
    NB06 berhenti dengan pesan eksplisit alih-alih melaporkan nol kendaraan.

> Referensi teknis hasil ekstraksi arsip kursus CV (rumus HSV, skor blur,
> luas bloom, NMS, GradCAM, dan pola yang ditolak): **`KNOWLEDGE_PIPELINE_CV.md`**.

> **Status bagian:** keputusan Silau/Berwarna berikut adalah eksplorasi historis. Taxonomy aktif 2026-11-12 adalah Standar/TidakStandar per jenis kendaraan; referensi visual lampu berwarna/silau tidak dipakai sebagai kelas output.
>
### 13.9a Definisi Operasional Label (Diputuskan 2026-10-28)

Empat keputusan pemilik riset, menutup Â§13 butir 9 dan blok "belum ditetapkan" di Â§1:

| Aspek | Keputusan |
|---|---|
| Dasar penilaian `Silau` | **acuan contoh visual**, bukan ambang fotometrik |
| Cakupan yang dinilai | hanya **lampu depan**; lampu rem/sein/belakang tidak dinilai warnanya |
| Warna yang dihitung `Berwarna` | hanya **biru/cyan** (hue OpenCV 95â€“135) |
| Kasus ragu | `BelumDitentukan`, memblokir export |

**Kenapa silau bukan ambang.** Diukur 2026-10-28 pada gold set saat itu (78 frame
reviewed, 1.125 box aktif) memakai `archive/one-off-scripts/probe_label_definition.py`; CSV mentah di
`output/annotation/label_probe.csv`.

| Fitur | Ambang terbaik | Silau tertangkap | Normal ikut tertandai | Pantulan ikut |
|---|---|---|---|---|
| `v240` (konstanta NB03) | 0,10 | 0,89 | 0,64 | 0,60 |
| `v200` | 0,21 | 0,89 | 0,55 | 0,61 |
| `v180` | 0,29 | 0,89 | 0,54 | 0,61 |
| `white_bloom_ratio` (Vâ‰¥200, S<40) | 0,22 | 0,82 | 0,42 | 0,52 |
| ring 1,6Ã— di luar box (`ring_v180`) | 0,01 | 0,91 | 0,85 | 0,87 |

Median `v240` bahkan terbalik: `MobilSilau` 0,139 vs `MobilNormal` 0,147. Tiga sebab:
box ikut melebar mengikuti bloom sehingga rasio di dalam box ternormalisasi sendiri;
kompresi H.264 memotong highlight sebelum sampai ke crop; dan yang dinilai manusia
adalah **bentuk** luapan, bukan distribusi nilai piksel. Karena itu `bright_ratio` tetap
dipakai sebagai **diagnostik** NB03, **bukan** sebagai definisi label.

**Aturan silau.** `Silau` bila ada halo/lingkaran putih yang keluar melewati batas fisik
lampu (menutupi bodi, plat, atau aspal tepat di sekitar lampu) **atau** ada streak/flare
memanjang dari inti lampu. Lampu terang dengan batas jelas **bukan** silau. Cahaya yang
jatuh di aspal atau bodi = `PantulanCahaya`.

**Aturan berwarna.** Piksel berwarna = `S â‰¥ 80`, `V â‰¥ 80`, hue 95â€“135; minimal 25% luas
box harus piksel tersebut. Merah, oranye, kuning, hijau, dan ungu tidak dihitung pada
revisi ini.

**Konsekuensi data yang harus dibaca sebelum training.**

| Kelas | Box | Catatan |
|---|---|---|
| `PantulanCahaya` | 561 | 49,9% gold set â€” kelas terbesar, hard-negative |
| `MotorSilau` | 229 | |
| `MotorNormal` | 181 | |
| `MobilNormal` | 99 | |
| `MobilSilau` | 42 | |
| `MotorBerwarna` | 4 | semua hue 110â€“120 (biru) |
| `MotorSilauBerwarna` | 2 | hue 110â€“120 |
| `MobilBerwarna` | 0 | |
| `MobilSilauBerwarna` | 0 | |

Dari 93 box yang pikselnya berwarna kuat (`colored_ratio â‰¥ 0,25`), **79 adalah
`PantulanCahaya`** dengan hue 0 (merah) â€” pantulan lampu rem di aspal. Inilah alasan
aturan "hanya lampu depan" dan "hanya biru": tanpa keduanya, `Berwarna` akan didominasi
pantulan dan lampu rem.

> **Kendaraan arah menjauh â€” pertanyaan tertutup (2026-10-28).** ROI tiap kamera sudah
> disesuaikan pemilik riset ke arah jalan sehingga frame hanya memuat tampak depan
> kendaraan. Tidak ada kendaraan yang hanya tampak belakang di data, jadi aturan khusus
> tidak diperlukan dan "jumlah kendaraan" tidak kehilangan arah menjauh.

**Hasil mining kandidat biru (2026-10-28).** `archive/one-off-scripts/mine_blue_candidates.py` mengukur
seluruh 4.309 box pseudo-label di 1.474 crop, ambang definisi (`Sâ‰¥80`, `Vâ‰¥80`, hue 95â€“135):

| Tahap | Box | Frame |
|---|---|---|
| kandidat longgar (biru â‰¥ 25% luas box) | 81 | 69 |
| kandidat ketat (inti terang `Vâ‰¥200` didominasi biru â‰¥ 50%) | 20 | 20 |

Sebaran kandidat ketat per split: **test 18 Â· train 2 Â· val 0**. Rinciannya `pdam2` 17,
`bappenda1` 2, `bappenda2` 1. Artinya dari 20 kandidat, **hanya 2 yang boleh dipakai
latihan**, dan keduanya dari satu kamera (`bappenda1`) dengan dua frame berdekatan
(005580, 006480) â€” kemungkinan besar kendaraan yang sama.

**Kenapa hue saja tidak cukup untuk menyatakan "lampu biru".** Ditambahkan kolom
`frame_blue_share` = fraksi piksel terang seluruh frame yang berhue biru, untuk mendeteksi
auto-white-balance yang bergeser:

| Frame | `frame_blue_share` | Tafsir |
|---|---|---|
| `bappenda2` 013560 | 0,998 | seluruh frame biru â†’ artefak AWB, bukan lampu |
| `bappenda1` 005580 | 0,992 | seluruh frame biru â†’ artefak AWB |
| `pdam2` 004665 | 0,86 | bingkai ikut biru â†’ artefak AWB |
| `pdam2` 000690 | 0,83 | bingkai ikut biru â†’ artefak AWB |
| `bappenda1` 006480 | 0,18 | frame hangat, inti box biru â†’ kandidat meyakinkan |
| `pdam2` 010980â€“011340 (10 frame) | 0,23â€“0,39 | frame hangat, inti box biru â†’ kandidat paling meyakinkan |

Dari 20 kandidat ketat, **4 box ditandai `awb_suspect`** (frame biru menyeluruh). Sisanya
bertahan sebagai kandidat, tetapi pada ukuran box 25â€“60 px batas antara "lampu putih dengan
halo kebiruan" dan "lampu biru" masih tipis â€” dan hampir semuanya ada di kamera `test`
(`pdam2`), yang menurut protokol split **tidak boleh** dipakai latihan.

**Kesimpulan yang harus dipakai untuk perencanaan.** Mining berbasis hue berhasil
menemukan kandidat, tetapi **tidak** menghasilkan data latih yang cukup: 2 box train,
0 box val, dan keduanya satu kamera. Dengan 1.548 crop yang ada, kelas `Berwarna` dan
`SilauBerwarna` tidak dapat dilatih maupun dievaluasi secara sah.

**Sisa pekerjaan definisi (urutan).**

1. Bangun set contoh `output/annotation/reference_patches/`: Â±30 patch silau + Â±30 patch
   normal yang disetujui pemilik riset. Sampai ini ada, definisi silau **belum
   operasional penuh** dan label silau yang lama belum bisa diaudit.
2. Uji konsistensi: tampilkan 30 crop tanpa label lama, ukur kesepakatan dengan label
   sebelumnya; target â‰¥ 80%. Bila di bawah itu, aturan silau perlu dipertajam.
3. ~~Tetapkan aturan kendaraan yang hanya tampak belakang.~~ **Selesai 2026-10-28**: ROI
   hanya memuat tampak depan.
4. ~~Tambah anotasi terarah untuk kelas biru dengan mining kandidat.~~ **Selesai
   2026-10-28**, hasil: 20 kandidat ketat, 2 di train, 0 di val â€” **tidak cukup**.
5. **Keputusan strategi kelas berwarna** (menunggu pemilik riset): turunkan target latih ke
   4 kelas `{Mobil,Motor} Ã— {Normal,Silau}` dan laporkan `Berwarna`/`SilauBerwarna` sebagai
   batasan riset; atau tambah sesi rekaman baru yang memuat kendaraan lampu biru.
6. **Dibatalkan 2026-11-06:** kelas Silau/Berwarna tidak termasuk taxonomy aktif. `builders/build_reference_patches.py` dihapus; klaim lama bahwa 120 patch telah dihasilkan tidak tervalidasi dan jangan dikutip.
7. Uji konsistensi relabel patch silau terhadap label lama; Bila di bawah 80%, pertajam aturan.

> **Status bagian:** snapshot audit 2026-10-29; jumlah anotasi/split di bawah bukan distribusi dataset aktif saat ini. Periksa manifest/store terkini sebelum memakai angka.
>
### 13.9b Audit eksternal (Codex) 2026-10-29 dan tindakannya

Bahan: `kritik-saran/PERBAIKAN.md`, `kritik-saran/STRATEGI_PERBAIKAN_NB02_NB02B.md`, dan
paket percobaan `kritik-saran/CONTOH-CODEX/` (tidak pernah menyentuh file aktif proyek).

**Angka yang diukur ulang pada gold set saat itu** (`builders/audit_gold_quality.py`,
keluaran `output/annotation/gold_quality_summary.json` + `gold_quality_audit.csv`):

| Ukuran | Nilai |
|---|---|
| Frame reviewed | 170 (semua split `train`, kamera `bappenda1`) |
| Frame reviewed di split `val` | **0 dari 83** |
| Box aktif | 1.897 |
| `PantulanCahaya` | 874 box (46,1%) |
| Box digambar manual dari nol | 1.017 |
| Box pseudo yang dihapus pemilik riset | 396 (â‰ˆ31% proposal DINO) |
| **Proxy recall proposal DINO (lampu kendaraan)** | **0,606** |
| Pasangan box duplikat (IoU â‰¥ 0,85) | 0 |
| Box dengan sisi < 8 px | 1 |
| Box menyentuh tepi crop | 28 |
| Refleksi yang menumpuk di atas box kendaraan | 0 |

Kesimpulan: kritik Codex soal **recall proposal DINO benar dan terukur** (0,606, jauh di bawah
ambang go/no-go 0,80 yang mereka usulkan), tetapi kritik soal **mutu anotasi tidak berlaku**
pada 170 frame yang sudah dikerjakan â€” tidak ada duplikat, tidak ada refleksi yang menumpuk,
hampir tidak ada box tak-terlathi. Artinya pemilik riset sudah bekerja pada **jalur
manual-first** (1.017 dari 1.897 box digambar sendiri), sehingga rekomendasi utama Codex
("pindah ke manual-first") sudah menjadi praktik, bukan rencana.

**Klaim Codex yang tidak sesuai kondisi repo** (dipakai sebagai peringatan, bukan acuan):

1. "Keputusan label final **9 kelas** dan kode NB02bâ€“NB06 selaras" â€” repo memakai **9 kelas
   anotasi + 5 kelas latih** (`TRAINING_CLASS_NAMES`); `pipeline_common.py` yang mereka salin
   sendiri pun berisi kontrak 5 kelas, dan NB04 mereka memakai `nc: 5`.
2. "`MobilBerwarna`/`MobilSilauBerwarna` belum punya box" â€” saat audit: 5 dan 7 box (semua
   `train`, 0 `val`). Klaim itu berasal dari state sebelum gold set bertambah.
3. "NB02b mengambil 1 dari 15 frame" â€” stride 15 frame ada di **NB01**
   (`SAMPLE_INTERVAL_FRAMES`); NB02b memakai kuota per kamera + `random.sample` atas 871
   kandidat.

**Yang diterapkan 2026-10-29.**

| Perubahan | Alasan |
|---|---|
| Label `?` pada preview NB02 â†’ **`UNSURE`** (nilai di `nb02_summary.json` tetap `TidakPasti`) | `?` terbaca seperti nama kelas; `UNSURE` menyatakan status, bukan taksonomi |
| `DETECTION_THRESHOLD` 0,30 â†’ **0,25**; `VEHICLE_SCORE_THRESHOLD` 0,40 â†’ **0,30** | recall proposal 0,606 adalah bottleneck terukur; melonggarkan proposal lebih murah daripada kehilangan kendaraan |
| `VEHICLE_VOTE_MIN_COVERAGE` **tetap 0,30** (Codex mengusulkan 0,15) | coverage = irisan box kendaraan Ã· box lampu; 0,15 berarti box kendaraan hanya menutup 15% box lampu â€” itu kendaraan lain, bukan bukti jenis. Menurunkannya menambah label jenis yang salah, bukan menambah recall |
| Gate NB04: **training ditolak** bila ada kelas `TRAINING_CLASS_NAMES` dengan < 20 box `train` | kelas kosong menghasilkan mAP yang tidak bisa ditafsirkan |
| Gate NB04: **peringatan** bila kelas < 5 box `val` (tidak memblokir) | AP dari < 5 box adalah estimasi lemah, cukup dilaporkan, bukan alasan menghentikan run |
| `builders/audit_gold_quality.py` | versi murah dari "naikkan kualitas anotasi": duplikat, box kecil, box tepi, refleksi tumpang tindih, proxy recall â€” tanpa biaya anotator kedua |
| NB05: kontrak `jumlah_box_lampu_berwarna_*` dan `jumlah_box_lampu_silau_dan_berwarna_*` kini **`n/a`**, bukan `0` | angka 0 terbaca sebagai "diukur, hasilnya nol"; padahal model memang tidak punya kelas itu (`UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS`) |

**Yang ditolak atau ditunda, dengan alasan.**

| Saran | Keputusan |
|---|---|
| Sampling bin temporal di NB02b (`CONTOH-CODEX`) | **Ditolak.** `sample_manifest.json` dipakai ulang oleh guard; mengganti algoritme mengharuskan manifest dihapus, dan export hanya membaca frame yang ada di manifest â†’ 170 frame reviewed **keluar dari export** |
| `run_manifest.json`, lock file ber-hash, CLI headless, dashboard observability | **Ditunda.** Baik untuk rekayasa, tidak mengubah angka riset; masuk daftar future work laporan |
| `audit_test_gold`, bootstrap CI per kamera/sesi, 2 anotator + kappa | **Ditunda.** Butuh sumber daya yang tidak ada (anotator tunggal); dilaporkan sebagai batasan |
| Gold counting clip untuk validasi ByteTrack | **Ditunda** sampai model lulus training; ini pekerjaan terbesar yang tersisa untuk 8 angka |
| Bobot hash `requirements.txt` + pemeriksaan CUDA saat startup | **Ditunda** bersama `run_manifest` |

**Gate publikasi yang disetujui** (dari `PERBAIKAN.md`, dipakai apa adanya): angka final tidak
diterbitkan bila (a) kelas `Berwarna`/`SilauBerwarna` dilaporkan sebagai hasil model, (b) split
`test` dipakai memilih threshold, (c) tidak ada metrik per kamera dan per kelas, (d) artefak
berasal dari smoke mode, atau (e) label masih memuat `BelumDitentukan`.

## Catatan historis 14 November 2026 â€” NB02e CNN pseudo-label (dipensiunkan)

NB02e adalah tahap bantu anotasi kondisi lampu, terpisah dari fine-tuning Grounding DINO di NB02d.
NB02e melatih CNN teacher menggunakan anotasi manusia NB02b pada split train dan val. Validation hanya
memilih checkpoint; notebook tidak memuat GT/test. Model memakai kontrak CNN NB04b yang sama:
`Standar`, `TidakStandar`, dan `PantulanCahaya`.

Teacher kemudian menginferensikan bbox kendaraan pending (`BelumDitentukan`) pada split train. Prediksi
tidak diterima otomatis. Reviewer harus memastikan crop sesuai cakupan anotasi kepala/pancaran lampu,
lalu menerima atau mengoreksi kondisi. Proposal FM yang masih mencakup bodi perlu diperbaiki di NB02b
sebelum kondisi lampunya dinilai. Skor model adalah skor prediksi, bukan confidence terkalibrasi dan
bukan pengukuran fotometri/regulasi.

Keputusan review disimpan terpisah di `output/nb02e/review_decisions.json`; annotation store NB02b tidak
ditulis oleh NB02e. NB04b menambahkan keputusan berlabel yang benar-benar dikonfirmasi manusia sebagai
contoh train tambahan, setelah memverifikasi split train, kelas, dan bbox masih cocok dengan annotation
store. Prediksi yang belum direview, label PantulanCahaya pada kandidat kendaraan, bbox yang berubah,
serta keputusan dari val/test tidak masuk ke training classifier.

Urutan untuk memakai fitur ini: NB02b (label manusia) â†’ NB02e (latih teacher dan review saran train) â†’
NB04b (latih classifier produksi dengan contoh train tambahan). YOLO NB04 tidak berubah karena
pseudo-label NB02e hanya menambah kondisi CNN, bukan kelas deteksi atau bbox detector.

---

## 14. Referensi

- Dataset: CCTV Kab Bogor â€” Raya Lintang Kita (Rawamurang) 2026-08-01
- Model: GroundingDINO-tiny (swin-t)
- Dokumentasi Grounded-SAM resmi: https://github.com/IDEA-Research/Grounded-SAM
  (dikutip 2026-10-15)
- Riset terkait: Grounded-SAM (lihat RISK_REGISTER.md Â§5)


## Catatan preprocessing CNN NB04b

NB04b mengambil seluruh area dalam bbox kendaraan. Crop di-resize proporsional dengan sisi terpanjang 224 px, kemudian dipadding memakai warna mean ImageNet sampai 224Ã—224. Tidak ada center-crop atau peregangan rasio. Tensor RGB diubah ke rentang 0â€“1 dan dinormalisasi memakai mean/std ImageNet. Augmentasi train (horizontal flip dan ColorJitter) diterapkan setelah resize/padding; val/test hanya memakai transformasi deterministik yang sama tanpa augmentasi acak. Fitur numerik dihitung dari crop/mask bersih dan diskalakan memakai statistik train saja. Bila horizontal flip dipakai, koordinat centroid-x mask dan assignment e-means diperbarui konsisten; ColorJitter hanya mengubah cabang citra CNN.


## Catatan historis 15 November 2026 — eksperimen Grounding DINO (status diperbarui 16 November)

Rencana awal sempat memiliki dua varian, tetapi keputusan aktif dibatasi ke base saja. Notebook continued dan checkpoint legacy tidak dipakai sebagai sumber pipeline aktif. Notebook base dibangun dari `builders/build_nb02d_finetune.py`:

- `NB02d_finetune_grounding_dino_base.ipynb`: memulai dari `IDEA-Research/grounding-dino-tiny`.
- Varian continued dari checkpoint lama tiga kelas dibatalkan dan tidak boleh dijalankan.

Eksperimen base mengoptimalkan lima label store anotasi: `MobilStandar`, `MobilTidakStandar`, `MotorStandar`, `MotorTidakStandar`, dan `PantulanCahaya`. Train dipakai untuk update bobot; validation untuk memilih checkpoint; test tidak disentuh. Output aktif berada di `output/nb02d_fiveclass/base/`.

Output FM menghasilkan pseudo-label lima kelas. Jika proposal akan dimuat ke NB02b, pilih `NB02D_PROPOSAL_VARIANT=base` sebelum kernel dimulai. Box hasil model bukan gold dan tidak boleh dipakai sebagai validation/test truth. Kontrak classifier juga memakai proposal train sebagai tambahan train setelah dibuatkan mask oleh NB02e.

**Status verifikasi:** builder membangun notebook base dan memeriksa sintaks sel. Train/validation loss dan preview proposal bukan bukti akurasi; kualitas kelas dan metrik harus diukur pada label gold terpisah. Checkpoint legacy tiga kelas hanya arsip historis, bukan sumber model aktif.

NB02e versi teacher/pseudo-label kondisi di atas dipensiunkan. **NB02e aktif per 16 November 2026 adalah notebook berbeda:** ia menghasilkan mask SAM2 untuk fitur RGB/mask CNN dan tidak melatih teacher atau membuat label kelas. Jalankan setelah NB02b, NB02c, dan NB02d selesai; NB04b menolak manifest mask yang belum berstatus `complete`.

### Kontrak fitur CNN aktif per 16 November 2026

Cabang numerik NB04b menggabungkan 35 statistik warna/HSV seluruh bbox, 35 statistik warna pada piksel foreground SAM2, dan 7 fitur geometri mask. Jumlah cluster dipilih dengan membandingkan model objective linear terhadap model segmented-linear satu elbow interior memakai BIC, dengan kandidat `k=2..12` dan tiga restart per kandidat. Jika tidak ada elbow interior yang didukung, training berhenti setelah menyimpan objective seluruh kandidat ke `output/attribute_classifier/emeans_k_selection.csv`; tidak ada fallback k tetap. Kurva terpilih juga disimpan ke `.png`. Mask kosong menghasilkan 35 fitur foreground nol, dicatat pada `mask_quality.json`, dan menghentikan training jika proporsinya melebihi 1%. Scaler serta centroid final dipelajari dari split train saja, lalu validation/test dipetakan ke centroid itu. CNN menerima crop RGB 224×224, 77 fitur numerik, dan one-hot dengan lebar sesuai `k` terpilih. Saat NB06, SAM2 membuat mask dan fitur pada koordinat crop ROI yang sama seperti training; mask dipetakan ke frame asli hanya untuk overlay. Centroid tersimpan di checkpoint dipakai untuk assignment. Perubahan store, proposal, atau model SAM2 mengharuskan NB02e → NB04b → NB06 dijalankan ulang. E-means tidak memakai label target dan tidak menjamin peningkatan akurasi.
