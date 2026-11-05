# Master Dokumentasi — Riset HampirMatang

**Proyek:** Deteksi Lampu Kendaraan Malam Hari di Jalan Raya Kabupaten Bogor
**Dataset:** `CCTV Kab Bogor_ Raya Lintang Kita (Rawamurang) (2026-08-01)`
**Repo lokal:** `TugasJere/KerjaPraktik/Riset-HampirMatang/`
**Revisi terakhir:** 2026-10-18 (ROI final + approval user, model DINO
ditetapkan, koreksi dokumentasi P1)

> **Snapshot kanonik 2026-11-05.** Anotasi aktif berisi 194 frame dan 2.082 box COCO dalam 9
> kelas: `MobilNormal 220`, `MobilSilau 193`, `MobilBerwarna 73`, `MobilSilauBerwarna 39`,
> `MotorNormal 382`, `MotorSilau 501`, `MotorBerwarna 102`, `MotorSilauBerwarna 35`,
> `PantulanCahaya 537`. COCO/store mempertahankan 9 kelas. NB04 sengaja memakai 5 kelas
> detector; NB04b memakai dua atribut multi-label (`silau`, `berwarna`), lalu NB06 membentuk
> event final. Angka lama seperti 170 frame, 1.897 box, 0 box warna, atau 2 kandidat train
> adalah snapshot historis sebelum anotasi aktif terakhir.

> **Implementasi dashboard/NB06 2026-11-05.** NB06 memproses video terbaru setiap kamera dan
> membedakan scope `official` untuk split `test` dari `presentation_only` untuk `train`/`val`.
> Threshold kelas turunan berwarna dipetakan ke threshold kelas dasar YOLO agar tidak terbuang saat
> counting. Overlay memakai warna event final dan label kecil `ID <track_id> · <confidence>%` secara
> default. Dashboard menampilkan video dan counting side-by-side serta galeri crop bbox pada
> `event_crops/`; video `event_clips/` disimpan sebagai konteks audit.

---

## 0. Ringkasan Revisi

### 0.1 Revisi 2026-10-18

1. **ROI final & disetujui user** — 5 polygon selesai digambar, diverifikasi
   tidak bersilangan dan seluruh verteks di dalam batas frame. Overlay visual
   di `output/roi/preview/` dicek user → benar.
2. **Model DINO ditetapkan: `grounding-dino-tiny`.** Ditemukan mismatch —
   builder sebelumnya menunjuk `grounding-dino-base` (swin-b), padahal
   dokumentasi riset konsisten menyebut tiny (swin-t). Sudah diperbaiki dan
   notebook di-regenerate. Jalur upgrade ke base tercatat di §5.5.
3. **Koreksi dokumentasi (P1)** — banyak angka/fakta yang salah:
   - Nama kamera: `B01`–`B05` → **`bogor-kab-bappenda1/2/3`, `pdam2`, `pemda2`**.
   - Resolusi: `768×576`/`1280×720` → **berbeda per kamera** (`704×576`,
     `640×480`, `640×360`); fps seragam 25.
   - Struktur berkas: banyak file yang didokumentasikan **tidak pernah ada**
     (seluruh `src/*.py`, folder `notebook/`, builder NB03/04/05/FM).
   - **NB02b (CLAHE) tidak ada** — NB02 mandiri: ROI + dedup di dalam notebook,
     tanpa CLAHE. **Catatan 2026-10-20:** nama `NB02b` kini dipakai untuk
     **tahap anotasi manual** (koreksi pseudo-label NB02) — lihat §10.0. Yang
     tidak pernah ada adalah NB02b **versi CLAHE**, bukan slot NB02b itu sendiri.
4. **Catatan:** `output/roi/roi_config.json.broken` adalah sisa file rusak lama,
   bisa dihapus bila tidak diperlukan.

### 0.2 Revisi 2026-10-15

Revisi besar setelah review menyeluruh (8 reviewer paralel) dan pembacaan
literatur Grounded-SAM. Isi utama:

1. **Redesain NB02 (pseudo-labelling)** — dari SAM+DINO+geometry (Mode A lama)
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

**Status eksekusi saat ini:** ROI final ✅ · model tiny ✅ · builder NB02
diperbaiki bertahap (threshold 0.15 → **0.30**, filter luas 40%, prompt bertitik,
verifier kendaraan FRCNN) · notebook NB02 (8 sel kode, lolos self-check) ·
**perlu HAPUS chunks & RE-RUN NB02** — lihat §0.3–§0.4.

### 0.3 Temuan & Perbaikan Kualitas Label NB02 (Run 2026-10-18)

**Status: akar masalah sudah ditemukan, perbaikan SUDAH diimplementasikan di
builder, tinggal RE-RUN NB02.**

**Masalah pada run pertama (1.548 frame unik → 3.336 box):**

1. **Box raksasa (kritis).** 1.537 dari 3.336 box (**46.1%**) menutupi **>80% luas
   crop** — tidak wajar untuk satu lampu. Score box raksasa justru **lebih tinggi**
   (median 0.372) daripada box normal (0.303), jadi menaikkan threshold tidak
   menyelesaikan.
2. **Grouping Mode A praktis mati.** `LampuMobil` dengan `lamp_count=2` (union
   pasangan) hanya **19 dari 1.101 (1.7%)** — 98% `LampuMobil` ditentukan vote DINO,
   bukan geometri — kebalikan dari desain §3.5.
3. **Deteksi sedikit.** Median ~2 box/frame; banyak kendaraan di citra tidak
   tertangkap. Rasio kelas terbalik (`LampuMotor` 2.235 > `LampuMobil` 1.101).

**Akar masalah (lewat eksperimen GPU 20 frame — sweep threshold × prompt ×
tiny-vs-base):**

- **`DETECTION_THRESHOLD` terlalu tinggi (0.25).** Di crop malam hari, 0.25 hanya
  menangkap ~2 box/frame. Turun ke **0.15** menaikkan box wajar ~4× (→ ~7–9/frame)
  tanpa menambah box raksasa secara proporsional.
- **`TEXT_THRESHOLD` ternyata NO-OP** di `transformers` 5.14 — `score` dihitung dari
  max probabilitas teks atas seluruh 256 dimensi kosakata, jadi `text_threshold`
  hanya dipakai ekstraksi label, bukan filter box. Tidak berpengaruh ke jumlah box.
- **Box raksasa itu "region" DINO-tiny yang lolos** karena pipeline tidak punya
  filter luas sama sekali (hanya filter box < 5 px). 92% box normal berada **di
  dalam** box besar — DINO memberi 1 box region + 1 box lampu.
- **Crop ROI TIDAK merusak** deteksi (crop justru ≥ full frame di 4/5 frame uji).
- Model alternatif `grounding-dino-base` menghasilkan ~0 box raksasa, tapi 2×
  lebih berat; jalur upgrade dicatat di §5.5.

**Perbaikan yang sudah diimplementasikan di builder (2026-10-18):**

- `DETECTION_THRESHOLD` 0.25 → **0.15**.
- Filter luas box baru: `MAX_BOX_AREA_RATIO = 0.40` — box menutupi **>40% luas
  crop** otomatis dibuang (dipakai di `clip_and_filter_boxes`).
- `CONFIG_SIGNATURE` ikut berubah (threshold + filter) → **chunk lama otomatis
  ditolak**; Anda harus **hapus isi `output/nb02/chunks/` lalu re-run NB02**.

**Verifikasi (5 frame, 1 per kamera, konfigurasi baru):** box wajar naik jadi
**~7.8 box/frame** (bappenda1 14, bappenda2 8, bappenda3 4, pdam2 6, pemda2 7),
box raksasa terbuang otomatis (2–3 per frame). Sebelumnya median 2 box/frame.

**Bukti visual proses:** sel visualisasi NB02 (sel 4) menampilkan overlay box per
kamera + histogram ukuran box/score langsung di notebook; file PNG di
`output/nb02/previews/`.

---

### 0.4 Paket 3-Sinyal & Threshold 0.30 (Implementasi 2026-10-19)

**Status: implementasi SELESAI di builder + notebook, tinggal hapus chunk & re-run.**

Konteks: run 2 (threshold 0.15) diverifikasi bersih — **1.529 frame unik → 4.967
box**, 0 box > 40% crop, grouping union naik ke **22,9%**, 19 kontaminasi run 1
terhapus. Setelah itu riset foundation models (disetujui 2026-10-19)
merekomendasikan perubahan berikut; seluruhnya sudah diimplementasikan:

1. **Threshold 0.15 → 0.30** (`DETECTION_THRESHOLD`) — 0.15 memicu pasangan
   lampu palsu (penyebab "2 motor jadi 1 mobil"); 0.30 mengikuti rekomendasi
   riset untuk menekan grouping keliru.
2. **Prompt bertitik + lowercase** — `"car headlight."` / `"motorcycle headlight."`
   (pola resmi Grounding DINO pada transformers 5.x).
3. **Verifier kendaraan torchvision** — `fasterrcnn_resnet50_fpn_v2` bobot COCO;
   kelas COCO di-remap (`car/bus/truck` → `LampuMobil`, `motorcycle` →
   `LampuMotor`), `box_score_thresh` 0.4, vote diambil dari deteksi dengan
   **coverage tertinggi ≥ 0.3** terhadap box lampu (IoU tidak dipakai — box
   lampu vs kendaraan selalu overlap kecil). Berjalan di **CUDA**: terukur
   **1,15 d/frame** (DINO+verifier gabungan), **peak VRAM 1,82 GiB** dari
   ~4,3 GiB. Bobot 167 MB diunduh sekali saat run pertama (butuh internet);
   bila bobot gagal dimuat, verifier dimatikan otomatis dan deteksi tetap jalan
   tanpa vote.
4. **Hierarki kelas baru**: (a) grup ≥ 2 lampu → `LampuMobil`; (b) vote FRCNN →
   kelas kendaraannya; (c) fallback argmax skor DINO multi-pass. Field baru per
   deteksi: `vehicle_vote` dan `signals_agree` (false = geometri/vote
   bertentangan → ditandai **oranye `[konflik]`** di overlay dan masuk statistik
   sel histogram — bahan review NB03).
5. **Verifier blob terang (connected components cv2) DITOLAK berdasarkan data**:
   diuji pada 400 frame nyata, aturan "≥2 blob → mobil" nyaris tidak memisahkan
   (dua blob ≥60 px: **52%** grup ≥2 lampu vs **43%** grup 1 lampu) dan justru
   membalik **61%** deteksi 1-lampu menjadi `LampuMobil`. Kode & konstanta blob
   dibuang; dipertimbangkan ulang di NB03 setelah gold-set ada.
6. **`CONFIG_SIGNATURE` berubah lagi** (threshold + prompt + `detection_model_id`
   + `chunk_frame_count` + parameter verifier; sekaligus perbaikan review P0-2)
   → **chunk run 2 otomatis ditolak. WAJIB hapus isi `output/nb02/chunks/`
   lalu re-run NB02 penuh.**

Referensi: benchmark `bench_nb02_verifier.py` (2026-10-19), statistik blob
`diag_blob_stats_nb02.py` (400 frame), `REVIEW_2026-10-18.md` (P0-2 ikut
diperbaiki).

---

## 1. Tujuan Riset

Mendeteksi **lampu kendaraan pada rekaman CCTV malam** dan mengklasifikasikan
**kondisi** lampu tersebut ke dalam tiga kategori yang menjadi fokus riset:

1. **lampu kendaraan yang silau** — sumber cahaya yang menyilaukan (luapan
   cahaya / bloom besar pada sensor);
2. **lampu kendaraan yang berwarna** — lampu yang tidak tampil putih standar
   (mis. biru, merah, hijau, ungu, kuning pekat);
3. **lampu kendaraan yang silau dan berwarna** — memenuhi kedua kriteria
   sekaligus.

Output akhir terdiri dari dua lapis:

1. **Deteksi** — bounding box per kejadian lampu `(x, y, width, height)`
   beserta kelas kondisinya.
2. **Sistem hitung** — agregasi per frame / per jam / per kamera:

| Keluaran hitung | Turunan |
|---|---|
| Jumlah kendaraan mobil | semua kelas `Mobil*` |
| Jumlah kendaraan motor | semua kelas `Motor*` |
| Jumlah lampu silau — mobil / motor | `MobilSilau`, `MotorSilau` (+ rollup `*SilauBerwarna`) |
| Jumlah lampu berwarna — mobil / motor | `MobilBerwarna`, `MotorBerwarna` (+ rollup) |
| Jumlah lampu silau + berwarna — mobil / motor | `MobilSilauBerwarna`, `MotorSilauBerwarna` |

Kontrak agregasi lengkap ada di **§10.6**. Satuan hitung (per kendaraan vs per
lampu fisik) masih terbuka di **§13.10**.

**Dimensi kondisi** (3 kategori riset yang wajib dibedakan):

| Kondisi | Definisi |
|---|---|
| `Silau` | menyilaukan, warna dasar putih/normal |
| `Berwarna` | berwarna, tidak menyilaukan |
| `SilauBerwarna` | menyilaukan **dan** berwarna |

**Dimensi jenis kendaraan:** `Mobil`, `Motor`.

> Kelas model yang dilatih adalah **kombinasi kedua dimensi** — 8 kelas
> `{Mobil,Motor} × {Normal,Silau,Berwarna,SilauBerwarna}` — karena "jumlah
> kendaraan mobil/motor" hanya bisa dihitung bila lampu normal pun terdeteksi.
> Opsi label space dan konsekuensinya: **§13.9**. `PantulanCahaya` dipertahankan
> sebagai hard-negative dan tidak pernah dihitung.

> **Definisi operasional label ditetapkan 2026-10-28 — lihat §13.9a.** Ringkasnya:
> "silau" dinilai dari **bentuk luapan cahaya** (bukan ambang fotometrik, yang
> terbukti tidak memisahkan), "berwarna" hanya **lampu depan berwarna biru/cyan**,
> dan kasus ragu → `BelumDitentukan`.

> ✅ **Status keselarasan implementasi (diperbarui 2026-10-28): SELARAS DI KODE,
> BELUM DI DATA.** NB02b–NB06 sudah memakai 9 kelas kondisi lampu
> (`{Mobil,Motor} × {Normal,Silau,Berwarna,SilauBerwarna}` + `PantulanCahaya`),
> bukan lagi `LampuMobil`/`LampuMotor`. Yang belum: dua kelas berwarna hampir
> kosong (`MobilBerwarna` 0 box, `MotorBerwarna` 4, `MotorSilauBerwarna` 2 dari
> 1.125 box), dan set contoh acuan "silau" belum dibuat. Bukti per notebook:
> **§10.6**; konsekuensi data: **§13.9a**.

---

## 2. Dataset

Sumber: `CCTV Kab Bogor_ Raya Lintang Kita (Rawamurang) (2026-08-01)` — folder
berisi 5 kamera:

- **B01** — Gerbang Tol Ciawi
- **B02** — Gerbang Tol Ciawi 2
- **B03** — Ciawi
- **B04** — Jalan Raya Lintang Kita, Gunung Sindurato
- **B05** — Jalan Raya Lintang Kita, Ciparigi

**Kunci nama file (`baca_footage.py`):**

- `atcs_<camera_id>_<YYYYMMDD>_<HHMMSS>.mp4` → kamera sesuai `<camera_id>`

**Aturan pemetaan nama file (penting — jangan diubah):**

- Huruf `O` pada nama file CCTV **selalu dianggap angka `0`**.
- Setelah substitusi, hanya **6 digit pertama** yang dipakai sebagai
  timestamp: `YYMMDDHHMMSS`.

### 2.1 Konfigurasi Video (Terverifikasi 2026-10-18)

Ukuran dan fps berbeda per kamera — **bukan satu resolusi seragam**:

| Kamera | Resolusi | FPS | Frames/video | Durasi |
|---|---|---|---|---|
| bogor-kab-bappenda1 | 704×576 | 25.00 | 15.000 | 600 s |
| bogor-kab-bappenda2 | 640×480 | 25.00 | 15.000 | 600 s |
| bogor-kab-bappenda3 | 640×480 | 25.00 | 15.000 | 600 s |
| bogor-kab-pdam2 | 640×360 | 24.92 | 14.950 | 600 s |
| bogor-kab-pemda2 | 640×480 | 25.00 | 7.840 | 314 s |

Konsekuensi: **jangan pernah mengasumsikan 1280×720 atau 768×576.** Semua
koordinat (ROI, box) harus di-clamp ke ukuran frame aktual per kamera — sudah
ditangani `crop_and_fit_frame` di NB02 via `Image.open(...).size`.

### 2.2 Ringkasan Dataset

| Item | Nilai | Sumber |
|---|---|---|
| Total file (dataset penuh) | 439 | Perencanaan awal |
| Durasi (dataset penuh) | 34.27 jam | Perencanaan awal |
| Kamera | 5 (`bogor-kab-bappenda1/2/3`, `pdam2`, `pemda2`) | Terverifikasi |
| Video di folder kerja saat ini | 5 (1 per kamera) | Terverifikasi 2026-10-18 |
| Periode | 1–31 Agustus 2026 | Perencanaan awal |
| Frame di `frames_manifest.csv` | 4.520 | Terverifikasi 2026-10-18 |

> **Catatan:** angka 439 file / 34.27 jam adalah rencana dataset penuh.
> Folder kerja saat ini (`output/videos/`) berisi **5 video contoh** (satu per
> kamera). Sesuaikan klaim laporan dengan cakupan yang benar-benar diproses.

---

## 3. Grounding untuk Model

Model yang digunakan: **GroundingDINO-tiny (swin-t)**.

### 3.1 Kategori dan Prompt

Taksonomi yang dipakai (2 kelas — diputuskan, lihat §3.3):

| Prompt DINO | Nama kelas akhir | Definisi |
|---|---|---|
| `car headlight.` | **LampuMobil** | Seluruh area terang lampu depan mobil (headlight), termasuk gabungan pasangan |
| `motorcycle headlight.` | **LampuMotor** | Seluruh area terang lampu depan sepeda motor (headlight) |

Catatan penamaan: nama kelas memakai awalan `Lampu` agar konsisten dan tidak
bertabrakan dengan nama kelas COCO lain (`car`, `motorcycle`,
`motorcycle headlight road`) saat eksperimen prompt. Prompt ditulis
**lowercase + diakhiri titik** — pola resmi Grounding DINO pada transformers 5.x
(berbentuk kalimat pendek; tanpa titik, skoring teksnya kurang stabil).

### 3.2 Prompt Engineering untuk Lampu

GroundingDINO menangani **satu frasa per kelas**. Untuk `text_threshold` bekerja
benar, setiap frasa hanya boleh merujuk **satu kelas**. Karena itu pemrosesan
dilakukan **multi-pass (satu frasa per kelas)**:

| Pass | Frasa tunggal | Hasil lalu digabung |
|---|---|---|
| 1 | `"car headlight."` | kelas `LampuMobil` |
| 2 | `"motorcycle headlight."` | kelas `LampuMotor` |

Duplikat dari dua pass digabung berdasarkan IoU ≥ 0.5 (`LAMP_MERGE_IOU_THRESHOLD`);
kelas grup ditentukan hierarki geometri → vote verifier → vote DINO (lihat §3.5).

### 3.3 Alasan Pemilihan Taksonomi 2 Kelas (Diputuskan)

Keputusan: **2 kelas `LampuMobil`/`LampuMotor`**. Pertimbangan:

- Dataset CCTV jarak jauh, malam hari, lampu hanya titik terang → membedakan
  mobil vs motor hanya dari bentuk/ukuran lampu.
- Menambah kelas `LampuMobilJauh`/`LampuMobilDekat` memecah proporsi sangat
  tidak seimbang (mobil ≈ 91% vs motor ≈ 9%) menjadi kelas-kelas kecil.
- Berisiko menambah noise, bukan mengurangi.

### 3.4 Mitigasi Bias Box — Mode A (Lampu-Sepasang)

Mode A yang dipakai (terpilih dari 3 opsi yang dibahas):

| Kelas | Definisi box |
|---|---|
| **LampuMobil** | **Satu box** hasil **union** dari pasangan lampu mobil (bbox gabungan) |
| **LampuMotor** | **Satu box** untuk lampu tunggal sepeda motor |

Rationale: pada mobil, DINO cenderung memberi 1–2 box per lampu atau 1 box
gabungan → berubah-ubah → noise saat training. Dengan union pasangan, output
lebih konsisten. Pada motor, lampu depannya satu → tidak ada merging.

### 3.5 Strategi Pemberian Kelas (Grouping Geometri + Vote DINO)

Keputusan: **grouping berbasis GEOMETRI, vote DINO sebagai tiebreak.**

Alasan menolak **Opsi 1 (vote DINO murni)**: dokumen resmi Grounded-SAM
menegaskan DINO hanya bisa mengenali **objek makro** (`car`, `motorcycle`)
dengan cukup akurasi, sedangkan `taillight`/`headlight` hanya diberi label
**"lumayan"** — justru kasus yang kita hadapi. Menjadikan vote sebagai
mekanisme utama berarti mewariskan kelemahan itu ke pseudo-label.

Alasan memilih geometri: pada citra malam jarak jauh, **jumlah lampu mobil
(2) vs motor (1) adalah pembeda paling andal** — lebih stabil daripada
keyakinan DINO yang lemah pada sub-kategori.

Aturan grouping (untuk kandidat LampuMotor — yaitu kandidat yang **tidak**
tergabung):

1. Kandidat lampu yang **tidak tergabung** dengan kandidat lain → **LampuMotor**
   (kecuali hierarki kelas di bawah menentukan lain).
2. Kandidat lampu yang tergabung menjadi **≥ 2** (pasangan) → union → **LampuMobil**.
3. Syarat digabung: jarak tengah ≤ `80 px`, selisih tinggi tengah ≤ `25 px`,
   rasio lebar ≤ `2.5x`. (Duplikat antar-pass digabung lebih dulu bila
   IoU ≥ 0.5 — lihat §3.2.)

**Hierarki penentuan kelas per grup (implementasi saat ini, 2026-10-19):**

1. Grup ≥ 2 lampu → **`LampuMobil`** (geometri — pembeda paling andal).
2. Grup 1 lampu → **vote verifier torchvision FasterRCNN** — kelas kendaraan
   COCO (`car/bus/truck` → `LampuMobil`, `motorcycle` → `LampuMotor`) yang
   box-nya menutupi ≥ 30% area box lampu (`VEHICLE_VOTE_MIN_COVERAGE`).
3. Tanpa vote → **argmax skor DINO multi-pass** (tiebreak terakhir).

Catatan implementasi (penting): skor DINO dinilai dari **nilai logit/score
mentah** yang diambil saat inferensi multi-pass, **bukan** dari `text_threshold`
(`text_threshold` hanya pemilih box biner, bukan pengukur keyakinan). Pemisahan
pasangan berdasarkan vote DINO (aturan lama) **tidak** diimplementasikan —
pasangan selalu `LampuMobil`. Semua sinyal yang tersedia disimpan per deteksi
(`vehicle_vote`, `signals_agree`) untuk review di NB03.

---

## 4. Alur Riset (Pipeline)

```
Scraping & Ekstraksi Frame (NB01) ✅
        │
        ▼
Pseudo-labelling (NB02) ✅ selesai (2.018 box, 1.482 gambar YOLO)
  ├─ ROI polygon per kamera (di dalam NB02)
  ├─ Dedup frame (dedup_manifest.csv)
  ├─ GroundingDINO multi-pass (Mode B)
  └─ Grouping geometri + vote DINO (Mode A)
        │
        ▼
Anotasi Manual (NB02b) ✅ dibuat 2026-10-20
  └─ Koreksi pseudo-label (400 frame) → COCO (fine-tune DINO) + gold-set YOLO
        │
        ▼
Validasi & Kalibrasi (NB03) ❌ belum dibuat
  └─ Gold-set + kalibrasi threshold
        │
        ▼
Training YOLO (NB04) ❌ belum dibuat
        │
        ▼
Evaluasi & Analisis Error (NB05) ❌ belum dibuat
```

> **Koreksi 2026-10-18:** tidak ada tahap NB02b **versi CLAHE** maupun NB02a
> (frame filtering terpisah). NB02 bersifat mandiri: ROI polygon dan dedup
> frame ditangani di dalam notebook yang sama. CLAHE tidak dipakai.
>
> **Koreksi 2026-10-20:** slot **NB02b** kini terisi sebagai **tahap anotasi
> manual** (koreksi pseudo-label NB02) — lihat §10.0. Urutan eksekusi nyata:
> `NB01 → NB02 (pseudo) → NB02b (koreksi) → fine-tune DINO → re-NB02 → NB03`.

### 4.1 Posisi FM pada Pipeline

FM (fitur *Fine-grained Matching*) adalah **tahap lanjutan** setelah YOLO
dilatih dan dievaluasi.

```
YOLOv11 Detection → Deep Feature Extraction → Fine-Grained Matching → Tracking
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
- GroundingDINO sudah memberi **box + confidence** → cukup untuk
  pseudo-labelling.
- SAM hanya berguna jika butuh **mask presisi** untuk crop / analisis shape.

Konsekuensi: **semua parameter `SAM_*` tidak dipakai** (dipertahankan di config
hanya sebagai referensi). Pipeline jadi lebih sederhana dan hemat VRAM.

### 5.2 Multi-pass per frasa kelas

Karena GroundingDINO menangani satu frasa per kelas, pemrosesan dilakukan dua
pass (lihat §3.2). Hasil dua pass digabung: duplikat IoU ≥ 0.9 disatukan,
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
| `DETECTION_THRESHOLD` | **0.30** | 0.25 (awal) → 0.15 (fix run 1) → **0.30** (paket 3-sinyal — 0.15 memicu pasangan lampu palsu). Kalibrasi ulang di NB03 |
| `TEXT_THRESHOLD` | 0.20 | No-op di transformers 5.x (hanya pemilih label, bukan filter box) |
| `MAX_BOX_AREA_RATIO` | 0.40 | Filter box raksasa (> 40% luas crop dibuang) |
| `VEHICLE_SCORE_THRESHOLD` | 0.4 | Ambang skor deteksi kendaraan verifier (lihat §5.6) |
| `VEHICLE_VOTE_MIN_COVERAGE` | 0.3 | Vote kendaraan hanya sah bila menutupi ≥ 30% area box lampu |
| Kalibrasi di NB03 | Optimal | Rencana — target F1 maksimum pada gold-set |

### 5.5 Model DINO (diputuskan 2026-10-18)

| Item | Nilai | Keterangan |
|---|---|---|
| `DETECTION_MODEL_ID` | `IDEA-Research/grounding-dino-tiny` | Dipakai untuk run pertama NB02 |
| Varian | swin-t | Sesuai dokumentasi riset, ringan di RTX 3050 4GB |

**Jalur upgrade:** bila hasil anotasi NB02 (dicek di NB03 lewat gold-set)
kurang bagus — miss detection tinggi pada lampu kecil atau banyak false
positive — ganti `DETECTION_MODEL_ID` di `builders/build_nb02_pseudo_label.py`
menjadi `IDEA-Research/grounding-dino-base` (swin-b), lalu regenerate notebook
dan re-run NB02. Pertimbangan biaya: base butuh download ~900 MB tambahan dan
VRAM lebih besar, tetapi akurasi sedikit lebih baik pada objek kecil.

### 5.6 Verifier Kendaraan (diputuskan 2026-10-19)

| Item | Nilai | Keterangan |
|---|---|---|
| Model verifier | `fasterrcnn_resnet50_fpn_v2` (torchvision, bobot `COCO_V1`) | Vote sekunder untuk grup 1 lampu. OWLv2 / Florence-2 / YOLO-World di-skip (lebih berat, tanpa keuntungan terukur) |
| `VEHICLE_CLASS_REMAP` | `car/bus/truck → LampuMobil`, `motorcycle → LampuMotor` | Hanya kelas kendaraan COCO yang dipakai; kelas lain diabaikan |
| `VEHICLE_SCORE_THRESHOLD` | 0.4 | Ambang skor deteksi kendaraan |
| `VEHICLE_VOTE_MIN_COVERAGE` | 0.3 | IoU box lampu vs kendaraan selalu kecil → dipakai **coverage** (luas irisan / luas box lampu); vote sah bila ≥ 0.3 |
| `VERIFIER_DEVICE` | `cuda` | Benchmark 2026-10-19: **1,15 d/frame** (DINO+verifier), **peak VRAM 1,82 GiB** dari ~4,3 GiB; bobot 167 MB diunduh sekali (butuh internet) |
| Gagal dimuat | Verifier dimatikan otomatis | `load_vehicle_verifier` menangkap exception → vote `None`, deteksi DINO tetap jalan |

Vote **sekonder** — hanya menentukan kelas bila geometri grup < 2 lampu, tidak
pernah menghapus box. Kelemahan domain malam (vote lemah/salah) dipantau lewat
flag `signals_agree` dan dikalibrasi di NB03 pada gold-set.

---

## 6. ROI (Region of Interest) — Polygon 5 Kamera

### 6.1 Keputusan Desain

ROI direkonstruksi menjadi **polygon per kamera** (sebelumnya 2 rectangle untuk
B01/B02). Setiap kamera mendapat **satu polygon** yang membatasi area jalan
yang relevan.

Alasan: CCTV kabupaten punya area noise besar (pohon, gedung, lampu jalan,
langit). ROI memotong jumlah frame/kandidat yang diproses dan mengurangi
false positive.

### 6.2 Posisi Saat Ini — FINAL ✅

- **Final dan disetujui user (2026-10-18).** `output/roi/roi_config.json`
  berisi polygon untuk semua 5 kamera:

  | Kamera | Frame aktual | Verteks | Catatan |
  |---|---|---|---|
  | bogor-kab-bappenda1 | 704×576 | 4 | trapesium |
  | bogor-kab-bappenda2 | 640×480 | 4 | trapesium |
  | bogor-kab-bappenda3 | 640×480 | 4 | trapesium |
  | bogor-kab-pdam2 | 640×360 | 8 | 3 verteks sengaja dibuat rapat agar watermark tidak ikut |
  | bogor-kab-pemda2 | 640×480 | 24 | mengikuti kurva jalan, disengaja |

- **Ukuran frame berbeda per kamera** — bukan 1280×720. Koordinat polygon
  mengikuti PNG asli tiap kamera; builder NB02 meng-clamp ke ukuran frame
  aktual (`Image.open` → `size`), jadi koordinat selalu valid.
- Hasil verifikasi: tidak ada polygon bersilangan, tidak ada verteks di luar
  batas frame. Overlay preview tersimpan di `output/roi/preview/` (5 file PNG).
- Field `roi` (rectangle) di config masih menyimpan nilai template 1280×720 —
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
- Kurang dari 3 verteks → `Enter` ditolak dengan pesan `butuh minimal 3 verteks`.
- Polygon bersilangan → `Enter` ditolak dengan pesan `poligon bersilangan`, minta undo lalu perbaiki.
- Verteks duplikat (jarak ≤ 2 px) dibersihkan otomatis sebelum disimpan.

### 6.4 Validasi ROI — SELESAI ✅

- **Geometri:** semua 5 polygon diverifikasi otomatis — tidak bersilangan,
  seluruh verteks di dalam batas frame masing-masing.
- **Visual:** overlay polygon di atas frame contoh per kamera dibuat di
  `output/roi/preview/*.png` dan **dicek langsung oleh user → benar**.
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

- **19 risiko** terdaftar (B1–B4, H1–H10, I1–I5).
- Semua isu hasil review 8 reviewer dipetakan sebagai risiko terkontrol.
- Kontrol pre-fix **C1–C4** (backup, self-check, config signature, gold-set)
  sudah dijalankan sebelum perubahan kode.
- **Gate verifikasi** ditetapkan sebelum run NB02 (lihat register).

---

## 8. Backup & Hasil Lama

### 8.1 Backup Sebelum Perubahan

`backup_sebelum_fix_20261015/` sudah dipindahkan ke
`trash_2026-10-26/audit_2026-10-28/` pada 2026-10-28 (audit temuan 14): isinya
duplikat byte-per-byte dari builder aktif, sehingga berisiko diedit orang lain.
Tidak ada rollback point terpisah lagi — riwayat perubahan mengandalkan dokumen
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
├── MASTER_DOKUMENTASI_HAMPIR.md      ← Dokumen ini
├── RISK_REGISTER.md                  ← Register risiko (v1.5)
├── NB01_acquisition.ipynb            → Scraping & ekstraksi frame
├── NB02_pseudo_label.ipynb           → Pseudo-label DINO (dihasilkan builder)
├── builders/
│   ├── build_nb01_acquisition.py     → Builder NB01 (+ template ROI, group split)
│   ├── build_nb02_pseudo_label.py    → Builder NB02 (DINO multi-pass, Mode B + A)
│   ├── build_roi_config.py           → Tool interaktif polygon ROI
│   └── manual_check_nb02.py          → Pemeriksaan manual logika crop + grouping (bukan test otomatis)
├── test_pipeline_logic.py            → Self-check tanpa GPU (kontrak hitung + fragment stitching)
├── requirements.txt                  → Pin dependency yang dipakai
├── README.md                         → Cara setup & urutan menjalankan
├── output/
│   ├── videos/                       → file .mp4 per rekaman (nama bertimestamp)
│   ├── frames/                       → folder frame PNG per video
│   ├── katalog/                      → bogor_kab_targets.json, cctv_all.json
│   ├── manifests/                    → scraping_manifest, frames_manifest, dedup_manifest
│   ├── split/group_split.csv         → Split train/val/test per kamera
│   ├── roi/                          → roi_config.json + preview/ (5 PNG)
│   └── nb02/                         → chunks/, previews/, yolo/
└── trash_2026-10-26/                 → Berkas yang dipindahkan keluar pipeline (bukan dihapus)
```

> **Penting:** modul `src/*.py` **kosong** — seluruh logika hidup sebagai string
> di dalam `builders/build_*.py` lalu dirender menjadi sel notebook. Modul
> NB03/NB04/NB05/FM **belum pernah dibuat** — hanya ada sebagai rencana.

### 9.1 Fungsi Berkas yang Ada

| File | Fungsi |
|---|---|
| `builders/build_nb02_pseudo_label.py` | Merender `NB02_pseudo_label.ipynb` (5 sel kode + 1 markdown). Memuat logika DINO-only multi-pass (Mode B) + Mode A grouping. SAM dihapus. |
| `builders/manual_check_nb02.py` | Pemeriksaan manual logika crop polygon + grouping + aturan kelas. Dijalankan manual; **bukan** test otomatis (test otomatis ada di `test_pipeline_logic.py`). |
| `builders/build_roi_config.py` | Tool interaktif pemilihan polygon ROI per kamera + konstanta ROI. |
| `builders/build_nb01_acquisition.py` | Scraping katalog, ekstraksi frame, group split, template ROI. |

### 9.2 Berkas yang Belum Dibuat

| Tahap | Status |
|---|---|
| NB02b (anotasi manual — koreksi pseudo-label) | ✅ Ada (2026-10-20) — perlu dijalankan manual |
| NB03 (gold-set + kalibrasi threshold) | ❌ Belum ada |
| NB04 (dataset YOLO + training) | ❌ Belum ada |
| NB05 (evaluasi + analisis error) | ❌ Belum ada |
| FM (fine-grained matching) | ❌ Belum ada |

---

## 10. Status Pipeline Saat Ini

| Tahap | Notebook | Status |
|---|---|---|
| Scraping & Ekstraksi Frame | `NB01_acquisition.ipynb` | ✅ Selesai (5 video, 10 folder frame, 4.520 frame) |
| Pseudo-labelling (DINO Mode B + A) | `NB02_pseudo_label.ipynb` | ✅ Selesai (1.548 frame unik → 6.490 box mentah → 2.018 box YOLO, 1.482 gambar). ⚠️ Output **UNSplit** — pemecahan train/val/test menunggu NB04, lihat §10.0b |
| Anotasi Manual (koreksi pseudo-label) | `NB02b_annotation.ipynb` | ✅ Dibuat 2026-10-20 (6 sel kode + 1 markdown, lolos self-check) → **siap dijalankan manual** (lihat §10.2) |
| Validasi & Kalibrasi Threshold | NB03 | ❌ Belum dibuat |
| Training YOLO | NB04 | ❌ Belum dibuat |
| Evaluasi & Analisis Error | NB05 | ❌ Belum dibuat |
| Fine-Grained Matching | FM | ❌ Belum dibuat |

### 10.0 NB02b — Anotasi Manual Pinpoint (Ditambahkan 2026-10-20)

**Tujuan.** Mengoreksi pseudo-label NB02 secara manual pada sampel frame, dan
memakai hasilnya untuk **dua hal sekaligus**: (a) dataset fine-tune Grounding
DINO (continue pre-trained), (b) gold-set validasi/kalibrasi untuk NB03.

**Posisi.** NB02b diletakkan **setelah** NB02 (karena pseudo-label adalah bahan
yang dikoreksi) meskipun nomornya `01b`. NB02 hanya butuh frame dari NB01, jadi
tidak ada siklus dependensi. Urutan eksekusi nyata:

```
NB01 (akuisisi) → NB02 (pseudo-label) → NB02b (koreksi manual)
                                      → fine-tune Grounding DINO
                                      → re-run NB02 (label otomatis naik kualitas)
                                      → NB03 (gold-set + kalibrasi threshold)
```

**Implementasi.**

| Aspek | Nilai |
|---|---|
| Builder | `builders/build_nb02b_annotation.py` |
| Notebook | `NB02b_annotation.ipynb` (6 sel kode + 1 markdown) |
| Sumber gambar | `output/nb02/yolo/images/*.png` (sudah ROI-crop + fit 640px) |
| Sumber koreksi | `output/nb02/chunks/*.jsonl` (box pseudo-label) |
| **Split gold-set** | Hanya `train` + `val` (kamera `test` dikecualikan). ⚠️ Gold-set **masih mencampur `train`+`val` dalam satu folder** — pecah per split sebelum evaluasi NB03 (lihat R8) |
| Mode | Koreksi pseudo-label: box DINO ditampilkan, user tambah/hapus/ubah kelas |
| Sampling | Stratified proporsional per kamera, `SAMPLE_TOTAL_FRAMES = 400`, seed `20261020` |
| **Anti-kebocoran** | **Kamera split `test` dikecualikan** (`SAMPLE_EXCLUDED_SPLITS = ["test"]`) — hanya `train` + `val` dianotasi + guard ganda di sel EXPORT |
| Alokasi nyata | `bappenda1` (train) 267 · `pemda2` (val) 87 · `bappenda3` (train) 46 = **400** |
| Frame dikecualikan | `bappenda2` (test) 195 · `pdam2` (test) 445 = **640** tidak tersentuh |
| UI | `ipympl` (`%matplotlib widget`), `RectangleSelector` + klik, `ToggleButtons` kelas |
| Store | `output/annotation/annotations.json` (per frame, tulis atomik, resume-aman) |
| Manifest | `output/annotation/sample_manifest.json` (disimpan sekali → sampling stabil) |
| Export COCO | `output/annotation/annotations_coco.json` (fine-tune DINO) |
| Export gold-set | `output/annotation/yolo/{images,labels,data.yaml}` |
| Summary | `output/annotation/annotation_summary.json` + 2 PNG preview |

**Alur sel.**

1. Konfigurasi + `mkdir` output + peta split per kamera.
2. Sampling stratified per kamera **setelah membuang split `test`** → manifest (reload bila sudah ada;
   manifest lama dengan `excluded_splits` berbeda akan **menolak** dipakai).
3. Muat store anotasi + `save_annotation_store()` + validasi (kelas, span box, frame asing).
4. Preview 3 frame pertama (box pseudo-label) — cek cepat sebelum anotasi.
5. UI interaktif (`ipympl`) — autosave tiap mutasi.
6. Export COCO + gold-set YOLO + summary + preview hasil.

**Provenance box.** Tiap box menyimpan `source`: `pseudo` (belum disentuh),
`manual` (ditambah user), `edited` (kelas diubah user). Berguna saat audit
kualitas gold-set di NB03 — box yang masih `pseudo` berarti belum diverifikasi
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

1. **Kelas deteksi** — ikut export COCO (fine-tune DINO) dan gold-set YOLO
   (`nc: 3`, index `2`). Model belajar membedakannya, bukan sekadar membuangnya.
2. **Hard-negative** — saat penghitungan kendaraan di hilir, deteksi kelas
   `PantulanCahaya` **difilter keluar**. Pantulan tidak pernah lagi terhitung
   sebagai kendaraan.
   ⚠️ Sejak audit 2026-10-26: daftar objek yang dihitung berubah menjadi 8 angka
   kondisi lampu (**§10.6**); `PantulanCahaya` tetap tidak dihitung.

**Migrasi otomatis.** Store lama yang masih memakai `kind: distractor`
dikonversi saat `load_annotation_store()` dijalankan: field `kind` dibuang,
`class_name` diganti `PantulanCahaya`. Sudah diuji pada store nyata — **319 box
dimigrasi, 940 box aktif tetap utuh** (183 mobil + 444 motor + 313 pantulan).

**Shortcut keyboard.** `n`/`→` next · `p`/`←` prev · `x`/`Delete` hapus box
terpilih · **`1`/`2`/`3`** set kelas box terpilih · `r` reviewed + next · `s` skip ·
`u` undo · **`d` set box terpilih ke `PantulanCahaya`** (shortcut tercepat) ·
**`T` sembunyikan/tampilkan label box** (nama kelas + nomor box). Klik kanan
dijeda oleh `RectangleSelector` (drag = tambah box).

**Yang masih terbuka untuk kelas ini.** NB02 **belum** punya prompt
`road reflection` — DINO tetap hanya memprediksi `LampuMobil`/`LampuMotor`,
sehingga box pantulan murni dari anotasi manual. NB04 wajib memfilter kelas
index `2` saat menghitung kendaraan.


**Cara pakai di VS Code.** Buka `NB02b_annotation.ipynb`, pilih kernel
`.venv312`, jalankan sel 1–4 berurutan (cepat), lalu sel 5 → **panel gambar
keluar di tab "Output"** (silakan pop-out). Anotasi bisa ditinggal kapan saja —
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
- Frame tanpa deteksi NB02 tidak ikut disampel (tidak ada di `yolo/images`) —
  ini disengaja agar UI tidak pernah menampilkan frame kosong.
- Gold-set (`output/annotation/yolo`) hanya berisi frame yang ber-`reviewed`;
  frame reviewed tanpa box tetap masuk COCO (sebagai contoh negatif) tetapi
  **tidak** masuk struktur YOLO.
- Mengubah `SAMPLE_TOTAL_FRAMES`/`RANDOM_SEED` atau menambah frame NB02 baru
  membuat manifest lama tidak cocok → hapus `output/annotation/sample_manifest.json`
  (store anotasi tetap aman selama `frame_stem` masih ada di manifest baru).

### 10.1a Aturan Visualisasi Wajib (Ditambahkan 2026-10-18)

> **Setiap notebook tahap (`NBxx`) WAJIB menampilkan visualisasi dari proses
> sampai hasil**, bukan hanya angka. Kebutuhan: tugas riset membutuhkan bukti
> visual di laporan.

Cakupan minimal per notebook:

| Tahap | Visualisasi wajib |
|---|---|
| NB01 | Frame hasil ekstraksi per kamera |
| NB02 | Crop ROI → box DINO (kandidat) → grouping → overlay label final + `vote`/`[konflik]` → histogram ukuran box/score + statistik sinyal (vote & konflik) |
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
| `output/nb02/chunks/*.jsonl` | UNSplit (5 kamera) | Sumber pseudo-label, audit coverage | Bukan dataset — jangan dipakai training langsung |
| `output/nb02/yolo/{images,labels}` | **UNSplit** (1.482 gambar, 640 di antaranya `test`) | Inspeksi visual, sumber gambar NB02b | 🔴 **Tidak boleh** training: `data.yaml` `train==val` |
| `output/nb02/yolo/data.yaml` | Ditandai `split_status: UNSplit` | — | 🔴 Jangan dipakai NB04 sebelum dipecah |
| `output/nb02/yolo/split_camera_map.csv` | Pemetaan resmi kamera→split | Dasar pemecahan di NB04 | — |
| `output/annotation/*` (NB02b) | `train` + `val` saja (bebas `test`) ✅ | Fine-tune DINO, gold-set | 🟠 `train`+`val` masih satu folder → metrik per split wajib |

**Peta split resmi** (dari `output/split/group_split.csv`, diduplikasi ke `yolo/split_camera_map.csv`):

| Kamera | Split | Frame di `yolo/images` |
|---|---|---|
| `bogor-kab-bappenda1` | train | 562 |
| `bogor-kab-bappenda3` | train | 98 |
| `bogor-kab-pemda2` | val | 182 |
| `bogor-kab-bappenda2` | test | 195 |
| `bogor-kab-pdam2` | test | 445 |

**Yang sudah diperbaiki:** NB02b mengecualikan split `test` saat sampling (+ guard di export).
NB02 tidak lagi menyamar sebagai dataset siap-training — `data.yaml` bertanda `UNSplit`,
peta split ditulis eksplisit, dan sel summary mencetak peringatan leakage.

**Diperbarui 2026-10-26 — NB04 sudah memecah dan mematerialisasi split:**

| Artefak | Split status | Aman untuk apa | Risiko |
|---|---|---|---|
| `output/nb04/dataset/{images,labels}/{train,val,test}` | **Tersplit** (660/182/640), 1:1 gambar:label | Training + evaluasi | 🟠 544 frame train dan 640 frame test masih berlabel pseudo; hanya 126 frame gold (semua train) |
| `output/nb04/dataset/data.yaml` | `train/val/test` tiga folder berbeda, `nc: 3` | Training YOLO | — |
| `output/nb04/split_audit.csv`, `class_balance.csv` | Audit per kamera + histogram kelas | Verifikasi kebocoran | — |
| `output/nb03/*` | Turunan langsung gold set per split | Kalibrasi threshold, diagnostik sinyal | 🟠 val 0 frame gold → kalibrasi belum sah |
| `output/nb05/*` | Kosong (menunggu bobot NB04) | — | Diisi saat NB05 dijalankan |

**Yang masih terbuka (wajib sebelum training/evaluasi):**

1. ✅ ~~**NB04** memecah `yolo/images` → `images/{train,val,test}` memakai `split_camera_map.csv`,
   lalu tulis `data.yaml` yang benar-benar menunjuk tiga folder berbeda.~~ Selesai 2026-10-26.
2. ✅ ~~**NB03** memecah gold-set NB02b per split dan **melaporkan metrik terpisah** untuk
   `train` vs `val`; kalibrasi threshold DINO hanya dari `val`. Jangan melaporkan mAP
   gabungan train+val sebagai generalisasi.~~ Selesai 2026-10-26 (NB03 selalu memisah split).
3. 🟠 **Anotasi 87 frame `val`** — tanpa ini NB03 tidak punya dasar kalibrasi dan NB04
   menolak training (guard `MINIMUM_GOLD_VAL_FRAMES = 30`).
4. 🟠 **Gold-set test terpisah** — split test tidak punya anotasi manual; lihat R13.

### 10.1 Catatan: NB02 Mandiri (Tanpa Preprocessing NB02b / CLAHE)

**NB02b versi preprocessing CLAHE tidak diperlukan untuk NB02.** NB02 bersifat
mandiri:

- Membaca frame **asli** dari `output/frames/` (bukan hasil CLAHE).
- Menerapkan **ROI polygon sendiri** (`crop_to_polygon`) di dalam notebook.
- **Tidak** menerapkan CLAHE.

Jadi NB02 tetap dijalankan **langsung setelah NB01** — tidak ada tahap
preprocessing yang mendahuluinya. Slot `NB02b` kini dipakai untuk anotasi
manual, yang justru berjalan **setelah** NB02 (§10.0). Bila riset ingin memakai
CLAHE, itu keputusan desain baru yang perlu dibahas terpisah (lihat §13).

### 10.3 Urutan Pekerjaan Berikutnya

0. ⛔ **PERBAIKI TAKSONOMI DULU (audit 2026-10-26)** — target riset adalah
   **silau / berwarna / silau+berwarna per jenis kendaraan**, sedangkan seluruh
   kode masih memakai `LampuMobil`/`LampuMotor`/`PantulanCahaya`. Jangan latih
   NB04 dan jangan kutip angka NB05 sebelum §13.9–§13.10 diputuskan. Rincian dan
   kontrak hitung: **§10.6**.
1. ✅ ~~Finalisasi ROI polygon~~ — selesai 2026-10-18.
2. ✅ ~~Validasi ROI + approval user~~ — selesai 2026-10-18.
3. ✅ ~~Jalankan NB02 (run 1)~~ — selesai 2026-10-18, **hasil bermasalah (R1)**.
4. ✅ ~~**Perbaiki NB02 (R1)**~~ — run 2 diverifikasi bersih (2026-10-18);
   paket 3-sinyal + threshold 0.30 + verifier FRCNN ditambahkan 2026-10-19
   (lihat §0.4) → **sudah di-run ulang 2026-10-19** (1.482 gambar / 2.018 box
   YOLO). Verifikasi box raksasa & statistik vote/konflik tercatat di
   sel visualisasi NB02.
5. 🔄 **Jalankan NB02b (anotasi manual)** — 400 frame stratified dari `train`+`val`
   saja, koreksi box pseudo-label, export COCO + gold-set (lihat §10.0). Butuh
   waktu manual ±3–5 jam; bisa dicicil, store resume-aman. **Status 2026-10-26:
   126/400 frame selesai, semuanya kamera `train` (`bappenda1`); 87 frame `val`
   belum tersentuh** — inilah penghambat NB03 (kalibrasi) dan NB04 (guard).
6. **Fine-tune Grounding DINO** — continue pre-trained dari
   `output/annotation/annotations_coco.json`, lalu **re-run NB02** agar label
   otomatis naik kualitas.
7. ✅ ~~**Pecah split (WAJIB sebelum NB03/NB04)**~~ — **selesai di NB04 (2026-10-26)**:
   `builders/build_nb04_train_yolo.py` mematerialisasi
   `output/nb04/dataset/{images,labels}/{train,val,test}` dari
   `output/nb02/yolo/split_camera_map.csv` (parser melewati baris komentar),
   hardlink gambar, label **gold menimpa pseudo** untuk frame teranotasi, dan
   `data.yaml` `nc: 3`. Hasil: **660 train / 182 val / 640 test**, jumlah
   gambar = label di tiap split, kamera `test` tidak pernah masuk train/val.
   Menutup sisa B1/R7; R8 (gold-set satu folder) ditutup karena metrik selalu
   dilaporkan per split.
8. ✅ ~~**Buat NB03**~~ — **dibuat 2026-10-26** (`NB03_gold_threshold.ipynb`,
   6 sel kode). EDA gold set + **kalibrasi threshold post-hoc** dari skor yang
   sudah tersimpan di `output/nb02/chunks/*.jsonl` (tanpa GPU), IoU-matching
   greedy per kelas, metrik per split, kalibrasi **hanya dari `val`**.
   Menjalankan pada data saat ini (126 frame gold, semuanya `train`) memberi
   F1 makro train **0,334** di threshold 0,30 dan **turun monoton** sampai
   0,00 di 0,75 — indikasi DINO menyimpan banyak TP berbobot rendah, dan
   **val belum bisa mengkalibrasi apa pun** (0 frame gold). `PantulanCahaya`
   dikeluarkan dari F1 makro karena pseudo-label masih 2 kelas.
9. ✅ ~~**Buat NB04**~~ — **dibuat & smoke-tested 2026-10-26**
   (`NB04_train_yolo.ipynb`, 7 sel kode). `yolo11n`, imgsz 640, batch 8,
   workers 0. **Guard aktif:** training ditolak selama frame val gold < 30
   (`MINIMUM_GOLD_VAL_FRAMES`), karena metrik val dari label pseudo tidak sah.
   Smoke test 1 epoch (guard dimatikan sementara) membuktikan alur training +
   val + penulisan bobot berjalan di RTX 3050 4 GB.
10. ✅ ~~**Buat NB05**~~ — **dibuat & smoke-tested 2026-10-26**
    (`NB05_evaluation.ipynb`, 7 sel kode). Metrik per split + confusion matrix
    dari objek `model.val()` (tanpa `pycocotools`), sapuan confidence
    sekali-jalan, **proxy counting** dengan filter kelas `PantulanCahaya` (R11),
    galeri error terburuk. Inferensi sapuan/counting berjalan di **CPU**
    (`PREDICT_DEVICE`) karena `predict` di GPU 4 GB OOM setelah dua `val`.
    ⚠️ Kontrak hitung ini **digantikan** oleh 8 angka kondisi lampu di §10.6
    setelah audit taksonomi 2026-10-26. **Sejak 2026-10-28** sel hitung NB05
    menerbitkan `jumlah_box_*` (`counting_unit: "box_per_frame"`), bukan
    `jumlah_kendaraan_*`; angka kendaraan unik hanya dari NB06 (§10.6.1).
11. **Selesaikan anotasi NB02b — prioritas: 87 frame `val`** — tanpa frame val
    gold, NB03 tidak bisa merekomendasikan threshold dan NB04 menolak training.
    Setelah 400 frame selesai, lanjut **gold-set test terpisah ±100 frame**
    dari kamera `test` (`pdam2`, `bappenda2`) agar R13 bisa ditutup.
12. **Fine-tune Grounding DINO** — continue pre-trained dari
    `output/annotation/annotations_coco.json`, lalu **re-run NB02** agar label
    otomatis naik kualitas, lalu jalankan ulang NB03 untuk kalibrasi.
13. ✅ ~~**Hitung hasil distractor NB02b**~~ — sudah diputuskan 2026-10-22:
    `PantulanCahaya` **naik jadi kelas resmi** (`nc: 3`, 313/940 box = 33,3%),
    dipakai sebagai hard-negative (lihat §10.0 dan §10.5).

---

### 10.5 Dataset Riset-4 — Inventaris & Pemanfaatan (Audit 2026-10-21)

Riset-4 (ATCS Bandung) punya aset anotasi manual. Sumber: `Riset-4/MASTER_DOKUMENTASI_RISET.md`.

**Hasil audit fisik (bukan sekadar dokumen):**

| Aset | Isi nyata | Bisa untuk fine-tune DINO? |
|---|---|---|
| `AnotasiManual/` | 490 crop PNG ±30×30 px, folder = kelas, **tanpa koordinat**, **tanpa frame asli** | ❌ Tidak |
| `anotasimanual-refined/` | 490 crop PNG, sama, versi diperjelas | ❌ Tidak |
| `golden_set_v3.csv` | 200 baris, **kolom koordinat KORUP** (Excel Indonesia: `19.05` → `19.050.863.647.460.900`) | ❌ Jangan dipakai |
| `golden_set_v3/candidates_index.csv` | koordinat asli sebagai JSON float (`bbox_xyxy`) | ✅ **Sumber koordinat** |
| `Hasil-2/output/frames_sampled/` | 1.000 frame PNG resolusi penuh (640×480) | ✅ **Sumber gambar** |
| `foundation_model/crops/` | 5.844 crop DINO+SAM | ❌ crop, bukan frame |

**Konverter (dibuat 2026-10-21).**

```
Temp/opencode/build_riset4_golden_to_coco.py    (--dry-run untuk cek tanpa tulis)
        │  baca koordinat dari candidates_index.csv (bypass golden_set_v3.csv yang korup)
        │  join ke Hasil-2/output/frames_sampled/frame_XXXXXX.png
        ▼
Riset-4/output/foundation_model/golden_set_v3_coco.json   (68,5 KB)
```

Hasil nyata: **199 anotasi / 182 gambar** (200 baris − 1 `LampuAtap` yang di-drop).
Distribusi: `LampuMobil` 121 · `LampuMotor` 77 · `PantulanCahaya` **1**.
0 koordinat hilang, 0 frame hilang, resolusi COCO cocok dengan file nyata (640×480).

**Batas pemanfaatan.** 199 box di 182 frame **terlalu sedikit** untuk fine-tune DINO
sendirian (butuh 1.000+). Layak sebagai **tambahan** ke dataset NB02b, bukan pengganti.
`PantulanCahaya` hanya **1 sampel** — tidak cukup untuk apa pun.

**Temuan penting untuk keputusan kelas pantulan (Riset-HampirMatang).** Riset-4
menolak kelas `bukan_lampu` berdasarkan data: dari 500 sampel yang diperiksa manual,
hanya **2 yang benar-benar bukan lampu** → FP rate kandidat DINO ≈ **0,4–0,8%**.
Implikasi: jumlah pantulan yang lolos deteksi di Riset-HampirMatang mungkin juga kecil,
sehingga kelas YOLO terpisah untuknya berisiko overfit. Karena itu NB02b memakai
mekanisme `kind: distractor` (§10.0) untuk **mengukur dulu**, bukan langsung membuat kelas.

---

### 10.6 Audit Keselarasan Taksonomi & Kontrak Hitung (2026-10-26)

**Pemicu.** Pemilik riset mengoreksi tujuan: yang dideteksi adalah **kondisi
lampu** (silau / berwarna / silau **dan** berwarna), dan keluarannya harus berupa
**sistem hitung**, bukan sekadar deteksi.

| Keluaran wajib | Satuan |
|---|---|
| Jumlah kendaraan mobil | kendaraan |
| Jumlah kendaraan motor | kendaraan |
| Jumlah lampu silau — mobil | kendaraan/lampu (§13.10) |
| Jumlah lampu silau — motor | kendaraan/lampu |
| Jumlah lampu berwarna — mobil | kendaraan/lampu |
| Jumlah lampu berwarna — motor | kendaraan/lampu |
| Jumlah lampu silau + berwarna — mobil | kendaraan/lampu |
| Jumlah lampu silau + berwarna — motor | kendaraan/lampu |

**Bukti audit per notebook:**

| NB | Bukti di kode/artefak | Selaras? |
|---|---|---|
| NB01 | Akuisisi frame + ROI polygon; tidak menyentuh kelas | ✅ netral |
| NB02b | `builders/build_nb02b_annotation.py:74` → `CLASS_NAMES = ["LampuMobil", "LampuMotor", "PantulanCahaya"]`; hotkey `1/2/3` + `d`; store 126 frame reviewed = **946 box** (182 mobil / 447 motor / 317 pantulan) | ❌ |
| NB02 | `builders/build_nb02_pseudo_label.py:75` → `CLASS_PROMPTS = [("LampuMobil", "car headlight."), ("LampuMotor", "motorcycle headlight.")]` | ❌ |
| NB03 | Kalibrasi threshold terhadap gold mobil/motor; F1 makro train 0,334 @0,30 | ❌ |
| NB04 | `build_nb04_train_yolo.py:94` + `dataset/data.yaml` `nc: 3`; split 660/182/640 | ❌ |
| NB05 | `COUNTED_CLASS_NAMES = ["LampuMobil", "LampuMotor"]` + `count_vehicles()` + filter indeks 2 (R11) | ❌ |

**Kesimpulan.** Ini **salah taksonomi pada level target**, bukan bug kecil. Yang
ada sekarang menjawab "berapa mobil dan motor", bukan "berapa lampu silau /
berwarna / silau+berwarna per jenis kendaraan". Akibatnya:

- Target `berwarna` dan `silau+berwarna` **tidak ada** di data, pseudo-label,
  model, maupun metrik.
- `PantulanCahaya` (317 box) **bukan** salah satu dari tiga target riset;
  posisinya hanya *hard-negative*, bukan objek yang dilaporkan.
- Seluruh angka NB03–NB05 saat ini **tidak boleh** dikutip sebagai hasil riset.

**Kontrak hitung yang diusulkan** (mekanis, tanpa asosiasi tambahan). Kelas
menyandikan **jenis × kondisi**, sehingga setiap angka di tabel atas hanya
penjumlahan:

| Kelas | Masuk "kendaraan mobil" | Masuk "kendaraan motor" | Masuk hitung silau | berwarna | silau+berwarna |
|---|---|---|---|---|---|
| `MobilNormal` | ✅ | — | — | — | — |
| `MobilSilau` | ✅ | — | ✅ | — | — |
| `MobilBerwarna` | ✅ | — | — | ✅ | — |
| `MobilSilauBerwarna` | ✅ | — | ✅ (rollup) | ✅ (rollup) | ✅ |
| `MotorNormal` | — | ✅ | — | — | — |
| `MotorSilau` | — | ✅ | ✅ | — | — |
| `MotorBerwarna` | — | ✅ | — | ✅ | — |
| `MotorSilauBerwarna` | — | ✅ | ✅ (rollup) | ✅ (rollup) | ✅ |
| `PantulanCahaya` | — | — | — | — | — |

**Urutan perbaikan (jangan dibalik).**

1. Tetapkan §13.9 (label space) dan §13.10 (satuan hitung).
2. Tetapkan definisi operasional: ambang "silau", daftar warna "berwarna",
   aturan tie-break bila ragu — keputusan pemilik riset.
3. Putuskan nasib 126 frame yang sudah dianotasi (opsi di §13.9).
4. Ubah `CLASS_PROMPTS` NB02 → frasa kondisi lampu, lalu **re-run NB02** (GPU).
5. Regenerasi NB02b (kelas + hotkey), NB04 (`data.yaml` + materialisasi),
   NB03 (kalibrasi), NB05 (kontrak hitung di atas).
6. Baru jalankan training penuh dan laporkan angka.

Sampai langkah 1–2 selesai: **jangan** latih NB04 dan **jangan** kutip angka NB05.

### 10.6.1 Tindak Lanjut Audit 2026-10-28

Audit 19 temuan (`AUDIT_ACTION_2026-10-28.md`) menemukan satu kebocoran makna: NB05
dan NB06 menerbitkan kunci JSON yang sama (`jumlah_kendaraan_*`) dengan satuan
berbeda. Perbaikannya:

- **NB05** menerbitkan `box_totals` dengan kunci `jumlah_box_*` + penanda
  `counting_unit: "box_per_frame"`. Sel hitung tetap ada sebagai pemeriksaan
  cakupan/keseimbangan kelas, **bukan** angka populasi.
- **NB06** tetap satu-satunya pemilik `counting_totals` / `jumlah_kendaraan_*`
  (kendaraan unik). `smoke_pipeline.py` memvalidasi kedua kontrak secara terpisah.

Sumber tunggal yang baru (`pipeline_common.py`): `CLASS_NAMES`,
`SPLIT_NAME_BY_CAMERA`, `COUNT_CONTRACT`, `BOX_COUNT_CONTRACT`,
`stitch_track_fragments`, `validate_tracker_thresholds`, konstanta `FRAGMENT_*`.
Seluruh builder mengimpornya; tidak ada lagi salinan lokal. Logika tersebut dites
tanpa GPU lewat `test_pipeline_logic.py`.

---

### 10.7 NB06 — Hitung Kendaraan Unik (Ditambahkan 2026-10-26)

**Pemicu.** §13.11. NB05 menjumlahkan box per **frame sampel** (1 dari setiap 15
frame), sehingga satu kendaraan muncul di puluhan frame dan terhitung puluhan
kali. Pemilik riset menetapkan satuan yang benar: **satu kendaraan dihitung satu
kali, sejak muncul di dalam ROI sampai keluar dari ROI.**

**Posisi stage.**

```
NB02 (lokalisasi lampu) -> NB02b (anotasi gold) -> NB03 (audit + fotometri)
  -> NB04 (training) -> NB05 (kalibrasi ambang + baseline per frame)
  -> NB06 (kendaraan unik)  <- angka resmi "jumlah kendaraan"
```

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
| 7. Hitung | kelas per track + kontrak §10.6 | 8 angka |

**Rantai koordinat.** Crop memakai bbox polygon yang area luarnya dihitamkan
(persis NB02), lalu `thumbnail(640)`. Pemetaan balik memakai
`frame = offset + crop / scale`. Skala saat ini 1,0 karena semua crop lebih kecil
dari 640 px, tetapi rumusnya tetap umum. Bukti empiris 2026-10-26: resolusi video
= resolusi PNG frame (704×576 / 640×480 / 640×360), dan crop NB06 = crop NB02
(412×377, 425×303, 315×184) dengan offset yang cocok.

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
6. Angka dijumlahkan dengan kontrak §10.6. `PantulanCahaya` tetap hard-negative:
   tercatat sebagai track, tetapi nol di kedelapan angka.

**Konfigurasi (nilai saat ini).**

| Konstanta | Nilai | Alasan |
|---|---|---|
| `COUNTED_SPLITS` | `["test"]` | keputusan pemilik riset 2026-10-26: hanya kamera yang tidak pernah dilihat model (`bappenda2`, `pdam2`, 29.950 frame) yang dilaporkan sebagai angka resmi |
| `FRAME_STRIDE` | 1 | setiap frame; tracking tidak boleh melompat |
| `IMAGE_SIZE` | 640 | sama dengan training NB04 dan kalibrasi NB05. Terbukti penting: pada `imgsz=320` deteksi di crop malam hilang total |
| `CONF_FLOOR` / `NMS_IOU` | 0,05 / 0,70 | sama dengan NB05, penyaringan akhir tetap per kelas |
| `TRACKER_CONFIG` | `builders/bytetrack_nb06.yaml` | lihat §13.12 |
| `MIN_TRACK_HITS` | 3 (smoke 2) | 0,12 detik pada 25 fps; membuang kedipan 1–2 frame |
| `FRAGMENT_MAX_GAP_FRAMES` | 50 | 2 detik; batas jarak waktu track pecahan boleh disambung |
| `FRAGMENT_MIN_IOU` | 0,50 | fragmen harus menempati kotak yang sama (tumpang tindih ≥50%) |
| `FRAGMENT_DISTANCE_RATIO` | 0,35 | alternatif IoU: pusat box berpindah < 0,35 × diagonal box |
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
| `counting_by_video.csv` | 8 angka kontrak per video + `vehicles_counted` + sembilan kolom kelas |
| `counting_summary.json` | agregat + `duplication_factor_vs_per_frame` + ambang yang dipakai |
| `fragment_merges.csv` | daftar kendaraan yang terbentuk dari >1 track (bukti penyambungan, untuk diperiksa manual) |
| `roi_mapping_audit.csv` | bukti offset/scale/crop per video |
| `counting_bar.png` | grafik 8 angka |

**Status verifikasi (2026-10-26).**

| Uji | Cakupan | Hasil |
|---|---|---|
| Smoke `builders/smoke_pipeline.py` | NB02 → NB06 di `output/smoke/` | ✅ exit 0, semua artefak terbentuk, kedua kontrak hitung konsisten |
| Uji logika sintetis | mayoritas, tie-break, `MIN_TRACK_HITS`, rollup pantulan, 4 kasus yang **harus ditolak** (motor vs mobil, berjauhan, dua track serentak) | ✅ |
| Uji integrasi sel TRACK+COUNT | 60 frame video asli `bappenda2`, ambang & `MIN_TRACK_HITS` sengaja diturunkan untuk probe | ✅ 151 deteksi → 151 `track_id` → hitung per track jalan, 0 fragmen disambung (semua track serentak, benar tidak disambung) |
| Uji API `model.track` CPU | yolo11n COCO + yaml kustom + source PIL | ✅ `boxes.id` terisi; `imgsz=320` deteksi hilang total |

Angka smoke seluruhnya nol karena bobot smoke hanya 1 epoch pada 7 gambar palsu: model
menghasilkan 877 box pada `conf=0.001` tetapi 0 box pada `conf=0.05`, jadi tidak ada
yang lolos `CONF_FLOOR`. Itu kelemahan bobot, bukan kegagalan pipeline — dan pipeline
melaporkannya lewat `untracked_detections`, bukan menghitung nol tanpa suara.

### 10.7.1 Jaminan Identitas — Apa yang Dijamin dan Apa yang Tidak

Pertanyaan pemilik riset 2026-10-26: *"kalau mobil A masuk ROI, pasti terhitung
1? Semua kendaraan punya ID berbeda sehingga tidak terhitung 2×?"*

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
| Kendaraan hilang dari pandangan lebih lama dari `track_buffer` (30 frame = 1,2 detik) lalu muncul lagi | ID baru → **terhitung 2×** | tergantung occlusion (tiang, kendaraan besar di depan) |
| Dua kendaraan sangat berdekatan/berpapasan dan box-nya bertukar | ID switch → satu kendaraan bisa terhitung 2× atau dua kendaraan menyatu jadi 1 | malam + lampu menyilaukan: risiko nyata |
| Kendaraan keluar ROI lalu masuk lagi | ID baru → terhitung 2× | jarang, tetapi mungkin di simpang |
| Kendaraan hanya terlihat 1–2 frame | dibuang `MIN_TRACK_HITS` → **tidak** terhitung | sering untuk kendaraan cepat di tepi ROI |
| Video berbeda | tracker di-reset → kendaraan yang sama terhitung lagi | pasti, karena tidak ada identitas lintas sesi |

**Konsekuensi penulisan laporan.** Angka NB06 adalah **jumlah kejadian kendaraan
terdeteksi di dalam ROI dan dilacak**, bukan jumlah kendaraan fisik di jalan.
Kalimat itu wajib ada, dan `duplication_factor_vs_per_frame` disertakan sebagai
pembanding agar pembaca melihat asumsi apa yang sedang dipakai.

**Mitigasi yang sudah terpasang (2026-10-26).**

1. **Penyambungan fragmen.** Sebelum dihitung, track yang terputus disambung
   kembali bila: jenisnya sama (mobil/motor), fragmen baru mulai **minimal 1 frame
   setelah** fragmen sebelumnya berakhir dan tidak lebih dari
   `FRAGMENT_MAX_GAP_FRAMES` (50 frame = 2 detik), serta box-nya bertumpuk
   (IoU ≥ 0,50) atau pusatnya berpindah < 0,35 × diagonal box. Syarat "minimal
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

Format notebook: `NB<studi>_<nama_tahap>.ipynb` — contoh `NB02_pseudo_label.ipynb`

---

## 12. Status Eksperimen Saat Ini

| Eksperimen | Model | Dataset | Status |
|---|---|---|---|
| Pseudo-labelling NB02 | GroundingDINO-tiny | `output/frames/` + ROI polygon final | ✅ Selesai 2026-10-26 (1.548 crop unik, 4.309 deteksi, 0 gagal) |
| Training YOLO | YOLOv11n | NB04 (gold 9 kelas) | 🔧 Dibangun, menunggu anotasi NB02b |
| Hitung kendaraan unik NB06 | YOLOv11n + ByteTrack | video asli kamera test (29.950 frame) | 🟡 Smoke NB02→NB06 lolos 2026-10-26, menunggu run penuh |
| Fine-Grained Matching | ResNet50 | NB04 | ❌ Dipertimbangkan ulang (§13.9 sudah dijawab: tidak perlu) |

---

## 13. Keputusan Terbuka

1. ~~Mode pseudo-labelling (A/B)~~ — **Sudah diputuskan: Mode B (DINO-only).**
2. ~~ROI~~ — **Sudah final & disetujui user 2026-10-18** (5 polygon, geometri
   tervalidasi, cek visual lolos).
3. ~~Taksonomi 2 vs 3+ kelas~~ — **Sudah diputuskan: 2 kelas.**
4. ~~Backup~~ — **Sudah dilakukan.**
5. ~~Definisi target deteksi~~ — **Sudah diputuskan: Mode A (lampu-sepasang).**
6. ~~Model DINO (tiny vs base)~~ — **Sudah diputuskan: `grounding-dino-tiny`
   untuk run pertama.** Jalur upgrade ke `grounding-dino-base` bila hasil
   anotasi di NB03 kurang bagus (lihat §5.5).
7. **Nilai konstanta grouping** — masih nilai awal, dikalibrasi di NB03.
8. **Apakah CLAHE perlu dipakai** — saat ini NB02 berjalan tanpa CLAHE
   (frame asli + ROI). Bila dianggap perlu, itu perubahan desain baru.
9. ~~Label space untuk target silau/berwarna~~ — **Sudah diputuskan user:
   9 kelas** (`{Mobil,Motor} × {Normal,Silau,Berwarna,SilauBerwarna}` +
   `PantulanCahaya` sebagai hard-negative). 946 box lama tidak dipakai apa
   adanya; NB02b menganotasi ulang pada label space ini.
10. ~~Satuan hitung "jumlah lampu"~~ — **Sudah diputuskan: (a) per kendaraan**
    (1 box = 1 kendaraan hasil union sepasang lampu). Kontrak penjumlahan
    angka ada di §10.6.
11. ~~Satuan "jumlah kendaraan" pada frame tersampling~~ — **Sudah diputuskan:
    (b) tracker pada frame berurutan.** Konsekuensinya NB05 tetap melaporkan
    *jumlah deteksi per frame* sebagai baseline dan **tidak boleh** dilabeli
    "jumlah kendaraan". Angka resmi kendaraan unik ada di **NB06**
    (`output/nb06/`), yang membaca video asli di `output/videos/` dan
    menjalankan ByteTrack di dalam crop ROI yang sama dengan NB02.
    Aturan hitung: satu `track_id` = satu kendaraan, dihitung sekali sejak
    muncul di dalam ROI; kelas per track = suara mayoritas; track di-reset per
    video. Batas: kendaraan yang keluar lalu masuk lagi terhitung dua kali, dan
    tidak ada identitas lintas sesi (lihat NB06).
12. **Ambang internal tracker** — ByteTrack bawaan memakai
    `new_track_thresh: 0.25`, sehingga box ber-confidence di bawah 0,25 tidak
    pernah bisa membentuk track baru dan kelas langka akan undercount
    sistematis. NB06 memakai `builders/bytetrack_nb06.yaml`
    (`new_track_thresh 0.05`, `track_high_thresh 0.05`, `track_low_thresh 0.02`,
    `track_buffer 30`), yaitu lantai yang sama dengan `CONF_FLOOR` NB06.
    Aturan yang mengikat: `new_track_thresh ≤ min(ambang per kelas)`; sel JOBS
    NB06 menolak jalan bila aturan ini dilanggar. Penyaringan akhir tetap
    dilakukan ambang per kelas hasil NB05 saat agregasi track, jadi tracker
    hanya mengurus asosiasi. Bila seluruh deteksi gagal mendapat `track_id`,
    NB06 berhenti dengan pesan eksplisit alih-alih melaporkan nol kendaraan.

> Referensi teknis hasil ekstraksi arsip kursus CV (rumus HSV, skor blur,
> luas bloom, NMS, GradCAM, dan pola yang ditolak): **`KNOWLEDGE_PIPELINE_CV.md`**.

### 13.9a Definisi Operasional Label (Diputuskan 2026-10-28)

Empat keputusan pemilik riset, menutup §13 butir 9 dan blok "belum ditetapkan" di §1:

| Aspek | Keputusan |
|---|---|
| Dasar penilaian `Silau` | **acuan contoh visual**, bukan ambang fotometrik |
| Cakupan yang dinilai | hanya **lampu depan**; lampu rem/sein/belakang tidak dinilai warnanya |
| Warna yang dihitung `Berwarna` | hanya **biru/cyan** (hue OpenCV 95–135) |
| Kasus ragu | `BelumDitentukan`, memblokir export |

**Kenapa silau bukan ambang.** Diukur 2026-10-28 pada gold set saat itu (78 frame
reviewed, 1.125 box aktif) memakai `builders/probe_label_definition.py`; CSV mentah di
`output/annotation/label_probe.csv`.

| Fitur | Ambang terbaik | Silau tertangkap | Normal ikut tertandai | Pantulan ikut |
|---|---|---|---|---|
| `v240` (konstanta NB03) | 0,10 | 0,89 | 0,64 | 0,60 |
| `v200` | 0,21 | 0,89 | 0,55 | 0,61 |
| `v180` | 0,29 | 0,89 | 0,54 | 0,61 |
| `white_bloom_ratio` (V≥200, S<40) | 0,22 | 0,82 | 0,42 | 0,52 |
| ring 1,6× di luar box (`ring_v180`) | 0,01 | 0,91 | 0,85 | 0,87 |

Median `v240` bahkan terbalik: `MobilSilau` 0,139 vs `MobilNormal` 0,147. Tiga sebab:
box ikut melebar mengikuti bloom sehingga rasio di dalam box ternormalisasi sendiri;
kompresi H.264 memotong highlight sebelum sampai ke crop; dan yang dinilai manusia
adalah **bentuk** luapan, bukan distribusi nilai piksel. Karena itu `bright_ratio` tetap
dipakai sebagai **diagnostik** NB03, **bukan** sebagai definisi label.

**Aturan silau.** `Silau` bila ada halo/lingkaran putih yang keluar melewati batas fisik
lampu (menutupi bodi, plat, atau aspal tepat di sekitar lampu) **atau** ada streak/flare
memanjang dari inti lampu. Lampu terang dengan batas jelas **bukan** silau. Cahaya yang
jatuh di aspal atau bodi = `PantulanCahaya`.

**Aturan berwarna.** Piksel berwarna = `S ≥ 80`, `V ≥ 80`, hue 95–135; minimal 25% luas
box harus piksel tersebut. Merah, oranye, kuning, hijau, dan ungu tidak dihitung pada
revisi ini.

**Konsekuensi data yang harus dibaca sebelum training.**

| Kelas | Box | Catatan |
|---|---|---|
| `PantulanCahaya` | 561 | 49,9% gold set — kelas terbesar, hard-negative |
| `MotorSilau` | 229 | |
| `MotorNormal` | 181 | |
| `MobilNormal` | 99 | |
| `MobilSilau` | 42 | |
| `MotorBerwarna` | 4 | semua hue 110–120 (biru) |
| `MotorSilauBerwarna` | 2 | hue 110–120 |
| `MobilBerwarna` | 0 | |
| `MobilSilauBerwarna` | 0 | |

Dari 93 box yang pikselnya berwarna kuat (`colored_ratio ≥ 0,25`), **79 adalah
`PantulanCahaya`** dengan hue 0 (merah) — pantulan lampu rem di aspal. Inilah alasan
aturan "hanya lampu depan" dan "hanya biru": tanpa keduanya, `Berwarna` akan didominasi
pantulan dan lampu rem.

> **Kendaraan arah menjauh — pertanyaan tertutup (2026-10-28).** ROI tiap kamera sudah
> disesuaikan pemilik riset ke arah jalan sehingga frame hanya memuat tampak depan
> kendaraan. Tidak ada kendaraan yang hanya tampak belakang di data, jadi aturan khusus
> tidak diperlukan dan "jumlah kendaraan" tidak kehilangan arah menjauh.

**Hasil mining kandidat biru (2026-10-28).** `builders/mine_blue_candidates.py` mengukur
seluruh 4.309 box pseudo-label di 1.474 crop, ambang definisi (`S≥80`, `V≥80`, hue 95–135):

| Tahap | Box | Frame |
|---|---|---|
| kandidat longgar (biru ≥ 25% luas box) | 81 | 69 |
| kandidat ketat (inti terang `V≥200` didominasi biru ≥ 50%) | 20 | 20 |

Sebaran kandidat ketat per split: **test 18 · train 2 · val 0**. Rinciannya `pdam2` 17,
`bappenda1` 2, `bappenda2` 1. Artinya dari 20 kandidat, **hanya 2 yang boleh dipakai
latihan**, dan keduanya dari satu kamera (`bappenda1`) dengan dua frame berdekatan
(005580, 006480) — kemungkinan besar kendaraan yang sama.

**Kenapa hue saja tidak cukup untuk menyatakan "lampu biru".** Ditambahkan kolom
`frame_blue_share` = fraksi piksel terang seluruh frame yang berhue biru, untuk mendeteksi
auto-white-balance yang bergeser:

| Frame | `frame_blue_share` | Tafsir |
|---|---|---|
| `bappenda2` 013560 | 0,998 | seluruh frame biru → artefak AWB, bukan lampu |
| `bappenda1` 005580 | 0,992 | seluruh frame biru → artefak AWB |
| `pdam2` 004665 | 0,86 | bingkai ikut biru → artefak AWB |
| `pdam2` 000690 | 0,83 | bingkai ikut biru → artefak AWB |
| `bappenda1` 006480 | 0,18 | frame hangat, inti box biru → kandidat meyakinkan |
| `pdam2` 010980–011340 (10 frame) | 0,23–0,39 | frame hangat, inti box biru → kandidat paling meyakinkan |

Dari 20 kandidat ketat, **4 box ditandai `awb_suspect`** (frame biru menyeluruh). Sisanya
bertahan sebagai kandidat, tetapi pada ukuran box 25–60 px batas antara "lampu putih dengan
halo kebiruan" dan "lampu biru" masih tipis — dan hampir semuanya ada di kamera `test`
(`pdam2`), yang menurut protokol split **tidak boleh** dipakai latihan.

**Kesimpulan yang harus dipakai untuk perencanaan.** Mining berbasis hue berhasil
menemukan kandidat, tetapi **tidak** menghasilkan data latih yang cukup: 2 box train,
0 box val, dan keduanya satu kamera. Dengan 1.548 crop yang ada, kelas `Berwarna` dan
`SilauBerwarna` tidak dapat dilatih maupun dievaluasi secara sah.

**Sisa pekerjaan definisi (urutan).**

1. Bangun set contoh `output/annotation/reference_patches/`: ±30 patch silau + ±30 patch
   normal yang disetujui pemilik riset. Sampai ini ada, definisi silau **belum
   operasional penuh** dan label silau yang lama belum bisa diaudit.
2. Uji konsistensi: tampilkan 30 crop tanpa label lama, ukur kesepakatan dengan label
   sebelumnya; target ≥ 80%. Bila di bawah itu, aturan silau perlu dipertajam.
3. ~~Tetapkan aturan kendaraan yang hanya tampak belakang.~~ **Selesai 2026-10-28**: ROI
   hanya memuat tampak depan.
4. ~~Tambah anotasi terarah untuk kelas biru dengan mining kandidat.~~ **Selesai
   2026-10-28**, hasil: 20 kandidat ketat, 2 di train, 0 di val — **tidak cukup**.
5. **Keputusan strategi kelas berwarna** (menunggu pemilik riset): turunkan target latih ke
   4 kelas `{Mobil,Motor} × {Normal,Silau}` dan laporkan `Berwarna`/`SilauBerwarna` sebagai
   batasan riset; atau tambah sesi rekaman baru yang memuat kendaraan lampu biru.
6. ~~Bangun set contoh `output/annotation/reference_patches/`.~~ **Bagian selesai 2026-10-28**:
   builder `builders/build_reference_patches.py` menghasilkan 120 patch (30 silau Mobil/Motor
   + 30 normal Mobil/Motor, ≤30 per kelas) di `output/annotation/reference_patches/` + index.
   **Perlu persetujuan pemilik riset** sebelum dipakai sebagai acuan resmi (target relabel ≥ 80%).
7. Uji konsistensi relabel patch silau terhadap label lama; Bila di bawah 80%, pertajam aturan.

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
| Box pseudo yang dihapus pemilik riset | 396 (≈31% proposal DINO) |
| **Proxy recall proposal DINO (lampu kendaraan)** | **0,606** |
| Pasangan box duplikat (IoU ≥ 0,85) | 0 |
| Box dengan sisi < 8 px | 1 |
| Box menyentuh tepi crop | 28 |
| Refleksi yang menumpuk di atas box kendaraan | 0 |

Kesimpulan: kritik Codex soal **recall proposal DINO benar dan terukur** (0,606, jauh di bawah
ambang go/no-go 0,80 yang mereka usulkan), tetapi kritik soal **mutu anotasi tidak berlaku**
pada 170 frame yang sudah dikerjakan — tidak ada duplikat, tidak ada refleksi yang menumpuk,
hampir tidak ada box tak-terlathi. Artinya pemilik riset sudah bekerja pada **jalur
manual-first** (1.017 dari 1.897 box digambar sendiri), sehingga rekomendasi utama Codex
("pindah ke manual-first") sudah menjadi praktik, bukan rencana.

**Klaim Codex yang tidak sesuai kondisi repo** (dipakai sebagai peringatan, bukan acuan):

1. "Keputusan label final **9 kelas** dan kode NB02b–NB06 selaras" — repo memakai **9 kelas
   anotasi + 5 kelas latih** (`TRAINING_CLASS_NAMES`); `pipeline_common.py` yang mereka salin
   sendiri pun berisi kontrak 5 kelas, dan NB04 mereka memakai `nc: 5`.
2. "`MobilBerwarna`/`MobilSilauBerwarna` belum punya box" — saat audit: 5 dan 7 box (semua
   `train`, 0 `val`). Klaim itu berasal dari state sebelum gold set bertambah.
3. "NB02b mengambil 1 dari 15 frame" — stride 15 frame ada di **NB01**
   (`SAMPLE_INTERVAL_FRAMES`); NB02b memakai kuota per kamera + `random.sample` atas 871
   kandidat.

**Yang diterapkan 2026-10-29.**

| Perubahan | Alasan |
|---|---|
| Label `?` pada preview NB02 → **`UNSURE`** (nilai di `nb02_summary.json` tetap `TidakPasti`) | `?` terbaca seperti nama kelas; `UNSURE` menyatakan status, bukan taksonomi |
| `DETECTION_THRESHOLD` 0,30 → **0,25**; `VEHICLE_SCORE_THRESHOLD` 0,40 → **0,30** | recall proposal 0,606 adalah bottleneck terukur; melonggarkan proposal lebih murah daripada kehilangan kendaraan |
| `VEHICLE_VOTE_MIN_COVERAGE` **tetap 0,30** (Codex mengusulkan 0,15) | coverage = irisan box kendaraan ÷ box lampu; 0,15 berarti box kendaraan hanya menutup 15% box lampu — itu kendaraan lain, bukan bukti jenis. Menurunkannya menambah label jenis yang salah, bukan menambah recall |
| Gate NB04: **training ditolak** bila ada kelas `TRAINING_CLASS_NAMES` dengan < 20 box `train` | kelas kosong menghasilkan mAP yang tidak bisa ditafsirkan |
| Gate NB04: **peringatan** bila kelas < 5 box `val` (tidak memblokir) | AP dari < 5 box adalah estimasi lemah, cukup dilaporkan, bukan alasan menghentikan run |
| `builders/audit_gold_quality.py` | versi murah dari "naikkan kualitas anotasi": duplikat, box kecil, box tepi, refleksi tumpang tindih, proxy recall — tanpa biaya anotator kedua |
| NB05: kontrak `jumlah_box_lampu_berwarna_*` dan `jumlah_box_lampu_silau_dan_berwarna_*` kini **`n/a`**, bukan `0` | angka 0 terbaca sebagai "diukur, hasilnya nol"; padahal model memang tidak punya kelas itu (`UNMEASURABLE_BOX_COUNT_CONTRACT_KEYS`) |

**Yang ditolak atau ditunda, dengan alasan.**

| Saran | Keputusan |
|---|---|
| Sampling bin temporal di NB02b (`CONTOH-CODEX`) | **Ditolak.** `sample_manifest.json` dipakai ulang oleh guard; mengganti algoritme mengharuskan manifest dihapus, dan export hanya membaca frame yang ada di manifest → 170 frame reviewed **keluar dari export** |
| `run_manifest.json`, lock file ber-hash, CLI headless, dashboard observability | **Ditunda.** Baik untuk rekayasa, tidak mengubah angka riset; masuk daftar future work laporan |
| `audit_test_gold`, bootstrap CI per kamera/sesi, 2 anotator + kappa | **Ditunda.** Butuh sumber daya yang tidak ada (anotator tunggal); dilaporkan sebagai batasan |
| Gold counting clip untuk validasi ByteTrack | **Ditunda** sampai model lulus training; ini pekerjaan terbesar yang tersisa untuk 8 angka |
| Bobot hash `requirements.txt` + pemeriksaan CUDA saat startup | **Ditunda** bersama `run_manifest` |

**Gate publikasi yang disetujui** (dari `PERBAIKAN.md`, dipakai apa adanya): angka final tidak
diterbitkan bila (a) kelas `Berwarna`/`SilauBerwarna` dilaporkan sebagai hasil model, (b) split
`test` dipakai memilih threshold, (c) tidak ada metrik per kamera dan per kelas, (d) artefak
berasal dari smoke mode, atau (e) label masih memuat `BelumDitentukan`.

---

## 14. Referensi

- Dataset: CCTV Kab Bogor — Raya Lintang Kita (Rawamurang) 2026-08-01
- Model: GroundingDINO-tiny (swin-t)
- Dokumentasi Grounded-SAM resmi: https://github.com/IDEA-Research/Grounded-SAM
  (dikutip 2026-10-15)
- Riset terkait: Grounded-SAM (lihat RISK_REGISTER.md §5)
