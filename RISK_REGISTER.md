# Risk Register — Riset-HampirMatang

> Dokumen manajemen risiko pipeline pseudo-labeling CCTV (Arsitektur V2: Grounding DINO → YOLO).
> Disusun **sebelum** perbaikan isu B1–H10, sebagai kontrol perubahan dan acuan gate.
> Sumber temuan: review 8 agent (code-reviewer, clean-code-engineer, python-debug-engineer, data-science-advanced, machine-learning-terapan, deep-learning-advanced, docs-writer, test-engineer).

| Field | Nilai |
|---|---|
| Proyek | Riset-HampirMatang (CCTV Kab. Bogor → Grounding DINO → YOLO 2 kelas) |
| Versi dokumen | 1.17 |
| Tanggal | 2026-10-26 |
| Status pipeline saat ini | NB01 selesai · ROI final & disetujui · Model `grounding-dino-tiny` · **NB02 selesai (1.482 gambar / 2.018 box YOLO) — output ditandai `UNSplit`, peta split ditulis** · **NB02b (anotasi manual) dibuat — bebas kamera `test`, punya 3 kelas termasuk `PantulanCahaya` (hard-negative) + toggle label** · Konverter `golden_set_v3` Riset-4 → COCO dibuat (199 box / 182 frame) · **NB03–NB05 dibuat & diverifikasi 2026-10-26** (NB03 jalan pada gold 126 frame; NB04 materialisasi split 660/182/640 selesai, training tertahan guard karena val belum dianotasi; NB05 lolos smoke test dengan bobot 1-epoch) · ⛔ **Audit 2026-10-26: taksonomi TIDAK SELARAS dengan tujuan riset** (target = silau / berwarna / silau+berwarna + sistem hitung; kode masih jenis kendaraan) — lihat **R14** |
| Cakupan | B1–H10 (temuan review) + risiko proyek (dokumentasi, testing, reproducibility, metode FM) |
| Pemilik | User (pemilik riset) |
| Aturan | Perbaikan kode TIDAK dimulai sebelum kontrol §6 dijalankan |

---

## 1. Tujuan & cakupan

Dokumen ini mengelola dua jenis risiko:

1. **Risiko kondisi saat ini** — kerusakan yang ditimbulkan isu B1–H10 bila dibiarkan.
2. **Risiko perubahan** — kerusakan yang bisa ditimbulkan oleh *proses perbaikan itu sendiri* (regresi, kehilangan data, pemborosan jam GPU, kehilangan reproducibility).

Pendekatan: identifikasi → penilaian → mitigasi → rencana urutan eksekusi → rollback → gate verifikasi.

---

## 2. Metode penilaian

**Likelihood (L)** — peluang risiko terjadi bila tidak dikendalikan:

| Skor | Label |
|---|---|
| 1 | Sangat jarang |
| 2 | Jarang |
| 3 | Mungkin |
| 4 | Kemungkinan besar |
| 5 | Hampir pasti |

**Impact (I)** — besarnya kerusakan bila terjadi:

| Skor | Label | Arti |
|---|---|---|
| 1 | Minor | Mengganggu, mudah dipulihkan |
| 2 | Kecil | Output sebagian salah, terlihat |
| 3 | Sedang | Output salah senyap, bisa dilacak |
| 4 | Besar | Hasil riset/latih tidak valid, perlu rerun |
| 5 | Katastrofik | Tujuan riset gagal / data rusak tak terpulihkan |

**Severity = L × I**

| Rentang | Level |
|---|---|
| 1–5 | 🟢 Rendah |
| 6–12 | 🟡 Sedang |
| 13–19 | 🟠 Tinggi |
| 20–25 | 🔴 Kritis |

---

## 3. Matriks risiko

| | I=1 | I=2 | I=3 | I=4 | I=5 |
|---|---|---|---|---|---|
| **L=5** | | H3, H6 | P2 | | **B1, B2** |
| **L=4** | | P7 | P1 | H1, H2, H4, H7, H8, P4, P5, P8 | |
| **L=3** | | | H10, P3 | P6 | |
| **L=2** | | | | | |
| **L=1** | | | | | |

> P1 (dokumentasi) L4×I3=12, P3 (reproducibility) L3×I3=9, P5 (validitas eksternal data) L4×I4=16, P7 (clean-code debt) L4×I2=8, P4 dinaikkan ke 16. Semua disertakan pada register di bawah.

---

## 4. Register risiko

| ID | Risiko | Kategori | L | I | Skor | Level | Mitigasi utama |
|---|---|---|---|---|---|---|---|
| **B1** | Split leakage: `data.yaml` `train==val`, `group_split.csv` tak dipakai → latih di kamera test | Validitas evaluasi | 5 | 5 | 25 | 🔴 Kritis | Materialisasi `images/{train,val,test}` dari `group_split.csv`; assert bebas kebocoran. **Untuk NB02b sudah ditutup 2026-10-20 (lihat R6):** sampling dan export membuang split `test` + guard `assert_no_excluded_split()`. **Sisa terbuka:** materialisasi struktur `images/{train,val,test}` untuk NB04 masih belum dibuat. **Diperbarui 2026-10-26:** materialisasi selesai di NB04 (`output/nb04/dataset`, 660/182/640, gambar:label 1:1, kamera test tidak pernah masuk train/val, audit ditulis ke `split_audit.csv`); sisa risiko turun ke kualitas label split `test` (lihat R13) |
| **B2** | Artefak disk STALE & cacat (204 frame, bappenda3 = 1 frame/0 deteksi); fix belum re-run | Integritas data | 5 | 5 | 25 | 🔴 Kritis | Hapus `output/nb02` penuh lalu re-run setelah semua fix; tandai artefak sebagai stale |
| **H1** | Label kelas dari token BPE noisy (`##light`) + substring `motorcycle` → kelas salah sistematis | Integritas label | 4 | 4 | 16 | 🟠 Tinggi | **Diperbaiki 2026-10-15 (reframed)** — multi-pass DINO per frasa, klasifikasi via **geometri kelompok lampu (A)**, bukan dari `text_labels`. **Diperkuat 2026-10-19** — vote verifier FRCNN sekunder + flag `signals_agree`. Perlu validasi NB03. |
| **H3** | Mask SAM dihitung (~1–2 jam GPU) tapi tak dipakai | Efisiensi/biaya | 5 | 2 | 10 | 🟡 Sedang | **Dihapus 2026-10-15** — mode B (DINO-only + multi-pass), SAM dihapus. |
| **H6** | `del sam_model` → `NameError` tiap run SAM | Correctness | 5 | 2 | 10 | 🟡 Sedang | **Terselesaikan (via penghapusan)** — SAM dihapus. |
| **H7** | `build_yolo_dataset` skip label yang ada → dataset basi setelah ROI/config berubah | Integritas data | 4 | 4 | 16 | 🟠 Tinggi | **Diperbaiki 2026-10-15** — skip `label_path.exists()` dihapus; hitungan `frames_with_detection` tetap. |
| **H8** | Resume-skip tanpa sidik jari konfigurasi → hasil lama terkunci | Integritas data | 4 | 4 | 16 | 🟠 Tinggi | **Diperbaiki 2026-10-15** — signature konfigurasi per chunk/summary; resume ditolak bila konfigurasi berubah. |
| **H9** | `crop_and_fit_frame` hanya rectangle → ROI polygon (LOCKED) memicu `KeyError` | Correctness | 4 | 3 | 12 | 🟡 Sedang | **Diterapkan 2026-10-15** — `crop_to_polygon` + `resolve_roi_polygon`; `builders/build_roi_config.py` (picker); self-check lulus. |
| **H10** | Tanpa isolasi error per-frame → 1 frame rusak menggagalkan seluruh fase | Robustness | 3 | 3 | 9 | 🟡 Sedang | **Diperbaiki 2026-10-15** — `try/except` per frame di proses DINO; frame gagal dicatat, fase tetap dilanjutkan. |
| **P1** | Dokumentasi memuat angka salah & kehilangan seksi environment/repro | Dokumentasi | 4 | 3 | 12 | 🟡 Sedang | **Sebagian besar dikerjakan 2026-10-18, diperbarui 2026-10-20** — koreksi nama kamera (B01–B05 → `bogor-kab-*`), resolusi/fps (per kamera, terverifikasi), struktur berkas (hapus file yang tidak ada), status pipeline (NB03–NB05 belum dibuat), NB02b/CLAHE diklarifikasi = NB02b bukan CLAHE melainkan **tahap anotasi manual** (MASTER §10.0/§10.1). Sisa: seksi Environment & Reproduction (versi paket, seed, hash input) |
| **P2** | Tidak ada test → perbaikan tanpa jaring pengaman | Testing | 5 | 3 | 15 | 🟠 Tinggi | Tulis test P0 (harness ekstraksi AST) sebelum fix. **2026-10-28:** `test_pipeline_logic.py` menutup logika hitung (kontrak, fragment stitching, guard ambang tracker) tanpa GPU; sisa celah: perilaku notebook & preprocessing gambar masih hanya dijaga smoke bermodel |
| **P3** | Reproducibility lemah (tanpa seed/versi paket/lockfile) | Reproducibility | 3 | 3 | 9 | 🟡 Sedang | Catat versi paket, hash input, seed; buat requirements. **2026-10-28:** `requirements.txt` (13 paket, versi dari `.venv312`) + `README.md` urutan menjalankan; belum ada lockfile transitif dan hash input |
| **P4** | Metode FM (Grounding DINO) tidak optimal untuk objek kecil low-light | Metode/desain | 4 | 4 | 16 | 🟠 Tinggi | Riset SELESAI (2026-10-15): buang SAM, perbaiki pemetaan label, threshold dari gold set — lihat §5 |
| **P5** | Validitas eksternal data: 1 malam, 1 sesi, beberapa kamera blur | Desain riset | 4 | 4 | 16 | 🟠 Tinggi | Scope-kan klaim; tambah sesi/kondisi bila memungkinkan |
| **P6** | Dependensi eksternal: path absolut `sam_b.pt` di `Riset-4` (read-only) | Ketergantungan | 3 | 4 | 12 | 🟡 Sedang | ✅ **Moot 2026-10-15** — SAM dihapus dari NB02; tak ada kode aktif yang memakai `sam_b.pt` (terverifikasi 2026-10-18) |
| **P7** | Clean-code debt (dead code, type hints, SRP, DRY) | Maintainability | 4 | 2 | 8 | 🟡 Sedang | Rapikan bertahap; ekstrak fungsi murni ke modul importable |
| **P8** | Risiko perubahan: fix B1–H10 menimbulkan regresi (tanpa test, re-run GPU mahal) | Perubahan | 4 | 4 | 16 | 🟠 Tinggi | Backup + test dulu + fix sekali jalan + gate verifikasi + rollback |
| **R1** | **Label NB02 tidak layak pakai: 46% box raksasa (>80% crop), grouping Mode A mati (1.7% union), rasio kelas terbalik, deteksi sedikit (~2 box/frame)** | Integritas label | 5 | 4 | 20 | 🟡 Sedang (fix diterapkan, re-run pending) | **Ditemukan 2026-10-18** — akar: threshold 0.25 terlalu tinggi + tak ada filter luas box. **FIX di builder:** `DETECTION_THRESHOLD`→0.15 + `MAX_BOX_AREA_RATIO=0.40` → **run 2 diverifikasi bersih** (1.529 frame, 4.967 box, 0 box >40%, union 22,9%). **2026-10-19:** threshold dikalibrasi lagi → **0.30** + prompt bertitik + hierarki kelas geometri → vote FRCNN → DINO (lihat MASTER §0.4); signature berubah → **hapus chunks & re-run NB02**, verifikasi box raksasa ≈ 0 + statistik vote/konflik via sel visualisasi |
| **R2** | **Vote verifier FRCNN lemah di domain malam → kelas grup 1 lampu bisa salah arah; bobotnya juga butuh internet saat run pertama** | Integritas label | 3 | 3 | 9 | 🟡 Sedang (dikontrol) | **Dikontrol 2026-10-19** — vote hanya **sekonder** (geometri ≥2 lampu tetap prioritas), tak pernah menghapus box; konflik ditandai `signals_agree=false` + `[konflik]` oranye di overlay & statistik sel histogram; bobot gagal dimuat → verifier mati otomatis (deteksi tetap jalan); kalibrasi NB03 |
| **R3** | **Label NB02 tetap pseudo-label: tanpa koreksi manusia, galat sistematis DINO terbawa ke gold-set & metrik NB03 (sirkularitas)** | Integritas label | 4 | 4 | 16 | 🟠 Tinggi (dikontrol) | **Dikontrol 2026-10-20** — NB02b anotasi manual: 400 frame stratified per kamera, tiap box menyimpan provenance `pseudo/manual/edited`; hanya frame ber-`reviewed` yang masuk gold-set. Audit NB03 wajib melaporkan fraksi box yang masih `pseudo` (belum diverifikasi manusia) |
| **R4** | **Sampel 400 frame tidak mewakili seluruh kondisi: setelah kamera `test` dibuang (anti-B1), hanya 3 kamera tersisa** | Validitas eksternal | 3 | 3 | 9 | 🟡 Sedang (dipantau) | **Dikontrol 2026-10-20** — sampling stratified proporsional pada `train`+`val` dengan seed tetap `20261020` + manifest tersimpan (stabil & dapat diaudit). Alokasi: `bappenda1` 267 · `pemda2` 87 · `bappenda3` 46 = 400 (rasio 47,5% dari 842 frame eligible). Batas: `bappenda3` hanya 98 frame eligible → kontribusi kecil; perbesar `SAMPLE_TOTAL_FRAMES` bila anggaran waktu anotasi memungkinkan |
| **R5** | **Anotasi manual tidak punya kontrol kualitas (single-annotator, tanpa re-check)** | Kualitas anotasi | 4 | 3 | 12 | 🟡 Sedang (terbuka) | **Terbuka** — mitigasi tersedia: export gold-set dapat di-split jadi subset **re-check** (annotator mengulang tanpa melihat label) untuk estimasi konsistensi; NB03 wajib melaporkan jumlah frame reviewed vs target 400 sebagai syarat gate |
| **R6** | **Kebocoran split ke gold-set/fine-tune: sampling awal NB02b menyampel 5 kamera tanpa melihat `group_split.csv` → ±175 frame kamera `test` ikut teranotasi** | Validitas evaluasi | 5 | 5 | 25 | 🔴 Kritis (diperbaiki 2026-10-20) | **Ditemukan & diperbaiki 2026-10-20** sebelum notebook dijalankan — `SPLIT_NAME_BY_CAMERA` + `SAMPLE_EXCLUDED_SPLITS=["test"]` memfilter kandidat saat sampling; 640 frame kamera test tidak tersentuh. **Guard kedua:** `assert_no_excluded_split()` di sel EXPORT `raise` bila kamera test ikut ter-export; manifest lama dengan `excluded_splits` berbeda ditolak. Diverifikasi dry-run pada data nyata + assertion di smoke test |
| **R7** | **NB02 menghasilkan `output/nb02/yolo` sebagai dataset "siap-training" padahal UNSplit: `data.yaml` menunjuk `train: images` + `val: images` (train==val) dan `images/` memuat 640 frame kamera split `test`** | Validitas evaluasi | 5 | 5 | 25 | 🔴 Kritis (dimigrasi 2026-10-20) | **Ditemukan 2026-10-20** saat audit lanjutan pengguna. **Dimigrasi:** `build_yolo_dataset` kini melaporkan `detections_by_split`; `write_data_yaml` menandai `split_status: UNSplit` + menulis `yolo/split_camera_map.csv` (peta kamera→split) dan sel summary NB02 mencetak peringatan leakage eksplisit. **BELUM SELESAI — NB04 wajib** memecah ke `images/{train,val,test}` dari peta itu; sampai itu terjadi, `data.yaml` NB02 **tidak boleh** dipakai training |
| **R8** | **Output NB02b → gold-set `MIXED`: gold-set (`output/annotation/yolo`) memuat frame `train`+`val` dalam satu folder, dan fine-tune DINO memakai kamera `val`** | Validitas evaluasi | 4 | 4 | 16 | 🟠 Tinggi (terbuka) | **Terbuka** — NB02b sudah bebas kamera `test` (R6), tetapi belum memisahkan `train`/`val` di dalam gold-set. Sebelum evaluasi NB03, gold-set harus dipecah: metrik dilaporkan per split (`train` vs `val`), dan kalibrasi threshold DINO **hanya** dari frame `val`. Jangan melaporkan mAP gabungan train+val sebagai generalisasi |
| **R9** | **Kelas `PantulanCahaya` dibangun dari sampel terlalu sedikit → overfit + noise ke training YOLO** | Kualitas model | 3 | 4 | 12 | 🟢 Rendah (dimigrasi 2026-10-22) | **Dimigrasi.** Kekhawatiran awal (2026-10-21) berbasis FP rate Riset-4 **0,4–0,8%** → dianggap kelas akan overfit. **Data nyata membantah:** pada 126 frame yang sudah dianotasi, **313 dari 940 box (33,3%)** adalah pantulan, muncul di **33% frame**, proyeksi **~994 box** pada 400 frame — jauh di atas ambang ~150. Keputusan 2026-10-22: `PantulanCahaya` **dinaikkan jadi kelas resmi ke-3** (`nc: 3`) dan dipakai sebagai hard-negative. Migrasi store otomatis (`kind: distractor` → `class_name: PantulanCahaya`), diuji: 319 box dimigrasi, 940 box aktif utuh |
| **R11** | **Kelas `PantulanCahaya` ikut terhitung sebagai kendaraan di hilir — penghitungan kendaraan jadi terlalu tinggi** | Validitas hasil | 4 | 4 | 16 | 🟠 Tinggi (terbuka) | **Terbuka — wajib ditutup di NB04/NB05.** Sejak 2026-10-22 pantulan **ikut ter-export** ke COCO dan gold-set YOLO (sebagai kelas index `2`). Semua konsumen hilir **wajib** memfilter kelas ini: penghitungan kendaraan hanya memakai `LampuMobil` + `LampuMotor`; `PantulanCahaya` hanya dipakai sebagai hard-negative/analisis error. Bila filter lupa dipasang, refleksi akan terhitung sebagai kendaraan. Tambahan: distribusinya condong — 313 pantulan vs 183 mobil — perlu dicek class imbalance saat training NB04. **Filter sudah dipasang di kode 2026-10-26:** NB04 menulis `PantulanCahaya` sebagai indeks kelas 2, NB05 memakai `COUNTED_CLASS_NAMES = ["LampuMobil", "LampuMotor"]` untuk proxy counting dan galeri error. Status tetap 🟠 sampai training nyata dijalankan dan metrik per kelas dilaporkan |
| **R12** | **Bug navigasi NB02b menimpa kelas box terakhir setiap frame (`sync_class_toggle` menulis, bukan membaca)** | Kualitas data | 5 | 5 | 25 | 🟢 Ditutup 2026-10-26 (data lama belum dipulihkan) | **Ditutup, bukti kuat.** `go_to_index()` memanggil `sync_class_toggle()` yang ternyata fungsi **penulis** (menerapkan nilai ToggleButtons ke box terpilih). Karena `d` tidak memperbarui toggle, kelas basi menimpa box terakhir (`available[-1]`) setiap kali sebuah frame dibuka kembali. Direproduksi: `PantulanCahaya` → `LampuMobil` setelah next+prev. **Bukti kuantitatif:** box terakhir berubah kelas **74/83 (89%)** vs box lain **11/161 (7%)**; dari 74 itu, `vehicle_vote` mendukung `LampuMobil` 50×, `None` 24×, **tidak pernah** mendukung `LampuMotor`. Perbaikan: `sync_class_toggle` dipecah menjadi `apply_toggle_to_selection` (observer, menulis) + `show_selected_class_in_toggle` (hanya membaca); `go_to_index`/`on_box_click`/`mark_selected_as_reflection` kini menyinkronkan toggle tanpa menulis. Regression test: 12× next + 12× prev mengubah **0** dari 400 frame. **Data lama belum diubah** — 74 box terdaftar di `output/annotation/audit_navigation_overwrite.csv` untuk diperiksa manual |
| **R10** | **`golden_set_v3.csv` Riset-4 rusak oleh Excel Indonesia — kolom koordinat berupa string ribuan (`19.05` → `19.050.863.647.460.900`)** | Kualitas data | 4 | 5 | 20 | 🟠 Tinggi (dimigrasi) | **Dimigrasi 2026-10-21.** Ditemukan saat audit. Konverter `Temp/opencode/build_riset4_golden_to_coco.py` **sengaja** membaca koordinat dari `candidates_index.csv` (JSON float asli), bukan `golden_set_v3.csv`. Skrip lain yang membaca `golden_set_v3.csv` **tidak boleh** mempercayai kolom `x1..y2` di sana |
| **R13** | **Split `test` tanpa ground truth — seluruh label test berasal dari pseudo-label DINO, sehingga metrik test bukan akurasi** | Validitas hasil | 4 | 5 | 20 | 🟠 Tinggi (terbuka) | **Terbuka sejak keputusan anti-kebocoran 2026-10-20.** Gold set NB02b sengaja mengecualikan kamera `test` (`bappenda2`, `pdam2`), sehingga split test tidak punya anotasi manual sama sekali (0/640 frame). Konsekuensi: NB05 hanya bisa mengukur **kesepakatan dengan DINO** di test, bukan performa sebenarnya; klaim "model bekerja di kamera baru" belum bisa dibuktikan. Mitigasi terpasang: NB05 menulis `label_source` per split dan peringatan eksplisit di JSON + sel summary; NB04 menolak training bila frame val gold < 30. **Penutupan:** anotasi gold-set test terpisah ±100 frame dari kamera test setelah 400 frame train/val selesai (keputusan pengguna 2026-10-26: "val dulu, test menyusul") |
| **R14** | **Taksonomi pipeline tidak sesuai tujuan riset — model mendeteksi jenis kendaraan (`LampuMobil`/`LampuMotor`), bukan kondisi lampu (`silau` / `berwarna` / `silau+berwarna`)** | Validitas hasil | 5 | 5 | 25 | 🔴 Kritis (mitigasi terpasang, menunggu verifikasi) | **Ditemukan 2026-10-26 dari koreksi pemilik riset.** Tujuan: deteksi lampu **silau**, lampu **berwarna**, dan lampu **silau+berwarna**, plus sistem hitung (kendaraan mobil/motor + 6 hitungan kondisi per jenis). Kenyataan: `CLASS_NAMES` NB02b/NB04 dan `CLASS_PROMPTS` NB02 hanya memuat jenis kendaraan + `PantulanCahaya`; kata "silau"/"berwarna" **tidak ada** di seluruh repo. Target `berwarna` dan `silau+berwarna` **tidak ada sama sekali**; target `silau` hanya tersentuh tidak langsung lewat pantulan/glare di aspal (bukan lampu kendaraan). Dampak: 126 frame gold (**946 box**) tidak bisa dipakai apa adanya, pseudo-label 1.548 frame harus digenerate ulang, dan seluruh angka NB03–NB05 tidak sah. Mitigasi: audit + kontrak hitung di MASTER **§10.6**; gate kerja di MASTER §10.3 langkah 0 melarang training NB04 dan mengutip angka NB05. **Progres 2026-10-26:** label space diputuskan (9 kelas, §13.9), NB02 ditulis ulang (hanya melokalisasi lampu, tidak pernah menebak kondisi), NB02b 9 kelas + sentinel, NB03 dialihkan ke audit gold + karakterisasi fotometrik, NB04/NB05 disesuaikan, NB06 dibuat untuk hitung kendaraan unik. **Penutupan:** smoke NB02→NB06 lolos + run penuh + anotasi NB02b selesai, lalu angka NB03–NB06 disahkan |
| **R15** | **Angka "kendaraan unik" bergantung pada mutu tracker: track di-reset per video, kendaraan yang keluar lalu masuk lagi terhitung dua kali, dan `MIN_TRACK_HITS` membuang kendaraan yang hanya sempat terlihat 1–2 frame** | Validitas hasil | 3 | 4 | 12 | 🟠 Terbuka (melekat pada metode) | **Muncul 2026-10-26** saat satuan hitung diubah dari "deteksi per frame" menjadi "kendaraan unik" (MASTER §13.11). Tidak bisa dihilangkan tanpa identitas lintas sesi (butuh ReID) atau garis hitung berarah (butuh kalibrasi homografi). Mitigasi terpasang: tracking pada **setiap frame** video asli (bukan frame sampel 0,6 detik), ambang internal tracker diturunkan (`builders/bytetrack_nb06.yaml`) agar kelas langka tidak undercount, ambang per kelas NB05 diterapkan saat agregasi track, `MIN_TRACK_HITS = 3` (0,12 detik pada 25 fps), dan NB06 selalu melaporkan `duplication_factor_vs_per_frame` sebagai pembanding baseline NB05. **Wajib ditulis di laporan:** angka NB06 adalah jumlah kejadian kendaraan masuk-tetap-dihitung-satu-kali per video, bukan jumlah kendaraan fisik di jalan |
| **R16** | **Berkas duplikat/usang di root proyek (backup builder identik, dua bobot YOLO tak terpakai) berisiko diedit atau dikutip sebagai versi aktif** | Maintainability | 2 | 3 | 6 | 🟢 Ditutup 2026-10-28 | **Ditutup.** `backup_sebelum_fix_20261015/` (14 file, ±2,8 MB) dan `yolo26n.pt` + `yolov8n.pt` (12 MB) dipindahkan ke `trash_2026-10-26/audit_2026-10-28/`, bukan dihapus permanen. Hanya `yolo11n.pt` yang benar-benar dirujuk NB04. Konsekuensi: tidak ada lagi rollback point snapshot; riwayat perubahan mengandalkan `AUDIT_ACTION_2026-10-28.md` + register ini |
| **R17** | **Definisi operasional "silau"/"berwarna" belum operasional: label silau tidak bisa direproduksi fitur piksel apa pun, dan dua kelas berwarna praktis kosong** | Validitas label | 4 | 5 | 20 | 🔴 Terbuka 2026-10-28 | **Bukti** (78 frame, 1.125 box, `builders/probe_label_definition.py`, CSV `output/annotation/label_probe.csv`): median `v240` `MobilSilau` 0,139 vs `MobilNormal` 0,147 (terbalik); ambang terbaik `white_bloom_ratio` 0,22 menangkap 82% silau tapi menandai 42% normal dan 52% pantulan; bloom di luar box (`ring_v180`) gagal total (normal ikut 85%). `MobilBerwarna`/`MobilSilauBerwarna` = 0 box; `MotorBerwarna` 4 + `MotorSilauBerwarna` 2, semuanya hue 110–120 (biru). Dari 93 box berpiksel berwarna kuat, 79 adalah `PantulanCahaya` hue 0 (merah). **Mitigasi (keputusan pemilik riset 2026-10-28, MASTER §13.9a):** silau dinilai **berbasis contoh** (bukan ambang), `Berwarna` hanya **lampu depan + biru/cyan** (hue 95–135, minimal 25% luas box), ragu → `BelumDitentukan`. Panduan ini sudah masuk markdown NB02b. **Masih terbuka:** (1) set contoh `output/annotation/reference_patches/` belum dibuat, jadi definisi silau belum operasional penuh; (2) uji kesepakatan ulang belum dijalankan (target ≥ 80%); **(3) ~~aturan kendaraan yang hanya tampak belakang~~ → TERTUTUP 2026-10-28: ROI tiap kamera sudah disesuaikan ke arah jalan, frame hanya memuat tampak depan, jadi kasus ini tidak ada di data;** **(4) 271 label silau lama belum diaudit ulang**; **(5) mining kandidat biru sudah dijalankan (`builders/mine_blue_candidates.py`, `output/annotation/blue_candidates.csv`): 81 kandidat longgar / 20 ketat, tetapi hanya 2 di train, 0 di val, 18 di test — kelas berwarna tidak dapat dilatih maupun dievaluasi secara sah dengan data yang ada. 4 kandidat ditandai `awb_suspect` (frame biru menyeluruh). Perlu keputusan strategi: turunkan target ke 4 kelas + laporkan warna sebagai batasan, atau tambah sesi rekaman baru** |
| **R18** | **Proposal NB02 hanya menangkap ±61% kendaraan: 1.017 dari 1.897 box gold digambar manual karena DINO tidak pernah mempropokenya** | Validitas data | 4 | 5 | 20 | 🔴 Terbuka 2026-10-29 | **Bukti** (`builders/audit_gold_quality.py`, 170 frame reviewed): 620 box berasal dari pseudo (`pseudo`+`edited`), 403 box kendaraan digambar manual, **proxy recall 0,606**; 396 box pseudo (31% proposal) dihapus pemilik riset. Risiko: kendaraan yang tidak pernah dipropose tidak pernah muncul di NB02b, sehingga gold set menyampel lokasi secara tidak acak dan detektor mewarisi prior DINO. **Mitigasi terpasang:** `DETECTION_THRESHOLD` 0,30→0,25 dan `VEHICLE_SCORE_THRESHOLD` 0,40→0,30 (2026-10-29); `VEHICLE_VOTE_MIN_COVERAGE` sengaja **tidak** diturunkan (0,15 akan melabeli jenis kendaraan lain). **Penutupan:** ukur ulang proxy recall setelah NB02 dijalankan dengan threshold baru; target ≥ 0,80 atau terima manual-first sebagai metode resmi dan tulis sebagai batasan |
| **R19** | **Split `val` belum dianotasi sama sekali (0/83 frame), sehingga NB03 gagal gate, NB04 menolak training, dan NB05 akan `raise`** | Jadwal + validitas | 5 | 5 | 25 | 🔴 Terbuka 2026-10-29 | **Bukti:** 170 frame reviewed semuanya kamera `bappenda1` (split `train`); 83 frame `pemda2` (`val`) masih 0. Godaan yang harus ditolak: melonggarkan `MINIMUM_VAL_FRAMES`/gate kelas agar "run jalan". **Mitigasi terpasang:** NB04 menolak training bila < 30 frame val **atau** ada kelas < 20 box train; NB03 menolak gold set tanpa frame reviewed. **Penutupan:** anotasi 83 frame `pemda2` di NB02b, lalu export ulang |

---

## 5. Analisis & mitigasi risiko prioritas

### 🔴 B1 — Split leakage (Kritis)
- **Dampak:** mAP val palsu; kamera test ikut terlatih; klaim generalisasi lintas-kamera gugur → tujuan inti riset tidak terpenuhi.
- **Akar:** `write_data_yaml` menulis `train: images` = `val: images`; `group_split.csv` tidak pernah dibaca NB02.
- **Mitigasi (preventif):**
  1. Materialisasi split di NB03: `images/{train,val,test}` + `labels/{train,val,test}` mengikuti `group_split.csv` (bukan random).
  2. `data.yaml` memakai `train/val/test` terpisah.
  3. Assert: `set(cam_train) ∩ set(cam_val) = ∅`, `∩ set(cam_test) = ∅`; simpan bukti.
- **Kontingensi:** bila NB04 sudah sempat dijalankan → batalkan hasil, jangan dilaporkan.
- **Trigger:** NB04 dijalankan tanpa split terpisah.

### 🔴 B2 — Artefak disk stale & cacat (Kritis)
- **Dampak:** training di atas 204 frame terdistorsi; train efektif 1 kamera (bappenda3 = 1 frame/0 deteksi).
- **Akar:** fix dedup hanya di builder; artefak on-disk masih window 4/dist 8.
- **Mitigasi (preventif):**
  1. Hapus **seluruh** `output/nb02` (bukan hanya `chunks`) sebelum re-run.
  2. Re-run **setelah** semua fix kode masuk (hindari run berulang yang mahal).
  3. Tandai artefak lama sebagai STALE di dokumentasi.
- **Kontingensi:** re-run parsial → verifikasi `frame_unique`, dan bahwa setiap kamera punya frame (khususnya bappenda3 > 0).
- **Trigger:** `nb02_summary.json.frame_unique == 204` atau bappenda3 == 0.

### 🟠 H1 — Label kelas dari token BPE noisy (Tinggi)
- **Dampak:** ~40,6% box (358/882) kelas berpotensi salah → mAP per-kelas tidak bermakna.
- **Akar:** `text_labels` = rekonstruksi token (artefak `##light`); pemetaan substring.
- **Mitigasi (preventif):** validasi confusion DINO vs 500 GT; jika akurasi kelas rendah → **kolaps ke 1 kelas `LampuKendaraan`** atau pindah ke pemetaan phrase-level (`labels`/phrase index) + prompt disjoint + naikkan `text_threshold`.
- **Kontingensi:** bila tak bisa dipastikan → laporkan sebagai limitasi, jangan klaim per-kelas.
- **Trigger:** akurasi kelas vs GT < ~85%.

### 🟠 H2 — Threshold di lantai distribusi / tanpa negatif (Tinggi)
- **Dampak:** presisi rendah, FP dari sumber cahaya non-kendaraan; model tak pernah diajar "tidak ada lampu".
- **Akar:** threshold 0.25 = min skor; tanpa confidence filter; 0 background.
- **Mitigasi (preventif):** audit FP rate (≥100 box manual, stratified); kurasi subset training (skor+sanity+buang near-dup); tambah frame negatif (label kosong).
- **Kontingensi:** bila presisi sangat rendah → naikkan threshold atau sinkronkan dengan kolaps kelas.
- **Trigger:** FP rate > ~15%.

### 🟠 H4 — Val 1 kamera + leakage domain di test (Tinggi)
- **Dampak:** varians antar-kamera tak terdefinisi; dua kamera test sebagian se-domain dengan train.
- **Mitigasi (preventif):** val ≥2–3 kamera atau **Leave-One-Camera-Out CV**; kelompokkan kamera ko-lokasi (Bappenda) dalam split sama; selaraskan distribusi val dan test.
- **Trigger:** laporan evaluasi memakai val = 1 kamera.

### 🟠 H7 / H8 — Dataset basi & resume terkunci (Tinggi)
- **Dampak:** pipeline melaporkan "sukses" tapi dataset/koordinat tidak berubah (silently wrong).
- **Mitigasi:** hapus/regenerasi `yolo/images`+`labels`; tambah `config_hash` (threshold+prompt+window+distance+ROI mtime) di summary; tolak resume bila hash beda.
- **Trigger:** re-run dengan ROI/threshold baru tanpa menghapus artefak.

### 🟠 P2 — Tidak ada jaring pengaman test (Tinggi)
- **Dampak:** memperbaiki B1–H10 tanpa test → regresi tak terdeteksi, terutama pada 95,5%-style bug dedup.
- **Mitigasi:** tulis test P0 lebih dulu (compute_duplicate_flags, map_dino_label_to_class, normalize+clip box, frame_dhash+hamming, build_frame_records) via harness ekstraksi AST; mock model GPU.
- **Trigger:** PR/fix tanpa test.

### 🟠 P5 — Validitas eksternal data (Tinggi)
- **Dampak:** kesimpulan terlalu luas; "multi-kamera" = 5 titik, 1 operator, 1 kota, 1 sesi, 1 malam.
- **Mitigasi:** scope-kan klaim eksplisit; tambah sesi/kondisi (terang, jam berbeda) bila memungkinkan; laporkan sebagai limitasi.

### 🟠 P8 — Risiko perubahan (Tinggi)
- **Dampak:** fix menimbulkan regresi / kehilangan data / pemborosan jam GPU.
- **Mitigasi:** kontrol §6 (backup, test, urutan, gate, rollback).

### 🟠 P4 — Kesesuaian metode FM Grounding DINO (Tinggi) — HASIL RISET 2026-10-15
- **Dampak:** pseudo-label berbasis Grounded-SAM (Grounding DINO + SAM — arsitektur saat temuan dicatat) menghasilkan label kelas berlumur noise dan deteksi ekor-rendah; jika tak diperbaiki, seluruh dataset training cacat.
- **Temuan riset (source-backed):**
  1. **Akar H1 terkonfirmasi dari source `transformers`:** `text_labels` = hasil decode id-token di atas `text_threshold` (fungsi `get_phrases_from_posmap` + `batch_decode`) → token-level, memunculkan artefak `##light` dan bisa menggabung antar-frasa. Ini limitasi yang diakui maintainer, bukan bug lokal.
  2. **Grounding DINO lemah pada objek kecil/malam.** Skor maksimum 0.441 = sinyal domain mismatch berat; kelas mobil vs motor dari lampunya sendiri adalah sinyal lemah.
  3. **Mask SAM tidak berguna untuk training YOLO-deteksi** (hanya untuk YOLO-seg / QA filter) → mengonfirmasi H3; SAM juga resize ke 1024² sehingga mask blob 19 px kasar.
  4. **Threshold benar dipilih dari gold set**, bukan dari lantai distribusi.
  5. **Alternatif:** multi-pass DINO per-frasa; YOLO-World/OWLv2 (cepat tapi lemah objek kecil); Grounding DINO 1.5 (API/gated); SAM 3 concept-prompt (ganti seluruh stack, tapi 3,45 GB — tak muat 4 GB).
- **Mitigasi (rekomendasi riset):**
  1. Keluarkan SAM dari loop pelabelan (hemat 1–2 jam); bila tetap perlu → MobileSAM (41 MB) atau SAM2.1-t (78 MB).
  2. Ganti pemetaan kelas: token-argmax → frasa (§ pembatas `[CLS]/[SEP]/"."`), atau **multi-pass** `"car"` / `"motorcycle"`.
  3. Pilih threshold via gold set 50–100 frame; pertimbangkan latih YOLO langsung dengan sedikit label manual dan pakai DINO hanya sebagai seed.
- **Status:** Riset selesai; **keputusan redesign pipeline menunggu approval user** (item keputusan §9).
- **Sumber kunci:** HF docs GroundingDINO; diskusi `remove_combined` (IDEA-Research/grounding-dino-tiny#6); repo IDEA-Research/GroundingDINO (`util/inference.py`); paper GSAM (arXiv 2406.05776); Ultralytics (MobileSAM/SAM2/SAM3/YOLO-World/SAHI).
- **Koreksi 4 GB (2026-10-15):** `sam_b.pt` + DINO **terbukti berjalan** di RTX 3050 4 GB milik user → kendala VRAM **diturunkan** (bukan risiko); sisa biaya hanya waktu (~1–2 jam). Pemilihan MobileSAM/SAM2.1-t jadi opsional (kecepatan), bukan keharusan.

---

## 6. Pengendalian SEBELUM perbaikan (pre-fix controls)

Wajib dijalankan berurutan sebelum menyentuh kode.

### C1 — Freeze & backup (anti kehilangan data)
- Snapshot ke folder bertimestamp: `backup_sebelum_fix_{YYYYMMDD}/` berisi `builders/`, `NB01_acquisition.ipynb`, `NB02_pseudo_label.ipynb`, `MASTER_DOKUMENTASI_HAMPIR.md`, `output/roi/`, `output/split/`, `output/manifests/`, `output/nb02/nb02_summary.json`.
- Catat hash (`sha256`) semua file yang akan diubah + dari folder `output/` sebagai baseline verifikasi.
- **Jangan** hapus `output/frames/` (sumber tak bisa direproduksi tanpa rekam ulang).
- **Status (2026-10-15):** ✅ SELESAI — `backup_sebelum_fix_20261015/` (12 file + `SHA256SUMS.txt`).

### C2 — Jaring pengaman test (anti regresi)
- Tulis test P0 lebih dulu (lihat H/P2). Minimal 5 test prioritas.
- Harness ekstraksi AST dari string builder (karena logika ada di dalam string).
- Test GPU di-mock.

### C3 — Keputusan desain difinalkan (anti rework)
Putuskan **sebelum** fix, agar re-run GPU hanya sekali:
- Dedup: konfirmasi `window=1` / `distance=4`.
- ROI: **polygon** (dan bangun `build_roi_config.py`) ATAU tunda dengan keputusan eksplisit "pakai full-frame".
- Taksonomi: 2 kelas (perbaiki pemetaan) ATAU 1 kelas (`LampuKendaraan`).
- `imgsz` target NB04.

### C4 — Aturan kontrol perubahan
- Satu perbaikan = satu unit test + verifikasi lulus.
- Tandai setiap `shortcut:` yang disengaja (level: komentar kode atau catatan di register ini bila di path final).
- Tidak ada perbaikan tanpa catatan di §10 (riwayat revisi).

---

## 7. Urutan eksekusi & rollback

Urutan dirancang agar re-run GPU mahal hanya terjadi sekali.

| Langkah | Aksi | Mengunci risiko | Verifikasi |
|---|---|---|---|
| 0 | Backup & hash baseline (C1) | Semua | Folder backup ada; hash tercatat |
| 1 | Tulis test P0 (C2) | P2, P8 | Test hijau di kode lama & setelah fix |
| 2 | Finalisasi keputusan desain (C3) | H1, H9, H5 | Keputusan tercatat di §10 |
| 3 | Fix kode non-GPU: H6, H7, H8, H10, H1(mapping), clean-code | H6–H10, P7 | Test P0 lulus; tidak mengubah output sampai re-run |
| 4 | (Jika polygon) bangun `build_roi_config.py` + upgrade `crop_and_fit_frame` (H9) | H9 | Unit test rect+polygon |
| 5 | Hapus total `output/nb02`, **re-run NB02 sekali** | B2 | `frame_unique` ≈1.500; bappenda3 > 0; setiap kamera ada |
| 6 | Materialisasi split + `data.yaml` (B1) | B1 | Assert bebas kebocoran |
| 7 | NB03 audit & kurasi label (H1, H2) + redesign val (H4) | H1, H2, H4 | FP rate terukur; audit kelas |
| 8 | NB04 + metrik per-kamera (H5, P3) | H5, P3 | mAP per kelas & per kamera |

**Rollback:**

| Titik gagal | Tindakan |
|---|---|
| Fix kode merusak output | Restore dari `backup_sebelum_fix_*` (**git tidak tersedia** — `Riset-HampirMatang/` untracked); test P0 menunjukkan regresi |
| Re-run NB02 gagal/parsial | Hapus chunk parsial, ulangi dari last good chunk (resume) atau dari nol; frame tetap aman |
| Split salah | Bangun ulang `images/`+`labels/` dari `group_split.csv` (tidak perlu re-run DINO/SAM) |
| Hasil NB03 tidak lulus QA | Jangan lanjut NB04; kembali ke langkah 3/5 |

---

## 8. Gate verifikasi (harus lulus)

1. Backup & hash baseline ada.
2. Test P0 lulus.
3. Keputusan desain (dedup, ROI, taksonomi, imgsz) tercatat.
4. Re-run NB02: `frame_unique` ≈1.500 **dan** setiap kamera > 0 (khususnya bappenda3).
5. Split terpisah & assert bebas kebocoran lulus.
6. Audit label: FP rate terukur + keputusan kurasi + validasi kelas.
7. NB04 dijalankan dengan metrik per-kelas **dan** per-kamera.

Belum lulus gate 1–5 → jangan mulai NB04.

---

## 9. Risiko yang masih terbuka

| ID | Isu terbuka | Status |
|---|---|---|
| H1 | Taksonomi **2 kelas (diputuskan user)** — lampu-sepasang (A), multi-pass DINO | ✅ Diterapkan di builder 2026-10-15; perlu validasi NB03 |
| H3 | SAM dihapus (mode B) | ✅ Diterapkan di builder 2026-10-15 |
| H7 | Skip fix, akuntansi perbaiki | ✅ Diterapkan di builder 2026-10-15 |
| H8 | Config signature + resume check | ✅ Diterapkan di builder 2026-10-15 |
| H9 | Polygon ROI | ✅ Selesai 2026-10-15 |
| H10 | Per-frame isolation | ✅ Diterapkan di builder 2026-10-15 |
| P4 | Redesign pipeline (mode B+A) | ✅ Diterapkan di builder 2026-10-15 (H8 signature terpasang, klasifikasi via geometri, union box) |
| P5 | Sesi data tambahan (kondisi terang) | Belum ada rencana |
| R17 | Definisi operasional label (silau berbasis contoh, berwarna = biru/lampu depan) | 🔴 Terbuka 2026-10-28 — set contoh acuan belum dibuat **dan** mining kandidat biru hanya menghasilkan 2 box train / 0 val; lihat baris R17 |
| R18 | Recall proposal DINO hanya ±0,61 pada 170 frame gold | 🔴 Terbuka 2026-10-29 — threshold dilonggarkan, perlu ukur ulang; lihat baris R18 |
| R19 | Split `val` 0/83 frame dianotasi | 🔴 Terbuka 2026-10-29 — memblokir NB03/NB04/NB05; lihat baris R19 |

### Keputusan terbuka (butuh approval user)
1. **FM/pipeline**: ✅ **DIPUTUSKAN 2026-10-15 → opsi (b)** Grounding DINO saja (tanpa SAM) + multi-pass. Opsi yang dievaluasi: (a) pertahankan Grounded-SAM (Grounding DINO + SAM) dengan perbaikan pemetaan label + threshold, (b) turunkan ke Grounding DINO saja (tanpa SAM) + multi-pass, (c) hybrid: sedikit label manual + DINO sebagai seed untuk latih YOLO langsung.
2. **ROI**: ✅ diputuskan — bangun polygon (selesai).
3. **Taksonomi**: ✅ diputuskan — **2 kelas** (perbaiki mapping + mitigasi bias box).
4. **Backup**: ✅ diputuskan — snapshot dibuat.
5. **Bias box 2 kelas (baru)**: definisi target deteksi — tetap per-lampu (ambigu) atau reframe ke **lampu-sepasang / region kendaraan** agar model bisa membedakan mobil vs motor. Menunggu keputusan user.
6. ~~**Kendaraan yang hanya tampak belakang**~~ ✅ **TERTUTUP 2026-10-28**: ROI tiap kamera sudah disesuaikan ke arah jalan sehingga hanya tampak depan kendaraan; kasus ini tidak ada di data, aturan khusus tidak diperlukan.
7. **Strategi kelas berwarna (baru, 2026-10-28)**: mining hue menemukan 20 kandidat ketat tetapi hanya 2 di `train` dan 0 di `val` (18 di `test`). Pilihan: (a) turunkan target latih ke 4 kelas `{Mobil,Motor} × {Normal,Silau}` dan laporkan `Berwarna`/`SilauBerwarna` sebagai batasan riset; (b) tambah sesi rekaman baru yang memuat kendaraan lampu biru sebelum training; (c) tetap latih 9 kelas dan laporkan mAP kelas berwarna sebagai tidak bermakna. Menunggu keputusan pemilik riset.
8. **Acuan silau (2026-10-28)**: builder `build_reference_patches.py` selesai menghasilkan 120 patch; butuh persetujuan pemilik riset sebelum dijadikan acuan resmi (lihat R17 / MASTER §13.9a).

---

## 10. Riwayat revisi

| Versi | Tanggal | Perubahan | Penulis |
|---|---|---|---|
| 1.0 | 2026-10-15 | Draft awal — register B1–H10 + P1–P8, kontrol pre-fix, urutan eksekusi | Orchestrator (review 8 agent) |
| 1.1 | 2026-10-15 | P4 dinaikkan ke Tinggi (16); tambah temuan riset Grounded-SAM; tambah daftar keputusan terbuka di §9 | Orchestrator (riset FM) |
| 1.2 | 2026-10-15 | C1 backup selesai; H9 ROI polygon selesai (kode + self-check); taksonomi 2 kelas diputuskan; koreksi kendala 4 GB; tambah keputusan bias box |
| 1.3 | 2026-10-15 | Builder NB02 direwrite: mode B (DINO-only + multi-pass), mode A (cluster lampu-sepasang, box union), signature konfigurasi, polygon crop, per-frame isolation, fix H1/H7/H8/H10/H3/H6/H9; NB02 lama dipindah ke trash | Orchestrator |
| 1.4 | 2026-10-18 | Rapikan penamaan arsitektur: "Grounded-SAM" → "Grounding DINO" pada header, baris P4, dan opsi keputusan §9; P6 ditandai moot (SAM dihapus, terverifikasi tak ada kode aktif memakai `sam_b.pt`); koreksi metadata header (versi 1.0 draft → 1.4, status pipeline); hapus saran rollback `git` (repo untracked) | Orchestrator |
| 1.10 | 2026-10-20 | NB02 ditandai selesai (run 2026-10-19); **NB02b anotasi manual** dibuat & lolos self-check (builder + notebook 6 sel, smoke test logika non-UI + cek statis sel UI); tambah risiko **R3** (sirkularitas pseudo-label → gold-set), **R4** (keterwakilan sampel 400 frame), **R5** (tanpa kontrol kualitas anotasi, terbuka) | Orchestrator |
| 1.11 | 2026-10-20 | **Perbaikan kebocoran split pada NB02b (R6, Kritis)** — ditemukan saat audit pengguna sebelum notebook dijalankan: sampling awal menyampel kelima kamera termasuk 2 kamera `test` (±175 frame). Ditambahkan `SPLIT_NAME_BY_CAMERA` + `SAMPLE_EXCLUDED_SPLITS`, penolakan manifest dengan `excluded_splits` berbeda, dan guard `assert_no_excluded_split()` di sel EXPORT. B1 ditutup untuk NB02b; R4 diperbarui dengan alokasi nyata 267/87/46 | Orchestrator |
| 1.12 | 2026-10-20 | **Audit lanjutan kebocoran lintas-notebook (atas permintaan pengguna).** Temuan: **R7 (Kritis)** output `output/nb02/yolo` ternyata UNSplit — `data.yaml` `train: images` + `val: images` (train==val) dan `images/` memuat 640 frame kamera `test`; builder NB02 diperbaiki (`detections_by_split`, `split_status: UNSplit`, tulis `split_camera_map.csv`, peringatan di sel summary + navbar). **R8 (Tinggi, terbuka)** gold-set NB02b masih mencampur `train`+`val` dalam satu folder → metrik NB03 wajib dilaporkan per split. B1 diperbarui: materialisasi `images/{train,val,test}` tetap terbuka, kini dijadikan syarat NB04 | Orchestrator |
| 1.13 | 2026-10-21 | **Kelas pantulan + audit dataset Riset-4.** NB02b ditambah penandaan `kind: distractor` (hotkey `d`) — box pantulan/glare tidak ikut export COCO/YOLO tapi tetap tersimpan untuk audit; label box kini bisa disembunyikan (tombol + hotkey `T`). Keputusan: kelas `PantulanCahaya` **tidak** langsung dibuat, diukur dulu (dasar: FP rate DINO 0,4–0,8% dari Riset-4). Audit fisik dataset Riset-4 kecil: `AnotasiManual` 490 crop **tanpa koordinat** (tak bisa fine-tune), `golden_set_v3.csv` kolom koordinat **korup** oleh Excel. Dibuat konverter `build_riset4_golden_to_coco.py` → **199 anotasi / 182 frame** dari `candidates_index.csv` + frame asli. Risiko baru **R9** dan **R10** | Orchestrator |
| 1.14 | 2026-10-22 | **`PantulanCahaya` dinaikkan jadi kelas resmi ke-3.** Data nyata membantah asumsi R9: dari 126 frame teranotasi, **313/940 box (33,3%)** adalah pantulan, ada di **33% frame**, proyeksi **~994 box** pada 400 frame. NB02b kini 3 kelas (`nc: 3`, index 2), hotkey `1/2/3` + `d` sebagai shortcut, migrasi otomatis store lama (`kind: distractor` → `class_name: PantulanCahaya`, teruji 319 box, 940 box utuh). Bug label kelas (`1: LampuMotor` ganda) diperbaiki. **R9 ditutup**, **R11 baru** (filter kelas pantulan wajib di NB04/NB05). Artefak export parsial sisa run tak sengaja dihapus. **Bug migrasi ditutup:** sel STORE ternyata memuat store dengan `json.load` mentah sehingga migrasi tidak pernah jalan — kini memakai `load_annotation_store()`, terverifikasi pada store asli (0 sisa field `kind`, 319 box dimigrasi, 940 box aktif utuh). Konverter Riset-4 dipindah ke `Temp/opencode/build_riset4_golden_to_coco.py` | Orchestrator |
| 1.15 | 2026-10-26 | **Bug navigasi yang menimpa kelas box ditutup.** Dilaporkan pengguna: `d` lalu `r` lalu `n`, saat kembali kelasnya balik. Akar masalah: `go_to_index()` memanggil `sync_class_toggle()` — fungsi itu ternyata **penulis** (menerapkan nilai ToggleButtons ke box terpilih), jadi setiap frame yang dibuka kembali menimpa box terakhirnya dengan kelas basi. Dipecah menjadi `apply_toggle_to_selection` (observer, menulis) dan `show_selected_class_in_toggle` (baca saja). Sesudah perbaikan, 12× next + 12× prev mengubah **0 dari 400 frame**. Dampak data lama diukur: box terakhir berubah kelas **74/83 (89%)** vs box lain **11/161 (7%)**, dan `vehicle_vote` tidak pernah mendukung kelas pengganti → **R12** dicatat. Data lama **tidak** diubah; 74 box masuk `output/annotation/audit_navigation_overwrite.csv` untuk cek manual. Ikut ditutup: `on_rectangle_selected` masih memakai `BOX_KIND_TARGET` yang sudah dihapus (menambah box baru = `NameError`) | Orchestrator |
| 1.16 | 2026-10-26 | **NB03–NB05 dibuat, diverifikasi, dan smoke-tested.** NB03: kalibrasi threshold post-hoc dari skor chunk (tanpa GPU), IoU-matching greedy per kelas, metrik per split, `PantulanCahaya` dikeluarkan dari F1 makro karena pseudo-label hanya 2 kelas. NB04: materialisasi split dari `split_camera_map.csv` (parser melewati baris komentar), hardlink gambar, label gold menimpa pseudo untuk frame teranotasi, `data.yaml` `nc: 3`, guard menolak training bila frame val gold < 30. NB05: metrik per split + confusion matrix dari objek `model.val()`, sapuan confidence sekali-jalan, proxy counting dengan filter R11, galeri error; inferensi sapuan di CPU karena VRAM 4 GB tidak cukup untuk `predict` setelah dua `val`. **B1 sisa (materialisasi split) ditutup**, **R11 difilter di kode**, **R13 baru** (test tanpa ground truth). Dua bug tertangkap smoke test: stem frame chunk tertimpa (`frame_000015` tanpa `video_stem`) dan parser `split_camera_map.csv` salah menganggap baris komentar sebagai header | Orchestrator |
| 1.17 | 2026-10-26 | **Audit keselarasan taksonomi — hasil: TIDAK SELARAS (R14 kritis 25).** Koreksi pemilik riset: pipeline dibangun untuk mendeteksi **lampu silau**, **lampu berwarna**, dan **lampu silau+berwarna**, dengan keluaran sistem hitung (jumlah kendaraan mobil/motor + jumlah lampu silau / berwarna / silau+berwarna per jenis). Penelusuran kode: tidak ada satu pun label/kelas/prompt "silau" atau "berwarna" — NB02b `CLASS_NAMES` = `LampuMobil`/`LampuMotor`/`PantulanCahaya`, NB02 `CLASS_PROMPTS` = `car headlight.`/`motorcycle headlight.`, NB03–NB05 mewarisi daftar yang sama, NB05 menghitung kendaraan (bukan kondisi lampu). Mitigasi: MASTER §1 ditulis ulang (tujuan + tabel 8 keluaran hitung), **§10.6 baru** (bukti per notebook + kontrak agregasi 8 kelas + urutan perbaikan), gate larangan training/klaim di §10.3 langkah 0, dua keputusan terbuka §13.9 (label space) dan §13.10 (satuan hitung) | Orchestrator |
| 1.18 | 2026-10-29 | **Audit eksternal Codex ditindaklanjuti + gold quality audit.** Diukur ulang pada 170 frame gold (`builders/audit_gold_quality.py`): 1.897 box, `PantulanCahaya` 46,1%, **proxy recall proposal DINO 0,606**, 396 box pseudo dihapus (31%), 0 pasangan duplikat, 1 box <8 px, 0 refleksi menumpuk di kendaraan. Diterapkan: label preview `?` → `UNSURE`, `DETECTION_THRESHOLD` 0,30→0,25, `VEHICLE_SCORE_THRESHOLD` 0,40→0,30 (`VEHICLE_VOTE_MIN_COVERAGE` tetap 0,30), gate kelas train NB04 (<20 box ditolak, <5 box val diperingatkan), serta kontrak box warna NB05 kini `n/a` (bukan `0`). Ditolak: bin sampling NB02b (mengorbitkan 170 frame reviewed dari export). Ditunda: run manifest, lock hash, CLI headless, bootstrap CI, anotator kedua, gold counting clip. Risiko baru **R18** (recall proposal) dan **R19** (val 0/83). Rincian: MASTER §13.9b | Orchestrator |
