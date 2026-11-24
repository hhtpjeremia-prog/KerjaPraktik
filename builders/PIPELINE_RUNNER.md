# Runner pipeline notebook

## Rerun setelah Ponytail Audit 3 (2026-11-18)

Audit 3 memperbaiki kode evaluasi NB05, tracking/counting dan pembuatan artefak NB06, serta caching dashboard.
Audit tersebut tidak mengubah pseudo-label, annotation store, Ground Truth, konfigurasi/model YOLO atau CNN,
mask SAM2, maupun fitur training. Jika semua artefak upstream itu berasal dari run yang sama dan lengkap,
rerun terarah cukup:

```powershell
.\builders\run_pipeline.ps1 -StartAt NB05
```

Runner menjalankan NB05 lalu NB06 secara berurutan. NB05 memperbarui evaluasi/threshold dan artefak evaluasi;
NB06 memakai hasil tersebut untuk memperbarui tracking, overlay, checkpoint/rekap, galeri, dan dashboard.
Jangan jalankan NB05 atau NB06 bersamaan dari VS Code dan runner. Sebelum memulai, pastikan tidak ada proses
pipeline atau kernel yang sedang mengeksekusi salah satu notebook tersebut.

NB02b/NB02c/NB02d/NB02e/NB03/NB04/NB04b tidak perlu diulang untuk perubahan Audit 3 ini. Sampling NB02b
yang kini dapat dikonfigurasi hanya berlaku jika memang ingin membuat ulang sampling/anotasi; perubahan itu
tidak mengharuskan regenerasi anotasi yang sudah ada. Jika prasyarat atau input model ternyata berubah, ikuti
dependensi pada bagian pipeline penuh di bawah, bukan rerun terarah ini.

`run_pipeline.ps1` menjalankan notebook komputasi secara berurutan dan menyimpan log setiap tahap di
`output/pipeline_runs/<timestamp>/`. Runner menjalankan notebook lewat kernel `.venv` dan menulis output
kembali ke notebook (`nbconvert --inplace`).

Pada Windows PowerShell 5.1, runner sementara mengatur `$ErrorActionPreference` ke `Continue` saat
menangkap stdout/stderr nbconvert. PowerShell dapat mengubah pesan status biasa pada stderr menjadi
`NativeCommandError` ketika preference bernilai `Stop`; keberhasilan notebook ditentukan dari exit code
proses Python.

Sebelum commit atau menjalankan pipeline panjang, jalankan `builders/run_checks.ps1`. Ini menjalankan
logic checks, dashboard state checks, dan membandingkan source semua 11 notebook builder dengan file
`.ipynb` di root. Jika source notebook tertinggal, sinkronkan dengan
`python builders/notebook_gate.py --sync-sources`, lalu jalankan kembali `run_checks.ps1`. Sel kode
yang berubah dibersihkan output-nya agar hasil lama tidak terlihat seolah berasal dari source baru.
NB01b dan NB01c tidak memiliki builder Python dan tidak termasuk pasangan gate.

NB04 memerlukan export YOLO NB02b dengan vocabulary lima kelas yang sama persis dengan `pipeline_common.py`.
Jika export perlu diperbarui, jalankan ulang sel export pada NB02b dari store anotasi aktif; sel itu
adalah satu-satunya writer resmi COCO dan YOLO. Export lama dipindahkan ke backup bertimestamp sebelum
data baru dipasang.

## Tahap dan pilihan mulai

Urutannya adalah NB02d base → NB02e mask SAM2 → NB03 → NB04 YOLO → NB04b CNN/e-means → NB05 → NB06.
NB02b anotasi manual dan NB02c ground truth adalah tahap interaktif, jadi harus selesai sebelum runner
dijalankan. Saat NB02d masih berjalan di VS Code, jangan mulai runner dari NB02d karena akan membuat
proses training kedua. Setelah proses itu selesai, mulai dari tahap berikutnya.

```powershell
# Mulai dari awal alur komputasi, hanya jika NB02d tidak sedang berjalan
.\builders\run_pipeline.ps1 -StartAt NB02d

# Pilihan yang sesuai bila NB02d sudah selesai dan proposal-nya complete
.\builders\run_pipeline.ps1 -StartAt NB02e

# Lanjutkan dari satu tahap tertentu setelah memperbaiki/menyiapkan prasyaratnya
.\builders\run_pipeline.ps1 -StartAt NB04b
```

NB02e memerlukan proposal NB02d berstatus `complete` dan GT NB02c. NB04b memerlukan mask manifest
NB02e berstatus `complete`. Jika source e-means belum ada, runner menjalankan installer resmi satu kali
sebelum NB04b; instalasi ini perlu akses internet.

Jalankan satu tahap notebook pada satu waktu. Jangan membuka eksekusi notebook yang sama lewat VS Code
dan runner secara bersamaan; dua kernel akan bersaing memakai memori/GPU dan dapat membuat output
parsial atau menghentikan kernel.

## Pemulihan NB06

NB06 menyimpan satu checkpoint atomik per video di `output/nb06/video_checkpoints/` setelah video
selesai diproses. Jika eksekusi terhenti, mulai lagi dari NB06: checkpoint yang cocok dengan input,
bobot, threshold, ROI, konfigurasi SAM2, dan tracker akan dipakai untuk menyusun ulang artefak global.
Video tanpa checkpoint valid, termasuk video yang berhenti di tengah, diproses ulang dari awal karena
state internal ByteTrack tidak disimpan per frame. Jangan menghapus checkpoint valid saat melanjutkan.
Overlay parsial yang belum punya checkpoint tidak menandakan video sudah selesai.

Pasangan overlay dengan/tanpa mask dan crop galeri bermask dibentuk pada tahap pasca-NB06. Toggle di
dashboard hanya mengubah tampilan; ia tidak memicu inferensi atau mengubah counting. Jika hanya
pasangan presentasi yang perlu dibangun ulang, gunakan builder `builders/build_nb06_display_variants.py`
setelah memastikan manifest deteksi, video sumber, dan bobot SAM2 tersedia.
Dalam mode smoke, builder tersebut menulis ke `output/smoke/nb06`, bukan artefak dashboard riset.

## Jika notebook gagal

Runner menyimpan log tahap, notebook dengan traceback, `run_summary.json`, dan `codex_debug_prompt.md`,
lalu berhenti pada kegagalan. Log notebook bisa berisi teks dari nama file, anotasi, atau data eksternal;
runner tidak mengirimkannya ke agent yang dapat mengubah working tree. Baca prompt/log secara manual,
perbaiki source, lalu jalankan ulang dari stage yang gagal. `-RetryFailedStageOnce` mengaktifkan satu
retry tanpa perubahan source, untuk kegagalan sementara saja; jangan gunakan untuk error data atau
konfigurasi yang deterministik.

Sebelum diagnosis, prompt menyarankan membaca [`CODEX_RUNNER_CONTEXT.md`](CODEX_RUNNER_CONTEXT.md),
README, kontrak aktif master, dan file export sesi yang tersedia. Export dapat berisi keputusan lama;
source dan kontrak aktif tetap menjadi acuan. Runner tidak menjalankan Codex CLI dan tidak dapat mengirim
pesan ke sesi ChatGPT.

## Batas runner

- Runner hanya menjalankan NB02d, NB02e, NB03, NB04, NB04b, NB05, dan NB06; anotasi manual/GT tetap
  dikerjakan lewat notebook interaktif sebelum run.
- Retry opsional dibatasi satu kali dan tidak mengubah source. Secara default runner berhenti setelah
  kegagalan agar operator dapat meninjau log sebelum stage mahal dijalankan ulang.
- Perbaikan yang bergantung pada data hilang, jaringan, kredensial, GPU/RAM, atau tindakan manual akan
  dilaporkan sebagai prasyarat; Codex tidak boleh mengarang atau mengubah data untuk menutupinya.
- Output log berada di bawah `output/`, bukan source dataset atau annotation store.
