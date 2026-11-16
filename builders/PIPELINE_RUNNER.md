# Runner pipeline notebook

`run_pipeline.ps1` menjalankan notebook komputasi secara berurutan dan menyimpan log setiap tahap di
`output/pipeline_runs/<timestamp>/`. Runner menjalankan notebook lewat kernel `.venv` dan menulis output
kembali ke notebook (`nbconvert --inplace`).

Pada Windows PowerShell 5.1, runner sementara mengatur `$ErrorActionPreference` ke `Continue` saat
menangkap stdout/stderr nbconvert. PowerShell dapat mengubah pesan status biasa pada stderr menjadi
`NativeCommandError` ketika preference bernilai `Stop`; keberhasilan notebook ditentukan dari exit code
proses Python.

NB04 memerlukan export YOLO NB02b dengan vocabulary lima kelas yang sama persis dengan `pipeline_common.py`.
Jika manifest export lama tidak mencatat vocabulary tersebut, refresh artifact dari store anotasi yang
sedang aktif menggunakan `python builders/refresh_nb02b_exports.py`, lalu jalankan ulang dari NB04.
Script menyimpan export YOLO lama dan COCO lama di folder backup bertimestamp.

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
