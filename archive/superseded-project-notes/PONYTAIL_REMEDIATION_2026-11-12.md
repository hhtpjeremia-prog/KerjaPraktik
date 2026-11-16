# Implementasi audit Ponytail — 12 November 2026

Dokumen ini mencatat status perbaikan atas `ponytail-audit.md` dan `ponytail-review.md`.
File audit asli dipertahankan sebagai rekaman temuan; kontrak aktif proyek ada di README dan
bagian pembuka Master Dokumentasi.

## Temuan yang diperbaiki

- **NB06 class counts:** kelas dihitung dari seluruh deteksi yang lolos kualifikasi; subset
  `TidakStandar` tetap menjadi sumber hitungan resmi bbox event. Jadi kelas Standar dan
  `PantulanCahaya` tidak lagi nol secara struktural.
- **Kamera tanpa bbox event:** tabel per video dibuat dari video yang benar-benar diproses,
  termasuk baris nol. Loader dashboard menerima CSV kosong dengan skema yang diketahui dan
  memeriksa versi taksonomi/kontrak agar artefak lama tidak ditampilkan sebagai angka nol.
- **Satuan dan kontrak hitung:** NB06 memakai `jumlah_bbox_mobil_tidak_standar` dan
  `jumlah_bbox_motor_tidak_standar`; `bbox_counted` tidak diduplikasi sebagai `vehicles_counted`.
  `unique_vehicles_audit` tetap terpisah sebagai audit tracking.
- **NB05:** hanya melaporkan bbox detector `Mobil` dan `Motor` per kelas/kamera. NB05 tidak
  mengklaim menghitung kondisi lampu, dan tabel `n/a` yang sebelumnya selalu kosong dihapus.
- **Klasifikasi CNN:** fitur numerik diambil dari crop bersih sebelum augmentasi citra. Transform
  tensor train dan inference memakai satu definisi. Checkpoint dengan `model_type` yang salah
  ditolak sebelum NB06 memproses video; prediksi kondisi dicache per track untuk konsistensi.
- **ROI inference:** sumber training YOLO adalah crop ROI, sedangkan NB06 sebelumnya menjalankan
  YOLO pada frame penuh yang dimasking. NB06 kini mendeteksi pada crop ROI lalu memetakan bbox
  kembali ke koordinat frame penuh untuk overlay/tracking. Ini mengatasi mismatch preprocessing.
- **Metadata anotasi ROI:** audit menemukan 81 entri store menyimpan lebar/tinggi lama walau gambar
  pada manifest aktif telah memakai dimensi ROI baru. Koordinat bbox berada di dalam gambar aktif;
  tidak ada dasar untuk menggeser bbox. Metadata store dan COCO diperbarui mengikuti gambar aktif,
  lalu export YOLO dibangun ulang. Backup dibuat oleh `builders/repair_annotation_dimensions.py`.
- **Export NB02b:** label/gambar YOLO dibangun di direktori staging, diverifikasi, kemudian dataset
  aktif dipindahkan ke backup dan staging dipromosikan. Kegagalan validasi sebelum promosi menjaga
  export lama tetap utuh.
- **Dashboard dan smoke tooling:** dead UI 9-kelas, legend merah yang tidak cocok, label hitungan
  yang ambigu, guard kolom epoch, fixture smoke, dan pemeriksaan dashboard diperbarui ke kontrak kini.
- **Lainnya:** builder patch referensi usang dihapus; penulisan CSV NB03 ganda dibuang; ignore
  `.streamlit/` diubah agar `config.toml` dapat disimpan di git.

## Berkas kerja yang tersinkron

Source notebook diselaraskan dengan builder untuk NB02, NB02b, NB03, NB04, NB04b, NB05, dan NB06.
Output dan execution count notebook lama dipertahankan. Perbaikan dimensi metadata berlaku pada
artefak lokal di `output/annotation/`; backup tidak dihapus.

## Verifikasi dan batasan

Pemeriksaan yang dilakukan pada sesi ini bersifat statis: compile source Python, pemeriksaan
kontrak/import, dan pembandingan source notebook terhadap builder. Training YOLO/CNN, inferensi
video NB06, smoke pipeline penuh, dan tes yang memerlukan artefak model tidak dijalankan. Karena
perbaikan ROI mengubah input detector NB06 agar identik dengan pipeline training, hasil video
overlay dan hitungan NB06 harus dibuat ulang. Dimensi yang dipakai untuk normalisasi label YOLO
juga berubah, sehingga dataset sudah diekspor ulang dan NB04 perlu dilatih ulang. NB05 lalu
mengukur dan mengkalibrasi bobot baru sebelum NB06 dijalankan.

## Belum diklaim selesai

- Temuan kompleksitas kuadratik penyambungan track tetap sebagai catatan optimasi, sesuai audit.
- Kualitas model setelah refactor belum dapat dinilai tanpa training dan evaluasi ulang.
- Audit tidak membuktikan akurasi overlay pada semua video; periksa hasil NB06 baru secara visual.


## Pembaruan dokumentasi lain

README, Master Dokumentasi, Risk Register, panduan ground truth, dokumentasi dashboard, referensi CV, catatan audit historis, dan source generator brief dosen diperbarui agar memakai kontrak aktif. File audit Ponytail asli dipertahankan sebagai bukti temuan dan diberi tautan ke status remediasi. File `.docx` brief tidak ada di workspace pada saat audit; generatornya diperbarui, tetapi file tersebut tidak dibuat ulang.

## Perbaikan NB05 dan ekspor ground truth (12 November 2026)

NB05 sempat memakai label GT YOLO yang tidak cocok dengan bobot detector: file mengandung ID kelas
`0–2`, sedangkan bobot NB04 hanya memiliki `Mobil` dan `Motor` (ID `0–1`). Sebagian koordinat
ternormalisasi memakai dimensi crop lama, sehingga Ultralytics mengabaikan sebagian besar gambar.
Selain itu, cell kalibrasi NB05 memanggil `training_class_name` tanpa mengimpornya.

Ekspor GT dibangun ulang dari annotation store NB02c dengan kontrak dua kelas yang saat ini dilatih:
box Standar/TidakStandar digabung menjadi `Mobil` atau `Motor`; `PantulanCahaya` tetap di store/COCO
untuk audit tetapi tidak menjadi target atau hard-negative berlabel khusus. Semua 254 pasangan gambar
dan label kini memiliki ID `0–1`, koordinat valid, dan metadata dimensi yang cocok. Ekspor sebelumnya
disimpan di `output/ground_truth/_backup_yolo_repair_20261112_035702/`. Jumlah target detector baru
adalah 701 bbox Mobil dan 1.452 bbox Motor; 829 anotasi pantulan tidak diekspor.

NB05 sekarang mengimpor helper yang hilang dan memvalidasi seluruh label GT sebelum memanggil
Ultralytics, agar evaluasi berhenti dengan pesan jelas jika kontrak/koordinat kembali rusak. Metrik
test lama ditandai stale karena dihitung dari ekspor yang tidak valid. NB04 tidak perlu dilatih ulang;
jalankan ulang NB05 untuk menghitung metrik test dan threshold dari bobot yang sudah ada.

## Pemulihan ByteTrack NB06

Traceback `numpy.linalg.LinAlgError: Singular matrix` terjadi saat ByteTrack memperbarui kovarians
Kalman XYAH. NB06 sekarang menyaring bbox non-finite atau berdimensi di bawah 1 piksel sebelum
tracker, lalu menyediakan satu kali reset/retry jika kovarians tetap singular. Reset mempertahankan
urutan ID agar track baru tidak memakai ulang ID yang telah tercatat. Jika retry gagal, frame itu
dilewati dan dihitung di ringkasan video; angka untuk video tersebut perlu ditinjau bila jumlah frame
terlewat tidak nol. Perubahan taksonomi PantulanCahaya belum diterapkan ke run aktif dan tetap
ditunda sampai run NB06 ini selesai.
