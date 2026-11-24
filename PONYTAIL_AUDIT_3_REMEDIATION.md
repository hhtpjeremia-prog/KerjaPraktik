# Remediasi Ponytail Audit 3

Tanggal: 18 November 2026. Dokumen ini memetakan 17 temuan audit ke perubahan source aktif. Audit ini
adalah code/data-safety review; tidak menjalankan training model, inferensi video riset, frame restore,
atau smoke pipeline end-to-end.

| # | Temuan | Penanganan |
|---:|---|---|
| 1 | Notebook dapat berbeda diam-diam dari builder | `builders/notebook_gate.py --check-all` membangun source notebook di memori lalu membandingkan seluruh source sel dengan notebook tersimpan. `run_checks.ps1` menjalankannya. `--sync-sources` memperbarui source sambil mempertahankan output sel yang tidak berubah dan menghapus output sel kode yang source-nya berubah. |
| 2 | Smoke NB06 menulis ke artefak dashboard riset | `build_nb06_display_variants.py` memilih `output/smoke/nb06` saat `PIPELINE_SMOKE=1`; video sumber tetap dibaca dari `output/videos`. |
| 3 | Hash restore berbeda dari NB01 dan manifest dapat berubah setelah kegagalan | `restore_frames_from_manifest.py` memakai `pipeline_common.difference_hash`. Kedua CSV manifest ditulis atomik setelah semua frame lolos verifikasi; mismatch mempertahankan manifest lama dan menghasilkan laporan kegagalan. |
| 4 | Gate sampling NB02b dipatok ke 826/190 | Komposisi keras dihapus. Target dapat diatur melalui `NB02B_SAMPLE_TOTAL_FRAMES`; default sampling historis tetap 1.016, dan komposisi aktual dicatat per manifest. `allocate_protected_sample_quota` menjaga semua frame reviewed/proposal dan mengalokasikan sisa budget proporsional. Target yang tidak dapat dicapai atau frame terlindungi yang melebihi target menghasilkan pesan eksplisit. |
| 5 | Dua writer ekspor NB02b berbeda | `builders/refresh_nb02b_exports.py` dihapus. Sel export NB02b menjadi writer tunggal COCO dan YOLO. |
| 6 | Event clip/crop dibuat untuk kelas yang tidak tampil | Pembuatan event NB06 dibatasi ke `MobilTidakStandar` dan `MotorTidakStandar`; kelas standar dan pantulan tidak menghasilkan clip/crop galeri. |
| 7 | Dashboard membaca ulang semua file di setiap interaksi | Loader snapshot diberi `st.cache_data(ttl=30)` dengan kunci root dan fingerprint mtime/ukuran lima artefak. |
| 8 | Logic hitung tidak dapat diuji tanpa GPU | `qualify_by_class_threshold`, `resolve_track_class`, `summarise_units`, dan `detect_line_crossings` dipindah ke `pipeline_common.py`; unit checks baru menguji threshold/roll-up dan event crossing, termasuk eksklusi kelas standar. |
| 9 | Skrip sekali pakai bercampur dengan pipeline | Sembilan skrip diagnostik/sekali pakai dipindah ke `archive/one-off-scripts/`. `build_nb02c_ground_truth.py` tetap aktif karena merupakan generator notebook yang sekarang diperiksa gate. Audit gold dan repair GT tetap aktif karena masih dirujuk. |
| 10 | Daftar kelas keluarga tidak dipakai | `MOBIL_CLASS_NAMES` dan `MOTOR_CLASS_NAMES` dihapus; kode tetap memakai satu vocabulary `CLASS_NAMES`. |
| 11 | Parameter threshold decoder CNN mati | Argumen threshold dihapus dari `decode_attributes` dan pemanggil NB06; keputusan multiclass tetap `argmax`. |
| 12 | Fallback threshold berulang dan tidak mungkin tercapai | Pembacaan menjadi `selected_thresholds[class_name]`; pemeriksaan kelas yang hilang dilakukan lebih dulu dengan error yang eksplisit. |
| 13 | Daftar kolom dashboard menduplikasi vocabulary | Loader membentuk daftar kolom langsung dari `CROSSINGS_COUNTED_COLUMN`, `CLASS_NAMES`, dan `COUNT_CONTRACT`. |
| 14 | Dekorasi UI dan grafik duplikat | Warna event di-inline, tab tunggal dihapus, grafik playback yang mengulang metrik frame dan chart ringkasan duplikat dihapus. |
| 15 | Test metric disalin ke setiap epoch; CSV pemilihan k ditulis berulang | Test metric hanya disimpan sebagai ringkasan evaluasi, tidak menjadi kolom tiap epoch. CSV kandidat e-means tetap ditulis per kandidat untuk recovery dan sekali lagi setelah skor/final pilihan tersedia; penulisan interim tanpa nilai dieliminasi. |
| 16 | Backup restore berpotensi collision; environment test bocor | Nama backup diberi suffix `_1`, `_2`, … bila timestamp sama. `verify_dashboard_states.py` mengembalikan `NB06_ROOT` ke nilai awal dalam `finally`. |
| 17 | Dependency YAML/OpenCV berlebih | NB05 memakai `ultralytics.utils.YAML`; pin langsung PyYAML dihapus. `opencv-contrib-python` diganti `opencv-python`. Environment `.venv` yang ada belum dibersihkan dari paket lama karena proses Python lain terdeteksi saat audit. |

## Verifikasi

- `python builders/notebook_gate.py --sync-sources`: source 11 notebook generator disamakan; output dibersihkan hanya pada sel kode yang source-nya berubah.
- `python builders/notebook_gate.py --check-all`: seluruh 11 pasangan builder/notebook cocok.
- `python test_pipeline_logic.py`: 22 pemeriksaan lolos.
- `python builders/verify_dashboard_states.py`: state empty/ready/partial/broken lolos; Streamlit mencetak warning `missing ScriptRunContext` saat dijalankan standalone, tetapi semua assertion lolos.
- `py_compile` untuk semua source Python yang disentuh lolos.
- Training, restore frame terhadap dataset, smoke end-to-end, dan inferensi NB06 penuh tidak dijalankan.

Notebook `NB01b_roi_config.ipynb` dan `NB01c_curated_sampling.ipynb` tidak mempunyai builder Python; keduanya bukan bagian dari pasangan yang bisa dibandingkan oleh builder gate. Source notebook tersebut tetap tersimpan di Git dan perlu direview manual bila diedit.
