# Pengetahuan Pipeline CV — Hasil Ekstraksi `Computer-Vision/`

> Sumber: `C:\Users\Jeremmy\LocalProjects\Python_Things\Computer-Vision`
> Diekstrak: 2026-10-26 · 103 notebook, 14.838 baris kode setelah output/plot dibuang.
> Skrip ekstraksi ulang: `builders/extract_cv_knowledge.py`.
> Dokumen ini **bukan** rangkuman kursus. Isinya hanya yang mengubah keputusan di pipeline ini.

---

## 1. Karakter arsip (batas kepercayaan)

| Aspek | Kenyataan |
|---|---|
| Bentuk | Kursus "Modern Computer Vision" — 2 folder (`OpenCV` 51 file, `Deep Learning CV` 77 file) + 1 PDF slide + 1 zip |
| Isi | Resep per-topik: thresholding, contour, tracking, YOLOv3/v4/v5, TFOD API, Detectron2, GAN, ViT |
| Yang **tidak** ada | Repo, test, config terpusat, penanganan malam/glare, pseudo-labelling, split anti-leakage, alat anotasi |
| Kesimpulan | Layak dipakai sebagai **sumber rumus dan konstanta**, tidak layak dipakai sebagai **cetak biru arsitektur** |

Semua pola di bawah sudah diverifikasi ada di kode notebook, bukan dari judul.

---

## 2. Pola yang diambil dan dipakai

| # | Pola | Sumber notebook | Dipakai di | Bentuk konkret |
|---|---|---|---|---|
| 1 | Filter warna HSV `inRange`, merah butuh dua mask | `OpenCV/21. Filtering Colors` | Definisi operasional "berwarna" (NB02b/NB03) | Biru `H 90–135`; merah `H 0–10` **dan** `170–180` (hue wrap) |
| 2 | K-Means dominan warna | `OpenCV/19. Histograms and K-Means` | Audit label "berwarna" (NB03) | 3 cluster pada piksel dalam box; ambil cluster dengan porsi terbesar, bukan rata-rata piksel |
| 3 | Skor blur = varian Laplacian | `OpenCV/39. Detect Blur in Images` | Saringan frame gold (NB03) | `cv2.Laplacian(gray, cv2.CV_64F).var()`; nilai **rendah** = blur |
| 4 | Adaptive/Otsu pada kanal V | `OpenCV/9. Thresholding & Adaptive` | Ukur luas bloom = dasar "silau" (NB03) | Ambang `V ≥ 240`, hitung luas wilayah terang / luas box |
| 5 | Contour area + moments | `OpenCV/12. Moments, Sorting, Approximating` | Pusat & luas blob lampu (NB03) | `cv2.contourArea`, centroid `m10/m00`, `m01/m00` |
| 6 | `SimpleBlobDetector` | `OpenCV/13. Line, Circle and Blob` | Pembanding tanpa model untuk bloom | Deteksi blob terang sebagai sanity check angka pola 4 |
| 7 | NMS `cv2.dnn.NMSBoxes` | `OpenCV/33. YOLOv3 in OpenCV` | Dedup box hasil pairing lampu (NB02) | `NMSBoxes(boxes, conf, 0.5, 0.3)` |
| 8 | Background subtraction MOG2/KNN + `accumulateWeighted` | `OpenCV/23. Background and Foreground` | Bukti pendukung gerak kendaraan, **bukan** label | `cv2.createBackgroundSubtractorKNN()`; hanya untuk validasi silang |
| 9 | EarlyStopping + ModelCheckpoint monitor `val` | `Deep Learning CV/19. Lightning`, `20. Transfer Learning` | Disiplin training NB04 | Pantau `val`, `patience` terbatas, simpan bobot terbaik saja |
| 10 | Aturan jumlah iterasi darknet | `Deep Learning CV/45. Pot Hole TinyYOLOv4` | Batas bawah epoch NB04 | `max_batches = kelas × 2000`, `steps` 80%/90% — **khusus darknet**, untuk ultralytics pakai `patience` pada `mAP50-95` val |
| 11 | GradCAM / GradCAM++ heatmap | `Deep Learning CV/11. CNN Visualisation` | Panel interpretability NB05 | Heatmap per box: model harus melihat lampu, bukan pantulan aspal |

Pola 1–6 adalah alasan utama dokumen ini ada: **"silau" dan "berwarna" harus jadi angka**, bukan perasaan anotator.

---

## 3. Pola yang sengaja tidak dipakai

| Pola | Alasan ditolak |
|---|---|
| Haar cascade kendaraan (`OpenCV/17`) | Berbasis tepi abu-abu; di malam hari bodi kendaraan tidak punya kontras, hanya lampunya |
| TFOD API / Detectron2 (`43`–`57`) | Dependency berat, tidak ada keunggulan di VRAM 4 GB; YOLO sudah jalan |
| Watershed (`OpenCV/22`) | Butuh batas objek yang jelas; bloom lampu justru menghapus batas itu |
| MeanShift / CAMSHIFT (`OpenCV/24`), tracking warna (`26`) | Butuh frame berurutan; NB01 melakukan sampling, bukan video kontinu |
| DeepSORT (`58. Yolov5 DeepSort`) | Sama: tracker identitas butuh temporal kontinu. Lihat §5 |
| GAN / style transfer / ViT klasifikasi (`25`–`36`, `60`) | Di luar cakupan riset |

---

## 4. Celah arsip — harus diputuskan sendiri, tidak bisa diwarisi

1. **Glare/bloom malam hari** tidak dibahas sama sekali. Pola 4 & 6 hanya memberi alat ukur, bukan ambang.
2. **Pseudo-labelling + koreksi manusia** tidak ada.
3. **Split per kamera & anti-leakage** tidak ada.
4. **Kontrak agregasi hitung** tidak ada.
5. **Alat anotasi** tidak ada.

---

## 5. Temuan yang mengubah desain: satuan hitung

Ini konsekuensi paling penting dari arsip ini.

DeepSORT (`58`) ada di arsip, dan itu mengingatkan satu hal: **identitas kendaraan hanya bisa dilacak pada frame berurutan**. NB01 mengambil *sampling* frame (jarak antar-frame dalam satu video tidak kontinu), sehingga:

- Menjumlahkan deteksi lintas frame **bukan** "jumlah kendaraan" — kendaraan yang sama muncul di banyak frame dan terhitung berkali-kali.
- Tanpa tracker, angka yang sah hanyalah **jumlah deteksi per frame** (dan agregatnya per kamera/jam).
- Untuk mendapat "kendaraan unik", perlu salah satu: (a) tracking pada frame berurutan, atau (b) garis hitung (counting line) dengan arah lintasan, atau (c) dedup lintas frame berbasis kemiripan box+waktu.

Status: **belum diputuskan** — dicatat sebagai keputusan terbuka di MASTER §13.11. Sampai diputuskan, keluaran NB05 harus diberi label satuan yang jujur (`deteksi per frame`), bukan `jumlah kendaraan`.

---

## 6. Lima aturan yang dipakai saat menulis builder

1. **Ukur, jangan tebak.** Setiap klaim atribut ("silau", "berwarna") punya angka pendukung: rasio luas bloom, hue dominan, varian Laplacian.
2. **Ambang dari `val` saja**, lalu dipakai apa adanya di `test`. Tidak pernah dari train+val gabungan.
3. **NMS sebelum menghitung.** Pairing lampu menghasilkan box bertumpuk; hitung setelah dedup.
4. **Interpretability wajib** untuk klaim atribut — GradCAM per box, bukan hanya mAP.
5. **Satuan hitung harus eksplisit.** Kalau frame di-sampling, jangan pernah menyebut hasilnya "jumlah kendaraan unik".
