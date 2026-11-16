from __future__ import annotations

from datetime import date
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile
from xml.sax.saxutils import escape


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "BRIEF_PEMAHAMAN_INFRA_PIPELINE_CCTV.docx"


def run(text: str) -> str:
    return escape(text).replace("\n", "<w:br/>")


def paragraph(text: str = "", style: str = "Normal") -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="{style}"/></w:pPr><w:r><w:t xml:space="preserve">{run(text)}</w:t></w:r></w:p>'


def bullet(text: str) -> str:
    return f'<w:p><w:pPr><w:pStyle w:val="ListBullet"/></w:pPr><w:r><w:t xml:space="preserve">{run(text)}</w:t></w:r></w:p>'


def heading(text: str, level: int = 1) -> str:
    return paragraph(text, f"Heading{level}")


def table(rows: list[list[str]], widths: list[int] | None = None) -> str:
    widths = widths or [3000] * len(rows[0])
    out = ['<w:tbl><w:tblPr><w:tblBorders><w:top w:val="single" w:sz="4"/><w:left w:val="single" w:sz="4"/><w:bottom w:val="single" w:sz="4"/><w:right w:val="single" w:sz="4"/><w:insideH w:val="single" w:sz="4"/><w:insideV w:val="single" w:sz="4"/></w:tblBorders></w:tblPr>']
    for row_index, row in enumerate(rows):
        out.append("<w:tr>")
        for index, cell in enumerate(row):
            bold = "<w:rPr><w:b/></w:rPr>" if row_index == 0 else ""
            out.append(f'<w:tc><w:tcPr><w:tcW w:w="{widths[index]}" w:type="dxa"/></w:tcPr><w:p><w:r>{bold}<w:t xml:space="preserve">{run(cell)}</w:t></w:r></w:p></w:tc>')
        out.append("</w:tr>")
    out.append("</w:tbl>")
    return "".join(out)


def page_break() -> str:
    return '<w:p><w:r><w:br w:type="page"/></w:r></w:p>'


def build_body() -> str:
    parts: list[str] = []
    parts += [heading("Brief Pemahaman Infrastruktur Pipeline Computer Vision CCTV", 1),
              paragraph("Bahan diskusi dengan dosen pembimbing", "Subtitle"),
              paragraph("Proyek: Deteksi kondisi lampu kendaraan malam hari dan penghitungan kendaraan pada beberapa titik CCTV di Kabupaten Bogor."),
              paragraph("Tanggal dokumen: 12 November 2026"),
              paragraph("Status dokumen: penjelasan teknis dan akademik; bukan laporan hasil akhir."),
              page_break()]

    parts += [heading("Cara membaca brief ini", 1),
              paragraph("Dokumen ini ditulis untuk pembaca yang belum memahami seluruh istilah Computer Vision, machine learning, object detection, tracking, dan dashboard. Baca berurutan dari bagian 1 sampai 5 untuk memahami gambaran besar. Bagian 6 sampai 12 membahas detail teknis dan validitas. Bagian terakhir berisi pertanyaan yang dapat dibawa ke dosen pembimbing."),
              heading("Ringkasan satu halaman", 1),
              paragraph("Penelitian ini membaca rekaman CCTV malam hari untuk mendeteksi mobil/motor, mempelajari pantulan cahaya sebagai kelas eksplisit, dan menilai kondisi lampu. Sistem menghitung bbox tidak standar per frame; ByteTrack memberi track_id untuk menghubungkan deteksi; hitungan resmi tetap bbox TidakStandar per frame."),
              paragraph("Kontrak aktif: anotasi, Grounding DINO/FM, YOLO, dan CNN memakai lima kelas: MobilStandar, MobilTidakStandar, MotorStandar, MotorTidakStandar, dan PantulanCahaya. Kelas PantulanCahaya dipelajari sebagai kelas eksplisit, lalu disaring dari ByteTrack dan counting kendaraan."),
              paragraph("Pipeline tidak langsung mengambil angka dari satu model. Sistem terdiri dari beberapa tahap: memilih frame dari video, membatasi area jalan dengan ROI, membuat proposal awal menggunakan Grounding DINO, mengoreksi sebagian data secara manual, melatih detector YOLO, melatih classifier atribut cahaya, menghubungkan deteksi antar-frame dengan ByteTrack, dan menampilkan hasil di dashboard."),
              table([["Pertanyaan", "Jawaban sementara"], ["Apa yang dideteksi?", "BBox dari lima kelas aktif."], ["Apa yang diklasifikasikan?", "Kendaraan standar/tidak standar serta pantulan cahaya."], ["Apa yang dihitung?", "Track kendaraan TidakStandar yang melintasi garis; pantulan disaring."], ["Apakah hasil sudah akurat di kamera test?", "Belum bisa disimpulkan dari source; metrik test memerlukan holdout ground truth yang selesai direview."], ["Apa status saat ini?", "Pipeline lima kelas perlu dijalankan ulang dan hasilnya dievaluasi sebelum dipresentasikan."]], [2600, 6500]),
              page_break()]

    parts += [heading("1. Masalah penelitian", 1),
              paragraph("Rekaman CCTV malam hari memiliki tantangan yang tidak muncul pada gambar siang hari. Lampu kendaraan dapat sangat kecil, tertutup kendaraan lain, melebar karena bloom atau flare, berubah warna akibat kamera dan white balance, serta muncul berulang kali dalam banyak frame."),
              heading("Tujuan utama", 2),
              bullet("Mendeteksi bbox kendaraan pada crop ROI area jalan."),
              bullet("Membedakan kendaraan mobil dan motor sebagai informasi dasar."),
              bullet("Menentukan apakah lampu standar atau tidak standar."),
              bullet("Menghitung bbox tidak standar pada frame aktif."),
              bullet("Menyajikan hasil agar dapat dipahami oleh pengguna non-teknis."),
              heading("Definisi operasional", 2),
              table([["Istilah", "Definisi kerja"], ["Standar", "Label visual kendaraan yang dinilai memenuhi definisi anotasi penelitian."], ["TidakStandar", "Label visual kendaraan yang dinilai tidak memenuhi definisi anotasi penelitian."], ["PantulanCahaya", "Cahaya pada aspal/bodi yang bukan sumber lampu kendaraan; kelas target supervised yang disaring sebelum tracking dan counting."]], [2500, 6600]),
              paragraph("Definisi kerja Standar/TidakStandar harus disepakati dan diuji konsistensi anotasinya. Label citra tidak sama dengan pengukuran lux/candela dan tidak membuktikan kepatuhan regulasi."),
              heading("Batas klaim", 2),
              paragraph("Sistem ini bukan alat ukur fisika cahaya yang menentukan tingkat silau dalam satuan lux atau candela. Sistem ini adalah sistem Computer Vision yang membuat keputusan berdasarkan citra CCTV, anotasi manusia, fitur visual, dan model statistik."),
              page_break()]

    parts += [heading("2. Gambaran besar arsitektur", 1),
              paragraph("Pipeline dapat dibayangkan sebagai jalur produksi data. Setiap tahap memiliki masukan, proses, keluaran, dan risiko sendiri."),
              table([["Tahap", "Input", "Proses", "Output"], ["NB01", "Video CCTV", "Sampling frame dan pencatatan manifest", "Video, frame, hash, split kamera"], ["NB01b", "Frame kamera", "Menggambar polygon ROI", "Konfigurasi ROI"], ["NB01c", "Frame kandidat", "Kurasi sampling manual", "Manifest frame curated"], ["NB02", "Frame + ROI", "Grounding DINO membuat proposal", "Crop dan pseudo-label"], ["NB02b", "Crop + proposal", "Review/koreksi manusia", "Store lima kelas COCO dan YOLO"], ["NB02d", "Label reviewed", "Fine-tune FM base dan usulkan frame", "Proposal lima kelas untuk train"], ["NB02e", "Box + citra", "Grounded-SAM 2 membuat mask", "Mask manual/GT/pseudo untuk classifier"], ["NB03", "Gold set", "Karakterisasi fotometri", "Audit fitur dan gold set"], ["NB04", "YOLO labels", "Training detector lima kelas", "Bobot YOLO dan metrik validation"], ["NB04b", "Crop, mask, fitur numerik", "ResNet18 weighted multiclass + e-means", "Bobot CNN dan histori train/val/test"], ["NB05", "Gold + bobot YOLO", "Evaluasi detector per bbox/frame", "Threshold dan baseline bbox"], ["NB06", "Video asli + model", "Mask SAM2, ByteTrack, klasifikasi dan counting", "Overlay, track, timeline, crop"], ["Dashboard", "Artefak NB06 dan model", "Playback, filter, chart, interpretasi", "Tampilan untuk analisis/presentasi"]], [1300, 2300, 3500, 2000]),
              heading("Alur konseptual", 2),
              paragraph("Video CCTV → frame terpilih → area ROI → proposal lampu → anotasi manusia → detector kendaraan → classifier atribut → tracking antar-frame → counting → dashboard."),
              paragraph("Pemisahan tahap ini membantu audit. Jika angka salah, kita dapat bertanya apakah masalahnya berasal dari frame, ROI, anotasi, detector, classifier, tracker, atau cara agregasi."),
              page_break()]

    parts += [heading("3. Tiga lapisan label yang sering tertukar", 1),
              paragraph("Salah satu sumber kebingungan terbesar dalam proyek ini adalah adanya tiga representasi label. Ketiganya memiliki fungsi berbeda dan tidak boleh dibandingkan secara langsung."),
              heading("3.1 Anotasi gold dan COCO: kontrak aktif", 2),
              paragraph("Anotasi aktif menyimpan lima kelas dan setiap kelas dipertahankan sebagai target supervised pada FM dan YOLO. CNN mengklasifikasikan crop kendaraan ke empat kelas kendaraan; sampel PantulanCahaya ikut dilatih sebagai kelas kelima."),
              table([["Keluarga", "Kelas"], ["Mobil", "MobilStandar, MobilTidakStandar"], ["Motor", "MotorStandar, MotorTidakStandar"], ["Pantulan", "PantulanCahaya"]], [2200, 6900]),
              heading("3.2 Detector YOLO dan FM: 5 kelas", 2),
              paragraph("YOLO dan Grounding DINO/FM memakai kelas MobilStandar, MobilTidakStandar, MotorStandar, MotorTidakStandar, dan PantulanCahaya. Pantulan dilatih sebagai target eksplisit, lalu disaring dari tracking/counting kendaraan."),
              heading("3.3 Classifier kondisi NB04b: 5 kelas", 2),
              paragraph("NB04b memakai ResNet18 dengan weighted CrossEntropy untuk lima kelas anotasi. Crop RGB, fitur numerik yang memakai mask, dan fitur cluster e-means menjadi masukan. Bobot kelas dihitung dari distribusi train; e-means di-fit hanya pada train. Histori train/validation dicatat per epoch; metrik test hanya sah jika tersedia GT test yang sesuai."),
              paragraph("Kesimpulan penting: detector menentukan lokasi dan tipe atau pantulan; CNN memprediksi kondisi kendaraan atau pantulan; NB06 menyaring pantulan dan hanya menggambar/menghitung bbox TidakStandar."),
              page_break()]

    parts += [heading("4. Penjelasan setiap notebook", 1),
              heading("NB01 — acquisition dan sampling", 2),
              paragraph("NB01 mengambil video atau frame dari sumber CCTV, menyimpan frame yang dipilih, mencatat nomor frame, path, dan hash. Sampling dilakukan karena video penuh sangat besar dan anotasi manual seluruh video tidak realistis."),
              paragraph("Risiko utama: frame sampling dapat kehilangan kendaraan, frame yang terlalu berdekatan dapat membuat data train dan validation terlalu mirip, dan path mesin dapat salah jika environment tidak konsisten."),
              heading("NB01b — ROI", 2),
              paragraph("ROI adalah polygon yang menentukan area jalan yang boleh dilihat model. Area di luar ROI dimatikan atau tidak dipakai. ROI mengurangi false positive dari bangunan, langit, lampu jalan, dan area yang tidak relevan."),
              heading("NB01c — curated sampling", 2),
              paragraph("NB01c membantu memilih subset frame secara manual. Manifest curated harus diperlakukan sebagai sumber data penting karena perubahan manifest dapat mengubah semua tahap berikutnya."),
              heading("NB02 — pseudo-labelling", 2),
              paragraph("NB02 menggunakan Grounding DINO-tiny untuk membuat proposal awal lokasi lampu/kendaraan. Proposal ini bukan ground truth. Ia hanya mempercepat pekerjaan anotator dan tetap harus diperiksa."),
              heading("NB02b — anotasi manual", 2),
              paragraph("NB02b memungkinkan anotator mengoreksi, menghapus, dan menambah box. Store/COCO memuat empat kelas kendaraan dan PantulanCahaya; export YOLO mereduksi kelas kendaraan menjadi Mobil/Motor dan melewati PantulanCahaya."),
              heading("NB03 — audit gold dan threshold", 2),
              paragraph("NB03 menghitung fitur dan memeriksa kualitas gold set. Ia bukan pengganti validasi manusia. Threshold harus dipilih menggunakan validation, bukan test."),
              heading("NB04 — detector YOLO", 2),
              paragraph("NB04 menyusun dataset train/val/test sesuai split kamera dan melatih YOLO. Bobot ini mendeteksi box dan kelas dasar detector. Ia bukan satu-satunya sumber label event akhir."),
              heading("NB04b — classifier atribut", 2),
              paragraph("NB04b memakai ResNet18 pretrained yang menggabungkan tensor citra, 77 fitur numerik dari photometry bbox, photometry foreground mask SAM2 dan geometri mask, serta vektor cluster e-means untuk memprediksi lima kelas aktif. Scaler dan centroid e-means hanya di-fit dari split train; validation memilih checkpoint dan test hanya dievaluasi bila holdout GT sudah direview. Crop menjaga rasio aspek, sisi terpanjang menjadi 224 px dan dipadding. Pixel dinormalisasi dengan mean/std ImageNet; ColorJitter hanya diterapkan pada cabang citra saat training. E-means adalah fitur eksperimen, bukan jaminan akurasi."),
              heading("NB05 — evaluasi detector", 2),
              paragraph("NB05 mengukur bbox detector Mobil/Motor dan memilih threshold dari validation. Hitungan kondisi lampu serta hitungan event resmi per frame tersedia di NB06."),
              heading("NB06 — tracking dan counting", 2),
              paragraph("NB06 membaca video, membentuk crop ROI aktif, menjalankan YOLO pada crop, memetakan bbox ke koordinat frame, mengklasifikasikan kondisi, dan menghubungkan deteksi dengan ByteTrack. Setiap bbox TidakStandar per frame masuk hitungan; ID track hanya untuk audit/galeri."),
              page_break()]

    parts += [heading("5. Model dan algoritma", 1),
              heading("Grounding DINO", 2),
              paragraph("Grounding DINO adalah foundation model open-vocabulary untuk menemukan objek berdasarkan teks prompt. Dalam proyek ini ia digunakan sebagai pembuat proposal awal, bukan sebagai labeler yang selalu benar."),
              heading("YOLO", 2),
              paragraph("YOLO adalah object detector. Ia menerima gambar dan menghasilkan bounding box, confidence, dan kelas. Confidence bukan probabilitas kebenaran yang sempurna; threshold harus dikalibrasi pada data yang relevan."),
              heading("ResNet18 NB04b", 2),
              paragraph("ResNet18 adalah convolutional neural network yang membaca crop gambar bersama fitur numerik bbox/mask dan e-means. Backbone pretrained diberi head lima kelas sesuai taxonomy anotasi: MobilStandar, MobilTidakStandar, MotorStandar, MotorTidakStandar, dan PantulanCahaya. Weighted CrossEntropy membantu kelas minoritas selama supervised training."),
              paragraph("Training menggunakan BCEWithLogitsLoss karena dua atribut dapat aktif bersamaan. Optimizer yang dipakai adalah AdamW. Threshold atribut dipilih dari validation."),
              heading("ByteTrack", 2),
              paragraph("ByteTrack menghubungkan box antar-frame menjadi track ID untuk audit. Karena unit resmi bbox per frame, kendaraan yang tampak pada beberapa frame memang menyumbang beberapa bbox."),
              paragraph("ByteTrack tidak memberikan jaminan identitas sempurna. Occlusion, kendaraan yang berpapasan, ID switch, dan kendaraan yang keluar lalu masuk kembali masih dapat menyebabkan double count atau missed count."),
              heading("Tracker versus counter", 2),
              paragraph("Tracker menghasilkan hubungan antar-frame. Counter menerapkan aturan agregasi: minimal kemunculan, mayoritas kelas, dan penyambungan fragmen. Jadi angka akhir adalah hasil gabungan model, tracker, dan aturan bisnis."),
              page_break()]

    parts += [heading("6. Apa arti metrik model", 1),
              paragraph("Object detection dan klasifikasi atribut tidak boleh diringkas menjadi satu angka akurasi tanpa konteks."),
              table([["Metrik", "Arti sederhana"], ["Precision", "Dari semua event yang diklaim model, berapa yang benar."], ["Recall", "Dari semua event yang benar-benar ada menurut label manusia, berapa yang ditemukan."], ["F1-score", "Keseimbangan precision dan recall."], ["mAP", "Ringkasan performa object detection pada beberapa threshold IoU/confidence."], ["Loss", "Ukuran kesalahan optimisasi selama training; bukan persentase akurasi."], ["Confidence", "Tingkat keyakinan internal model pada satu deteksi; bukan jaminan benar."]], [2200, 6900]),
              paragraph("Contoh: recall 80% berarti dari 100 objek yang diketahui ada berdasarkan ground truth, model menemukan sekitar 80. Itu berbeda dengan mengatakan model benar 80% dari seluruh keputusan."),
              heading("Train, validation, dan test", 2),
              table([["Split", "Fungsi", "Status proyek"], ["Train", "Untuk memperbarui bobot model.", "Dipakai pada kamera train."], ["Validation", "Untuk memilih threshold, memantau overfitting, dan membandingkan model.", "Memiliki gold set terbatas."], ["Test", "Untuk evaluasi final yang tidak dipakai memilih model.", "Ketersediaan dan status review GT harus dicek; metrik hanya sah untuk label holdout yang sudah direview."]], [1800, 4200, 3100]),
              paragraph("Jika holdout GT belum direview, hasil test hanya inferensi/counting dan bukan metrik akurasi. Periksa status GT aktual sebelum menafsirkan panel dashboard."),
              heading("Pertanyaan akademik penting", 2),
              bullet("Berapa banyak frame validation yang cukup untuk mendukung klaim performa?"),
              bullet("Apakah validation hanya satu kamera cukup mewakili variasi CCTV?"),
              bullet("Apakah ground truth untuk counting perlu dibuat pada beberapa klip test?"),
              bullet("Apakah definisi Standar/TidakStandar cukup konsisten, atau perlu anotator kedua?"),
              page_break()]

    parts += [heading("7. Counting bbox per frame dan keterbatasannya", 1),
              paragraph("Hitungan resmi bukan jumlah kendaraan unik. Setiap deteksi MobilTidakStandar atau MotorTidakStandar yang lolos threshold menyumbang satu bbox pada frame tersebut. Kendaraan yang terlihat pada banyak frame dapat menyumbang banyak bbox."),
              heading("Cara membaca hitungan", 2),
              bullet("Unit analisis adalah bbox event per frame, bukan track atau kendaraan fisik."),
              bullet("Hanya kelas TidakStandar yang digambar merah dan masuk hitungan event resmi."),
              bullet("Mobil dan motor dibedakan dari prediksi detector YOLO."),
              bullet("ByteTrack menghubungkan deteksi dan memberi track_id untuk audit; ia tidak mengubah satuan hitung."),
              bullet("Ambang detector dipilih dari validation; test tidak boleh dipakai memilih threshold."),
              heading("Apa yang dijamin dan tidak dijamin", 2),
              table([["Kondisi", "Konsekuensi"], ["Deteksi lolos threshold", "Satu bbox event dicatat untuk frame itu."], ["Kendaraan tampak di banyak frame", "Bisa dihitung sebagai beberapa bbox, sesuai unit per-frame."], ["ID switch/occlusion", "Dapat mengganggu riwayat track, tetapi tidak mengubah definisi unit bbox."], ["GT test belum direview", "Belum ada dasar untuk mengklaim akurasi test."], ["Video test berbeda", "Hasil tidak otomatis mewakili kamera atau waktu lain."]], [3000, 6100]),
              paragraph("Gunakan istilah jumlah bbox TidakStandar per frame. Jangan menyebut angka ini sebagai jumlah kendaraan unik, arus kendaraan per jam, atau tingkat kepatuhan regulasi."),
              heading("Validasi counting yang disarankan", 2),
              bullet("Pilih klip holdout dari kamera test dan anotasi bbox/event secara manual."),
              bullet("Samakan definisi frame, confidence threshold, dan kelas sebelum membandingkan."),
              bullet("Laporkan selisih hitungan per frame/klip dan contoh false positive/false negative."),
              bullet("Evaluasi ID tracking secara terpisah bila kontinuitas identitas menjadi pertanyaan riset."),
              page_break()]

    parts += [heading("8. Dashboard untuk orang awam", 1),
              paragraph("Dashboard tidak hanya menampilkan angka. Ia harus menjelaskan apa arti angka, dari mana angka berasal, dan apa batasannya."),
              heading("Halaman ringkasan", 2),
              bullet("Jumlah bbox event TidakStandar, rincian mobil/motor, dan video yang diproses."),
              bullet("Dua angka kontrak bbox TidakStandar: mobil dan motor."),
              bullet("Status kamera: dihitung atau di luar split counting."),
              heading("Playback dan timeline", 2),
              paragraph("Video overlay dan counting ditampilkan side-by-side. Counting timeline menggunakan satu baris per bbox tidak standar pada frame aktif; track_id dan first_frame tetap disediakan untuk audit temporal. Slider posisi disediakan untuk audit deterministik. Train/val dapat ditampilkan sebagai presentation_only, sedangkan counting resmi tetap memakai test."),
              heading("Footage per kelas", 2),
              paragraph("NB06 menghasilkan video konteks di event_clips dan crop bbox di event_crops. Dashboard memakai crop bbox sebagai galeri: filter kelas/CCTV, detail ID, jenis kendaraan, menit, frame, tombol sebelumnya/berikutnya, serta grid 8×8."),
              heading("Visualisasi kelas", 2),
              paragraph("Overlay hanya menggambar bbox TidakStandar dengan warna merah. Legend dashboard menjelaskan kelas event; warna kelas lama Silau/Berwarna tidak digunakan."),
              heading("Panel performa", 2),
              paragraph("Panel performa menampilkan train/validation per epoch. Metrik test hanya dapat ditampilkan jika holdout GT telah direview; kurva train/val bukan akurasi test. Counting utama adalah bbox per frame, bukan jumlah kendaraan."),
              heading("Panel cara kerja", 2),
              paragraph("Panel edukasi menjelaskan video → ROI → proposal → anotasi → detector → classifier → tracker → counting → dashboard dengan contoh visual pada setiap tahap."),
              page_break()]

    parts += [heading("9. Artefak penting", 1),
              table([["Artefak", "Fungsi"], ["output/annotation/annotations.json", "Store anotasi manual lima kelas."], ["output/annotation/annotations_coco.json", "Anotasi COCO sesuai taxonomy aktif."], ["output/annotation/yolo/labels", "Label YOLO lima kelas sesuai urutan taxonomy aktif."], ["output/attribute_classifier/best.pt", "Bobot ResNet18 classifier lima kelas."], ["output/attribute_classifier/training_history.csv", "Histori train/validation dan metrik test bila GT tersedia."], ["output/nb04/.../best.pt", "Bobot detector YOLO."], ["output/nb05/thresholds.json", "Threshold confidence detector."], ["output/nb06/tracking_detections.csv", "Deteksi per frame yang memiliki track."], ["output/nb06/track_summary.csv", "Ringkasan track untuk audit temporal."], ["output/nb06/counting_timeline.csv", "Satu baris per crossing track TidakStandar."], ["output/nb06/overlay_manifest.json", "Daftar video overlay, FPS, durasi, split, dan path."], ["output/nb06/event_clip_manifest.json", "Daftar konteks event per track ID."], ["output/nb06/event_crops/<kelas>/*.jpg", "Crop bbox untuk galeri dashboard."], ["output/nb06/counting_summary.json", "Ringkasan counting dan kontrak angka."]], [3600, 5500]),
              heading("Status rerun terakhir", 2),
              paragraph("Setelah metadata ROI diperbaiki, jalankan NB03, latih ulang NB04 dan NB04b, evaluasi/kalibrasi NB05, lalu jalankan NB06. Sumber video, frame, ROI, crop, manifest, dan store anotasi tidak perlu dihapus. Hasil overlay/count lama bukan keluaran dari kombinasi source dan bobot terbaru."),
              page_break()]

    parts += [heading("10. Risiko dan pertanyaan untuk dosen pembimbing", 1),
              heading("Pertanyaan metodologi", 2),
              bullet("Apakah label visual Standar/TidakStandar memerlukan rubric dan anotator kedua?"),
              bullet("Berapa banyak frame/klip holdout yang memadai untuk mengevaluasi detector, classifier, dan counting?"),
              bullet("Apakah lima kamera dari satu wilayah cukup untuk mendukung klaim penelitian?"),
              bullet("Apakah satu anotator cukup, atau perlu anotator kedua untuk mengukur agreement?"),
              bullet("Apakah satu sesi video per kamera cukup untuk pengujian?"),
              heading("Pertanyaan evaluasi", 2),
              bullet("Berapa frame dan klip test yang harus dianotasi manual untuk validasi counting?"),
              bullet("Apakah metrik sebaiknya dilaporkan per kelas, per kamera, atau keduanya?"),
              bullet("Apakah model berhasil memisahkan PantulanCahaya dari target kendaraan pada holdout test?"),
              bullet("Apakah classifier atribut perlu dibandingkan dengan baseline aturan HSV sederhana?"),
              heading("Pertanyaan implementasi", 2),
              bullet("Apakah dashboard akan dipakai offline saat sidang atau dideploy online?"),
              bullet("Apakah video boleh disimpan di server/cloud, dan bagaimana kebijakan privasinya?"),
              bullet("Apakah clip per event perlu disimpan permanen atau dibuat on-demand?"),
              page_break()]

    parts += [heading("11. Rencana kerja yang disarankan", 1),
              table([["Prioritas", "Pekerjaan", "Alasan"], ["1", "Jalankan NB03, latih ulang NB04/NB04b, lalu NB05 dan NB06.", "Menghasilkan artefak detector, CNN, evaluasi, overlay, timeline, dan crop yang konsisten."], ["2", "Periksa beberapa overlay dan crop.", "Memastikan bbox merah, confidence, kelas kondisi, dan event tidak salah secara visual."], ["3", "Jalankan dashboard lokal.", "Memastikan playback side-by-side, filter, counting bbox/frame, galeri, dan panel performa bekerja."], ["4", "Buat gold counting clip di kamera test.", "Mengubah counting dari estimasi menjadi evaluasi yang dapat diukur."], ["5", "Validasi kondisi Standar/TidakStandar.", "Memeriksa apakah classifier generalisasi ke kamera test."], ["6", "Rapikan laporan dan klaim.", "Memisahkan hasil terukur dari keterbatasan dan asumsi."], ["7", "Pertimbangkan deployment online.", "Dilakukan setelah hasil dan kebutuhan privasi jelas."]], [1000, 4500, 3600]),
              heading("Checklist sebelum presentasi", 2),
              bullet("Bisa menjelaskan rubric Standar/TidakStandar dan keterbatasan label berbasis citra."),
              bullet("Bisa menjelaskan perbedaan detector YOLO, classifier atribut, dan tracker."),
              bullet("Tidak menyebut hasil test sebagai akurasi tanpa ground truth yang telah direview."),
              bullet("Bisa menunjukkan satu video overlay dan satu clip per kelas."),
              bullet("Bisa menjelaskan bahwa angka resmi adalah bbox event per frame, bukan kendaraan unik."),
              bullet("Bisa menyebutkan keterbatasan ID switch dan kendaraan yang hilang."),
              bullet("Dashboard memiliki banner atau keterangan tentang status validasi."),
              page_break()]

    parts += [heading("12. Glosarium", 1),
              table([["Istilah", "Penjelasan"], ["Bounding box", "Kotak koordinat yang mengelilingi objek."], ["Confidence", "Skor keyakinan model terhadap satu deteksi."], ["COCO", "Format dataset anotasi object detection."], ["Detector", "Model yang mencari lokasi dan kelas objek."], ["Event", "Kondisi akhir yang dibentuk dari tipe kendaraan dan kondisi lampu."], ["Ground truth", "Label acuan manusia yang dipakai untuk evaluasi."], ["Hard-negative", "Contoh yang mirip objek target tetapi tidak boleh dihitung sebagai target."], ["Inference", "Menjalankan model pada data baru untuk menghasilkan prediksi."], ["IoU", "Ukuran tumpang tindih dua bounding box."], ["mAP", "Metrik umum untuk object detection."], ["CNN/ResNet18", "Neural network konvolusional untuk membaca crop citra."], ["Pseudo-label", "Label awal dari model yang belum dianggap ground truth."], ["ROI", "Region of Interest, area gambar yang menjadi fokus sistem."], ["Track", "Rangkaian deteksi antar-frame yang dianggap objek sama."], ["Track ID", "Identitas sementara dari tracker dalam satu video."], ["Validation", "Data untuk memilih model/threshold dan memantau generalisasi."], ["Test", "Data evaluasi final yang tidak digunakan memilih model."]], [2200, 6900]),
              page_break(),
              heading("13. UML dan model arsitektur", 1),
              paragraph("Bagian ini menjelaskan UML dari tingkat paling tinggi sampai tingkat detail implementasi. UML bukan gambar hiasan; ia membantu menjawab siapa yang memakai sistem, komponen apa yang bekerja, data apa yang mengalir, dan kapan sebuah keputusan dibuat."),
              heading("13.1 Level 0 — system context diagram", 2),
              paragraph("Pada level paling tinggi, sistem dianggap sebagai satu kotak besar. Aktor dan sistem eksternal berada di luar kotak. Unit hitungan aktif adalah bbox TidakStandar per frame, bukan kendaraan unik."),
              table([["Entitas", "Hubungan dengan sistem"], ["Dosen/presenter", "Melihat ringkasan, footage, metrik, dan keterbatasan."], ["Peneliti/anotator", "Memilih frame, menggambar ROI, memberi label, dan memeriksa event."], ["Video CCTV", "Sumber video malam hari."], ["Model YOLO", "Mendeteksi bbox kendaraan Mobil/Motor."], ["Classifier NB04b", "Memprediksi Standar/TidakStandar dari crop kendaraan."], ["ByteTrack", "Menghubungkan deteksi antar-frame."], ["Dashboard", "Menyajikan hasil untuk manusia."]], [2800, 6300]),
              paragraph('''Representasi PlantUML level context:\n@startuml\nactor Peneliti\nactor Dosen\nrectangle "Pipeline CCTV" as P\nrectangle "Video CCTV" as V\nrectangle "Model YOLO" as Y\nrectangle "Classifier Atribut" as A\nrectangle "ByteTrack" as B\nPeneliti --> P : anotasi dan konfigurasi\nDosen --> P : membaca hasil\nV --> P : video malam\nP --> Y\nP --> A\nP --> B\n@enduml'''),
              heading("13.2 Level 1 — use case diagram", 2),
              paragraph("Use case menunjukkan tujuan pengguna, bukan detail fungsi internal."),
              table([["Aktor", "Use case", "Hasil"], ["Peneliti", "Menyiapkan data", "Frame, manifest, ROI, split"], ["Anotator", "Membuat gold set", "Box dan kelas kondisi lampu"], ["Engineer", "Melatih model", "Bobot YOLO dan classifier"], ["Pengguna dashboard", "Melihat playback", "Video overlay dan timeline"], ["Pengguna dashboard", "Membandingkan kelas", "Chart mobil/motor dan event"], ["Dosen", "Mengaudit klaim", "Metrik dan batasan validitas"]], [2300, 3500, 3300]),
              paragraph('''PlantUML use case:\n@startuml\nleft to right direction\nactor Peneliti\nactor Anotator\nactor Dosen\nactor "Pengguna Dashboard" as User\nrectangle "Pipeline CCTV" {\nusecase "Siapkan data dan ROI" as U1\nusecase "Anotasi gold" as U2\nusecase "Latih detector dan atribut" as U3\nusecase "Playback footage" as U4\nusecase "Baca counting dan metrik" as U5\n}\nPeneliti --> U1\nAnotator --> U2\nPeneliti --> U3\nUser --> U4\nUser --> U5\nDosen --> U5\n@enduml'''),
              heading("13.3 Level 2 — component diagram", 2),
              paragraph("Component diagram memecah sistem berdasarkan tanggung jawab teknis."),
              table([["Komponen", "Tanggung jawab", "Artefak utama"], ["Acquisition", "Membaca video dan sampling frame", "videos, frames, manifest"], ["ROI manager", "Menyimpan polygon per kamera", "roi_config.json"], ["Pseudo-labeler", "Membuat proposal awal", "nb02 chunks, crops"], ["Annotation store", "Menyimpan keputusan manusia", "annotations.json"], ["COCO exporter", "Mengekspor taxonomy aktif", "annotations_coco.json"], ["YOLO exporter", "Mengekspor lima kelas", "yolo/images, yolo/labels"], ["Detector trainer", "Melatih YOLO lima kelas", "best.pt, results.csv"], ["FM trainer", "Fine-tune Grounding DINO base lima kelas", "nb02d/checkpoints"], ["Attribute trainer", "Melatih ResNet18 lima kelas berbobot + mask/e-means", "attribute_classifier/best.pt, training_history.csv"], ["Evaluator", "Memilih threshold", "thresholds.json"], ["Tracker/counter", "Memfilter pantulan dan menghitung track TidakStandar yang melintasi garis", "tracking, timeline, summary"], ["Dashboard", "Membaca dan menjelaskan artefak", "Streamlit UI"]], [2200, 4200, 2700]),
              paragraph("Relasi penting: Annotation store adalah sumber gold. COCO dan YOLO adalah dua export dengan tujuan berbeda. Dashboard tidak melatih model dan tidak membuat angka baru; ia membaca hasil NB06."),
              heading("13.4 Level 3 — data flow diagram", 2),
              paragraph("Aliran data utama:\nVideo -> FrameManifest -> ROIConfig -> ROICrop -> ProposalBox -> HumanReviewedAnnotation -> COCO/YOLOExport -> YOLODetector -> DetectionOnCrop -> FrameCoordinates -> AttributeClassifier -> ByteTrackID -> NonstandardBBoxEventPerFrame -> DashboardArtifact."),
              table([["Data object", "Contoh field"], ["FrameManifest", "video_path, frame_path, frame_index, frame_hash"], ["ROIConfig", "camera_id, polygon, frame_size"], ["AnnotationBox", "box, class_name, deleted, source"], ["DetectionFrame", "video_stem, frame_index, score, box, track_id"], ["AttributeEvent", "vehicle_type, condition (Standar/TidakStandar), confidence"], ["BBoxEvent", "video_stem, frame_index, track_id, class_name, bbox"], ["DashboardArtifact", "summary, timeline, overlay_manifest, clip_manifest"]], [2700, 6400]),
              heading("13.5 Sequence diagram — satu frame pada NB06", 2),
              paragraph("Urutan ini menjelaskan apa yang terjadi pada satu frame video."),
              paragraph("PlantUML sequence:\n@startuml\nparticipant VideoCapture\nparticipant ROI\nparticipant YOLO\nparticipant ResNet18\nparticipant ByteTrack\nparticipant ResNet18\nparticipant Counter\nVideoCapture -> ROI : baca frame\nROI -> YOLO : crop ROI sama dengan training\nYOLO --> ROI : box crop + confidence\nROI -> ByteTrack : deteksi Mobil/Motor\nByteTrack --> ROI : box + track_id\nROI -> ResNet18 : map box ke frame, lalu crop kendaraan\nResNet18 --> ROI : Standar/TidakStandar\nROI -> Counter : kondisi + box pada frame aktif\nCounter -> Counter : catat bbox TidakStandar\n@enduml"),
              heading("13.6 Sequence diagram — akhir video", 2),
              paragraph("Setelah semua frame selesai, sistem membuat ringkasan track untuk audit dan menghitung bbox TidakStandar per frame."),
              bullet("Filter deteksi sesuai confidence detector dan threshold classifier."),
              bullet("Kelompokkan track untuk audit; jangan mengganti agregasi resmi bbox per frame dengan jumlah track."),
              bullet("Pakai hasil klasifikasi yang dicache per track untuk konsistensi kondisi."),
              bullet("Penyambungan fragmen hanya untuk audit ID track, bukan untuk menghitung total event resmi."),
              bullet("Jangan menerapkan MIN_TRACK_HITS sebagai filter hitungan resmi bbox per frame."),
              bullet("Hitung satu bbox TidakStandar pada setiap frame; gunakan track_id untuk audit temporal, bukan sebagai agregasi angka utama."),
              bullet("Tulis summary, timeline, manifest overlay, dan clip event."),
              heading("13.7 Activity diagram — pembuatan event", 2),
              paragraph("Start → baca frame → frame valid? Jika tidak, lanjut ke frame berikutnya → jalankan detector → ada box? Jika tidak, lanjut → klasifikasikan atribut → tracker memberi ID? Jika tidak, catat untracked → jika ya, simpan event → semua frame selesai? Jika tidak, ulangi → agregasi track → tulis output → selesai."),
              heading("13.8 Class diagram level implementasi", 2),
              table([["Class/module", "Atribut penting", "Operasi penting"], ["ROIConfig", "camera_id, polygon, frame_size", "build_roi_crop, map crop box to frame"], ["Detection", "box, score, class_name, frame_index", "qualify_by_class_threshold"], ["AttributeFusionCNN", "ResNet18 + 77 fitur bbox/mask + cluster e-means", "forward, decode_attributes"], ["Track", "track_id, hits, first_frame, last_frame", "resolve_track_class"], ["BBoxEvent", "bbox_id, counting_frame, class_name", "count per frame"], ["CountingSnapshot", "state, summary, timeline, manifest", "load_counting_snapshot"], ["Dashboard", "filters, selected_video, selected_class", "render_playback, render_event_footage"]], [2300, 3800, 3000]),
              heading("13.9 Deployment diagram", 2),
              paragraph("Pada penggunaan lokal, semua komponen berada di satu komputer. Video besar tidak perlu dikirim ke cloud."),
              paragraph("Deployment lokal:\n[Komputer peneliti]\n  .venv Python\n  Notebook NB01–NB06\n  output/artefak\n  Streamlit dashboard\n  Browser pengguna"),
              paragraph("Deployment online:\n[Server Streamlit] — membaca JSON/CSV dan menjalankan UI\n[Object storage] — menyimpan overlay dan clip event\n[Browser end user] — playback video dan membaca dashboard\nDeployment online membutuhkan autentikasi, pengaturan privasi, dan pengelolaan bandwidth."),
              heading("13.10 State diagram dashboard", 2),
              table([["State", "Pemicu", "Perilaku"], ["Empty", "Belum ada summary dan CSV", "Tampilkan instruksi menjalankan pipeline."], ["Partial", "Sebagian artefak hilang", "Tampilkan artefak yang belum ada."], ["Error", "Artefak rusak/tidak terbaca", "Tampilkan pesan error, jangan menampilkan angka palsu."], ["Ready", "Artefak wajib terbaca", "Tampilkan ringkasan, playback, timeline, dan footage."], ["Smoke", "summary.smoke_mode=true", "Tampilkan banner bahwa angka sintetis."]], [1800, 4500, 2800]),
              heading("13.11 Aturan integritas antar-komponen", 2),
              bullet("Setiap frame_stem harus memiliki hubungan yang jelas ke video dan kamera."),
              bullet("Kamera tidak boleh muncul di dua split berbeda."),
              bullet("COCO, Grounding DINO/FM, YOLO, dan CNN memakai lima label supervised yang sama; mask SAM2 dan e-means menjadi fitur classifier.") ,
              bullet("Counting resmi adalah bbox TidakStandar per frame; jangan menyebutnya kendaraan unik."),
              bullet("Dashboard tidak boleh mengubah n/a menjadi nol."),
              bullet("Test tanpa ground truth tidak boleh diberi label accuracy."),
              bullet("Overlay dan timeline harus merujuk video_stem yang sama."),
              page_break(),
              heading("Kesimpulan untuk diskusi", 1),
              paragraph("Infra ini sudah memiliki pemisahan yang jelas antara anotasi, detector, classifier atribut, tracker, counting, dan dashboard. Nilai akademik utama yang perlu dibahas bukan hanya apakah dashboard dapat menampilkan angka, tetapi apakah definisi label, kualitas ground truth, pemilihan split, dan validasi counting cukup kuat untuk mendukung klaim penelitian."),
              paragraph("Pertanyaan penting untuk pembimbing: apakah definisi operasional Standar/TidakStandar dan kualitas holdout GT cukup kuat untuk mendukung klaim performa model? Unit angka sistem sudah ditetapkan sebagai bbox event per frame, bukan jumlah kendaraan unik."),
              paragraph("Dokumen ini sengaja membedakan tiga hal: hasil yang sudah diproduksi kode, hasil yang masih menunggu rerun, dan hasil yang belum dapat diklaim karena belum memiliki ground truth.")]
    return "".join(parts)


def document_xml() -> str:
    body = build_body()
    sect = '<w:sectPr><w:pgSz w:w="11906" w:h="16838"/><w:pgMar w:top="1100" w:right="1100" w:bottom="1100" w:left="1100"/><w:footerReference w:type="default" r:id="rId1"/></w:sectPr>'
    return f'''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
<w:body>{body}{sect}</w:body></w:document>'''


def styles_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
<w:docDefaults><w:rPrDefault><w:rPr><w:rFonts w:ascii="Aptos" w:hAnsi="Aptos"/><w:sz w:val="22"/></w:rPr></w:rPrDefault><w:pPrDefault><w:pPr><w:spacing w:after="100" w:line="276"/></w:pPr></w:pPrDefault></w:docDefaults>
<w:style w:type="paragraph" w:default="1" w:styleId="Normal"><w:name w:val="Normal"/></w:style>
<w:style w:type="paragraph" w:styleId="Title"><w:name w:val="Title"/><w:basedOn w:val="Normal"/><w:rPr><w:b/><w:sz w:val="34"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Subtitle"><w:name w:val="Subtitle"/><w:basedOn w:val="Normal"/><w:rPr><w:i/><w:sz w:val="24"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading1"><w:name w:val="heading 1"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="30"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="Heading2"><w:name w:val="heading 2"/><w:basedOn w:val="Normal"/><w:next w:val="Normal"/><w:rPr><w:b/><w:sz w:val="25"/></w:rPr></w:style>
<w:style w:type="paragraph" w:styleId="ListBullet"><w:name w:val="List Bullet"/><w:basedOn w:val="Normal"/><w:pPr><w:ind w:left="500" w:hanging="250"/></w:pPr></w:style>
</w:styles>'''


def rels_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/footer" Target="footer1.xml"/><Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''


def footer_xml() -> str:
    return '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:ftr xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main"><w:p><w:pPr><w:jc w:val="center"/></w:pPr><w:r><w:t>Brief Infrastruktur Pipeline CCTV — 2026</w:t></w:r></w:p></w:ftr>'''


def create() -> None:
    content_types = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types"><Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/><Default Extension="xml" ContentType="application/xml"/><Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/><Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/><Override PartName="/word/footer1.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.footer+xml"/></Types>'''
    root_rels = '''<?xml version="1.0" encoding="UTF-8" standalone="yes"?><Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships"><Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/></Relationships>'''
    with ZipFile(OUT, "w", ZIP_DEFLATED) as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", root_rels)
        archive.writestr("word/document.xml", document_xml())
        archive.writestr("word/styles.xml", styles_xml())
        archive.writestr("word/footer1.xml", footer_xml())
        archive.writestr("word/_rels/document.xml.rels", rels_xml())
    print(OUT)


if __name__ == "__main__":
    create()
