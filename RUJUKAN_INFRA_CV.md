# Rujukan Ilmiah untuk Infrastruktur Computer Vision

> **Kecocokan implementasi (12 November 2026).** Gunakan README untuk kontrak model/ROI aktif. Rujukan semi-supervised, Grad-CAM, dan metode lain di bawah adalah landasan atau kemungkinan pengembangan kecuali dinyatakan telah diimplementasikan. Model classifier aktif memakai CNN ResNet18 + fitur HSV numerik; evaluasi test hanya sah saat holdout GT direview.


Daftar ini menghubungkan tahapan pipeline NewStyle dengan paper primer, prosiding, dokumentasi resmi, dan standar yang relevan. Tujuannya sebagai bahan diskusi dengan dosen pembimbing dan landasan penulisan laporan.

## Cara menggunakan daftar ini

- Rujukan implementasi menjelaskan komponen yang memang digunakan atau dievaluasi.
- Rujukan pengembangan menjelaskan metode yang belum diterapkan, terutama semi-supervised object detection.
- Google Scholar adalah mesin indeks/pencarian, bukan penerbit atau lisensi. ScienceDirect adalah platform penerbit Elsevier. Untuk sitasi, gunakan metadata pada penerbit/prosiding atau DOI.
- Open Access berarti tersedia untuk dibaca di sumber tersebut. Itu tidak otomatis memberi izin memakai ulang gambar, tabel, kode, checkpoint, atau dataset. Periksa lisensi tiap aset secara terpisah.

## Peta rujukan menurut tahap pipeline

| Tahap | Komponen project | Rujukan | Batas interpretasi |
|---|---|---|---|
| Proposal deteksi | Grounding DINO mencari kandidat bbox dari prompt | Liu et al. (2024) | Kandidat model bukan ground truth; manusia memverifikasi label dan bbox. |
| Anotasi dan ekspor | Anotasi bbox manual, ekspor COCO/YOLO | Lin et al. (2014) | COCO adalah format/dataset/benchmark, bukan model pembuat label. |
| Deteksi kendaraan | YOLO mendeteksi bbox dan tipe kendaraan | Redmon et al. (2016); dokumentasi Ultralytics YOLO11 | Paper YOLO asli bukan deskripsi persis YOLO11. |
| Klasifikasi kondisi | ResNet18 dan fitur numerik HSV/statistik piksel | He et al. (2016); Yosinski et al. (2014) | Fitur warna dari video adalah proksi visual, bukan lux atau candela terkalibrasi. |
| Penggabungan prediksi | Hasil YOLO dan classifier dipasangkan pada bbox yang sama di NB06 | Literatur fusion | Penggabungan terjadi pada pipeline inferensi; tidak ada joint end-to-end training kedua model. |
| Tracking | ByteTrack menghubungkan deteksi antar-frame | Zhang et al. (2022) | ByteTrack membuat asosiasi/ID track, bukan kelas kondisi lampu. |
| Evaluasi bbox | Precision, recall, mAP@50, mAP@50:95, error ukuran bbox | COCO evaluator; VOC challenge | Perlu GT bbox yang direview dan dipisahkan dari data training. |
| Evaluasi classifier | Precision, recall, F1, MSE/Brier, RMSE | Brier (1950); Guo et al. (2017) | MSE/RMSE probabilitas kelas bukan metrik geometri bbox. |
| Semi-supervised | Belum diterapkan sebagai jalur training | Soft Teacher; Unbiased Teacher | Relevan hanya jika data tanpa label/pseudo-label benar-benar masuk proses training. |

## A. Proposal deteksi, anotasi, dan format data

### 1. Grounding DINO

Liu, S., Zeng, Z., Ren, T., et al. (2024). Grounding DINO: Marrying DINO with Grounded Pre-Training for Open-Set Object Detection. European Conference on Computer Vision (ECCV).

Relevansi: deteksi open-set menggunakan prompt bahasa. Ini mendukung penjelasan tahap pencarian proposal objek sebelum anotasi manual. Hasil model tetap perlu diperiksa manusia.

- [Paper resmi ECCV (PDF)](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06319.pdf)
- [arXiv](https://arxiv.org/abs/2303.05499)

### 2. COCO dan anotasi objek

Lin, T.-Y., Maire, M., Belongie, S., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.

Relevansi: anotasi objek per-instance, bbox, dan benchmark deteksi. COCO JSON adalah cara merepresentasikan anotasi; COCO tidak dengan sendirinya menghasilkan label project.

- [Microsoft Research](https://www.microsoft.com/en-us/research/publication/microsoft-coco-common-objects-in-context/)
- [arXiv](https://arxiv.org/abs/1405.0312)

## B. Deteksi kendaraan dan classifier

### 3. YOLO

Redmon, J., Divvala, S., Girshick, R., dan Farhadi, A. (2016). You Only Look Once: Unified, Real-Time Object Detection. CVPR 2016, 779–788.

Relevansi: paper fondasional untuk keluarga metode YOLO dan deteksi bbox satu tahap. Ini bukan paper arsitektur YOLO11 yang persis digunakan project.

- [CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/html/Redmon_You_Only_Look_CVPR_2016_paper.html)
- [Dokumentasi YOLO11 dan sitasi software resmi](https://docs.ultralytics.com/models/yolo11)

Dokumentasi Ultralytics menyatakan belum ada paper formal YOLO11. Untuk laporan, sebut versi/model software dan checkpoint yang dipakai, serta sitasi software resmi.

### 4. ResNet dan transfer learning

He, K., Zhang, X., Ren, S., dan Sun, J. (2016). Deep Residual Learning for Image Recognition. CVPR 2016.

Relevansi: dasar residual network yang dipakai sebagai backbone CNN.

- [Paper CVF Open Access](https://openaccess.thecvf.com/content_cvpr_2016/papers/He_Deep_Residual_Learning_CVPR_2016_paper.pdf)

Yosinski, J., Clune, J., Bengio, Y., dan Lipson, H. (2014). How Transferable Are Features in Deep Neural Networks? NeurIPS 2014.

Relevansi: mendukung pembahasan penggunaan bobot pretrained dan fine-tuning pada tugas target.

- [NeurIPS Proceedings](https://papers.nips.cc/paper_files/paper/2014/hash/532a2f85b6977104bc93f8580abbb330-Abstract.html)
- [arXiv](https://arxiv.org/abs/1411.1792)

### 5. Preprocessing dan augmentasi

Buslaev, A., Iglovikov, V., Khvedchenya, E., et al. (2020). Albumentations: Fast and Flexible Image Augmentations. Information, 11(2), 125.

Relevansi: rujukan umum untuk augmentasi computer vision. Paper ini tidak berarti project memakai library Albumentations; implementasi project memakai transformasi Torchvision.

- [MDPI dan DOI](https://doi.org/10.3390/info11020125)
- [arXiv](https://arxiv.org/abs/1809.06839)
- [Dokumentasi transform Torchvision](https://docs.pytorch.org/vision/stable/transforms.html)

### 6. Ketimpangan kelas

Lin, T.-Y., Goyal, P., Girshick, R., He, K., dan Dollár, P. (2017). Focal Loss for Dense Object Detection. ICCV 2017.

Relevansi: menjelaskan masalah ketimpangan foreground/background pada object detection. Project memakai weighted binary cross-entropy untuk classifier; Focal Loss adalah metode berbeda dan hanya menjadi rujukan konseptual masalah imbalance.

- [Paper ICCV Open Access (PDF)](https://openaccess.thecvf.com/content_ICCV_2017/papers/Lin_Focal_Loss_for_ICCV_2017_paper.pdf)
- [arXiv](https://arxiv.org/abs/1708.02002)

## C. Video malam, lampu kendaraan, dan fotometri

### 7. CCTV malam dan lampu kendaraan

Lashkov, I., Yuan, R., dan Zhang, G. (2025). Machine learning-based vehicle detection and tracking based on headlight extraction and GMM clustering under low illumination conditions. Expert Systems with Applications, 267, 126240.

Relevansi: sangat dekat dengan domain penelitian: CCTV malam, lampu kendaraan, deteksi, dan tracking.

- [ScienceDirect/Elsevier, DOI](https://doi.org/10.1016/j.eswa.2024.126240)

Chen, Y.-L., Wu, B.-F., Huang, H.-Y., dan Fan, C.-J. (2011). A Real-Time Vision System for Nighttime Vehicle Detection and Traffic Surveillance. IEEE Transactions on Industrial Electronics.

Relevansi: sistem deteksi dan tracking kendaraan malam pada kamera surveillance tetap.

- [DOI IEEE](https://doi.org/10.1109/TIE.2010.2055771)
- [Repository institusi](https://ir.lib.nycu.edu.tw/handle/11536/8964)

### 8. Batas klaim HSV/RGB terhadap intensitas fisik cahaya

Commission Internationale de l’Éclairage (CIE). (2016). CIE Technical Note 004:2016 — The Use of Terms and Units in Photometry.

Relevansi: sumber standar untuk membedakan besaran fotometri dan satuannya. HSV, brightness ratio, dan histogram warna dari video biasa tidak dapat dilaporkan sebagai lux/candela tanpa instrumen, kalibrasi fotometrik/radiometrik, dan geometri pengukuran yang sesuai.

- [Dokumen resmi CIE (PDF)](https://files.cie.co.at/841_CIE_TN_004-2016.pdf)

## D. Fusion dan tracking

### 9. Penggabungan keluaran YOLO dan CNN

Deep Multimodal Data Fusion. ACM Computing Surveys (2024).

Relevansi: taksonomi fusion pada level data, fitur, dan keputusan.

- [DOI ACM](https://doi.org/10.1145/3649447)

Pada project, embedding citra CNN digabung dengan fitur HSV/statistik piksel di classifier. Kemudian NB06 memasangkan prediksi kondisi classifier dengan bbox/jenis kendaraan dari YOLO. ByteTrack menghubungkan track ID antar-frame. Ini bukan training YOLO dan CNN secara bersama.

### 10. ByteTrack

Zhang, Y., Sun, P., Jiang, Y., et al. (2022). ByteTrack: Multi-Object Tracking by Associating Every Detection Box. ECCV 2022.

Relevansi: metode untuk mempertahankan identitas objek antar-frame melalui asosiasi bbox.

- [DOI Springer](https://doi.org/10.1007/978-3-031-20047-2_1)
- [Repository penulis](https://github.com/FoundationVision/ByteTrack)

Bewley, A., Ge, Z., Ott, L., Ramos, F., dan Upcroft, B. (2016). Simple Online and Realtime Tracking. ICIP 2016.

Relevansi tambahan: baseline tracking-by-detection online.

- [arXiv](https://arxiv.org/abs/1602.00763)

## E. Evaluasi deteksi, classifier, dan tracking

### 11. mAP dan IoU bbox

Lin, T.-Y., et al. (2014). Microsoft COCO: Common Objects in Context. ECCV 2014.

Gunakan bersama evaluator resmi COCO untuk definisi AP pada IoU 0.50:0.95, AP@0.50, dan metrik terkait.

- [Definisi metrik evaluasi COCO](https://cocodataset.github.io/dataset/detection-eval.htm)

Everingham, M., Van Gool, L., Williams, C. K. I., Winn, J., dan Zisserman, A. (2010). The Pascal Visual Object Classes (VOC) Challenge. International Journal of Computer Vision, 88, 303–338.

Relevansi: benchmark dan prosedur evaluasi deteksi; AP pada IoU 0.50 sering dikaitkan dengan protokol VOC.

- [Springer, DOI](https://doi.org/10.1007/s11263-009-0275-4)

### 12. MSE/RMSE probabilitas dan kalibrasi confidence

Brier, G. W. (1950). Verification of Forecasts Expressed in Terms of Probability. Monthly Weather Review, 78(1), 1–3.

Relevansi: dasar Brier score, yaitu mean squared error antara probabilitas prediksi dan label biner. Pada classifier project, MSE/RMSE melengkapi precision/recall/F1; keduanya bukan metrik bbox.

- [American Meteorological Society, DOI](https://doi.org/10.1175/1520-0493%281950%29078%3C0001%3AVOFEIT%3E2.0.CO%3B2)

Guo, C., Pleiss, G., Sun, Y., dan Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. Proceedings of the 34th International Conference on Machine Learning, PMLR 70, 1321–1330.

Relevansi: skor confidence jaringan saraf belum tentu setara dengan probabilitas prediksi benar yang terkalibrasi.

- [PMLR](https://proceedings.mlr.press/v70/guo17a.html)

### 13. Metrik tracking

Luiten, J., Osep, A., Dendorfer, P., et al. (2021). HOTA: A Higher Order Metric for Evaluating Multi-object Tracking. International Journal of Computer Vision, 129, 548–578.

Relevansi: HOTA mengevaluasi deteksi, asosiasi identitas, dan lokalisasi.

Batas pada project: HOTA/IDF1 membutuhkan GT track ID per frame. Anotasi bbox dan kelas tanpa ID lintas-frame belum cukup.

- [Paper Open Access](https://pmc.ncbi.nlm.nih.gov/articles/PMC7881978/)
- [DOI Springer](https://doi.org/10.1007/s11263-020-01375-2)

## F. Kamera berbeda dan validitas split

Lu, Y., Zhong, Z., dan Shu, Y. (2023). Multi-View Domain Adaptive Object Detection on Camera Networks. Proceedings of the AAAI Conference on Artificial Intelligence, 37(7), 8966–8974.

Relevansi: perbedaan domain antar kamera dan adaptasi detektor di camera networks.

- [AAAI resmi dan DOI](https://doi.org/10.1609/aaai.v37i7.26077)

Kapoor, S., dan Narayanan, A. (2023). Leakage and the Reproducibility Crisis in Machine-Learning-Based Science. Patterns.

Relevansi: prinsip untuk mencegah data test memengaruhi training atau pemilihan model. Untuk data video, pemisahan berdasarkan kamera/rekaman membantu mengurangi risiko frame yang sangat mirip tersebar di train dan test.

- [Artikel Open Access](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/)
- [DOI](https://doi.org/10.1016/j.patter.2023.100804)

## G. Semi-supervised learning — rujukan pengembangan, bukan deskripsi training saat ini

Xu, M., Zhang, Z., Hu, H., et al. (2021). End-to-End Semi-Supervised Object Detection with Soft Teacher. ICCV 2021, 3060–3069.

Relevansi: teacher–student dan pseudo-label dengan pembobotan confidence untuk melatih detektor memakai data berlabel dan tidak berlabel.

- [CVF Open Access](https://openaccess.thecvf.com/content/ICCV2021/html/Xu_End-to-End_Semi-Supervised_Object_Detection_With_Soft_Teacher_ICCV_2021_paper.html)
- [DOI IEEE](https://doi.org/10.1109/ICCV48922.2021.00305)

Liu, Y.-C., Ma, C.-Y., He, Z., et al. (2021). Unbiased Teacher for Semi-Supervised Object Detection. ICCV 2021.

Relevansi: membahas bias pseudo-label dan framework teacher–student dalam semi-supervised detection.

- [arXiv](https://arxiv.org/abs/2102.09480)
- [Repository implementasi](https://github.com/facebookresearch/unbiased-teacher)

**Pemisahan istilah:** proposal otomatis yang kemudian diberi label atau dikoreksi manusia adalah bantuan anotasi/human-in-the-loop. Itu sendiri belum berarti training semi-supervised. Untuk menyatakan semi-supervised training, data tanpa label atau pseudo-label harus benar-benar memengaruhi objective/proses training. Evaluasi akhir tetap memerlukan GT holdout yang tidak dipakai saat training.

## Catatan ketahanan terhadap overfitting

Referensi arsitektur/augmentasi tidak membuktikan bahwa model project bebas overfitting. Pada trainer classifier saat ini terdapat weighted BCE, pretrained ResNet18, augmentasi flip dan ColorJitter, dropout, weight decay, warm-up head, serta pemilihan checkpoint berdasarkan validation loss. Namun belum ada early stopping berbasis patience. Semua itu adalah mitigasi, bukan bukti performa.

Variasi visual dalam kelas TidakStandar berbeda dari class imbalance. Weighted BCE menangani perbedaan jumlah contoh; ia tidak otomatis membuat variasi kondisi kamera, jenis lampu, jarak, blur, pantulan, atau exposure terwakili. Bukti overfitting harus dinilai dari kurva train-vs-val, metrik per kelas/kamera, hasil holdout test, serta pemeriksaan contoh false positive/false negative.

## Catatan lisensi kode, bobot, dan data

Sitasi paper dan hak penggunaan perangkat lunak/bobot adalah hal berbeda. Dokumentasi resmi Ultralytics menyatakan YOLO11 belum mempunyai paper formal dan mencantumkan sitasi software. Halaman lisensi Ultralytics menyebut penggunaan code/model/trained weights tunduk pada AGPL-3.0 atau lisensi Enterprise sesuai penggunaan.

- [Dokumentasi YOLO11 dan sitasi software](https://docs.ultralytics.com/models/yolo11)
- [Lisensi resmi Ultralytics](https://www.ultralytics.com/license)

Untuk setiap aset, catat secara terpisah: sitasi paper/model, lisensi kode, lisensi checkpoint pretrained dan bobot hasil training, izin dataset/video CCTV, serta izin penggunaan ulang gambar/tabel pada laporan.

## Bacaan prioritas untuk diskusi

1. Grounding DINO — kandidat deteksi dari prompt.
2. YOLO original dan dokumentasi YOLO11 — bedakan metode umum dengan implementasi aktual.
3. ResNet — backbone classifier.
4. ByteTrack — asosiasi bbox dan ID antar-frame.
5. COCO evaluator — makna mAP@50 dan mAP@50:95.
6. Paper CCTV malam dari Expert Systems with Applications — kedekatan domain.
7. CIE Technical Note — batas klaim fitur piksel dibanding pengukuran fotometri.
8. Soft Teacher dan Unbiased Teacher — hanya jika semi-supervised training akan benar-benar diuji.\n