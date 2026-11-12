# Bar Desain Dashboard

Aturan mengikat untuk `dashboard/app.py`. Diturunkan dari skill `appllama-app-design-skill`
(`.agents/skills/appllama-app-design-skill/SKILL.md`), lalu diterjemahkan ke primitif Streamlit.

## Yang dipinjam dan yang dibuang

Skill aslinya menargetkan aplikasi mobile Expo / React Native dengan Apple HIG. Yang **bisa
ditegakkan** di Streamlit adalah lapisan disiplin, bukan lapisan platform:

| Hukum asli | Status di Streamlit |
|---|---|
| Anti-slop laws 1–9 | **Dipakai penuh** — semua larangan visual di bawah berasal dari sini |
| Typography is hierarchy | **Dipakai** — satu `st.title` per halaman, sisanya `st.subheader` |
| Tabular numerals untuk angka | **Dipakai** — lewat CSS `font-variant-numeric` |
| Format numbers like a product | **Dipakai** — `format_count()` di `dashboard/data.py` |
| Spacing rhythm (base 8) | **Dipakai** — jarak antar blok selalu `st.divider()` atau grid `st.columns`, tidak ada padding manual |
| Shadows for elevation logic | **Dibuang** — Streamlit `border=True` sudah jadi satu-satunya elevasi |
| Semantic colors, light + dark | **Dipakai sebagian** — tema UI berasal dari `.streamlit/config.toml`; merah event bbox adalah warna semantik yang dipetakan eksplisit dan harus identik dengan overlay TidakStandar |
| Native controls, SF Symbols, haptics, motion, navigation, simulator loop | **Dibuang** — tidak ada padanannya di Streamlit. Jangan dipaksakan. |

## Token terkunci

| Token | Nilai | Aturan |
|---|---|---|
| Aksen | `#1F7A4D` | Satu-satunya hue non-abu. Hanya untuk aksi utama, state aktif, seri chart. Tidak boleh muncul hue kedua. |
| Keluarga abu | cool grey (`#F4F6F5` → `#1A1D1B`) | Tidak boleh campur warm grey. |
| Radius | `baseRadius = "medium"` dari tema | Tidak ada radius yang ditulis manual di kode. |
| Spacing | kelipatan 8 px | Jarak antar seksi = `st.divider()`, bukan margin CSS. |
| Tipografi | sans serif sistem | `st.title` satu kali, `st.subheader` untuk seksi, `st.caption` untuk metadata. Tidak ada heading buatan CSS. |

## Larangan

1. **Tanpa gradient, glow, glassmorphism, mesh, confetti, sparkle.** Nol.
2. **Tanpa emoji di chrome.** Tidak ada emoji di judul, tombol, label metrik, atau nama kolom.
   Emoji hanya boleh muncul di dalam teks penjelasan bila benar-benar perlu.
3. **Tanpa ikon dari keluarga campuran.** Belum ada ikon sama sekali; label teks lebih baik
   daripada tiga keluarga ikon.
4. **Tanpa warna dekoratif hardcoded di `app.py`.** Warna event semantik boleh dipetakan eksplisit di
   legend karena harus identik dengan warna bbox NB06; warna dekoratif UI tetap berasal dari
   `.streamlit/config.toml`.
5. **Satuan harus eksplisit.** Label hitung resmi adalah "bbox TidakStandar per frame"; jangan
   menyebutnya jumlah kendaraan unik. Hitungan kelas pada timeline mengikuti bbox di frame terpilih.
6. **Angka tidak boleh berbohong.** Nilai `"n/a"` tetap `n/a`, tidak pernah diubah menjadi `0`.
   Angka nol hanya ditampilkan bila memang nol.
7. **Tanpa animasi dekoratif.** Tidak ada `st.spinner` untuk operasi yang sudah selesai,
   tidak ada progress palsu.

## Siklus state wajib

Playback adalah lapisan presentasi di atas artefak NB06, bukan sumber angka baru. Video dipilih
dari `overlay_manifest.json`; counting aktif membaca `counting_timeline.csv` dan hanya menampilkan
baris dengan `counting_frame` sama dengan frame yang sedang dipilih. Satu kendaraan boleh muncul
berulang karena satuan utama memang bbox per frame. `track_id` tetap ditampilkan untuk audit.

Playback dan counting ditampilkan side-by-side. Legend warna bbox bersifat permanen dan menjelaskan
keluarga kendaraan (mobil/motor) serta status tidak standar. Galeri event memakai crop bbox dari
`event_crops/`, dengan filter kelas/CCTV, detail event, navigasi sequential, dan grid 8×8. Video
full-frame pada `event_clips/` tetap boleh disimpan sebagai konteks audit, tetapi bukan tampilan
utama galeri untuk end user.

Model YOLO/FM memiliki kelas detector `Mobil`, `Motor`, `PantulanCahaya`; CNN memiliki label
`Standar`, `TidakStandar`, `PantulanCahaya`. Pantulan ikut dilatih dan dievaluasi, tetapi disaring
sebelum ByteTrack serta tidak masuk overlay, galeri event, atau hitungan kendaraan. Overlay end user
tetap hanya menggambar bbox `TidakStandar` merah.

Bagian performa model membaca sumber evaluasi secara terpisah dari hitungan NB06:
`output/ground_truth/annotation_summary.json` untuk cakupan review GT NB02c,
`output/nb05/test_metrics.json` untuk evaluasi holdout YOLO, dan
`output/attribute_classifier/metrics.json` untuk evaluasi holdout CNN. Pesan bahwa kamera test
belum memiliki GT hanya boleh ditampilkan jika artefak GT memang tidak tersedia. Kurva epoch
hanya menampilkan train dan validation; metrik test final ditampilkan terpisah setelah checkpoint
terbaik dipilih. Evaluasi GT tidak mengubah angka counting/inferensi NB06.

Halaman tidak boleh hanya menangani jalur bahagia. Empat state ini harus ada dan terlihat beda:

| State | Pemicu | Tampilan |
|---|---|---|
| `empty` | kedua artefak NB06 belum ada | peringatan + instruksi urutan NB03→NB06 |
| `partial` | hanya satu artefak ada | peringatan berisi nama artefak yang hilang |
| `error` | artefak ada tapi gagal dibaca | `st.error` berisi pesan aslinya |
| `ready` | keduanya terbaca | dashboard penuh |

State tambahan di dalam `ready`: `smoke_mode = true` wajib memunculkan banner bahwa angka
berasal dari data sintetis. Dashboard tidak boleh menampilkan angka smoke tanpa penanda itu.

Titik CCTV yang tidak masuk split hitung tetap ditampilkan, dengan status
`di luar split hitung (<split>)`. Menyembunyikannya membuat dashboard tampak seperti sudah
mencakup semua kamera padahal belum.

## Pre-flight anti-slop (mekanis, bukan selera)

Sebelum dashboard dianggap selesai, hitung:

- hue aksen berbeda: **harus 1**. Varian terang/gelap dari hue yang sama untuk mode gelap
  (`#3E9F70` terhadap `#1F7A4D`) dihitung sebagai aksen yang sama, bukan hue kedua.
- gradient / glow / glassmorphism: **harus 0**
- emoji di chrome: **harus 0**
- radius yang ditulis manual di kode UI: **harus 0**
- label berbeda untuk satu maksud: **harus 0**
- state yang belum ditangani dari tabel di atas: **harus 0**
- angka `n/a` yang tampil sebagai `0`: **harus 0**

Hitungan yang gagal adalah bug, bukan penilaian rasa.
