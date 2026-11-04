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
| Semantic colors, light + dark | **Dipakai sebagian** — semua warna lewat `.streamlit/config.toml`, tidak ada hex di kode UI |
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
4. **Tanpa warna hardcoded di `app.py`.** Warna hanya dari `.streamlit/config.toml`.
5. **Satu label per maksud.** "Kendaraan dihitung" dipakai konsisten; jangan berganti menjadi
   "total unit", "jumlah mobil+motor", atau "kendaraan masuk" untuk hal yang sama.
6. **Angka tidak boleh berbohong.** Nilai `"n/a"` tetap `n/a`, tidak pernah diubah menjadi `0`.
   Angka nol hanya ditampilkan bila memang nol.
7. **Tanpa animasi dekoratif.** Tidak ada `st.spinner` untuk operasi yang sudah selesai,
   tidak ada progress palsu.

## Siklus state wajib

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
