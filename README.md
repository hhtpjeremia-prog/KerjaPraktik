# Dashboard Pemantauan Lampu Kendaraan Tidak Standar — CCTV Kabupaten Bogor

Dashboard interaktif berbasis **Streamlit** dengan pemutar video *real-time synced playback* (penghitung kumulatif kendaraan berlampu tidak standar bertambah otomatis saat kendaraan melintasi garis hitung virtual kuning di layar), galeri *crop* kendaraan dengan/tanpa masker segmentasi SAM 2, serta ringkasan evaluasi model (YOLO11n dan ResNet-18 Multi-Modal Fusion).

## Arsitektur Deployment Gratis (Rp0)
- **Aplikasi & Artefak Ringan (~9 MB):** Hosted di **Streamlit Community Cloud** (repo GitHub ini; hanya butuh `streamlit`, `pandas`, `pillow` — tanpa PyTorch/YOLO/SAM2).
- **10 Video Overlay H.264 (669 MB):** Disimpan di **HuggingFace Dataset publik** yang mendukung streaming HTTPS, CORS, dan `HTTP 206 Partial Content` (*byte-range seeking* instan).

## Cara Deploy ke Cloud (3 Langkah)

### 1. Upload 10 File Video MP4 ke HuggingFace Dataset (Gratis, Tanpa Kartu Kredit)
1. Buat akun/login di [huggingface.co](https://huggingface.co) → klik **+ New → Dataset** (pilih **Public**, misal `Olannnn/bogor-headlight-overlays`).
2. Upload isi folder lokal `output/nb06/overlays_browser/` ke root dataset tersebut dengan struktur persis sama:
   - `atcs_bogor-kab-bappenda1_20261002_185759_overlay.mp4`
   - `atcs_bogor-kab-bappenda2_20261002_190817_overlay.mp4`
   - `atcs_bogor-kab-bappenda3_20261002_191842_overlay.mp4`
   - `atcs_bogor-kab-pdam2_20261002_192854_overlay.mp4`
   - `atcs_bogor-kab-pemda2_20261002_193906_overlay.mp4`
   - `without_mask/atcs_bogor-kab-bappenda1_20261002_185759_overlay_without_mask.mp4`
   - `without_mask/atcs_bogor-kab-bappenda2_20261002_190817_overlay_without_mask.mp4`
   - `without_mask/atcs_bogor-kab-bappenda3_20261002_191842_overlay_without_mask.mp4`
   - `without_mask/atcs_bogor-kab-pdam2_20261002_192854_overlay_without_mask.mp4`
   - `without_mask/atcs_bogor-kab-pemda2_20261002_193906_overlay_without_mask.mp4`

   Atau lewat CLI/Python (di Windows set `HF_HUB_DISABLE_XET=1` bila transfer Xet tersendat):
   ```powershell
   $env:HF_HUB_DISABLE_XET="1"
   $env:HF_HUB_ENABLE_HF_TRANSFER="0"
   hf auth login
   hf upload Olannnn/bogor-headlight-overlays "d:\Jere\NewStyle\output\nb06\overlays_browser" . --repo-type dataset
   ```
3. URL dasar video publik:
   ```text
   https://huggingface.co/datasets/Olannnn/bogor-headlight-overlays/resolve/main
   ```

### 2. Branch Deployment (`deploy`) di Repo `hhtpjeremia-prog/KerjaPraktik`
Branch `deploy` adalah *orphan branch* terpisah yang hanya berisi isi folder `deploy_streamlit/` (sekitar ~8,17 MB, tanpa `requirements.txt` CUDA dari branch `main`).

### 3. Deploy di Streamlit Community Cloud
1. Buka [share.streamlit.io](https://share.streamlit.io) → **Create app** → pilih repo `hhtpjeremia-prog/KerjaPraktik`.
2. Atur konfigurasi:
   - **Branch:** `deploy`
   - **Main file path:** `app.py`
3. Klik **Advanced settings → Secrets**, lalu tempel:
   ```toml
   VIDEO_BASE_URL = "https://huggingface.co/datasets/Olannnn/bogor-headlight-overlays/resolve/main"
   ```
4. Klik **Deploy** — selesai!

## Menjalankan Lokal (Untuk Uji Sebelum Push)
```bash
streamlit run app.py
```
Saat `VIDEO_BASE_URL` tidak diisi, aplikasi otomatis memakai `LocalVideoServer` (HTTP 206) dan mencari video lokal di `artifacts/nb06/overlays_browser/` atau `../output/nb06/overlays_browser/`.
