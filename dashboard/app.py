"""Dashboard Streamlit hasil hitung kendaraan per titik CCTV dari artefak NB06."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.data import (
    MEASURABLE_CONTRACT_KEYS,
    NOT_AVAILABLE_LABEL,
    STATE_EMPTY,
    STATE_ERROR,
    STATE_PARTIAL,
    CountingSnapshot,
    build_camera_status_table,
    format_count,
    load_counting_snapshot,
    resolve_nb06_root,
)
from pipeline_common import CLASS_NAMES, TRAINING_CLASS_NAMES, UNMEASURABLE_COUNT_CONTRACT_KEYS

PAGE_TITLE = "Hasil Hitung Kendaraan — CCTV Kab. Bogor"
RATIO_DECIMAL_PLACES = 1

CONTRACT_LABELS = {
    "jumlah_kendaraan_mobil": "Kendaraan mobil",
    "jumlah_kendaraan_motor": "Kendaraan motor",
    "jumlah_lampu_silau_mobil": "Lampu silau — mobil",
    "jumlah_lampu_silau_motor": "Lampu silau — motor",
    "jumlah_lampu_berwarna_mobil": "Lampu berwarna — mobil",
    "jumlah_lampu_berwarna_motor": "Lampu berwarna — motor",
    "jumlah_lampu_silau_dan_berwarna_mobil": "Silau dan berwarna — mobil",
    "jumlah_lampu_silau_dan_berwarna_motor": "Silau dan berwarna — motor",
}

CLASS_LABELS = {
    "MobilNormal": "Mobil normal",
    "MobilSilau": "Mobil silau",
    "MobilBerwarna": "Mobil berwarna",
    "MobilSilauBerwarna": "Mobil silau dan berwarna",
    "MotorNormal": "Motor normal",
    "MotorSilau": "Motor silau",
    "MotorBerwarna": "Motor berwarna",
    "MotorSilauBerwarna": "Motor silau dan berwarna",
    "PantulanCahaya": "Pantulan cahaya",
}

EVENT_COLORS = {
    "Normal": "#F4F4F4",
    "Silau": "#FF9F1C",
    "Berwarna": "#00B8D9",
    "SilauBerwarna": "#D946EF",
    "PantulanCahaya": "#9CA3AF",
}

STYLE_BLOCK = """
<style>
[data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.8rem;
  letter-spacing: 0.02em;
}
</style>
"""


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown(STYLE_BLOCK, unsafe_allow_html=True)
    snapshot = load_counting_snapshot()
    render_header(snapshot.summary)
    if not render_state_notice(snapshot):
        return
    render_smoke_notice(snapshot.summary)
    render_totals(snapshot.summary)
    st.divider()
    render_playback(snapshot)
    st.divider()
    footage_tab = st.tabs(["Footage per kelas"])[0]
    with footage_tab:
        render_event_footage(snapshot)
    st.divider()
    render_model_validation(snapshot)
    st.divider()
    render_camera_section(snapshot)
    st.divider()
    render_contract_section(snapshot.summary)
    st.divider()
    render_class_section(snapshot.summary)
    st.divider()
    render_quality_section(snapshot.summary)
    render_footer()


def render_header(summary: dict | None) -> None:
    st.title(PAGE_TITLE)
    unit_note = "Satu baris = satu kendaraan unik di dalam ROI, bukan satu box per frame."
    if summary is None:
        st.caption(unit_note)
        return
    counted_splits = ", ".join(summary.get("counted_splits", []))
    st.caption(
        f"{unit_note} Split dihitung: {counted_splits} · "
        f"minimal {format_count(summary.get('min_track_hits'))} kemunculan per kendaraan."
    )


def render_state_notice(snapshot: CountingSnapshot) -> bool:
    if snapshot.state == STATE_EMPTY:
        st.warning(
            "Belum ada hasil model. Dashboard membaca `output/nb06/counting_summary.json` dan "
            "`output/nb06/counting_by_video.csv`; keduanya belum ada."
        )
        st.markdown(
            "Jalankan **NB03 → NB04 → NB05 → NB06** sampai selesai, lalu muat ulang halaman ini. "
            "Urutan lengkap ada di `README.md`."
        )
        return False
    if snapshot.state == STATE_PARTIAL:
        missing_list = ", ".join(f"`{name}`" for name in snapshot.missing_artifacts)
        st.warning(f"Artefak NB06 belum lengkap — belum ada: {missing_list}. Jalankan ulang NB06.")
        return False
    if snapshot.state == STATE_ERROR:
        st.error(f"Artefak NB06 ada tetapi tidak bisa dibaca: {snapshot.error_message}")
        return False
    return True


def render_smoke_notice(summary: dict) -> None:
    if summary.get("smoke_mode"):
        st.info("Mode SMOKE: angka di halaman ini berasal dari data sintetis, bukan hasil riset.")


def render_totals(summary: dict) -> None:
    counting_totals = summary["counting_totals"]
    total_columns = st.columns(4)
    total_columns[0].metric(
        "Kendaraan dihitung", format_count(summary.get("vehicles_counted")), border=True
    )
    total_columns[1].metric(
        "Mobil", format_count(counting_totals.get("jumlah_kendaraan_mobil")), border=True
    )
    total_columns[2].metric(
        "Motor", format_count(counting_totals.get("jumlah_kendaraan_motor")), border=True
    )
    total_columns[3].metric(
        "Video dihitung", format_count(summary.get("videos_counted")), border=True
    )


def render_playback(snapshot: CountingSnapshot) -> None:
    st.subheader("Playback dan counting timeline")
    if not snapshot.overlay_manifest or snapshot.timeline_frame is None:
        st.info(
            "Playback interaktif tersedia setelah NB06 selesai dengan artefak "
            "`overlay_manifest.json` dan `counting_timeline.csv`."
        )
        return
    manifest_frame = pd.DataFrame(snapshot.overlay_manifest)
    counted_splits = set(snapshot.summary.get("counted_splits", []))
    counted_cameras = sorted(
        manifest_frame.loc[manifest_frame["split"].isin(counted_splits), "camera_id"].unique().tolist()
    )
    other_cameras = sorted(
        set(manifest_frame["camera_id"].unique().tolist()) - set(counted_cameras)
    )
    camera_options = ["Semua titik", *counted_cameras, *other_cameras]
    default_camera_index = 1 if counted_cameras else 0
    selected_camera = st.selectbox(
        "Titik CCTV", camera_options, index=default_camera_index, key="playback_camera"
    )
    filtered_manifest = manifest_frame if selected_camera == "Semua titik" else manifest_frame[manifest_frame["camera_id"] == selected_camera]
    if filtered_manifest.empty:
        st.info("Tidak ada video overlay untuk titik ini.")
        return
    selected_video = st.selectbox(
        "Video overlay", filtered_manifest["video_stem"].tolist(), key="playback_video"
    )
    video_row = filtered_manifest[filtered_manifest["video_stem"] == selected_video].iloc[0]
    video_path = Path(video_row["overlay_path"])
    if not video_path.is_absolute():
        video_path = PROJECT_ROOT / video_path
    video_timeline = snapshot.timeline_frame[
        snapshot.timeline_frame["video_stem"] == selected_video
    ].copy()
    if video_timeline.empty:
        st.warning("Belum ada event tracking pada video ini.")
    elif video_row["split"] not in set(snapshot.summary.get("counted_splits", [])):
        st.info(
            f"Video ini berasal dari split `{video_row['split']}`. Counting yang tampil adalah "
            "mode presentasi/inspeksi, bukan angka resmi evaluasi test."
        )
    max_frame = max(int(video_row["frames_processed"]) - 1, 0)
    selected_frame = st.slider(
        "Posisi counting timeline (frame)", 0, max_frame, 0, key=f"playback_frame_{selected_video}"
    )
    fps = float(video_row["fps"])
    st.caption(f"Posisi: {selected_frame / fps:.1f} detik dari {float(video_row['duration_seconds']):.1f} detik")
    visible_timeline = video_timeline[video_timeline["first_frame"] <= selected_frame]
    video_column, count_column = st.columns([1.55, 1], gap="large")
    with video_column:
        st.markdown("#### Video overlay")
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"File overlay belum ditemukan: `{video_path}`")
    with count_column:
        st.markdown("#### Counting aktif")
        count_columns = st.columns(2)
        count_columns[0].metric("Kendaraan", format_count(len(visible_timeline)))
        count_columns[1].metric(
            "Mobil", format_count(int(visible_timeline["class_name"].str.startswith("Mobil").sum()))
        )
    event_groups = {
        "Normal": {"MobilNormal", "MotorNormal"},
        "Silau": {"MobilSilau", "MotorSilau"},
        "Berwarna": {"MobilBerwarna", "MotorBerwarna"},
        "Silau + berwarna": {"MobilSilauBerwarna", "MotorSilauBerwarna"},
    }
    with count_column:
        count_columns = st.columns(2)
        for column, (label, classes) in zip(count_columns, event_groups.items()):
            value = int(visible_timeline["class_name"].isin(classes).sum())
            column.metric(label, format_count(value))
        st.markdown("**Arti warna bbox dan tipe kendaraan**")
        render_class_legend()
    class_counts = visible_timeline["class_name"].value_counts().reindex(CLASS_NAMES, fill_value=0)
    st.bar_chart(pd.DataFrame({"Jumlah": class_counts}, index=[CLASS_LABELS[name] for name in CLASS_NAMES]), height=240)
    if not visible_timeline.empty:
        st.dataframe(
            visible_timeline.sort_values("first_frame")[["vehicle_id", "class_name", "first_frame", "hits", "mean_score"]],
            hide_index=True,
            width="stretch",
        )


def render_class_legend() -> None:
    legend_rows = []
    for class_name in CLASS_NAMES:
        if class_name == "PantulanCahaya":
            suffix = "PantulanCahaya"
        elif "SilauBerwarna" in class_name:
            suffix = "SilauBerwarna"
        elif "Silau" in class_name:
            suffix = "Silau"
        elif "Berwarna" in class_name:
            suffix = "Berwarna"
        else:
            suffix = "Normal"
        vehicle = "Mobil" if class_name.startswith("Mobil") else "Motor" if class_name.startswith("Motor") else "—"
        legend_rows.append(
            f'<div style="display:flex;align-items:center;gap:.45rem;margin:.2rem 0;">'
            f'<span style="width:12px;height:12px;border-radius:3px;background:{EVENT_COLORS[suffix]};display:inline-block;border:1px solid #777;"></span>'
            f'<span>{CLASS_LABELS[class_name]} <small>({vehicle})</small></span></div>'
        )
    st.markdown("".join(legend_rows), unsafe_allow_html=True)


def render_model_validation(snapshot: CountingSnapshot) -> None:
    st.subheader("Performa model dan batas interpretasi")
    nb06_root = resolve_nb06_root()
    nb04_root = nb06_root.parent / "nb04"
    results_paths = list((nb04_root / "runs").glob("*/results.csv")) if (nb04_root / "runs").exists() else []
    metrics_path = nb04_root / "val_metrics.json"
    if not results_paths and not metrics_path.exists():
        st.info("Artefak metrik NB04 belum tersedia.")
        return
    if results_paths:
        results_frame = pd.read_csv(sorted(results_paths)[-1])
        preferred_columns = [
            column for column in (
                "epoch", "train/box_loss", "train/cls_loss", "val/box_loss", "val/cls_loss",
                "metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)",
            ) if column in results_frame.columns
        ]
        if preferred_columns:
            st.caption("Kurva train dan validation per epoch. Ini bukan kurva akurasi test.")
            st.line_chart(results_frame[preferred_columns].set_index("epoch"), height=300)
    if metrics_path.exists():
        metrics_payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        overall = metrics_payload.get("overall", {})
        metric_frame = pd.DataFrame(
            [{"Metrik validation": key, "Nilai": value} for key, value in overall.items()]
        )
        if not metric_frame.empty:
            st.dataframe(metric_frame, hide_index=True, width="stretch")
    st.info(
        "Kamera test dipakai untuk inferensi dan counting, tetapi belum memiliki ground truth manual. "
        "Karena itu dashboard tidak menyebut hasil test sebagai akurasi model."
    )


def render_event_footage(snapshot: CountingSnapshot) -> None:
    st.subheader("Galeri crop event")
    st.caption("Setiap item hanya menampilkan crop bbox kendaraan agar event mudah dibandingkan.")
    if not snapshot.event_clip_manifest:
        st.info(
            "Event belum tersedia. Jalankan NB06 versi terbaru untuk membuat "
            "`event_clip_manifest.json` dan folder `event_crops/`."
        )
        return
    clips = pd.DataFrame(snapshot.event_clip_manifest)
    class_filter = st.selectbox("Kelas event", ["Semua kelas", *CLASS_NAMES], key="clip_class")
    camera_filter = st.selectbox(
        "Titik CCTV", ["Semua titik", *sorted(clips["camera_id"].unique().tolist())], key="clip_camera"
    )
    filtered = clips.copy()
    if class_filter != "Semua kelas":
        filtered = filtered[filtered["class_name"] == class_filter]
    if camera_filter != "Semua titik":
        filtered = filtered[filtered["camera_id"] == camera_filter]
    if filtered.empty:
        st.info("Tidak ada footage untuk filter tersebut.")
        return
    filtered = filtered.reset_index(drop=True)
    st.caption(f"{len(filtered)} event cocok dengan filter.")
    render_event_gallery(filtered)
    return
    selected_index = st.selectbox(
        "Pilih vehicle ID untuk membuka footage",
        filtered.index.tolist(),
        format_func=lambda index: (
            f"{CLASS_LABELS[filtered.loc[index, 'class_name']]} · "
            f"{filtered.loc[index, 'camera_id']} · vehicle {filtered.loc[index, 'vehicle_id']}"
        ),
        key="clip_selection",
    )
    selected_clip = filtered.loc[selected_index]
    clip_path = Path(selected_clip["clip_path"])
    if not clip_path.is_absolute():
        clip_path = PROJECT_ROOT / clip_path
    if clip_path.exists():
        st.video(str(clip_path))
    else:
        st.warning(f"File clip belum ditemukan: `{clip_path}`")


def render_event_gallery(filtered: pd.DataFrame) -> None:
    gallery_tab, grid_tab = st.tabs(["Detail event", "Grid 8 × 8"])
    with gallery_tab:
        if "gallery_index" not in st.session_state or st.session_state.gallery_index >= len(filtered):
            st.session_state.gallery_index = 0
        selected_index = st.session_state.gallery_index
        selected_clip = filtered.iloc[selected_index]
        crop_path = Path(str(selected_clip.get("crop_path", "")))
        if not crop_path.is_absolute():
            crop_path = PROJECT_ROOT / crop_path
        image_column, detail_column = st.columns([1.25, 1], gap="large")
        with image_column:
            if crop_path.exists():
                st.image(str(crop_path), caption=f"ID {selected_clip['vehicle_id']} · {CLASS_LABELS.get(selected_clip['class_name'], selected_clip['class_name'])}", width="stretch")
            else:
                st.warning(f"Crop belum ditemukan: `{crop_path}`. Rerun NB06 terbaru diperlukan.")
        with detail_column:
            class_name = str(selected_clip["class_name"])
            vehicle_type = "Mobil" if class_name.startswith("Mobil") else "Motor"
            st.markdown(f"### {CLASS_LABELS.get(class_name, class_name)}")
            st.write(f"**ID kendaraan:** {selected_clip['vehicle_id']}")
            st.write(f"**Jenis kendaraan:** {vehicle_type}")
            st.write(f"**CCTV:** {selected_clip['camera_id']}")
            st.write(f"**Menit:** {float(selected_clip.get('timestamp_seconds', 0)) / 60:.2f}")
            st.write(f"**Frame:** {selected_clip['first_frame']}–{selected_clip['last_frame']}")
            st.write(f"**Confidence rata-rata:** {float(selected_clip['mean_score']) * 100:.1f}%")
            previous_column, next_column = st.columns(2)
            if previous_column.button("← Sebelumnya", disabled=selected_index == 0, use_container_width=True):
                st.session_state.gallery_index = max(0, selected_index - 1)
                st.rerun()
            if next_column.button("Berikutnya →", disabled=selected_index == len(filtered) - 1, use_container_width=True):
                st.session_state.gallery_index = min(len(filtered) - 1, selected_index + 1)
                st.rerun()
    with grid_tab:
        page_count = max(1, (len(filtered) - 1) // 64 + 1)
        page = st.number_input("Halaman grid", min_value=1, max_value=page_count, value=1, step=1, key="gallery_page")
        page_frame = filtered.iloc[(page - 1) * 64: page * 64]
        st.caption(f"Menampilkan {len(page_frame)} crop pada halaman {page}/{page_count}.")
        for start in range(0, len(page_frame), 8):
            columns = st.columns(8, gap="small")
            for column, (_, item) in zip(columns, page_frame.iloc[start:start + 8].iterrows()):
                item_path = Path(str(item.get("crop_path", "")))
                if not item_path.is_absolute():
                    item_path = PROJECT_ROOT / item_path
                with column:
                    if item_path.exists():
                        st.image(str(item_path), caption=f"ID {item['vehicle_id']}\n{CLASS_LABELS.get(item['class_name'], item['class_name'])}", use_container_width=True)


def render_camera_section(snapshot: CountingSnapshot) -> None:
    st.subheader("Per titik CCTV")
    status_frame = build_camera_status_table(
        snapshot.camera_table, snapshot.summary.get("counted_splits", [])
    )
    st.dataframe(camera_display_frame(status_frame), hide_index=True, width="stretch")
    st.caption(
        "Titik CCTV di luar split hitung tetap ditampilkan agar cakupan dashboard tidak "
        "tampak lebih luas dari yang benar-benar dihitung NB06."
    )
    if snapshot.camera_table is None:
        return
    with st.expander("Rincian per kelas untuk tiap titik CCTV"):
        st.dataframe(class_display_frame(snapshot.camera_table), hide_index=True, width="stretch")


def camera_display_frame(status_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Titik CCTV": status_frame["camera_id"],
            "Split": status_frame["split"],
            "Status": status_frame["status"],
            "Video": formatted_column(status_frame, "videos_counted"),
            "Kendaraan": formatted_column(status_frame, "vehicles_counted"),
            "Mobil": formatted_column(status_frame, "jumlah_kendaraan_mobil"),
            "Motor": formatted_column(status_frame, "jumlah_kendaraan_motor"),
            "Silau mobil": formatted_column(status_frame, "jumlah_lampu_silau_mobil"),
            "Silau motor": formatted_column(status_frame, "jumlah_lampu_silau_motor"),
        }
    )


def class_display_frame(camera_table: pd.DataFrame) -> pd.DataFrame:
    class_columns = {
        CLASS_LABELS[class_name]: formatted_column(camera_table, class_name)
        for class_name in CLASS_NAMES
    }
    return pd.DataFrame({"Titik CCTV": camera_table["camera_id"], **class_columns})


def formatted_column(frame: pd.DataFrame, column_name: str) -> list[str]:
    if column_name not in frame.columns:
        return [NOT_AVAILABLE_LABEL for _ in range(len(frame))]
    return [format_count(value) for value in frame[column_name]]


def ratio_label(value: float | None) -> str:
    if value is None:
        return NOT_AVAILABLE_LABEL
    return f"{float(value):.{RATIO_DECIMAL_PLACES}f}x"


def render_contract_section(summary: dict) -> None:
    counting_totals = summary["counting_totals"]
    st.subheader("Delapan angka kontrak")
    measured_column, unmeasured_column = st.columns(2)
    with measured_column:
        st.caption("Terukur dari model")
        st.dataframe(
            contract_display_frame(counting_totals, MEASURABLE_CONTRACT_KEYS),
            hide_index=True,
            width="stretch",
        )
    if UNMEASURABLE_COUNT_CONTRACT_KEYS:
        with unmeasured_column:
            pass
        st.caption("Belum terukur — menunggu classifier atribut")
        st.dataframe(
            contract_display_frame(counting_totals, list(UNMEASURABLE_COUNT_CONTRACT_KEYS)),
            hide_index=True,
            width="stretch",
        )


def contract_display_frame(counting_totals: dict, contract_keys: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Keluaran": [CONTRACT_LABELS[contract_key] for contract_key in contract_keys],
            "Jumlah": [
                format_count(counting_totals.get(contract_key, NOT_AVAILABLE_LABEL))
                for contract_key in contract_keys
            ],
        }
    )


def render_class_section(summary: dict) -> None:
    st.subheader("Kendaraan per kelas")
    class_counts = summary.get("class_counts", {})
    class_frame = pd.DataFrame(
        {"Jumlah": [int(class_counts.get(class_name, 0)) for class_name in CLASS_NAMES]},
        index=[CLASS_LABELS[class_name] for class_name in CLASS_NAMES],
    )
    st.bar_chart(class_frame, height=280)
    st.caption(
        "Lima kelas detector YOLO diperkaya oleh dua atribut NB04b. Karena itu dashboard "
        "menampilkan sembilan event akhir: mobil/motor normal, silau, berwarna, silau+berwarna, "
        "serta pantulan cahaya."
    )


def render_quality_section(summary: dict) -> None:
    with st.expander("Cara angka ini dihasilkan"):
        quality_rows = [
            ("Frame diproses", format_count(summary.get("frames_processed"))),
            ("Deteksi mentah", format_count(summary.get("raw_detections"))),
            ("Deteksi tanpa track", format_count(summary.get("untracked_detections"))),
            ("Deteksi lolos ambang", format_count(summary.get("thresholded_detections"))),
            ("Track mentah ByteTrack", format_count(summary.get("tracks_total"))),
            ("Fragmen disambung", format_count(summary.get("fragments_stitched"))),
            ("Kendaraan setelah penyambungan", format_count(summary.get("vehicles_total"))),
            ("Rasio duplikasi vs hitung per frame", ratio_label(summary.get("duplication_factor_vs_per_frame"))),
            ("Video ditemukan", format_count(summary.get("videos_found"))),
        ]
        st.dataframe(
            pd.DataFrame(quality_rows, columns=["Tahap", "Nilai"]),
            hide_index=True,
            width="stretch",
        )
        render_threshold_table(summary.get("selected_thresholds", {}))


def render_threshold_table(selected_thresholds: dict) -> None:
    if not selected_thresholds:
        return
    st.dataframe(
        pd.DataFrame(
            [
                {"Kelas": CLASS_LABELS.get(class_name, class_name), "Ambang": threshold}
                for class_name, threshold in selected_thresholds.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_footer() -> None:
    st.caption(
        "Sumber: `output/nb06/counting_summary.json` dan `output/nb06/counting_by_video.csv`. "
        "Angka resmi hanya berasal dari NB06; NB05 melaporkan box per frame, bukan populasi kendaraan."
    )


if __name__ == "__main__":
    main()
