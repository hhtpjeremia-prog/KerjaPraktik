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
from pipeline_common import CLASS_NAMES

PAGE_TITLE = "Hasil Hitung Kendaraan — CCTV Kab. Bogor"

CLASS_LABELS = {
    "MobilStandar": "Mobil standar",
    "MobilTidakStandar": "Mobil tidak standar",
    "MotorStandar": "Motor standar",
    "MotorTidakStandar": "Motor tidak standar",
    "PantulanCahaya": "Pantulan cahaya",
}

EVENT_COLORS = {
    "TidakStandar": "#FF0000",
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
    st.sidebar.caption(f"Sumber artefak NB06: `{resolve_nb06_root().resolve()}`")
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
    unit_note = "Counting utama = kendaraan tidak standar yang melintasi counting line; setiap vehicle track dihitung sekali per video."
    if summary is None:
        st.caption(unit_note)
        return
    counted_splits = ", ".join(summary.get("counted_splits", []))
    st.caption(
        f"{unit_note} Split dihitung: {counted_splits}. "
        f"Filter minimal {format_count(summary.get('min_track_hits'))} kemunculan hanya berlaku "
        "untuk menyaring track singkat dan audit temporal. Bbox-per-frame tersedia sebagai metrik pendamping."
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
        "Track tidak standar crossing", format_count(summary.get("line_crossings_counted")), border=True
    )
    total_columns[1].metric(
        "Mobil tidak standar crossing", format_count(counting_totals.get("jumlah_kendaraan_mobil_tidak_standar_melintasi_garis")), border=True
    )
    total_columns[2].metric(
        "Motor tidak standar crossing", format_count(counting_totals.get("jumlah_kendaraan_motor_tidak_standar_melintasi_garis")), border=True
    )
    total_columns[3].metric(
        "Video diproses", format_count(summary.get("videos_counted")), border=True
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
    timeline_frame_column = "crossing_frame" if "crossing_frame" in video_timeline.columns else "frame_index"
    visible_timeline = video_timeline[video_timeline[timeline_frame_column] == selected_frame]
    video_column, count_column = st.columns([1.55, 1], gap="large")
    with video_column:
        st.markdown("#### Video overlay")
        if video_path.exists():
            st.video(str(video_path))
        else:
            st.warning(f"File overlay belum ditemukan: `{video_path}`")
    with count_column:
        st.markdown("#### Kendaraan melintasi garis pada frame ini")
        count_columns = st.columns(3)
        count_columns[0].metric("Total crossing", format_count(len(visible_timeline)))
        count_columns[1].metric("Mobil tidak standar", format_count(int(visible_timeline["class_name"].eq("MobilTidakStandar").sum())))
        count_columns[2].metric("Motor tidak standar", format_count(int(visible_timeline["class_name"].eq("MotorTidakStandar").sum())))
        if "crossing_direction" in visible_timeline:
            directions = visible_timeline["crossing_direction"].value_counts().to_dict()
            st.caption(f"Arah: turun {directions.get('down', 0)} · naik {directions.get('up', 0)}")
        st.markdown("**Arti warna bbox dan tipe kendaraan**")
        render_class_legend()
    class_counts = pd.Series(
        {
            "Mobil tidak standar": int(visible_timeline["class_name"].isin({"MobilTidakStandar"}).sum()),
            "Motor tidak standar": int(visible_timeline["class_name"].isin({"MotorTidakStandar"}).sum()),
        }
    )
    st.bar_chart(pd.DataFrame({"Track melintasi garis": class_counts}), height=220)
    if not visible_timeline.empty:
        st.dataframe(
            visible_timeline.sort_values(timeline_frame_column)[
                [column for column in ("vehicle_id", "track_id", "class_name", timeline_frame_column, "crossing_direction", "score") if column in visible_timeline.columns]
            ],
            hide_index=True,
            width="stretch",
        )


def render_class_legend() -> None:
    legend_rows = [
        f'<div style="display:flex;align-items:center;gap:.45rem;margin:.2rem 0;">'
        f'<span style="width:12px;height:12px;background:{EVENT_COLORS["TidakStandar"]};display:inline-block;"></span>'
        '<span>Merah = lampu tidak standar (mobil atau motor)</span></div>',
        '<div style="display:flex;align-items:center;gap:.45rem;margin:.2rem 0;">'
        '<span style="width:12px;height:12px;background:transparent;display:inline-block;outline:1px solid currentColor;"></span>'
        '<span>Tanpa bbox = standar / tidak ditampilkan</span></div>',
    ]
    st.markdown("".join(legend_rows), unsafe_allow_html=True)


def render_model_validation(snapshot: CountingSnapshot) -> None:
    st.subheader("Performa model dan batas interpretasi")
    nb06_root = resolve_nb06_root()
    nb04_root = nb06_root.parent / "nb04"
    nb05_root = nb06_root.parent / "nb05"
    ground_truth_root = nb06_root.parent / "ground_truth"
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
        if "epoch" in results_frame.columns and len(preferred_columns) > 1:
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

    # NB02c holdout GT lives separately from the NB06 inference/counting artifacts.
    # Read its review/export summary and the NB05 final detector evaluation explicitly.
    gt_summary_path = ground_truth_root / "annotation_summary.json"
    detector_test_path = nb05_root / "test_metrics.json"
    gt_summary = None
    if gt_summary_path.exists():
        try:
            gt_summary = json.loads(gt_summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            st.warning(f"Ringkasan GT NB02c tidak bisa dibaca: {error}")
    detector_test = None
    if detector_test_path.exists():
        try:
            detector_test = json.loads(detector_test_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            st.warning(f"Metrik test detector NB05 tidak bisa dibaca: {error}")

    st.markdown("#### Evaluasi holdout test dengan ground truth NB02c")
    if gt_summary:
        reviewed_frames = int(gt_summary.get("reviewed_frames", 0))
        vehicle_boxes = int(gt_summary.get("gold_boxes", 0))
        reflection_boxes = int(
            gt_summary.get(
                "detector_gt_reflection_target_boxes",
                gt_summary.get("gold_boxes_by_class", {}).get("PantulanCahaya", 0),
            )
        )
        st.caption(
            f"GT direview: {reviewed_frames:,} frame · {vehicle_boxes:,} bbox target detector "
            f"(termasuk {reflection_boxes:,} PantulanCahaya). Pantulan ikut evaluasi kelas, "
            "lalu disaring sebelum ByteTrack, overlay, dan counting kendaraan."
        )
    else:
        st.info("Ringkasan GT NB02c belum ditemukan. Periksa `output/ground_truth/annotation_summary.json`.")

    if detector_test and detector_test.get("status") == "computed":
        detector_overall = detector_test.get("overall", {})
        st.markdown(
            f"**YOLO test** · {int(detector_test.get('frames', 0)):,} frame berlabel "
            f"· sumber: {detector_test.get('label_source', 'ground truth')}"
        )
        detector_metric_names = {
            "precision": "Precision",
            "recall": "Recall",
            "mAP50": "mAP@50",
            "mAP50-95": "mAP@50:95",
            "bbox_coordinate_mse_normalized": "Koordinat bbox MSE (normalisasi)",
            "bbox_coordinate_rmse_normalized": "Koordinat bbox RMSE (normalisasi)",
            "bbox_size_mae_normalized": "Ukuran bbox MAE (normalisasi)",
            "bbox_iou_error_mean": "Rata-rata error IoU (1 − IoU)",
            "bbox_matched_count": "Pasangan prediksi–GT",
        }
        detector_rows = [
            {"Metrik test YOLO": label, "Nilai": detector_overall[key]}
            for key, label in detector_metric_names.items()
            if key in detector_overall
        ]
        if detector_rows:
            st.dataframe(pd.DataFrame(detector_rows), hide_index=True, width="stretch")
    elif gt_summary:
        st.info(
            "GT sudah tersedia, tetapi artefak evaluasi test YOLO belum ada atau belum berstatus "
            "computed. Jalankan tahap evaluasi test NB05, lalu muat ulang dashboard."
        )
    else:
        st.info("Metrik test YOLO belum dapat ditampilkan karena ringkasan GT belum tersedia.")

    attribute_root = nb06_root.parent / "attribute_classifier"
    attribute_history_path = attribute_root / "training_history.csv"
    attribute_metrics_path = attribute_root / "metrics.json"
    if attribute_history_path.exists():
        st.markdown("#### CNN kondisi lampu")
        attribute_history = pd.read_csv(attribute_history_path)
        history_columns = [
            column for column in (
                "epoch", "train_loss", "val_loss", "train_f1", "val_f1",
                "train_accuracy", "val_accuracy",
            ) if column in attribute_history.columns
        ]
        if "epoch" in attribute_history.columns and len(history_columns) > 1:
            st.line_chart(attribute_history[history_columns].set_index("epoch"), height=280)
        if attribute_metrics_path.exists():
            attribute_metrics = json.loads(attribute_metrics_path.read_text(encoding="utf-8"))
            test_metrics = attribute_metrics.get("test_metrics")
            if test_metrics is None:
                if gt_summary:
                    st.info(
                        "GT holdout tersedia, tetapi metrik test CNN belum tercatat. "
                        "Periksa atau jalankan ulang evaluasi test NB04b."
                    )
                else:
                    st.info("Metrik test CNN belum tersedia karena GT holdout belum ditemukan.")
            else:
                st.markdown("**CNN kondisi lampu — test holdout**")
                st.dataframe(
                    pd.DataFrame([test_metrics]),
                    hide_index=True,
                    width="stretch",
                )
    if gt_summary and detector_test and detector_test.get("status") == "computed":
        st.caption(
            "Metrik di atas membandingkan prediksi model dengan label holdout yang direview. "
            "Counting NB06 tetap merupakan keluaran inferensi pada video; metrik ini tidak "
            "mengubah atau menggantikan angka counting."
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
    nonstandard_classes = {"MobilTidakStandar", "MotorTidakStandar"}
    clips = clips[clips["class_name"].isin(nonstandard_classes)].copy()
    class_filter = st.selectbox("Kelas event", ["Semua tidak standar", "Mobil tidak standar", "Motor tidak standar"], key="clip_class")
    camera_filter = st.selectbox(
        "Titik CCTV", ["Semua titik", *sorted(clips["camera_id"].unique().tolist())], key="clip_camera"
    )
    filtered = clips.copy()
    if class_filter == "Mobil tidak standar":
        filtered = filtered[filtered["class_name"].str.startswith("Mobil")]
    elif class_filter == "Motor tidak standar":
        filtered = filtered[filtered["class_name"].str.startswith("Motor")]
    if camera_filter != "Semua titik":
        filtered = filtered[filtered["camera_id"] == camera_filter]
    if filtered.empty:
        st.info("Tidak ada footage untuk filter tersebut.")
        return
    filtered = filtered.reset_index(drop=True)
    st.caption(f"{len(filtered)} event cocok dengan filter.")
    render_event_gallery(filtered)


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
        st.caption("Rincian kelas adalah bbox deteksi per frame; angka utama adalah track tidak standar yang melintasi garis.")


def camera_display_frame(status_frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Titik CCTV": status_frame["camera_id"],
            "Split": status_frame["split"],
            "Status": status_frame["status"],
            "Video": formatted_column(status_frame, "videos_counted"),
            "Kendaraan crossing": formatted_column(status_frame, "crossings_counted"),
            "Mobil tidak standar crossing": formatted_column(status_frame, "jumlah_kendaraan_mobil_tidak_standar_melintasi_garis"),
            "Motor tidak standar crossing": formatted_column(status_frame, "jumlah_kendaraan_motor_tidak_standar_melintasi_garis"),
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


def render_contract_section(summary: dict) -> None:
    counting_totals = summary["counting_totals"]
    st.subheader("Kendaraan tidak standar melintasi garis")
    st.caption("Satu track dihitung sekali saat center bbox melewati garis hitung; arah crossing ikut dicatat.")
    st.dataframe(
        pd.DataFrame({
            "Keluaran": ["Mobil tidak standar", "Motor tidak standar"],
            "Jumlah": [
                format_count(counting_totals.get("jumlah_kendaraan_mobil_tidak_standar_melintasi_garis", 0)),
                format_count(counting_totals.get("jumlah_kendaraan_motor_tidak_standar_melintasi_garis", 0)),
            ],
        }),
        hide_index=True,
        width="stretch",
    )
def render_class_section(summary: dict) -> None:
    st.subheader("Track tidak standar melintasi garis per tipe kendaraan")
    class_counts = summary.get("event_class_counts", {})
    class_frame = pd.DataFrame(
        {"Jumlah crossing": [
            sum(int(class_counts.get(class_name, 0)) for class_name in ("MobilTidakStandar",)),
            sum(int(class_counts.get(class_name, 0)) for class_name in ("MotorTidakStandar",)),
        ]},
        index=["Mobil tidak standar", "Motor tidak standar"],
    )
    st.bar_chart(class_frame, height=280)
    st.caption("Satu track hanya dihitung sekali per video. Kendaraan standar dan pantulan tidak masuk counting utama.")


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
            ("Crossing line utama", format_count(summary.get("line_crossings_counted"))),
            ("BBox event per frame (audit)", format_count(summary.get("bbox_per_frame_nonstandard_detections"))),
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
