"""Dashboard Streamlit hasil hitung kendaraan per titik CCTV dari artefak NB06."""

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
)
from pipeline_common import TRAINING_CLASS_NAMES, UNMEASURABLE_COUNT_CONTRACT_KEYS

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
        for class_name in TRAINING_CLASS_NAMES
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
    with unmeasured_column:
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
        {"Jumlah": [int(class_counts.get(class_name, 0)) for class_name in TRAINING_CLASS_NAMES]},
        index=[CLASS_LABELS[class_name] for class_name in TRAINING_CLASS_NAMES],
    )
    st.bar_chart(class_frame, height=280)
    st.caption(
        "Hanya lima kelas ini yang dilatih model. Mobil berwarna dan motor berwarna dilipat ke "
        "kelas normal, silau berwarna dilipat ke kelas silau, jadi empat angka kontrak berwarna "
        "belum bisa diukur."
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
