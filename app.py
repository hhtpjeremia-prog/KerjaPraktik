"""Dashboard Streamlit hasil hitung kendaraan per titik CCTV (siap deploy Streamlit Cloud & lokal)."""

import base64
import html
import importlib
import json
import os
import sys
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

APP_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = APP_ROOT
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))

import data as _data_mod

if getattr(_data_mod, "ROLE_INDEPENDENT_TEST", None) != "Lokasi Uji Model":
    _data_mod = importlib.reload(_data_mod)

from data import (
    CLASS_NAMES,
    NOT_AVAILABLE_LABEL,
    ROLE_INDEPENDENT_TEST,
    STATE_EMPTY,
    STATE_ERROR,
    STATE_PARTIAL,
    CountingSnapshot,
    build_all_cameras_crossing_table,
    camera_display_name,
    format_count,
    format_exact_count,
    load_counting_snapshot,
    resolve_nb06_root,
)
from media_server import LocalVideoServer

PAGE_TITLE = "Pendeteksi Objek Lampu Kendaraan Tidak Standar — CCTV Kab. Bogor"

SCOPE_ALL_LOCATIONS = "Semua Lokasi (5 Titik CCTV)"
SCOPE_TEST_LOCATIONS = "Lokasi Uji Model (2 Titik CCTV)"

CLASS_LABELS = {
    "MobilStandar": "Mobil standar",
    "MobilTidakStandar": "Mobil tidak standar",
    "MotorStandar": "Motor standar",
    "MotorTidakStandar": "Motor tidak standar",
    "PantulanCahaya": "Pantulan cahaya",
}

SNAPSHOT_ARTIFACTS = (
    "counting_summary.json",
    "counting_by_video.csv",
    "counting_timeline.csv",
    "overlay_manifest.json",
    "event_clip_manifest.json",
)

STYLE_BLOCK = """
<style>
[data-testid="stMetricValue"] {
  font-variant-numeric: tabular-nums;
}
[data-testid="stMetricLabel"] p {
  font-size: 0.82rem;
  letter-spacing: 0.01em;
}
</style>
"""


def resolve_video_base_url() -> str | None:
    """Ambil VIDEO_BASE_URL dari Streamlit Secrets atau environment variable untuk mode cloud."""
    env_url = os.environ.get("VIDEO_BASE_URL", "").strip()
    if env_url:
        return env_url.rstrip("/")
    try:
        secret_url = str(st.secrets.get("VIDEO_BASE_URL", "")).strip()
        if secret_url and "<username>" not in secret_url:
            return secret_url.rstrip("/")
    except Exception:
        pass
    return None


def main() -> None:
    st.set_page_config(page_title=PAGE_TITLE, layout="wide")
    st.markdown(STYLE_BLOCK, unsafe_allow_html=True)
    video_base_url = resolve_video_base_url()
    st.sidebar.caption("CCTV Kabupaten Bogor · 5 Lokasi Pantau")
    st.sidebar.caption(
        "Mode pemutaran video: Streaming Cloud (HTTPS)"
        if video_base_url
        else "Mode pemutaran video: Lokal (HTTP 206)"
    )
    snapshot = load_cached_counting_snapshot()
    render_header(snapshot.summary)
    if not render_state_notice(snapshot):
        return
    render_smoke_notice(snapshot.summary)

    all_cameras_frame = build_all_cameras_crossing_table(
        snapshot.timeline_frame,
        snapshot.summary.get("counted_splits", []),
        snapshot.overlay_manifest,
    )

    tab_overview, tab_locations = st.tabs(
        ["Pemantauan Umum", "Performa Model pada Data Uji"]
    )

    with tab_overview:
        selected_scope = st.radio(
            "Pilih cakupan lokasi",
            [SCOPE_ALL_LOCATIONS, SCOPE_TEST_LOCATIONS],
            horizontal=True,
            key="overview_scope",
        )
        only_test_locations = selected_scope == SCOPE_TEST_LOCATIONS
        if only_test_locations:
            st.caption(
                "Menampilkan khusus 2 titik yang dipakai buat menguji performa model (**Bappenda 2** dan **PDAM 2**) — "
                "keduanya tidak ikut saat training, jadi hasilnya menggambarkan performa model di lokasi baru."
            )
        else:
            st.caption(
                "Menampilkan gabungan hasil deteksi dari 5 titik CCTV Kabupaten Bogor. "
                "Pindah ke **Lokasi Uji Model** kalau mau fokus ke 2 titik yang dipakai buat menguji model."
            )
        render_totals(all_cameras_frame, only_test_locations=only_test_locations)
        st.divider()
        render_playback(snapshot, all_cameras_frame, only_test_locations=only_test_locations)
        st.divider()
        render_event_footage(snapshot, all_cameras_frame, only_test_locations=only_test_locations)
        st.divider()
        render_camera_section(all_cameras_frame, only_test_locations=only_test_locations)

    with tab_locations:
        render_test_locations_overview(all_cameras_frame)
        st.divider()
        render_model_validation(snapshot)
        st.divider()
        render_contract_section(snapshot.summary, snapshot.camera_table)
        render_quality_section(snapshot.summary)

    st.divider()
    render_footer()


def snapshot_fingerprint(root: Path) -> tuple[tuple[str, int | None, int | None], ...]:
    fingerprints = []
    for name in SNAPSHOT_ARTIFACTS:
        try:
            stat = (root / name).stat()
            fingerprints.append((name, stat.st_mtime_ns, stat.st_size))
        except OSError:
            fingerprints.append((name, None, None))
    return tuple(fingerprints)


@st.cache_data(ttl=30, show_spinner=False)
def _load_counting_snapshot_cached(root_value: str, fingerprint: tuple) -> CountingSnapshot:
    del fingerprint  # Fingerprints participate in Streamlit's cache key.
    return load_counting_snapshot(Path(root_value))


@st.cache_resource(show_spinner=False)
def get_local_video_server() -> LocalVideoServer:
    return LocalVideoServer()


def load_cached_counting_snapshot() -> CountingSnapshot:
    root = resolve_nb06_root().resolve()
    return _load_counting_snapshot_cached(str(root), snapshot_fingerprint(root))


def render_header(summary: dict | None) -> None:
    del summary
    st.title(PAGE_TITLE)
    st.caption(
        "Deteksi otomatis kendaraan berlampu tidak standar di 5 titik CCTV Kabupaten Bogor. "
        "Tiap kendaraan dihitung sekali saat melintasi garis hitung kuning."
    )


def render_state_notice(snapshot: CountingSnapshot) -> bool:
    if snapshot.state == STATE_EMPTY:
        st.warning(
            "Belum ada hasil pemantauan. Dashboard membutuhkan `counting_summary.json` dan "
            "`counting_by_video.csv`; keduanya belum ditemukan."
        )
        return False
    if snapshot.state == STATE_PARTIAL:
        missing_list = ", ".join(f"`{name}`" for name in snapshot.missing_artifacts)
        st.warning(f"Data pemantauan belum lengkap — berkas berikut belum ada: {missing_list}.")
        return False
    if snapshot.state == STATE_ERROR:
        st.error(f"Data pemantauan ada tetapi tidak dapat dibaca: {snapshot.error_message}")
        return False
    return True


def render_smoke_notice(summary: dict) -> None:
    if summary.get("smoke_mode"):
        st.info("Mode simulasi: angka di halaman ini berasal dari data contoh, bukan rekaman asli.")


def render_totals(all_cameras_frame: pd.DataFrame, *, only_test_locations: bool) -> None:
    active_frame = (
        all_cameras_frame[all_cameras_frame["is_test_location"]]
        if only_test_locations
        else all_cameras_frame
    )
    total_crossings = int(active_frame["crossings_counted"].sum()) if not active_frame.empty else 0
    mobil_crossings = int(active_frame["mobil_crossings"].sum()) if not active_frame.empty else 0
    motor_crossings = int(active_frame["motor_crossings"].sum()) if not active_frame.empty else 0
    locations_count = int(len(active_frame))

    total_columns = st.columns(4)
    total_columns[0].metric(
        "Total Kendaraan Tidak Standar", format_exact_count(total_crossings), border=True
    )
    total_columns[1].metric(
        "Mobil Tidak Standar", format_exact_count(mobil_crossings), border=True
    )
    total_columns[2].metric(
        "Motor Tidak Standar", format_exact_count(motor_crossings), border=True
    )
    total_columns[3].metric(
        "Titik CCTV Dipantau", format_exact_count(locations_count), border=True
    )


def _resolve_project_path(raw_path: object) -> Path | None:
    text = str(raw_path or "").strip()
    if not text or text.lower() == "nan":
        return None
    normalized = text.replace("\\", "/")
    candidate = Path(normalized)
    if candidate.is_absolute():
        return candidate
    in_app = APP_ROOT / candidate
    if in_app.exists():
        return in_app
    in_nb06 = resolve_nb06_root() / candidate
    if in_nb06.exists():
        return in_nb06
    in_parent_output = APP_ROOT.parent / "output" / "nb06" / candidate
    if in_parent_output.exists():
        return in_parent_output
    return in_app


def _build_remote_video_url(base_url: str, relpath: object) -> str | None:
    text = str(relpath or "").strip().replace("\\", "/").lstrip("/")
    if not text or text.lower() == "nan":
        return None
    if text.startswith("overlays_browser/"):
        text = text[len("overlays_browser/") :]
    encoded = "/".join(quote(part, safe="") for part in text.split("/"))
    return f"{base_url}/{encoded}"


@st.cache_data(show_spinner=False)
def _load_video_crop_data_uris(
    nb06_root_str: str,
    crop_entries: tuple[tuple[int, str, str], ...],
) -> dict[int, dict[str, str]]:
    """Muat foto crop (dengan & tanpa highlight) untuk satu video menjadi data URI ringan (~1 KB/foto)."""
    del nb06_root_str
    result: dict[int, dict[str, str]] = {}
    for vehicle_id, mask_raw, plain_raw in crop_entries:
        entry: dict[str, str] = {}
        for key, raw_val in (("mask", mask_raw), ("raw", plain_raw)):
            crop_path = _resolve_project_path(raw_val)
            if crop_path is not None and crop_path.exists():
                try:
                    encoded = base64.b64encode(crop_path.read_bytes()).decode("ascii")
                    mime = "image/png" if crop_path.suffix.lower() == ".png" else "image/jpeg"
                    entry[key] = f"data:{mime};base64,{encoded}"
                except OSError:
                    continue
        if entry:
            result[vehicle_id] = entry
    return result


def _serialize_timeline_events(
    video_timeline: pd.DataFrame,
    fps: float,
    crop_data_uris: dict[int, dict[str, str]] | None = None,
) -> list[dict]:
    if video_timeline.empty:
        return []
    frame_column = "crossing_frame" if "crossing_frame" in video_timeline.columns else "frame_index"
    ordered = video_timeline.sort_values([frame_column, "vehicle_id"])
    crop_lookup = crop_data_uris or {}
    events = []
    for _, row in ordered.iterrows():
        frame_index = int(row[frame_column])
        if "timestamp_seconds" in row and pd.notna(row["timestamp_seconds"]):
            timestamp_seconds = float(row["timestamp_seconds"])
        else:
            timestamp_seconds = frame_index / max(fps, 1e-6)
        class_name = str(row["class_name"])
        vehicle_id = int(row["vehicle_id"]) if "vehicle_id" in row and pd.notna(row["vehicle_id"]) else 0
        crops = crop_lookup.get(vehicle_id, {})
        events.append({
            "vehicle_id": vehicle_id,
            "track_id": int(row["track_id"]) if "track_id" in row and pd.notna(row["track_id"]) else 0,
            "class_name": class_name,
            "class_label": CLASS_LABELS.get(class_name, class_name),
            "frame": frame_index,
            "ts": round(timestamp_seconds, 3),
            "direction": str(row.get("crossing_direction", "down")),
            "score": round(float(row.get("score", 0.0)) * 100.0, 1),
            "cropMask": crops.get("mask", ""),
            "cropRaw": crops.get("raw", ""),
        })
    return events


def build_realtime_playback_html(
    *,
    video_url_with_mask: str | None,
    video_url_without_mask: str | None,
    camera_id: str,
    camera_label: str,
    video_stem: str,
    fps: float,
    duration_seconds: float,
    events: list[dict],
) -> str:
    payload = {
        "urlWithMask": video_url_with_mask or "",
        "urlWithoutMask": video_url_without_mask or "",
        "cameraId": camera_id,
        "cameraLabel": camera_label,
        "videoStem": video_stem,
        "fps": fps,
        "duration": duration_seconds,
        "events": events,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_camera = html.escape(camera_label)
    has_pair = bool(video_url_with_mask and video_url_without_mask)
    toggle_markup = (
        '<div class="seg-toggle" role="group" aria-label="Tampilan highlight lampu">'
        '<button type="button" id="btnMaskOn" class="seg-btn active">Dengan highlight</button>'
        '<button type="button" id="btnMaskOff" class="seg-btn">Tanpa highlight</button>'
        "</div>"
        if has_pair
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="id">
<head>
<meta charset="utf-8" />
<style>
  :root {{
    --accent: #1F7A4D;
    --accent-soft: #E8F3ED;
    --bg: #FFFFFF;
    --surface: #F4F6F5;
    --border: #DCE2DF;
    --text: #1A1D1B;
    --muted: #5A635E;
    --event-red: #FF0000;
    --line-yellow: #D4A100;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: transparent;
    color: var(--text);
    line-height: 1.4;
  }}
  .layout {{
    display: grid;
    grid-template-columns: 1.55fr 1fr;
    gap: 20px;
    align-items: start;
  }}
  @media (max-width: 920px) {{
    .layout {{ grid-template-columns: 1fr; }}
  }}
  .panel {{
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
  }}
  .toolbar {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 12px;
  }}
  .title-block {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }}
  .meta-line {{
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .controls-row {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
  }}
  .seg-toggle {{
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
    background: var(--surface);
  }}
  .seg-btn {{
    border: none;
    background: transparent;
    color: var(--text);
    padding: 5px 10px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }}
  .seg-btn.active {{
    background: var(--accent);
    color: #FFFFFF;
  }}
  .speed-select {{
    border: 1px solid var(--border);
    border-radius: 6px;
    background: var(--bg);
    color: var(--text);
    padding: 4px 8px;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }}
  video {{
    width: 100%;
    max-height: 460px;
    background: #111312;
    border-radius: 6px;
    display: block;
  }}
  .progress-wrap {{
    margin-top: 10px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
    font-size: 12px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
  }}
  .legend {{
    display: flex;
    flex-wrap: wrap;
    gap: 14px;
    margin-top: 10px;
    padding-top: 10px;
    border-top: 1px solid var(--border);
    font-size: 12px;
    color: var(--muted);
  }}
  .legend-item {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
  }}
  .swatch {{
    width: 11px;
    height: 11px;
    display: inline-block;
    border-radius: 2px;
  }}
  .metrics-grid {{
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 12px;
  }}
  .metric-card {{
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 10px 12px;
    transition: border-color 140ms ease, background-color 140ms ease;
  }}
  .metric-card.pulse {{
    border-color: var(--accent);
    background: var(--accent-soft);
  }}
  .metric-label {{
    font-size: 11px;
    color: var(--muted);
    text-transform: none;
    margin-bottom: 4px;
  }}
  .metric-value {{
    font-size: 24px;
    font-weight: 700;
    color: var(--text);
    font-variant-numeric: tabular-nums;
    line-height: 1.15;
  }}
  .metric-sub {{
    font-size: 11px;
    color: var(--muted);
    font-variant-numeric: tabular-nums;
    margin-top: 2px;
  }}
  .status-strip {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 10px;
    font-variant-numeric: tabular-nums;
  }}
  .live-banner {{
    border: 1px solid var(--border);
    background: var(--surface);
    border-radius: 6px;
    padding: 8px 10px;
    font-size: 12px;
    color: var(--muted);
    margin-bottom: 12px;
    min-height: 36px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    font-variant-numeric: tabular-nums;
  }}
  .live-banner.active {{
    border-color: var(--accent);
    background: var(--accent-soft);
    color: var(--text);
    font-weight: 500;
  }}
  .feed-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 8px;
    margin-bottom: 8px;
  }}
  .feed-title {{
    font-size: 13px;
    font-weight: 600;
  }}
  .feed-tabs {{
    display: inline-flex;
    border: 1px solid var(--border);
    border-radius: 6px;
    overflow: hidden;
  }}
  .feed-tab {{
    border: none;
    background: var(--surface);
    color: var(--muted);
    padding: 4px 8px;
    font-size: 11px;
    cursor: pointer;
  }}
  .feed-tab.active {{
    background: var(--accent);
    color: #FFFFFF;
  }}
  .feed-wrap {{
    max-height: 295px;
    overflow-y: auto;
    border: 1px solid var(--border);
    border-radius: 6px;
  }}
  table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
    font-variant-numeric: tabular-nums;
  }}
  thead th {{
    position: sticky;
    top: 0;
    background: var(--surface);
    color: var(--muted);
    font-weight: 600;
    text-align: left;
    padding: 7px 8px;
    border-bottom: 1px solid var(--border);
  }}
  tbody tr {{
    border-bottom: 1px solid var(--border);
    cursor: pointer;
  }}
  tbody tr:hover {{
    background: var(--surface);
  }}
  tbody tr.recent {{
    background: var(--accent-soft);
  }}
  tbody td {{
    padding: 6px 8px;
    white-space: nowrap;
  }}
  .empty-row {{
    text-align: center;
    color: var(--muted);
    padding: 18px 8px !important;
    cursor: default;
  }}
  .jump-hint {{
    font-size: 11px;
    color: var(--muted);
    margin-top: 8px;
  }}
  .id-crop-btn {{
    display: inline-flex;
    align-items: center;
    gap: 4px;
    border: 1px solid #BBD8C8;
    background: var(--accent-soft);
    color: var(--accent);
    font-weight: 600;
    font-size: 11.5px;
    padding: 2px 7px;
    border-radius: 5px;
    cursor: pointer;
    font-variant-numeric: tabular-nums;
    transition: background-color 120ms ease, color 120ms ease;
  }}
  .id-crop-btn:hover {{
    background: var(--accent);
    color: #FFFFFF;
    border-color: var(--accent);
  }}
  .banner-crop-btn {{
    border: 1px solid var(--accent);
    background: #FFFFFF;
    color: var(--accent);
    font-size: 11px;
    font-weight: 600;
    padding: 2px 7px;
    border-radius: 5px;
    cursor: pointer;
    white-space: nowrap;
  }}
  .banner-crop-btn:hover {{
    background: var(--accent);
    color: #FFFFFF;
  }}
  .crop-modal-backdrop {{
    position: fixed;
    inset: 0;
    background: rgba(15, 23, 20, 0.56);
    display: none;
    align-items: center;
    justify-content: center;
    padding: 16px;
    z-index: 50;
  }}
  .crop-modal-backdrop.open {{
    display: flex;
  }}
  .crop-modal {{
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    width: min(560px, 100%);
    box-shadow: 0 18px 42px rgba(0, 0, 0, 0.24);
    overflow: hidden;
  }}
  .crop-modal-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 16px;
    background: var(--surface);
    border-bottom: 1px solid var(--border);
  }}
  .crop-modal-title {{
    font-size: 14px;
    font-weight: 600;
    color: var(--text);
  }}
  .crop-modal-close {{
    border: none;
    background: transparent;
    color: var(--muted);
    font-size: 20px;
    line-height: 1;
    cursor: pointer;
    padding: 2px 6px;
    border-radius: 4px;
  }}
  .crop-modal-close:hover {{
    background: #E2E7E4;
    color: var(--text);
  }}
  .crop-modal-body {{
    display: grid;
    grid-template-columns: 1.15fr 1fr;
    gap: 16px;
    padding: 16px;
    align-items: start;
  }}
  @media (max-width: 540px) {{
    .crop-modal-body {{ grid-template-columns: 1fr; }}
  }}
  .crop-img-box {{
    background: #141715;
    border-radius: 8px;
    min-height: 220px;
    max-height: 270px;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
    padding: 8px;
  }}
  .crop-img-box img {{
    width: 100%;
    height: 240px;
    object-fit: contain;
    border-radius: 4px;
    image-rendering: auto;
  }}
  .crop-img-empty {{
    color: #CBD4CF;
    font-size: 12px;
    text-align: center;
    padding: 16px;
  }}
  .crop-meta {{
    display: flex;
    flex-direction: column;
    gap: 8px;
    font-size: 12.5px;
  }}
  .crop-meta-row {{
    display: flex;
    justify-content: space-between;
    gap: 8px;
    border-bottom: 1px solid #EEF2F0;
    padding-bottom: 5px;
  }}
  .crop-meta-label {{
    color: var(--muted);
  }}
  .crop-meta-val {{
    color: var(--text);
    font-weight: 600;
    text-align: right;
    font-variant-numeric: tabular-nums;
  }}
  .crop-modal-footer {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 10px;
    padding: 10px 16px 14px;
    border-top: 1px solid var(--border);
    flex-wrap: wrap;
  }}
  .modal-actions {{
    display: flex;
    gap: 8px;
  }}
  .modal-btn {{
    border: 1px solid var(--border);
    background: var(--surface);
    color: var(--text);
    border-radius: 6px;
    padding: 6px 11px;
    font-size: 12px;
    font-weight: 500;
    cursor: pointer;
  }}
  .modal-btn.primary {{
    background: var(--accent);
    border-color: var(--accent);
    color: #FFFFFF;
  }}
  .modal-btn:hover {{
    filter: brightness(0.96);
  }}
</style>
</head>
<body>
<div class="layout">
  <div class="panel">
    <div class="toolbar">
      <div>
        <div class="title-block">Pemutar CCTV sinkron — {safe_camera}</div>
        <div class="meta-line" id="videoClock">Detik 0.0 / {duration_seconds:.1f} · Frame 0</div>
      </div>
      <div class="controls-row">
        {toggle_markup}
        <select id="speedSelect" class="speed-select" aria-label="Kecepatan putar">
          <option value="0.5">0.5x</option>
          <option value="1" selected>1.0x</option>
          <option value="1.5">1.5x</option>
          <option value="2">2.0x</option>
        </select>
      </div>
    </div>
    <video id="overlayVideo" controls playsinline preload="metadata" crossorigin="anonymous"></video>
    <div class="progress-wrap">
      <span id="statusText">Siap diputar — angka otomatis bertambah saat kendaraan melewati garis kuning.</span>
      <span id="eventTotalLabel">Total video: {len(events)} kendaraan</span>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:#FF0000;"></span>Kotak merah = lampu tidak standar</span>
      <span class="legend-item"><span class="swatch" style="background:#D4A100;"></span>Garis kuning = garis hitung virtual</span>
      <span class="legend-item"><span class="swatch" style="background:transparent;border:1px solid #5A635E;"></span>Tanpa kotak = lampu standar</span>
    </div>
  </div>

  <div class="panel">
    <div class="metrics-grid">
      <div class="metric-card" id="cardTotal">
        <div class="metric-label">Total melintas</div>
        <div class="metric-value" id="valTotal">0</div>
        <div class="metric-sub" id="subTotal">dari {len(events)}</div>
      </div>
      <div class="metric-card" id="cardMobil">
        <div class="metric-label">Mobil tidak standar</div>
        <div class="metric-value" id="valMobil">0</div>
        <div class="metric-sub" id="subMobil">terhitung</div>
      </div>
      <div class="metric-card" id="cardMotor">
        <div class="metric-label">Motor tidak standar</div>
        <div class="metric-value" id="valMotor">0</div>
        <div class="metric-sub" id="subMotor">terhitung</div>
      </div>
    </div>

    <div class="status-strip">
      <span id="directionSummary">Arah: turun 0 · naik 0</span>
      <span id="nextEventHint">Berikutnya: —</span>
    </div>

    <div class="live-banner" id="liveBanner">
      <span>Menunggu kendaraan melewati garis hitung...</span>
    </div>

    <div class="feed-header">
      <span class="feed-title">Log pelintasan garis</span>
      <div class="feed-tabs">
        <button type="button" id="tabPassed" class="feed-tab active">Sudah lewat</button>
        <button type="button" id="tabAll" class="feed-tab">Semua</button>
      </div>
    </div>

    <div class="feed-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Detik</th>
            <th>ID (Foto)</th>
            <th>Kategori</th>
            <th>Yakin</th>
          </tr>
        </thead>
        <tbody id="feedBody"></tbody>
      </table>
    </div>
    <div class="jump-hint">Klik <strong>ID kendaraan</strong> untuk menjeda video dan langsung melihat foto crop. Klik kolom lain untuk melompat ke detik melintas.</div>
  </div>
</div>

<div class="crop-modal-backdrop" id="cropModalBackdrop" role="dialog" aria-modal="true" aria-labelledby="cropModalTitle">
  <div class="crop-modal">
    <div class="crop-modal-header">
      <span class="crop-modal-title" id="cropModalTitle">Foto Kendaraan</span>
      <button type="button" class="crop-modal-close" id="cropModalClose" aria-label="Tutup">&times;</button>
    </div>
    <div class="crop-modal-body">
      <div class="crop-img-box" id="cropImgBox">
        <img id="cropModalImg" alt="Foto crop kendaraan" />
        <div class="crop-img-empty" id="cropImgEmpty" style="display:none;">Foto crop tidak tersedia untuk kendaraan ini.</div>
      </div>
      <div class="crop-meta">
        <div class="crop-meta-row"><span class="crop-meta-label">ID Kendaraan</span><span class="crop-meta-val" id="metaVehicleId">—</span></div>
        <div class="crop-meta-row"><span class="crop-meta-label">Kategori</span><span class="crop-meta-val" id="metaClass">—</span></div>
        <div class="crop-meta-row"><span class="crop-meta-label">Lokasi CCTV</span><span class="crop-meta-val" id="metaCamera">{safe_camera}</span></div>
        <div class="crop-meta-row"><span class="crop-meta-label">Waktu Melintas</span><span class="crop-meta-val" id="metaTime">—</span></div>
        <div class="crop-meta-row"><span class="crop-meta-label">Arah Melintas</span><span class="crop-meta-val" id="metaDir">—</span></div>
        <div class="crop-meta-row"><span class="crop-meta-label">Tingkat Keyakinan</span><span class="crop-meta-val" id="metaConf">—</span></div>
      </div>
    </div>
    <div class="crop-modal-footer">
      <div class="seg-toggle" id="modalMaskToggle" role="group" aria-label="Mode foto crop">
        <button type="button" id="modalBtnMask" class="seg-btn active">Dengan highlight</button>
        <button type="button" id="modalBtnRaw" class="seg-btn">Foto asli</button>
      </div>
      <div class="modal-actions">
        <button type="button" class="modal-btn primary" id="modalJumpPlay">Lompat &amp; Putar</button>
        <button type="button" class="modal-btn" id="modalDismiss">Tutup (Tetap Jeda)</button>
      </div>
    </div>
  </div>
</div>

<script>
const DATA = {payload_json};
const video = document.getElementById("overlayVideo");
const videoClock = document.getElementById("videoClock");
const valTotal = document.getElementById("valTotal");
const valMobil = document.getElementById("valMobil");
const valMotor = document.getElementById("valMotor");
const cardTotal = document.getElementById("cardTotal");
const cardMobil = document.getElementById("cardMobil");
const cardMotor = document.getElementById("cardMotor");
const directionSummary = document.getElementById("directionSummary");
const nextEventHint = document.getElementById("nextEventHint");
const liveBanner = document.getElementById("liveBanner");
const feedBody = document.getElementById("feedBody");
const speedSelect = document.getElementById("speedSelect");
const btnMaskOn = document.getElementById("btnMaskOn");
const btnMaskOff = document.getElementById("btnMaskOff");
const tabPassed = document.getElementById("tabPassed");
const tabAll = document.getElementById("tabAll");

const cropModalBackdrop = document.getElementById("cropModalBackdrop");
const cropModalTitle = document.getElementById("cropModalTitle");
const cropModalImg = document.getElementById("cropModalImg");
const cropImgEmpty = document.getElementById("cropImgEmpty");
const metaVehicleId = document.getElementById("metaVehicleId");
const metaClass = document.getElementById("metaClass");
const metaTime = document.getElementById("metaTime");
const metaDir = document.getElementById("metaDir");
const metaConf = document.getElementById("metaConf");
const modalBtnMask = document.getElementById("modalBtnMask");
const modalBtnRaw = document.getElementById("modalBtnRaw");
const modalMaskToggle = document.getElementById("modalMaskToggle");
const cropModalClose = document.getElementById("cropModalClose");
const modalDismiss = document.getElementById("modalDismiss");
const modalJumpPlay = document.getElementById("modalJumpPlay");

const CAM_SVG = '<svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/><circle cx="12" cy="13" r="4"/></svg>';

const events = DATA.events || [];
let feedMode = "passed";
let lastCountedIndex = -1;
let lastRenderedSignature = "";
let pulseTimeout = null;
let activeModalEvent = null;
let modalShowMask = true;

function formatClock(sec) {{
  const total = Math.max(0, Math.floor(sec));
  const m = Math.floor(total / 60);
  const s = total % 60;
  return String(m).padStart(2, "0") + ":" + String(s).padStart(2, "0");
}}

function renderModalImage() {{
  if (!activeModalEvent) return;
  const hasMask = Boolean(activeModalEvent.cropMask);
  const hasRaw = Boolean(activeModalEvent.cropRaw);
  modalMaskToggle.style.display = (hasMask && hasRaw) ? "inline-flex" : "none";
  const src = modalShowMask
    ? (activeModalEvent.cropMask || activeModalEvent.cropRaw)
    : (activeModalEvent.cropRaw || activeModalEvent.cropMask);
  modalBtnMask.classList.toggle("active", modalShowMask);
  modalBtnRaw.classList.toggle("active", !modalShowMask);
  if (src) {{
    cropModalImg.src = src;
    cropModalImg.style.display = "block";
    cropImgEmpty.style.display = "none";
  }} else {{
    cropModalImg.removeAttribute("src");
    cropModalImg.style.display = "none";
    cropImgEmpty.style.display = "block";
  }}
}}

function openCropModal(ev) {{
  if (!ev) return;
  // Jeda video otomatis saat pengguna membuka foto crop ID kendaraan
  if (!video.paused) {{
    video.pause();
  }}
  activeModalEvent = ev;
  modalShowMask = !(btnMaskOff && btnMaskOff.classList.contains("active"));
  cropModalTitle.textContent = "Foto Kendaraan — ID " + ev.vehicle_id;
  metaVehicleId.textContent = "ID " + ev.vehicle_id;
  metaClass.textContent = ev.class_label;
  metaTime.textContent = formatClock(ev.ts) + " (" + ev.ts.toFixed(1) + " dtk)";
  metaDir.textContent = ev.direction === "up" ? "Menuju atas" : "Menuju bawah";
  metaConf.textContent = ev.score.toFixed(1) + "%";
  renderModalImage();
  cropModalBackdrop.classList.add("open");
}}

function closeCropModal() {{
  cropModalBackdrop.classList.remove("open");
  activeModalEvent = null;
}}

modalBtnMask.addEventListener("click", () => {{
  modalShowMask = true;
  renderModalImage();
}});
modalBtnRaw.addEventListener("click", () => {{
  modalShowMask = false;
  renderModalImage();
}});
cropModalClose.addEventListener("click", closeCropModal);
modalDismiss.addEventListener("click", closeCropModal);
cropModalBackdrop.addEventListener("click", (e) => {{
  if (e.target === cropModalBackdrop) closeCropModal();
}});
document.addEventListener("keydown", (e) => {{
  if (e.key === "Escape" && cropModalBackdrop.classList.contains("open")) {{
    closeCropModal();
  }}
}});
modalJumpPlay.addEventListener("click", () => {{
  if (!activeModalEvent) return;
  const targetTime = Math.max(0, activeModalEvent.ts - 0.6);
  closeCropModal();
  video.currentTime = targetTime;
  video.play().catch(() => {{}});
  updateDisplay(true);
}});

function setVideoSource(url, preserveTime) {{
  if (!url) return;
  const savedTime = preserveTime ? video.currentTime : 0;
  const wasPlaying = preserveTime && !video.paused && !video.ended;
  const savedRate = parseFloat(speedSelect.value || "1");
  video.src = url;
  video.load();
  const restore = () => {{
    video.playbackRate = savedRate;
    if (savedTime > 0 && Number.isFinite(video.duration)) {{
      video.currentTime = Math.min(savedTime, Math.max(0, video.duration - 0.05));
    }}
    if (wasPlaying) {{
      video.play().catch(() => {{}});
    }}
    updateDisplay(true);
  }};
  video.addEventListener("loadedmetadata", restore, {{ once: true }});
}}

if (btnMaskOn && btnMaskOff) {{
  btnMaskOn.addEventListener("click", () => {{
    if (btnMaskOn.classList.contains("active")) return;
    btnMaskOn.classList.add("active");
    btnMaskOff.classList.remove("active");
    setVideoSource(DATA.urlWithMask, true);
  }});
  btnMaskOff.addEventListener("click", () => {{
    if (btnMaskOff.classList.contains("active")) return;
    btnMaskOff.classList.add("active");
    btnMaskOn.classList.remove("active");
    setVideoSource(DATA.urlWithoutMask, true);
  }});
}}

speedSelect.addEventListener("change", () => {{
  video.playbackRate = parseFloat(speedSelect.value || "1");
}});

tabPassed.addEventListener("click", () => {{
  feedMode = "passed";
  tabPassed.classList.add("active");
  tabAll.classList.remove("active");
  updateDisplay(true);
}});

tabAll.addEventListener("click", () => {{
  feedMode = "all";
  tabAll.classList.add("active");
  tabPassed.classList.remove("active");
  updateDisplay(true);
}});

function upperBoundByTime(currentTime) {{
  let lo = 0;
  let hi = events.length;
  while (lo < hi) {{
    const mid = (lo + hi) >> 1;
    if (events[mid].ts <= currentTime + 0.02) {{
      lo = mid + 1;
    }} else {{
      hi = mid;
    }}
  }}
  return lo;
}}

function triggerPulse(className) {{
  cardTotal.classList.add("pulse");
  if (className === "MobilTidakStandar") cardMobil.classList.add("pulse");
  if (className === "MotorTidakStandar") cardMotor.classList.add("pulse");
  if (pulseTimeout) clearTimeout(pulseTimeout);
  pulseTimeout = setTimeout(() => {{
    cardTotal.classList.remove("pulse");
    cardMobil.classList.remove("pulse");
    cardMotor.classList.remove("pulse");
  }}, 550);
}}

function renderFeed(passedCount, currentTime) {{
  const sourceList = feedMode === "passed"
    ? events.slice(0, passedCount).map((e, idx) => ({{ ...e, seq: idx + 1 }})).reverse()
    : events.map((e, idx) => ({{ ...e, seq: idx + 1 }}));

  if (sourceList.length === 0) {{
    feedBody.innerHTML = '<tr><td colspan="5" class="empty-row">Belum ada kendaraan melewati garis hingga detik ini.</td></tr>';
    return;
  }}

  const visibleRows = sourceList.slice(0, 120);
  const fragment = document.createDocumentFragment();
  for (const ev of visibleRows) {{
    const tr = document.createElement("tr");
    if (ev.ts <= currentTime + 0.02 && currentTime - ev.ts <= 2.5) {{
      tr.className = "recent";
    }}
    tr.innerHTML =
      "<td>" + ev.seq + "</td>" +
      "<td>" + ev.ts.toFixed(1) + " dtk</td>" +
      '<td><button type="button" class="id-crop-btn" title="Klik untuk menjeda video &amp; melihat foto crop">' +
        "ID " + ev.vehicle_id + " " + CAM_SVG +
      "</button></td>" +
      "<td>" + ev.class_label + "</td>" +
      "<td>" + ev.score.toFixed(1) + "%</td>";

    const idBtn = tr.querySelector(".id-crop-btn");
    idBtn.addEventListener("click", (e) => {{
      e.stopPropagation();
      openCropModal(ev);
    }});

    tr.addEventListener("click", () => {{
      const targetTime = Math.max(0, ev.ts - 0.6);
      video.currentTime = targetTime;
      updateDisplay(true);
    }});
    fragment.appendChild(tr);
  }}
  feedBody.innerHTML = "";
  feedBody.appendChild(fragment);
}}

function updateDisplay(forceFeed) {{
  const now = Number.isFinite(video.currentTime) ? video.currentTime : 0;
  const frame = Math.round(now * (DATA.fps || 25));
  videoClock.textContent = "Detik " + now.toFixed(1) + " / " + DATA.duration.toFixed(1) + " · Frame " + frame;

  const passedCount = upperBoundByTime(now);
  let mobilCount = 0;
  let motorCount = 0;
  let downCount = 0;
  let upCount = 0;
  for (let i = 0; i < passedCount; i++) {{
    const ev = events[i];
    if (ev.class_name === "MobilTidakStandar") mobilCount++;
    else if (ev.class_name === "MotorTidakStandar") motorCount++;
    if (ev.direction === "up") upCount++;
    else downCount++;
  }}

  valTotal.textContent = String(passedCount);
  valMobil.textContent = String(mobilCount);
  valMotor.textContent = String(motorCount);
  directionSummary.textContent = "Arah: turun " + downCount + " · naik " + upCount;

  if (passedCount < events.length) {{
    const nxt = events[passedCount];
    nextEventHint.textContent = "Berikutnya: " + nxt.ts.toFixed(1) + " dtk (ID " + nxt.vehicle_id + ")";
  }} else {{
    nextEventHint.textContent = "Seluruh data video selesai";
  }}

  if (passedCount > 0) {{
    const latest = events[passedCount - 1];
    const age = now - latest.ts;
    liveBanner.className = (age >= -0.05 && age <= 2.5) ? "live-banner active" : "live-banner";
    const prefix = (age >= -0.05 && age <= 2.5) ? "Baru terhitung" : "Terakhir melintas";
    liveBanner.innerHTML =
      "<span>" + prefix + ": ID " + latest.vehicle_id + " · " + latest.class_label + " (" + latest.ts.toFixed(1) + " dtk)</span>" +
      '<button type="button" class="banner-crop-btn" id="bannerViewCrop">Lihat foto</button>';
    const bannerBtn = document.getElementById("bannerViewCrop");
    if (bannerBtn) {{
      bannerBtn.addEventListener("click", () => openCropModal(latest));
    }}
  }} else {{
    liveBanner.className = "live-banner";
    liveBanner.innerHTML = "<span>Menunggu kendaraan melewati garis hitung...</span>";
  }}

  if (passedCount > lastCountedIndex && lastCountedIndex >= 0 && passedCount - lastCountedIndex <= 3) {{
    triggerPulse(events[passedCount - 1].class_name);
  }}
  lastCountedIndex = passedCount;

  const sig = feedMode + ":" + passedCount;
  if (forceFeed || sig !== lastRenderedSignature) {{
    lastRenderedSignature = sig;
    renderFeed(passedCount, now);
  }}
}}

function tick() {{
  updateDisplay(false);
  if (!video.paused && !video.ended) {{
    requestAnimationFrame(tick);
  }}
}}

video.addEventListener("play", () => requestAnimationFrame(tick));
video.addEventListener("timeupdate", () => updateDisplay(false));
video.addEventListener("seeked", () => updateDisplay(true));

setVideoSource(DATA.urlWithMask || DATA.urlWithoutMask, false);
updateDisplay(true);
</script>
</body>
</html>
"""


def render_playback(
    snapshot: CountingSnapshot,
    all_cameras_frame: pd.DataFrame,
    *,
    only_test_locations: bool,
) -> None:
    st.subheader("Pemutar rekaman CCTV & penghitung otomatis")
    if not snapshot.overlay_manifest or snapshot.timeline_frame is None:
        st.info("Rekaman video pemantauan belum tersedia.")
        return
    manifest_frame = pd.DataFrame(snapshot.overlay_manifest)
    allowed_cameras = set(
        all_cameras_frame.loc[all_cameras_frame["is_test_location"], "camera_id"].tolist()
        if only_test_locations
        else all_cameras_frame["camera_id"].tolist()
    )
    manifest_frame = manifest_frame[manifest_frame["camera_id"].isin(allowed_cameras)].copy()
    if manifest_frame.empty:
        st.info("Tidak ada rekaman video untuk cakupan lokasi yang dipilih.")
        return

    test_camera_ids = set(
        all_cameras_frame.loc[all_cameras_frame["is_test_location"], "camera_id"].tolist()
    )
    available_cameras = sorted(manifest_frame["camera_id"].unique().tolist())
    # Urutkan lokasi uji independen di atas agar mudah diinspeksi
    ordered_cameras = [c for c in available_cameras if c in test_camera_ids] + [
        c for c in available_cameras if c not in test_camera_ids
    ]

    def _format_camera_option(cam_id: str) -> str:
        name = camera_display_name(cam_id)
        badge = "Lokasi Uji Model" if cam_id in test_camera_ids else "Lokasi Pantau Reguler"
        return f"{name} — {badge}"

    selector_cols = st.columns([1.2, 1.3], gap="medium")
    selected_camera = selector_cols[0].selectbox(
        "Pilih lokasi CCTV",
        ordered_cameras,
        format_func=_format_camera_option,
        key=f"playback_camera_{'test' if only_test_locations else 'all'}",
    )
    filtered_manifest = manifest_frame[manifest_frame["camera_id"] == selected_camera]
    if filtered_manifest.empty:
        st.info("Tidak ada video rekaman untuk lokasi ini.")
        return
    selected_video = selector_cols[1].selectbox(
        "Rekaman video",
        filtered_manifest["video_stem"].tolist(),
        key=f"playback_video_{'test' if only_test_locations else 'all'}",
    )
    video_row = filtered_manifest[filtered_manifest["video_stem"] == selected_video].iloc[0]
    video_timeline = snapshot.timeline_frame[
        snapshot.timeline_frame["video_stem"] == selected_video
    ].copy()
    if video_timeline.empty:
        st.warning("Belum ada kendaraan yang tercatat melintas pada video ini.")

    video_base_url = resolve_video_base_url()
    if video_base_url:
        rel_with = video_row.get("video_relpath_with_mask") or video_row.get("overlay_path_with_mask")
        rel_without = video_row.get("video_relpath_without_mask") or video_row.get("overlay_path_without_mask")
        url_with_mask = _build_remote_video_url(video_base_url, rel_with)
        url_without_mask = _build_remote_video_url(video_base_url, rel_without)
        if not url_with_mask and not url_without_mask:
            st.warning("Tautan video streaming untuk lokasi ini belum tersedia.")
            return
    else:
        with_mask_path = _resolve_project_path(
            video_row.get("overlay_path_with_mask") or video_row.get("overlay_path")
        )
        without_mask_path = _resolve_project_path(video_row.get("overlay_path_without_mask"))
        with_mask_exists = with_mask_path is not None and with_mask_path.exists()
        without_mask_exists = without_mask_path is not None and without_mask_path.exists()

        if not with_mask_exists and not without_mask_exists:
            st.info(
                "Rekaman video belum terhubung. Tambahkan `VIDEO_BASE_URL` pada pengaturan secrets "
                "atau pastikan file video lokal tersedia."
            )
            return

        media_server = get_local_video_server()
        url_with_mask = media_server.register(with_mask_path) if with_mask_exists else None
        url_without_mask = media_server.register(without_mask_path) if without_mask_exists else None

    # Siapkan pasangan foto crop (dengan & tanpa highlight) khusus untuk video yang sedang dipilih
    crop_entries_list: list[tuple[int, str, str]] = []
    if snapshot.event_clip_manifest:
        for item in snapshot.event_clip_manifest:
            if str(item.get("camera_id", "")) == str(video_row["camera_id"]) and str(
                item.get("video_stem", "")
            ) == str(selected_video):
                vid = int(item.get("vehicle_id", 0))
                mask_p = str(item.get("crop_path_with_mask") or "").strip()
                raw_p = str(item.get("crop_path_without_mask") or item.get("crop_path") or "").strip()
                crop_entries_list.append((vid, mask_p, raw_p))
    crop_data_uris = _load_video_crop_data_uris(
        str(resolve_nb06_root().resolve()),
        tuple(crop_entries_list),
    )

    fps = float(video_row.get("fps", 25.0))
    duration_seconds = float(video_row.get("duration_seconds", 0.0))
    events = _serialize_timeline_events(video_timeline, fps, crop_data_uris)

    player_html = build_realtime_playback_html(
        video_url_with_mask=url_with_mask,
        video_url_without_mask=url_without_mask,
        camera_id=str(video_row["camera_id"]),
        camera_label=camera_display_name(str(video_row["camera_id"])),
        video_stem=str(selected_video),
        fps=fps,
        duration_seconds=duration_seconds,
        events=events,
    )
    if hasattr(st, "iframe"):
        st.iframe(player_html, height=640)
    else:
        components.html(player_html, height=640, scrolling=False)
    st.caption(
        "Detik video otomatis sinkron sama daftar kendaraan yang lewat garis kuning. "
        "Klik **ID kendaraan** di log kalau mau jeda video dan langsung lihat foto crop-nya."
    )


def render_model_validation(snapshot: CountingSnapshot) -> None:
    del snapshot
    st.subheader("Performa model dan batas interpretasi")
    nb06_root = resolve_nb06_root()
    nb04_root = nb06_root.parent / "nb04"
    nb05_root = nb06_root.parent / "nb05"
    ground_truth_root = nb06_root.parent / "ground_truth"
    results_paths = list((nb04_root / "runs").glob("*/results.csv")) if (nb04_root / "runs").exists() else []
    metrics_path = nb04_root / "val_metrics.json"
    if not results_paths and not metrics_path.exists():
        st.info("Artefak metrik detektor belum tersedia.")
        return

    st.markdown("#### Detektor objek (YOLO11n) — split validasi")
    if results_paths:
        results_frame = pd.read_csv(sorted(results_paths)[-1])
        results_frame.columns = [column.strip() for column in results_frame.columns]
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
            [{"Metrik validation": key, "Nilai": round(float(value), 4)} for key, value in overall.items()]
        )
        per_class = metrics_payload.get("per_class", [])
        val_cols = st.columns([1, 1.8], gap="medium")
        with val_cols[0]:
            if not metric_frame.empty:
                st.dataframe(metric_frame, hide_index=True, width="stretch")
        with val_cols[1]:
            if per_class:
                per_class_frame = pd.DataFrame(per_class)
                st.dataframe(per_class_frame, hide_index=True, width="stretch")

    gt_summary_path = ground_truth_root / "annotation_summary.json"
    detector_test_path = nb05_root / "test_metrics.json"
    gt_summary = None
    if gt_summary_path.exists():
        try:
            gt_summary = json.loads(gt_summary_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            st.warning(f"Ringkasan GT holdout tidak bisa dibaca: {error}")
    detector_test = None
    if detector_test_path.exists():
        try:
            detector_test = json.loads(detector_test_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            st.warning(f"Metrik test detektor tidak bisa dibaca: {error}")

    st.markdown("#### Evaluasi holdout test (2 kamera uji terpisah)")
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
            f"GT holdout direview: {reviewed_frames:,} frame · {vehicle_boxes:,} objek berlabel "
            f"(termasuk {reflection_boxes:,} PantulanCahaya). Pada inferensi video, pantulan disaring "
            "sebelum ByteTrack, overlay, dan counting kendaraan."
        )
        exported_gt_classes = set(gt_summary.get("gold_boxes_by_class", {}).keys())
        uses_legacy_detector_export = bool(exported_gt_classes) and exported_gt_classes != set(CLASS_NAMES)
        if uses_legacy_detector_export:
            st.info(
                "Pada arsitektur inferensi akhir, YOLO11n berperan sebagai lokalisator objek kendaraan, "
                "sedangkan penetapan akhir 5 kelas kondisi lampu dikerjakan oleh klasifier atribut "
                "ResNet-18 + SAM2 + e-means. Evaluasi holdout 5-kelas penuh atas seluruh 2.982 objek "
                "direpresentasikan secara langsung oleh metrik klasifier atribut di bawah."
            )
        elif detector_test and detector_test.get("status") == "computed":
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
    else:
        st.info("Ringkasan GT holdout belum ditemukan. Periksa `output/ground_truth/annotation_summary.json`.")

    attribute_root = nb06_root.parent / "attribute_classifier"
    attribute_history_path = attribute_root / "training_history.csv"
    attribute_metrics_path = attribute_root / "metrics.json"
    if attribute_history_path.exists():
        st.markdown("#### Klasifier kondisi lampu (ResNet-18 + SAM2 + e-means)")
        attribute_history = pd.read_csv(attribute_history_path)
        history_columns = [
            column for column in (
                "epoch",
                "train_loss", "val_loss",
                "train_f1_macro", "val_f1_macro",
                "train_f1", "val_f1",
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
                    st.info("GT holdout tersedia, tetapi metrik test klasifier belum tercatat.")
                else:
                    st.info("Metrik test klasifier belum tersedia karena GT holdout belum ditemukan.")
            else:
                st.markdown("**Klasifier kondisi lampu — evaluasi test holdout (5 kelas)**")
                overall_rows = [
                    ("Accuracy", f"{float(test_metrics.get('accuracy', 0.0)) * 100:.2f}%"),
                    ("Macro Precision", f"{float(test_metrics.get('precision_macro', 0.0)) * 100:.2f}%"),
                    ("Macro Recall", f"{float(test_metrics.get('recall_macro', 0.0)) * 100:.2f}%"),
                    ("Macro F1-Score", f"{float(test_metrics.get('f1_macro', 0.0)) * 100:.2f}%"),
                    ("Brier MSE", f"{float(test_metrics.get('brier_mse', 0.0)):.4f}"),
                    ("Brier RMSE", f"{float(test_metrics.get('brier_rmse', 0.0)):.4f}"),
                    ("Total objek uji", format_count(test_metrics.get("support"))),
                ]
                per_class_rows = []
                for class_name in CLASS_NAMES:
                    precision = float(test_metrics.get(f"precision_{class_name}", 0.0))
                    recall = float(test_metrics.get(f"recall_{class_name}", 0.0))
                    f1_val = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
                    per_class_rows.append({
                        "Kelas": CLASS_LABELS.get(class_name, class_name),
                        "Precision": f"{precision * 100:.1f}%",
                        "Recall": f"{recall * 100:.1f}%",
                        "F1-Score": f"{f1_val * 100:.1f}%",
                        "Support": format_count(test_metrics.get(f"support_{class_name}", 0)),
                    })
                cnn_cols = st.columns([1, 1.6], gap="medium")
                with cnn_cols[0]:
                    st.dataframe(
                        pd.DataFrame(overall_rows, columns=["Metrik holdout", "Nilai"]),
                        hide_index=True,
                        width="stretch",
                    )
                with cnn_cols[1]:
                    st.dataframe(pd.DataFrame(per_class_rows), hide_index=True, width="stretch")
    if gt_summary:
        st.caption(
            "Metrik di atas membandingkan prediksi model dengan label holdout yang direview. "
            "Counting video tetap merupakan keluaran inferensi pipeline; metrik ini tidak "
            "mengubah atau menggantikan angka counting."
        )


def render_event_footage(
    snapshot: CountingSnapshot,
    all_cameras_frame: pd.DataFrame,
    *,
    only_test_locations: bool,
) -> None:
    st.subheader("Galeri foto kendaraan")
    st.caption(
        "Kumpulan foto crop kendaraan berlampu tidak standar. Mau cek kendaraan tertentu? "
        "Ketik langsung ID-nya di kotak pencarian di bawah."
    )
    if not snapshot.event_clip_manifest:
        st.info("Galeri foto kendaraan belum tersedia.")
        return
    clips = pd.DataFrame(snapshot.event_clip_manifest)
    nonstandard_classes = {"MobilTidakStandar", "MotorTidakStandar"}
    clips = clips[clips["class_name"].isin(nonstandard_classes)].copy()
    allowed_cameras = set(
        all_cameras_frame.loc[all_cameras_frame["is_test_location"], "camera_id"].tolist()
        if only_test_locations
        else all_cameras_frame["camera_id"].tolist()
    )
    clips = clips[clips["camera_id"].isin(allowed_cameras)].copy()
    if clips.empty:
        st.info("Belum ada foto kendaraan di cakupan lokasi ini.")
        return

    filter_cols = st.columns([1.2, 1, 1], gap="medium")
    search_query = filter_cols[0].text_input(
        "Cari ID kendaraan",
        placeholder="Ketik ID, misal 9521 atau 10...",
        key="clip_id_search",
    )
    class_filter = filter_cols[1].selectbox(
        "Kategori kendaraan",
        ["Semua tidak standar", "Mobil tidak standar", "Motor tidak standar"],
        key="clip_class",
    )
    camera_options = ["Semua lokasi", *sorted(clips["camera_id"].unique().tolist())]
    camera_filter = filter_cols[2].selectbox(
        "Lokasi CCTV",
        camera_options,
        format_func=lambda cid: cid if cid == "Semua lokasi" else camera_display_name(cid),
        key=f"clip_camera_{'test' if only_test_locations else 'all'}",
    )

    filtered = clips.copy()
    if class_filter == "Mobil tidak standar":
        filtered = filtered[filtered["class_name"].str.startswith("Mobil")]
    elif class_filter == "Motor tidak standar":
        filtered = filtered[filtered["class_name"].str.startswith("Motor")]
    if camera_filter != "Semua lokasi":
        filtered = filtered[filtered["camera_id"] == camera_filter]

    cleaned_query = search_query.strip().lower().replace("id", "").replace("#", "").strip()
    is_searching = bool(cleaned_query)
    if is_searching:
        vid_series = filtered["vehicle_id"].astype(str).str.strip()
        exact_mask = vid_series == cleaned_query
        partial_mask = vid_series.str.contains(cleaned_query, regex=False)
        filtered = filtered[partial_mask].copy()
        if not filtered.empty:
            filtered["_exact"] = filtered["vehicle_id"].astype(str).str.strip() == cleaned_query
            filtered = filtered.sort_values(["_exact", "vehicle_id"], ascending=[False, True]).drop(columns=["_exact"])

    if filtered.empty:
        if is_searching:
            st.info(f"ID `{search_query.strip()}` nggak ketemu di filter saat ini. Coba cek lokasi atau kategori lain.")
        else:
            st.info("Nggak ada foto kendaraan yang cocok dengan filter tersebut.")
        return
    filtered = filtered.reset_index(drop=True)
    if is_searching:
        st.caption(f"Ketemu **{format_exact_count(len(filtered))}** foto kendaraan yang cocok dengan pencarian `{search_query.strip()}`.")
    else:
        st.caption(f"Total **{format_exact_count(len(filtered))}** foto kendaraan sesuai filter.")
    render_event_gallery(filtered, is_searching=is_searching)


def render_event_gallery(filtered: pd.DataFrame, *, is_searching: bool = False) -> None:
    mask_columns_available = False
    if "crop_path_with_mask" in filtered.columns and "crop_path_without_mask" in filtered.columns:
        crop_values = filtered[["crop_path_with_mask", "crop_path_without_mask"]].fillna("").astype(str)
        mask_columns_available = bool(crop_values.apply(lambda column: column.str.strip().ne("").any()).all())
    if mask_columns_available:
        mask_view = st.radio(
            "Tampilan foto",
            ["Dengan highlight", "Foto asli"],
            horizontal=True,
            key="gallery_mask_view",
        )
        crop_column = "crop_path_with_mask" if mask_view == "Dengan highlight" else "crop_path_without_mask"
    else:
        crop_column = "crop_path"

    def _resolve_crop_path(item: pd.Series) -> Path | None:
        item_value = item.get(crop_column, "")
        if pd.isna(item_value) or not str(item_value).strip():
            item_value = item.get("crop_path", "") if crop_column == "crop_path_without_mask" else ""
        return _resolve_project_path(item_value)

    # Tampilan adaptif: kalau hasil pencarian ID sedikit (1-4 kendaraan), tampilkan kartu lebih besar + detail
    if is_searching and len(filtered) <= 4:
        card_cols = st.columns(min(len(filtered), 4), gap="medium")
        for col, (_, item) in zip(card_cols, filtered.iterrows()):
            item_path = _resolve_crop_path(item)
            class_name = str(item["class_name"])
            ts_sec = float(item.get("timestamp_seconds", 0.0))
            mins, secs = divmod(int(max(0, ts_sec)), 60)
            with col:
                with st.container(border=True):
                    if item_path is not None and item_path.exists():
                        st.image(str(item_path), width="stretch")
                    else:
                        st.warning("Foto belum ditemukan.")
                    st.markdown(f"**ID {item['vehicle_id']}** · {CLASS_LABELS.get(class_name, class_name)}")
                    st.caption(
                        f"Lokasi: **{camera_display_name(str(item['camera_id']))}**  \n"
                        f"Waktu: **{mins:02d}:{secs:02d}** ({ts_sec:.1f} dtk) · "
                        f"Keyakinan: **{float(item.get('mean_score', 0.0)) * 100:.1f}%**"
                    )
        return

    page_count = max(1, (len(filtered) - 1) // 64 + 1)
    if "gallery_page" in st.session_state and st.session_state.gallery_page > page_count:
        st.session_state.gallery_page = 1
    page = st.number_input(
        "Halaman grid",
        min_value=1,
        max_value=page_count,
        value=1,
        step=1,
        key="gallery_page",
    )
    page_frame = filtered.iloc[(page - 1) * 64: page * 64]
    st.caption(f"Menampilkan {len(page_frame)} foto pada halaman {page}/{page_count}.")
    for start in range(0, len(page_frame), 8):
        columns = st.columns(8, gap="small")
        for column, (_, item) in zip(columns, page_frame.iloc[start:start + 8].iterrows()):
            item_path = _resolve_crop_path(item)
            with column:
                if item_path is not None and item_path.exists():
                    st.image(
                        str(item_path),
                        caption=f"ID {item['vehicle_id']}\n{CLASS_LABELS.get(item['class_name'], item['class_name'])}",
                        width="stretch",
                    )


def render_camera_section(all_cameras_frame: pd.DataFrame, *, only_test_locations: bool) -> None:
    st.subheader("Ringkasan per titik CCTV")
    active_frame = (
        all_cameras_frame[all_cameras_frame["is_test_location"]].copy()
        if only_test_locations
        else all_cameras_frame.copy()
    )
    rows = []
    for _, row in active_frame.iterrows():
        rows.append(
            {
                "Lokasi CCTV": row["display_name"],
                "Peran Lokasi": row["role_label"],
                "Video": format_exact_count(row["videos_counted"]),
                "Total Kendaraan": format_exact_count(row["crossings_counted"]),
                "Mobil Tidak Standar": format_exact_count(row["mobil_crossings"]),
                "Motor Tidak Standar": format_exact_count(row["motor_crossings"]),
            }
        )
    scope_label = (
        f"Total ({len(active_frame)} Lokasi Uji Model)"
        if only_test_locations
        else f"Total Gabungan ({len(active_frame)} Lokasi)"
    )
    rows.append(
        {
            "Lokasi CCTV": scope_label,
            "Peran Lokasi": "—",
            "Video": format_exact_count(int(active_frame["videos_counted"].sum())),
            "Total Kendaraan": format_exact_count(int(active_frame["crossings_counted"].sum())),
            "Mobil Tidak Standar": format_exact_count(int(active_frame["mobil_crossings"].sum())),
            "Motor Tidak Standar": format_exact_count(int(active_frame["motor_crossings"].sum())),
        }
    )
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption(
        "Semua lokasi sudah masuk ke rekap di atas. Kalau mau tahu titik mana saja yang dipakai buat "
        "menguji model beserta grafik performanya, cek tab **Performa Model pada Data Uji** di bagian atas."
    )


def render_test_locations_overview(all_cameras_frame: pd.DataFrame) -> None:
    st.subheader("Lokasi yang Dipakai untuk Menguji Performa Model")
    test_frame = all_cameras_frame[all_cameras_frame["is_test_location"]]
    other_frame = all_cameras_frame[~all_cameras_frame["is_test_location"]]

    test_names = " dan ".join(f"**{name}**" for name in test_frame["display_name"].tolist())
    other_names = ", ".join(f"**{name}**" for name in other_frame["display_name"].tolist())

    st.info(
        f"Dari 5 titik CCTV yang dipantau, ada 2 titik yang khusus dipakai untuk menguji performa model — "
        f"yaitu {test_names}. Kedua lokasi ini memang **tidak dipakai saat training**, jadi hasilnya benar-benar "
        f"menggambarkan performa model di kondisi baru. Sementara 3 titik lainnya ({other_names}) "
        f"ikut dipakai buat melatih dan menyetel model."
    )

    compare_cols = st.columns(3)
    compare_cols[0].metric(
        "Total Semua Lokasi (5 Titik)",
        format_exact_count(int(all_cameras_frame["crossings_counted"].sum())),
        help=(
            f"Mobil: {format_exact_count(int(all_cameras_frame['mobil_crossings'].sum()))} · "
            f"Motor: {format_exact_count(int(all_cameras_frame['motor_crossings'].sum()))}"
        ),
        border=True,
    )
    compare_cols[1].metric(
        "Lokasi Uji Model (2 Titik)",
        format_exact_count(int(test_frame["crossings_counted"].sum())),
        help=(
            f"Mobil: {format_exact_count(int(test_frame['mobil_crossings'].sum()))} · "
            f"Motor: {format_exact_count(int(test_frame['motor_crossings'].sum()))}"
        ),
        border=True,
    )
    compare_cols[2].metric(
        "Lokasi Training & Penyetelan (3 Titik)",
        format_exact_count(int(other_frame["crossings_counted"].sum())),
        help=(
            f"Mobil: {format_exact_count(int(other_frame['mobil_crossings'].sum()))} · "
            f"Motor: {format_exact_count(int(other_frame['motor_crossings'].sum()))}"
        ),
        border=True,
    )

    detail_rows = []
    # Tampilkan Lokasi Uji Model terlebih dahulu agar langsung menonjol
    ordered = pd.concat([test_frame, other_frame], ignore_index=True)
    for _, row in ordered.iterrows():
        detail_rows.append(
            {
                "Lokasi CCTV": row["display_name"],
                "Kode Kamera": row["camera_id"],
                "Peran": ROLE_INDEPENDENT_TEST if row["is_test_location"] else "Lokasi Training / Penyetelan",
                "Total Kendaraan": format_exact_count(row["crossings_counted"]),
                "Mobil": format_exact_count(row["mobil_crossings"]),
                "Motor": format_exact_count(row["motor_crossings"]),
                "Keterangan": row["role_detail"],
            }
        )
    st.dataframe(pd.DataFrame(detail_rows), hide_index=True, width="stretch")


def class_display_frame(camera_table: pd.DataFrame) -> pd.DataFrame:
    class_columns = {
        CLASS_LABELS[class_name]: formatted_column(camera_table, class_name)
        for class_name in CLASS_NAMES
    }
    return pd.DataFrame(
        {"Lokasi CCTV": [camera_display_name(str(c)) for c in camera_table["camera_id"]], **class_columns}
    )


def formatted_column(frame: pd.DataFrame, column_name: str) -> list[str]:
    if column_name not in frame.columns:
        return [NOT_AVAILABLE_LABEL for _ in range(len(frame))]
    return [format_count(value) for value in frame[column_name]]


def render_contract_section(summary: dict, camera_table: pd.DataFrame | None = None) -> None:
    counting_totals = summary["counting_totals"]
    st.subheader("Rekap khusus 2 Lokasi Uji Model")
    st.caption("Tiap kendaraan dihitung sekali waktu melewati garis hitung kuning; arah melintasnya ikut dicatat.")
    st.dataframe(
        pd.DataFrame({
            "Kategori": ["Mobil tidak standar", "Motor tidak standar"],
            "Jumlah (2 Lokasi Uji)": [
                format_exact_count(counting_totals.get("jumlah_kendaraan_mobil_tidak_standar_melintasi_garis", 0)),
                format_exact_count(counting_totals.get("jumlah_kendaraan_motor_tidak_standar_melintasi_garis", 0)),
            ],
        }),
        hide_index=True,
        width="stretch",
    )
    if camera_table is not None and not camera_table.empty:
        with st.expander("Rincian deteksi per kategori di 2 lokasi uji"):
            st.dataframe(class_display_frame(camera_table), hide_index=True, width="stretch")
            st.caption("Angka di tabel ini adalah total kemunculan per frame video sebagai bahan cek ulang.")


def render_quality_section(summary: dict) -> None:
    with st.expander("Rincian teknis proses penyaringan & penghitungan"):
        quality_rows = [
            ("Frame diproses", format_count(summary.get("frames_processed"))),
            ("Deteksi mentah", format_count(summary.get("raw_detections"))),
            ("Deteksi tanpa jejak", format_count(summary.get("untracked_detections"))),
            ("Deteksi lolos ambang", format_count(summary.get("thresholded_detections"))),
            ("Jejak awal pelacakan", format_count(summary.get("tracks_total"))),
            ("Fragmen disambung", format_count(summary.get("fragments_stitched"))),
            ("Kendaraan setelah penyambungan", format_count(summary.get("vehicles_total"))),
            ("Kendaraan melintasi garis (2 lokasi uji)", format_exact_count(summary.get("line_crossings_counted"))),
            ("Video lokasi uji", format_exact_count(summary.get("videos_counted"))),
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
                {"Kategori": CLASS_LABELS.get(class_name, class_name), "Ambang": threshold}
                for class_name, threshold in selected_thresholds.items()
            ]
        ),
        hide_index=True,
        width="stretch",
    )


def render_footer() -> None:
    st.caption(
        "Dashboard Pemantauan Lampu Kendaraan Tidak Standar — Dinas Komunikasi dan Informatika Kabupaten Bogor."
    )


if __name__ == "__main__":
    main()
