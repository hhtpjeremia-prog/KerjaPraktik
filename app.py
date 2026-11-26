"""Dashboard Streamlit hasil hitung kendaraan per titik CCTV (siap deploy Streamlit Cloud & lokal)."""

import html
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

from data import (
    CLASS_NAMES,
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
from media_server import LocalVideoServer

PAGE_TITLE = "Hasil Hitung Kendaraan — CCTV Kab. Bogor"

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
  font-size: 0.8rem;
  letter-spacing: 0.02em;
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
    st.sidebar.caption(f"Sumber artefak: `{resolve_nb06_root().name}`")
    st.sidebar.caption(
        "Mode video: Cloud Streaming (HTTPS)"
        if video_base_url
        else "Mode video: Lokal (HTTP 206 Byte-Range)"
    )
    snapshot = load_cached_counting_snapshot()
    render_header(snapshot.summary)
    if not render_state_notice(snapshot):
        return
    render_smoke_notice(snapshot.summary)
    render_totals(snapshot)
    st.divider()
    render_playback(snapshot)
    st.divider()
    render_event_footage(snapshot)
    st.divider()
    render_model_validation(snapshot)
    st.divider()
    render_camera_section(snapshot)
    st.divider()
    render_contract_section(snapshot.summary)
    st.divider()
    render_quality_section(snapshot.summary)
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
            "Jalankan pipeline hingga tahap tracking dan counting selesai, lalu muat ulang halaman ini. "
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


def render_totals(snapshot: CountingSnapshot) -> None:
    summary = snapshot.summary
    counting_totals = summary["counting_totals"]
    official_video_count = (
        int(snapshot.per_video_frame["video_stem"].nunique())
        if snapshot.per_video_frame is not None and not snapshot.per_video_frame.empty
        else summary.get("videos_counted")
    )
    total_columns = st.columns(4)
    total_columns[0].metric(
        "Track tidak standar crossing", format_count(summary.get("line_crossings_counted")), border=True
    )
    total_columns[1].metric(
        "Mobil tidak standar crossing",
        format_count(counting_totals.get("jumlah_kendaraan_mobil_tidak_standar_melintasi_garis")),
        border=True,
    )
    total_columns[2].metric(
        "Motor tidak standar crossing",
        format_count(counting_totals.get("jumlah_kendaraan_motor_tidak_standar_melintasi_garis")),
        border=True,
    )
    total_columns[3].metric(
        "Video diproses", format_count(official_video_count), border=True
    )
    total_overlay_videos = len(snapshot.overlay_manifest) if snapshot.overlay_manifest else summary.get("videos_found")
    if (
        isinstance(official_video_count, int)
        and isinstance(total_overlay_videos, int)
        and total_overlay_videos > official_video_count
    ):
        st.caption(
            f"Total crossing resmi berasal dari {official_video_count} video split test; "
            f"{total_overlay_videos - official_video_count} video split train/val lainnya tetap tersedia "
            "pada pemutar playback di bawah untuk inspeksi presentasi."
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


def _serialize_timeline_events(video_timeline: pd.DataFrame, fps: float) -> list[dict]:
    if video_timeline.empty:
        return []
    frame_column = "crossing_frame" if "crossing_frame" in video_timeline.columns else "frame_index"
    ordered = video_timeline.sort_values([frame_column, "vehicle_id"])
    events = []
    for _, row in ordered.iterrows():
        frame_index = int(row[frame_column])
        if "timestamp_seconds" in row and pd.notna(row["timestamp_seconds"]):
            timestamp_seconds = float(row["timestamp_seconds"])
        else:
            timestamp_seconds = frame_index / max(fps, 1e-6)
        class_name = str(row["class_name"])
        events.append({
            "vehicle_id": int(row["vehicle_id"]) if "vehicle_id" in row and pd.notna(row["vehicle_id"]) else 0,
            "track_id": int(row["track_id"]) if "track_id" in row and pd.notna(row["track_id"]) else 0,
            "class_name": class_name,
            "class_label": CLASS_LABELS.get(class_name, class_name),
            "frame": frame_index,
            "ts": round(timestamp_seconds, 3),
            "direction": str(row.get("crossing_direction", "down")),
            "score": round(float(row.get("score", 0.0)) * 100.0, 1),
        })
    return events


def build_realtime_playback_html(
    *,
    video_url_with_mask: str | None,
    video_url_without_mask: str | None,
    camera_id: str,
    video_stem: str,
    fps: float,
    duration_seconds: float,
    events: list[dict],
) -> str:
    payload = {
        "urlWithMask": video_url_with_mask or "",
        "urlWithoutMask": video_url_without_mask or "",
        "cameraId": camera_id,
        "videoStem": video_stem,
        "fps": fps,
        "duration": duration_seconds,
        "events": events,
    }
    payload_json = json.dumps(payload, ensure_ascii=False).replace("</", "<\\/")
    safe_camera = html.escape(camera_id)
    has_pair = bool(video_url_with_mask and video_url_without_mask)
    toggle_markup = (
        '<div class="seg-toggle" role="group" aria-label="Tampilan segmentasi">'
        '<button type="button" id="btnMaskOn" class="seg-btn active">Mask tampil</button>'
        '<button type="button" id="btnMaskOff" class="seg-btn">Mask disembunyikan</button>'
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
</style>
</head>
<body>
<div class="layout">
  <div class="panel">
    <div class="toolbar">
      <div>
        <div class="title-block">Pemutar overlay sinkron — {safe_camera}</div>
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
      <span id="statusText">Siap diputar — angka counting akan bertambah otomatis saat kendaraan melewati garis kuning.</span>
      <span id="eventTotalLabel">Total video: {len(events)} crossing</span>
    </div>
    <div class="legend">
      <span class="legend-item"><span class="swatch" style="background:#FF0000;"></span>BBox merah = lampu tidak standar</span>
      <span class="legend-item"><span class="swatch" style="background:#D4A100;"></span>Garis kuning = garis hitung virtual</span>
      <span class="legend-item"><span class="swatch" style="background:transparent;border:1px solid #5A635E;"></span>Tanpa bbox = standar / tidak ditampilkan</span>
    </div>
  </div>

  <div class="panel">
    <div class="metrics-grid">
      <div class="metric-card" id="cardTotal">
        <div class="metric-label">Total crossing</div>
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
      <span id="nextEventHint">Event berikutnya: —</span>
    </div>

    <div class="live-banner" id="liveBanner">
      <span>Menunggu kendaraan melewati garis hitung...</span>
    </div>

    <div class="feed-header">
      <span class="feed-title">Log pelintasan garis</span>
      <div class="feed-tabs">
        <button type="button" id="tabPassed" class="feed-tab active">Sudah lewat</button>
        <button type="button" id="tabAll" class="feed-tab">Semua event</button>
      </div>
    </div>

    <div class="feed-wrap">
      <table>
        <thead>
          <tr>
            <th>#</th>
            <th>Detik</th>
            <th>ID</th>
            <th>Kelas</th>
            <th>Conf</th>
          </tr>
        </thead>
        <tbody id="feedBody"></tbody>
      </table>
    </div>
    <div class="jump-hint">Klik baris mana saja pada log untuk melompatkan video langsung ke momen kendaraan melewati garis.</div>
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

const events = DATA.events || [];
let feedMode = "passed";
let lastCountedIndex = -1;
let lastRenderedSignature = "";
let pulseTimeout = null;

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
      "<td>ID " + ev.vehicle_id + "</td>" +
      "<td>" + ev.class_label + "</td>" +
      "<td>" + ev.score.toFixed(1) + "%</td>";
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
    nextEventHint.textContent = "Seluruh event video selesai";
  }}

  if (passedCount > 0) {{
    const latest = events[passedCount - 1];
    const age = now - latest.ts;
    if (age >= -0.05 && age <= 2.5) {{
      liveBanner.className = "live-banner active";
      liveBanner.innerHTML =
        "<span>Baru terhitung: ID " + latest.vehicle_id + " · " + latest.class_label + "</span>" +
        "<span>" + latest.ts.toFixed(1) + " dtk · conf " + latest.score.toFixed(1) + "%</span>";
    }} else {{
      liveBanner.className = "live-banner";
      liveBanner.innerHTML =
        "<span>Terakhir melintas: ID " + latest.vehicle_id + " · " + latest.class_label + "</span>" +
        "<span>" + latest.ts.toFixed(1) + " dtk</span>";
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


def render_playback(snapshot: CountingSnapshot) -> None:
    st.subheader("Playback real-time dan counting timeline")
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
    selector_cols = st.columns([1, 1.4], gap="medium")
    selected_camera = selector_cols[0].selectbox(
        "Titik CCTV", camera_options, index=default_camera_index, key="playback_camera"
    )
    filtered_manifest = (
        manifest_frame
        if selected_camera == "Semua titik"
        else manifest_frame[manifest_frame["camera_id"] == selected_camera]
    )
    if filtered_manifest.empty:
        st.info("Tidak ada video overlay untuk titik ini.")
        return
    selected_video = selector_cols[1].selectbox(
        "Video overlay", filtered_manifest["video_stem"].tolist(), key="playback_video"
    )
    video_row = filtered_manifest[filtered_manifest["video_stem"] == selected_video].iloc[0]
    video_timeline = snapshot.timeline_frame[
        snapshot.timeline_frame["video_stem"] == selected_video
    ].copy()
    if video_timeline.empty:
        st.warning("Belum ada event tracking pada video ini.")
    elif video_row["split"] not in counted_splits:
        st.info(
            f"Video ini berasal dari split `{video_row['split']}`. Counting yang tampil adalah "
            "mode presentasi/inspeksi, bukan angka resmi evaluasi test."
        )

    video_base_url = resolve_video_base_url()
    if video_base_url:
        rel_with = video_row.get("video_relpath_with_mask") or video_row.get("overlay_path_with_mask")
        rel_without = video_row.get("video_relpath_without_mask") or video_row.get("overlay_path_without_mask")
        url_with_mask = _build_remote_video_url(video_base_url, rel_with)
        url_without_mask = _build_remote_video_url(video_base_url, rel_without)
        if not url_with_mask and not url_without_mask:
            st.warning("Manifest video tidak memuat path relatif untuk mode cloud streaming.")
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
                "Video overlay belum terhubung. Untuk deployment Streamlit Cloud, tambahkan "
                "`VIDEO_BASE_URL` di **App Settings → Secrets** yang mengarah ke HuggingFace Dataset "
                "berisi 10 file MP4 `overlays_browser/` (lihat `README.md`)."
            )
            return

        media_server = get_local_video_server()
        url_with_mask = media_server.register(with_mask_path) if with_mask_exists else None
        url_without_mask = media_server.register(without_mask_path) if without_mask_exists else None
    fps = float(video_row.get("fps", 25.0))
    duration_seconds = float(video_row.get("duration_seconds", 0.0))
    events = _serialize_timeline_events(video_timeline, fps)

    player_html = build_realtime_playback_html(
        video_url_with_mask=url_with_mask,
        video_url_without_mask=url_without_mask,
        camera_id=str(video_row["camera_id"]),
        video_stem=str(selected_video),
        fps=fps,
        duration_seconds=duration_seconds,
        events=events,
    )
    if hasattr(st, "iframe"):
        st.iframe(player_html, height=625)
    else:
        components.html(player_html, height=625, scrolling=False)
    st.caption(
        "Pemutar ini menyinkronkan posisi detik video secara real-time dengan artefak pelintasan garis "
        "(`counting_timeline.csv`). Toggle mask hanya mengganti lapisan tampilan tanpa mengubah hasil "
        "klasifikasi maupun jumlah kendaraan."
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


def render_event_footage(snapshot: CountingSnapshot) -> None:
    st.subheader("Galeri crop event")
    st.caption("Setiap item hanya menampilkan crop bbox kendaraan agar event mudah dibandingkan.")
    if not snapshot.event_clip_manifest:
        st.info(
            "Event belum tersedia. Jalankan tahap tracking/counting versi terbaru untuk membuat "
            "`event_clip_manifest.json` dan folder `event_crops/`."
        )
        return
    clips = pd.DataFrame(snapshot.event_clip_manifest)
    nonstandard_classes = {"MobilTidakStandar", "MotorTidakStandar"}
    clips = clips[clips["class_name"].isin(nonstandard_classes)].copy()
    filter_cols = st.columns(2, gap="medium")
    class_filter = filter_cols[0].selectbox(
        "Kelas event", ["Semua tidak standar", "Mobil tidak standar", "Motor tidak standar"], key="clip_class"
    )
    camera_filter = filter_cols[1].selectbox(
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
    mask_columns_available = False
    if "crop_path_with_mask" in filtered.columns and "crop_path_without_mask" in filtered.columns:
        crop_values = filtered[["crop_path_with_mask", "crop_path_without_mask"]].fillna("").astype(str)
        mask_columns_available = bool(crop_values.apply(lambda column: column.str.strip().ne("").any()).all())
    if mask_columns_available:
        mask_view = st.radio(
            "Tampilan segmentasi crop", ["Mask tampil", "Mask disembunyikan"],
            horizontal=True, key="gallery_mask_view",
        )
        crop_column = "crop_path_with_mask" if mask_view == "Mask tampil" else "crop_path_without_mask"
        st.caption("Pilihan ini hanya mengubah tampilan crop; hasil klasifikasi dan hitungan tetap sama.")
    else:
        crop_column = "crop_path"
        st.info("Pasangan crop dengan dan tanpa mask akan tersedia setelah artefak galeri dibangun.")
    with gallery_tab:
        if "gallery_index" not in st.session_state or st.session_state.gallery_index >= len(filtered):
            st.session_state.gallery_index = 0
        selected_index = st.session_state.gallery_index
        selected_clip = filtered.iloc[selected_index]
        crop_value = selected_clip.get(crop_column, "")
        if pd.isna(crop_value) or not str(crop_value).strip():
            crop_value = selected_clip.get("crop_path", "") if crop_column == "crop_path_without_mask" else ""
        crop_path = _resolve_project_path(crop_value)
        image_column, detail_column = st.columns([1.25, 1], gap="large")
        with image_column:
            if crop_path is not None and crop_path.exists():
                st.image(
                    str(crop_path),
                    caption=f"ID {selected_clip['vehicle_id']} · {CLASS_LABELS.get(selected_clip['class_name'], selected_clip['class_name'])}",
                    width="stretch",
                )
            else:
                st.warning(f"Crop belum ditemukan: `{crop_path}`.")
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
            if previous_column.button("← Sebelumnya", disabled=selected_index == 0, width="stretch"):
                st.session_state.gallery_index = max(0, selected_index - 1)
                st.rerun()
            if next_column.button("Berikutnya →", disabled=selected_index == len(filtered) - 1, width="stretch"):
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
                item_value = item.get(crop_column, "")
                if pd.isna(item_value) or not str(item_value).strip():
                    item_value = item.get("crop_path", "") if crop_column == "crop_path_without_mask" else ""
                item_path = _resolve_project_path(item_value)
                with column:
                    if item_path is not None and item_path.exists():
                        st.image(
                            str(item_path),
                            caption=f"ID {item['vehicle_id']}\n{CLASS_LABELS.get(item['class_name'], item['class_name'])}",
                            width="stretch",
                        )


def render_camera_section(snapshot: CountingSnapshot) -> None:
    st.subheader("Per titik CCTV")
    status_frame = build_camera_status_table(
        snapshot.camera_table, snapshot.summary.get("counted_splits", [])
    )
    st.dataframe(camera_display_frame(status_frame), hide_index=True, width="stretch")
    st.caption(
        "Titik CCTV di luar split hitung tetap ditampilkan agar cakupan dashboard tidak "
        "tampak lebih luas dari yang benar-benar dihitung pada evaluasi test."
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
        "Angka resmi hanya berasal dari tahap tracking dan counting video kontinu."
    )


if __name__ == "__main__":
    main()
