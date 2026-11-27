"""Muat artefak hitung NB06 untuk dashboard deployment (mandiri, tanpa dependensi PyTorch/OpenCV)."""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

APP_ROOT = Path(__file__).resolve().parent
DEFAULT_ARTIFACTS_ROOT = APP_ROOT / "artifacts"
DEFAULT_NB06_ROOT = DEFAULT_ARTIFACTS_ROOT / "nb06"
NB06_ROOT_ENV_VAR = "NB06_ROOT"

CLASS_NAMES = [
    "MobilStandar",
    "MobilTidakStandar",
    "MotorStandar",
    "MotorTidakStandar",
    "PantulanCahaya",
]

SPLIT_NAME_BY_CAMERA = {
    "bogor-kab-bappenda1": "train",
    "bogor-kab-bappenda2": "test",
    "bogor-kab-bappenda3": "train",
    "bogor-kab-pdam2": "test",
    "bogor-kab-pemda2": "val",
}

COUNT_CONTRACT = {
    "jumlah_kendaraan_mobil_tidak_standar_melintasi_garis": ["MobilTidakStandar"],
    "jumlah_kendaraan_motor_tidak_standar_melintasi_garis": ["MotorTidakStandar"],
}

COUNTING_SUMMARY_FILENAME = "counting_summary.json"
COUNTING_BY_VIDEO_FILENAME = "counting_by_video.csv"
COUNTING_TIMELINE_FILENAME = "counting_timeline.csv"
OVERLAY_MANIFEST_FILENAME = "overlay_manifest.json"
EVENT_CLIP_MANIFEST_FILENAME = "event_clip_manifest.json"
EXPECTED_ARTIFACT_NAMES = (COUNTING_SUMMARY_FILENAME, COUNTING_BY_VIDEO_FILENAME)

CAMERA_ID_COLUMN = "camera_id"
VIDEO_STEM_COLUMN = "video_stem"
SPLIT_COLUMN = "split"
STATUS_COLUMN = "status"
CROSSINGS_COUNTED_COLUMN = "crossings_counted"
VIDEOS_COUNTED_COLUMN = "videos_counted"

STATE_READY = "ready"
STATE_EMPTY = "empty"
STATE_PARTIAL = "partial"
STATE_ERROR = "error"

CAMERA_STATUS_COUNTED = "dihitung"
CAMERA_STATUS_NO_VIDEO = "tidak ada video"
CAMERA_STATUS_OUTSIDE = "di luar split hitung"

CAMERA_DISPLAY_NAMES = {
    "bogor-kab-bappenda1": "Bappenda 1",
    "bogor-kab-bappenda2": "Bappenda 2",
    "bogor-kab-bappenda3": "Bappenda 3",
    "bogor-kab-pdam2": "PDAM 2",
    "bogor-kab-pemda2": "Pemda 2",
}

ROLE_INDEPENDENT_TEST = "Lokasi Uji Model"
ROLE_REGULAR_MONITORING = "Lokasi Pantau Reguler"

NOT_AVAILABLE_LABEL = "n/a"
THOUSAND_SEPARATOR = "."
THOUSAND_SUFFIXES = ((1_000_000.0, "M"), (1_000.0, "k"))
SUFFIX_DECIMAL_PLACES = 1


@dataclass(frozen=True)
class CountingSnapshot:
    """Satu pembacaan artefak NB06 beserta status kelengkapannya."""

    state: str
    summary: dict | None = None
    per_video_frame: pd.DataFrame | None = None
    camera_table: pd.DataFrame | None = None
    timeline_frame: pd.DataFrame | None = None
    overlay_manifest: list[dict] | None = None
    event_clip_manifest: list[dict] | None = None
    missing_artifacts: tuple[str, ...] = ()
    error_message: str | None = None


def missing_artifact_names(nb06_root: Path = DEFAULT_NB06_ROOT) -> tuple[str, ...]:
    """Nama artefak wajib yang belum ada di direktori artefak NB06."""
    return tuple(
        artifact_name
        for artifact_name in EXPECTED_ARTIFACT_NAMES
        if not (nb06_root / artifact_name).exists()
    )


def resolve_nb06_root() -> Path:
    """Direktori artefak NB06 aktif (default: deploy_streamlit/artifacts/nb06)."""
    override_root = os.environ.get(NB06_ROOT_ENV_VAR)
    if override_root:
        return Path(override_root)
    return DEFAULT_NB06_ROOT


def build_camera_table(per_video_frame: pd.DataFrame) -> pd.DataFrame:
    """Jumlahkan hitungan per titik CCTV dari tabel per video."""
    video_counts = (
        per_video_frame.groupby(CAMERA_ID_COLUMN, sort=True)[VIDEO_STEM_COLUMN]
        .nunique()
        .rename(VIDEOS_COUNTED_COLUMN)
        .reset_index()
    )
    measured_columns = [
        column_name
        for column_name in (CROSSINGS_COUNTED_COLUMN, *CLASS_NAMES, *COUNT_CONTRACT.keys())
        if column_name in per_video_frame.columns
    ]
    if not measured_columns:
        return video_counts
    numeric_frame = per_video_frame[[CAMERA_ID_COLUMN, *measured_columns]].copy()
    for column_name in measured_columns:
        numeric_frame[column_name] = (
            pd.to_numeric(numeric_frame[column_name], errors="coerce").fillna(0).astype(int)
        )
    camera_table = (
        numeric_frame.groupby(CAMERA_ID_COLUMN, sort=True)[measured_columns].sum().reset_index()
    )
    return camera_table.merge(video_counts, on=CAMERA_ID_COLUMN, how="left")


def build_camera_status_table(
    camera_table: pd.DataFrame | None, counted_splits: list[str]
) -> pd.DataFrame:
    """Daftar semua titik CCTV yang dikenal pipeline, dengan status penghitungannya."""
    counted_camera_ids = set() if camera_table is None else set(camera_table[CAMERA_ID_COLUMN])
    status_rows = [
        {
            CAMERA_ID_COLUMN: camera_id,
            SPLIT_COLUMN: split_name,
            STATUS_COLUMN: camera_status(camera_id, split_name, counted_camera_ids, counted_splits),
        }
        for camera_id, split_name in sorted(SPLIT_NAME_BY_CAMERA.items())
    ]
    status_frame = pd.DataFrame(status_rows)
    if camera_table is None:
        return status_frame
    return status_frame.merge(camera_table, on=CAMERA_ID_COLUMN, how="left")


def camera_status(
    camera_id: str, split_name: str, counted_camera_ids: set[str], counted_splits: list[str]
) -> str:
    """Status satu titik CCTV terhadap split yang dihitung."""
    if camera_id in counted_camera_ids:
        return CAMERA_STATUS_COUNTED
    if split_name in counted_splits:
        return CAMERA_STATUS_NO_VIDEO
    return f"{CAMERA_STATUS_OUTSIDE} ({split_name})"


def format_count(value: int | float | str | None) -> str:
    """Format angka gaya produk: 1.4k, 38k, n/a."""
    if isinstance(value, str):
        return value
    if value is None:
        return NOT_AVAILABLE_LABEL
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return NOT_AVAILABLE_LABEL
    for threshold, suffix in THOUSAND_SUFFIXES:
        if abs(numeric_value) >= threshold:
            scaled_value = f"{numeric_value / threshold:.{SUFFIX_DECIMAL_PLACES}f}"
            return trim_trailing_zeros(scaled_value) + suffix
    return f"{int(numeric_value):,}".replace(",", THOUSAND_SEPARATOR)


def format_exact_count(value: int | float | str | None) -> str:
    """Format angka eksak dengan pemisah ribuan titik (mis. 1.672) tanpa disingkat k/M."""
    if isinstance(value, str):
        return value
    if value is None:
        return NOT_AVAILABLE_LABEL
    numeric_value = float(value)
    if math.isnan(numeric_value):
        return NOT_AVAILABLE_LABEL
    return f"{int(round(numeric_value)):,}".replace(",", THOUSAND_SEPARATOR)


def camera_display_name(camera_id: str) -> str:
    """Nama lokasi CCTV yang ramah pengguna umum."""
    return CAMERA_DISPLAY_NAMES.get(camera_id, camera_id)


def build_all_cameras_crossing_table(
    timeline_frame: pd.DataFrame | None,
    counted_splits: list[str],
    overlay_manifest: list[dict] | None = None,
) -> pd.DataFrame:
    """Agregasi lengkap seluruh titik CCTV dari counting_timeline.csv beserta peran non-teknikal."""
    counted_split_set = set(counted_splits or [])
    manifest_videos_by_cam: dict[str, set[str]] = {}
    if overlay_manifest:
        for row in overlay_manifest:
            cam = str(row.get(CAMERA_ID_COLUMN, "")).strip()
            stem = str(row.get(VIDEO_STEM_COLUMN, "")).strip()
            if cam and stem:
                manifest_videos_by_cam.setdefault(cam, set()).add(stem)

    rows = []
    for camera_id, split_name in sorted(SPLIT_NAME_BY_CAMERA.items()):
        is_test_loc = split_name in counted_split_set
        if timeline_frame is not None and not timeline_frame.empty:
            cam_tl = timeline_frame[timeline_frame[CAMERA_ID_COLUMN] == camera_id]
        else:
            cam_tl = pd.DataFrame()
        mobil = int((cam_tl["class_name"] == "MobilTidakStandar").sum()) if not cam_tl.empty else 0
        motor = int((cam_tl["class_name"] == "MotorTidakStandar").sum()) if not cam_tl.empty else 0
        total = mobil + motor
        if not cam_tl.empty and VIDEO_STEM_COLUMN in cam_tl.columns:
            videos_count = int(cam_tl[VIDEO_STEM_COLUMN].nunique())
        else:
            videos_count = len(manifest_videos_by_cam.get(camera_id, set()))

        if is_test_loc:
            role_detail = "Dipakai buat menguji performa model (tidak dipakai saat training)"
        elif split_name == "val":
            role_detail = "Dipakai buat menyetel sensitivitas & threshold model"
        else:
            role_detail = "Dipakai saat proses training pengenalan lampu kendaraan"

        rows.append(
            {
                CAMERA_ID_COLUMN: camera_id,
                "display_name": camera_display_name(camera_id),
                SPLIT_COLUMN: split_name,
                "is_test_location": is_test_loc,
                "role_label": ROLE_INDEPENDENT_TEST if is_test_loc else ROLE_REGULAR_MONITORING,
                "role_detail": role_detail,
                VIDEOS_COUNTED_COLUMN: videos_count,
                CROSSINGS_COUNTED_COLUMN: total,
                "mobil_crossings": mobil,
                "motor_crossings": motor,
            }
        )
    return pd.DataFrame(rows)


def trim_trailing_zeros(formatted_value: str) -> str:
    """Buang nol dan titik desimal yang tidak menambah informasi."""
    trimmed_value = formatted_value.rstrip("0").rstrip(THOUSAND_SEPARATOR)
    return trimmed_value or "0"


def load_counting_snapshot(nb06_root: Path | None = None) -> CountingSnapshot:
    """Baca artefak NB06 menjadi snapshot, atau kembalikan state yang menjelaskan kegagalannya."""
    nb06_root = resolve_nb06_root() if nb06_root is None else nb06_root
    missing_artifacts = missing_artifact_names(nb06_root)
    if len(missing_artifacts) == len(EXPECTED_ARTIFACT_NAMES):
        return CountingSnapshot(state=STATE_EMPTY, missing_artifacts=missing_artifacts)
    if missing_artifacts:
        return CountingSnapshot(state=STATE_PARTIAL, missing_artifacts=missing_artifacts)
    try:
        summary_payload = json.loads(
            (nb06_root / COUNTING_SUMMARY_FILENAME).read_text(encoding="utf-8")
        )
        try:
            per_video_frame = pd.read_csv(nb06_root / COUNTING_BY_VIDEO_FILENAME)
        except pd.errors.EmptyDataError:
            per_video_frame = pd.DataFrame(
                columns=[
                    VIDEO_STEM_COLUMN,
                    CAMERA_ID_COLUMN,
                    CROSSINGS_COUNTED_COLUMN,
                    *CLASS_NAMES,
                    *COUNT_CONTRACT.keys(),
                ]
            )
    except (OSError, ValueError) as artifact_error:
        return CountingSnapshot(
            state=STATE_ERROR,
            missing_artifacts=missing_artifacts,
            error_message=str(artifact_error),
        )
    if set(summary_payload.get("class_counts", {})) != set(CLASS_NAMES):
        return CountingSnapshot(
            state=STATE_ERROR,
            error_message="Artefak memakai taksonomi lama/tidak lengkap.",
        )
    if set(summary_payload.get("counting_totals", {})) != set(COUNT_CONTRACT):
        return CountingSnapshot(
            state=STATE_ERROR,
            error_message="Kontrak hitung tidak cocok dengan versi pipeline.",
        )
    timeline_frame = None
    timeline_path = nb06_root / COUNTING_TIMELINE_FILENAME
    if timeline_path.exists():
        timeline_frame = pd.read_csv(timeline_path)
    overlay_manifest = None
    manifest_path = nb06_root / OVERLAY_MANIFEST_FILENAME
    if manifest_path.exists():
        overlay_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    event_clip_manifest = None
    event_clip_manifest_path = nb06_root / EVENT_CLIP_MANIFEST_FILENAME
    if event_clip_manifest_path.exists():
        event_clip_manifest = json.loads(event_clip_manifest_path.read_text(encoding="utf-8"))
    return CountingSnapshot(
        state=STATE_READY,
        summary=summary_payload,
        per_video_frame=per_video_frame,
        camera_table=build_camera_table(per_video_frame),
        timeline_frame=timeline_frame,
        overlay_manifest=overlay_manifest,
        event_clip_manifest=event_clip_manifest,
        missing_artifacts=missing_artifacts,
    )
