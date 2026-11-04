"""Muat artefak hitung NB06 menjadi tabel siap tampil, tanpa dependensi Streamlit."""

import json
import math
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from pipeline_common import (
    COUNT_CONTRACT,
    SPLIT_NAME_BY_CAMERA,
    TRAINING_CLASS_NAMES,
    UNMEASURABLE_COUNT_CONTRACT_KEYS,
)

DEFAULT_NB06_ROOT = PROJECT_ROOT / "output" / "nb06"
NB06_ROOT_ENV_VAR = "NB06_ROOT"
COUNTING_SUMMARY_FILENAME = "counting_summary.json"
COUNTING_BY_VIDEO_FILENAME = "counting_by_video.csv"
EXPECTED_ARTIFACT_NAMES = (COUNTING_SUMMARY_FILENAME, COUNTING_BY_VIDEO_FILENAME)

CAMERA_ID_COLUMN = "camera_id"
VIDEO_STEM_COLUMN = "video_stem"
SPLIT_COLUMN = "split"
STATUS_COLUMN = "status"
VEHICLES_COUNTED_COLUMN = "vehicles_counted"
VIDEOS_COUNTED_COLUMN = "videos_counted"

STATE_READY = "ready"
STATE_EMPTY = "empty"
STATE_PARTIAL = "partial"
STATE_ERROR = "error"

CAMERA_STATUS_COUNTED = "dihitung"
CAMERA_STATUS_NO_VIDEO = "tidak ada video"
CAMERA_STATUS_OUTSIDE = "di luar split hitung"

NOT_AVAILABLE_LABEL = "n/a"
THOUSAND_SEPARATOR = "."
THOUSAND_SUFFIXES = ((1_000_000.0, "M"), (1_000.0, "k"))
SUFFIX_DECIMAL_PLACES = 1

MEASURABLE_CONTRACT_KEYS = [
    contract_key
    for contract_key in COUNT_CONTRACT
    if contract_key not in UNMEASURABLE_COUNT_CONTRACT_KEYS
]


@dataclass(frozen=True)
class CountingSnapshot:
    """Satu pembacaan artefak NB06 beserta status kelengkapannya."""

    state: str
    summary: dict | None = None
    per_video_frame: pd.DataFrame | None = None
    camera_table: pd.DataFrame | None = None
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
    """Direktori artefak NB06 aktif, bisa diarahkan lewat env NB06_ROOT untuk hasil smoke."""
    override_root = os.environ.get(NB06_ROOT_ENV_VAR)
    return Path(override_root) if override_root else DEFAULT_NB06_ROOT


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
        for column_name in (VEHICLES_COUNTED_COLUMN, *TRAINING_CLASS_NAMES, *MEASURABLE_CONTRACT_KEYS)
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
    """Status satu titik CCTV terhadap split yang dihitung NB06."""
    if camera_id in counted_camera_ids:
        return CAMERA_STATUS_COUNTED
    if split_name in counted_splits:
        return CAMERA_STATUS_NO_VIDEO
    return f"{CAMERA_STATUS_OUTSIDE} ({split_name})"


def format_count(value: int | float | str | None) -> str:
    """Format angka gaya produk: 1.4k, 38k, n/a. Nol tetap nol, n/a tidak pernah jadi nol."""
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
        per_video_frame = pd.read_csv(nb06_root / COUNTING_BY_VIDEO_FILENAME)
    except (OSError, ValueError) as artifact_error:
        return CountingSnapshot(
            state=STATE_ERROR,
            missing_artifacts=missing_artifacts,
            error_message=str(artifact_error),
        )
    return CountingSnapshot(
        state=STATE_READY,
        summary=summary_payload,
        per_video_frame=per_video_frame,
        camera_table=build_camera_table(per_video_frame),
        missing_artifacts=missing_artifacts,
    )
