import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from streamlit.testing.v1 import AppTest

from pipeline_common import CLASS_NAMES, COUNT_CONTRACT, TRAINING_CLASS_NAMES

APP_PATH = str(PROJECT_ROOT / "dashboard" / "app.py")


def build_summary() -> dict:
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_counts["MobilStandar"] = 7
    class_counts["MobilTidakStandar"] = 3
    class_counts["MotorStandar"] = 5
    class_counts["MotorTidakStandar"] = 2
    class_counts["PantulanCahaya"] = 1
    event_class_counts = {name: 0 for name in CLASS_NAMES}
    event_class_counts["MobilTidakStandar"] = 3
    event_class_counts["MotorTidakStandar"] = 2
    counting_totals = {key: sum(event_class_counts[name] for name in classes) for key, classes in COUNT_CONTRACT.items()}
    return {
        "smoke_mode": False,
        "counted_splits": ["test"],
        "min_track_hits": 3,
        "bbox_counted": 5,
        "videos_counted": 2,
        "videos_found": 2,
        "frames_processed": 1840,
        "raw_detections": 4210,
        "untracked_detections": 37,
        "thresholded_detections": 3980,
        "tracks_total": 96,
        "vehicles_total": 71,
        "fragments_stitched": 25,
        "duplication_factor_vs_per_frame": 221.1,
        "class_counts": class_counts,
        "event_class_counts": event_class_counts,
        "counting_totals": counting_totals,
        "selected_thresholds": {name: 0.35 for name in TRAINING_CLASS_NAMES},
    }


def build_per_video_frame() -> pd.DataFrame:
    rows = []
    for camera_id, video_stem, offset in (
        ("bogor-kab-bappenda2", "video_alpha", 0),
        ("bogor-kab-pdam2", "video_beta", 3),
    ):
        row = {"video_stem": video_stem, "camera_id": camera_id, "bbox_counted": 4 + offset}
        for class_index, class_name in enumerate(CLASS_NAMES):
            row[class_name] = 1 if class_index % 3 == 0 else 0
        for key, classes in COUNT_CONTRACT.items():
            row[key] = sum(row[name] for name in classes)
        rows.append(row)
    return pd.DataFrame(rows)


def write_artifacts(nb06_root: Path) -> None:
    nb06_root.mkdir(parents=True, exist_ok=True)
    (nb06_root / "counting_summary.json").write_text(
        json.dumps(build_summary(), indent=2), encoding="utf-8"
    )
    build_per_video_frame().to_csv(nb06_root / "counting_by_video.csv", index=False)


def run_app() -> AppTest:
    app_test = AppTest.from_file(APP_PATH, default_timeout=60)
    return app_test.run()


def main() -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)

        os.environ["NB06_ROOT"] = str(temporary_root / "empty_nb06")
        empty_test = run_app()
        assert not empty_test.exception, [item.value for item in empty_test.exception]
        assert any("CCTV" in item.value for item in empty_test.title), [item.value for item in empty_test.title]
        assert len(empty_test.warning) >= 1, "state kosong tidak memunculkan peringatan"
        assert len(empty_test.metric) == 0, len(empty_test.metric)
        print(f"[check] empty  title={empty_test.title[0].value!r} warnings={len(empty_test.warning)}")

        ready_root = temporary_root / "ready_nb06"
        write_artifacts(ready_root)
        os.environ["NB06_ROOT"] = str(ready_root)
        ready_test = run_app()
        assert not ready_test.exception, [item.value for item in ready_test.exception]
        assert len(ready_test.warning) == 0, [item.value for item in ready_test.warning]
        metric_labels = {item.label for item in ready_test.metric}
        assert {"BBox tidak standar", "BBox mobil", "BBox motor"}.issubset(metric_labels), metric_labels
        metric_values = [item.value for item in ready_test.metric]
        camera_frames = [item.value for item in ready_test.dataframe if {"Titik CCTV", "Status"}.issubset(item.value.columns)]
        assert camera_frames, "tabel status titik CCTV tidak dirender"
        camera_frame = camera_frames[0]
        print(f"[check] ready  metrics={metric_values}")
        print(f"[check] ready  cameras={camera_frame['Titik CCTV'].tolist()}")
        print(f"[check] ready  status={camera_frame['Status'].tolist()}")
        measured_columns = [
            column
            for column in ("BBox tidak standar", "BBox mobil tidak standar", "BBox motor tidak standar")
            if column in camera_frame.columns
        ]
        assert measured_columns, camera_frame.columns.tolist()
        print(f"[check] ready  measured={camera_frame[measured_columns].to_dict('records')}")

        partial_root = temporary_root / "partial_nb06"
        partial_root.mkdir(parents=True, exist_ok=True)
        (partial_root / "counting_summary.json").write_text("{}", encoding="utf-8")
        os.environ["NB06_ROOT"] = str(partial_root)
        partial_test = run_app()
        assert not partial_test.exception, [item.value for item in partial_test.exception]
        assert len(partial_test.warning) >= 1, "state parsial tidak memunculkan peringatan"
        print(f"[check] partial warnings={len(partial_test.warning)}")

        broken_root = temporary_root / "broken_nb06"
        broken_root.mkdir(parents=True, exist_ok=True)
        (broken_root / "counting_summary.json").write_text("{bukan json", encoding="utf-8")
        (broken_root / "counting_by_video.csv").write_text("camera_id\n", encoding="utf-8")
        os.environ["NB06_ROOT"] = str(broken_root)
        broken_test = run_app()
        assert not broken_test.exception, [item.value for item in broken_test.exception]
        assert len(broken_test.error) >= 1, "state rusak tidak memunculkan error"
        print(f"[check] broken errors={len(broken_test.error)}")

    print("[check] SEMUA STATE DASHBOARD OK")


if __name__ == "__main__":
    main()
