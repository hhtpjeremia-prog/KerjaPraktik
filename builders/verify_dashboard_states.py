import json
import os
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from streamlit.testing.v1 import AppTest

from pipeline_common import CLASS_NAMES, COUNT_CONTRACT, TRAINING_CLASS_NAMES, UNMEASURABLE_COUNT_CONTRACT_KEYS

APP_PATH = str(PROJECT_ROOT / "dashboard" / "app.py")


def build_summary() -> dict:
    class_counts = {name: 0 for name in CLASS_NAMES}
    class_counts["MobilNormal"] = 7
    class_counts["MobilSilau"] = 3
    class_counts["MotorNormal"] = 5
    class_counts["MotorSilau"] = 2
    class_counts["PantulanCahaya"] = 1
    counting_totals = {
        key: "n/a" if key in UNMEASURABLE_COUNT_CONTRACT_KEYS
        else sum(class_counts[name] for name in classes)
        for key, classes in COUNT_CONTRACT.items()
    }
    return {
        "smoke_mode": False,
        "counted_splits": ["test"],
        "min_track_hits": 3,
        "vehicles_counted": 18,
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
        "counting_totals": counting_totals,
        "selected_thresholds": {name: 0.35 for name in TRAINING_CLASS_NAMES},
    }


def build_per_video_frame() -> pd.DataFrame:
    rows = []
    for camera_id, video_stem, offset in (
        ("bogor-kab-bappenda2", "video_alpha", 0),
        ("bogor-kab-pdam2", "video_beta", 3),
    ):
        row = {"video_stem": video_stem, "camera_id": camera_id, "vehicles_counted": 4 + offset}
        for class_index, class_name in enumerate(CLASS_NAMES):
            row[class_name] = 1 if class_index % 3 == 0 else 0
        for key, classes in COUNT_CONTRACT.items():
            row[key] = "n/a" if key in UNMEASURABLE_COUNT_CONTRACT_KEYS else sum(
                row[name] for name in classes
            )
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
        assert len(empty_test.title) == 1, len(empty_test.title)
        assert len(empty_test.warning) >= 1, "state kosong tidak memunculkan peringatan"
        assert len(empty_test.metric) == 0, len(empty_test.metric)
        print(f"[check] empty  title={empty_test.title[0].value!r} warnings={len(empty_test.warning)}")

        ready_root = temporary_root / "ready_nb06"
        write_artifacts(ready_root)
        os.environ["NB06_ROOT"] = str(ready_root)
        ready_test = run_app()
        assert not ready_test.exception, [item.value for item in ready_test.exception]
        assert len(ready_test.metric) == 4, len(ready_test.metric)
        assert len(ready_test.warning) == 0, [item.value for item in ready_test.warning]
        assert len(ready_test.dataframe) == 5, len(ready_test.dataframe)
        chart_elements = ready_test.get("vega_lite_chart")
        assert len(chart_elements) == 1, len(chart_elements)
        metric_values = [item.value for item in ready_test.metric]
        camera_frame = ready_test.dataframe[0].value
        print(f"[check] ready  metrics={metric_values}")
        print(f"[check] ready  cameras={camera_frame['Titik CCTV'].tolist()}")
        print(f"[check] ready  status={camera_frame['Status'].tolist()}")
        print(f"[check] ready  kendaraan={camera_frame['Kendaraan'].tolist()}")

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
