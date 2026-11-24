#!/usr/bin/env python3
"""Self-check logika NB02: crop ROI (rect + polygon), pengelompokan lampu, dan klasifikasi bertingkat."""

import ast
import importlib.util
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

BUILDERS_DIR = Path(__file__).resolve().parent
NB02_BUILDER_PATH = BUILDERS_DIR / "build_nb02_pseudo_label.py"

EXTRACTED_FUNCTION_NAMES = (
    "resolve_roi_polygon",
    "resolve_roi_rectangle",
    "crop_to_polygon",
    "crop_and_fit_frame",
    "intersection_over_union",
    "merge_overlapping_lamps",
    "lamp_center",
    "lamps_are_same_vehicle",
    "group_lamps_into_vehicles",
    "collect_group_signal_classes",
    "resolve_group_class",
    "group_signals_agree",
    "box_coverage",
    "vehicle_vote_for_box",
    "group_bounding_box",
    "build_group_detections",
)
EXTRACTED_CONSTANT_NAMES = (
    "LAMP_MERGE_IOU_THRESHOLD",
    "GROUP_MAX_CENTER_DISTANCE_PIXELS",
    "GROUP_MAX_VERTICAL_OFFSET_PIXELS",
    "GROUP_SIZE_RATIO_LIMIT",
    "MULTI_LAMP_MIN_COUNT",
    "VEHICLE_VOTE_MIN_COVERAGE",
)

FRAME_SIZE = (100, 100)
POLYGON_TEST_POINTS = [[25, 25], [75, 75], [25, 75]]
POLYGON_INSIDE_POINT = (5, 45)
POLYGON_OUTSIDE_POINT = (45, 5)
FULL_WHITE = (255, 255, 255)
FULL_BLACK = (0, 0, 0)

LAMP_BOX_NEAR_LEFT = [0.0, 0.0, 10.0, 10.0]
LAMP_BOX_NEAR_RIGHT = [30.0, 0.0, 40.0, 10.0]
LAMP_BOX_FAR_RIGHT = [300.0, 0.0, 310.0, 10.0]
IOU_OVERLAP_EXPECTED = 1.0 / 3.0


def load_module_from_path(module_path: Path):
    module_specification = importlib.util.spec_from_file_location(module_path.stem, module_path)
    loaded_module = importlib.util.module_from_spec(module_specification)
    module_specification.loader.exec_module(loaded_module)
    return loaded_module


def extract_named_sources(source_text: str, function_names: tuple[str, ...], constant_names: tuple[str, ...]):
    parsed_tree = ast.parse(source_text)
    function_sources = []
    constant_sources = []
    found_functions = set()
    found_constants = set()
    for tree_node in parsed_tree.body:
        if isinstance(tree_node, ast.FunctionDef) and tree_node.name in function_names:
            function_sources.append(ast.get_source_segment(source_text, tree_node))
            found_functions.add(tree_node.name)
        if isinstance(tree_node, ast.Assign):
            for assignment_target in tree_node.targets:
                if isinstance(assignment_target, ast.Name) and assignment_target.id in constant_names:
                    constant_sources.append(ast.get_source_segment(source_text, tree_node))
                    found_constants.add(assignment_target.id)
    missing_names = (set(function_names) - found_functions) | (set(constant_names) - found_constants)
    if missing_names:
        raise SystemExit(f"Sumber tidak ditemukan di builder: {sorted(missing_names)}")
    return constant_sources, function_sources


def build_nb02_namespace(nb02_builder_module) -> dict:
    config_constants, _ = extract_named_sources(nb02_builder_module.CONFIG_SOURCE, (), EXTRACTED_CONSTANT_NAMES)
    _, dino_functions = extract_named_sources(nb02_builder_module.DINO_SOURCE, EXTRACTED_FUNCTION_NAMES, ())
    namespace = {"np": np, "Image": Image, "ImageDraw": ImageDraw, "Path": Path}
    exec("\n\n".join(config_constants + dino_functions), namespace)
    return namespace


def write_white_frame(frame_path: Path) -> None:
    Image.new("RGB", FRAME_SIZE, FULL_WHITE).save(frame_path)


def pixel_at(image: Image.Image, x_position: int, y_position: int) -> tuple[int, int, int]:
    return image.convert("RGB").getpixel((x_position, y_position))


def check_polygon_crop(namespace: dict, frame_path: Path) -> None:
    result_image = namespace["crop_and_fit_frame"](frame_path, {"polygon": POLYGON_TEST_POINTS}, 640)
    assert result_image.size == (50, 50), f"ukuran crop polygon salah: {result_image.size}"
    assert pixel_at(result_image, *POLYGON_OUTSIDE_POINT) == FULL_BLACK, "area luar polygon seharusnya hitam"
    assert pixel_at(result_image, *POLYGON_INSIDE_POINT) == FULL_WHITE, "area dalam polygon seharusnya putih"


def check_rectangle_crop(namespace: dict, frame_path: Path) -> None:
    rectangle_entry = {"roi": {"x": 50, "y": 50, "width": 50, "height": 50}}
    result_image = namespace["crop_and_fit_frame"](frame_path, rectangle_entry, 640)
    assert result_image.size == (50, 50), f"ukuran crop rectangle salah: {result_image.size}"
    assert pixel_at(result_image, 25, 25) == FULL_WHITE, "crop rectangle seharusnya putih penuh"


def check_full_frame_crop(namespace: dict, frame_path: Path) -> None:
    result_image = namespace["crop_and_fit_frame"](frame_path, None, 640)
    assert result_image.size == FRAME_SIZE, f"ukuran full-frame salah: {result_image.size}"


def check_polygon_precedence(namespace: dict, frame_path: Path) -> None:
    combined_entry = {"roi": {"x": 0, "y": 0, "width": 10, "height": 10}, "polygon": POLYGON_TEST_POINTS}
    result_image = namespace["crop_and_fit_frame"](frame_path, combined_entry, 640)
    assert result_image.size == (50, 50), "polygon harus menang atas rectangle"


def check_intersection_over_union(namespace: dict) -> None:
    intersection_over_union = namespace["intersection_over_union"]
    assert intersection_over_union(LAMP_BOX_NEAR_LEFT, LAMP_BOX_NEAR_LEFT) == 1.0, "IoU box identik harus 1.0"
    assert intersection_over_union(LAMP_BOX_NEAR_LEFT, LAMP_BOX_FAR_RIGHT) == 0.0, "IoU box lepas harus 0.0"
    overlap_value = intersection_over_union([0.0, 0.0, 10.0, 10.0], [5.0, 0.0, 15.0, 10.0])
    assert abs(overlap_value - IOU_OVERLAP_EXPECTED) < 1e-9, f"IoU overlap salah: {overlap_value}"


def check_merge_overlapping_lamps(namespace: dict) -> None:
    merge_overlapping_lamps = namespace["merge_overlapping_lamps"]
    merged_same_lamp = merge_overlapping_lamps(
        [
            {"class_name": "LampuMobil", "score": 0.6, "box": LAMP_BOX_NEAR_LEFT},
            {"class_name": "LampuMotor", "score": 0.5, "box": LAMP_BOX_NEAR_LEFT},
        ]
    )
    assert len(merged_same_lamp) == 1, "dua kandidat pada lampu sama harus menyatu"
    assert merged_same_lamp[0]["class_scores"] == {"LampuMobil": 0.6, "LampuMotor": 0.5}, "skor kelas hasil merge salah"
    merged_separate_lamps = merge_overlapping_lamps(
        [
            {"class_name": "LampuMobil", "score": 0.6, "box": LAMP_BOX_NEAR_LEFT},
            {"class_name": "LampuMobil", "score": 0.5, "box": LAMP_BOX_FAR_RIGHT},
        ]
    )
    assert len(merged_separate_lamps) == 2, "lampu terpisah tidak boleh menyatu"


def build_test_lamp(box: list[float], score: float, class_scores: dict) -> dict:
    return {"box": box, "score": score, "class_scores": class_scores}


def check_grouping(namespace: dict) -> None:
    group_lamps_into_vehicles = namespace["group_lamps_into_vehicles"]
    near_pair = [
        build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.6, {"LampuMobil": 0.6}),
        build_test_lamp(LAMP_BOX_NEAR_RIGHT, 0.5, {"LampuMobil": 0.5}),
    ]
    assert len(group_lamps_into_vehicles(near_pair)) == 1, "dua lampu berdekatan harus jadi satu grup"
    far_pair = [
        build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.6, {"LampuMobil": 0.6}),
        build_test_lamp(LAMP_BOX_FAR_RIGHT, 0.5, {"LampuMobil": 0.5}),
    ]
    assert len(group_lamps_into_vehicles(far_pair)) == 2, "lampu berjauhan harus jadi dua grup"


def check_class_rule(namespace: dict) -> None:
    resolve_group_class = namespace["resolve_group_class"]
    two_lamp_group = [
        build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.6, {}),
        build_test_lamp(LAMP_BOX_NEAR_RIGHT, 0.5, {}),
    ]
    assert resolve_group_class(two_lamp_group) == "LampuMobil", "grup 2 lampu harus LampuMobil"
    motor_vote_lamp = build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.5, {"LampuMobil": 0.3, "LampuMotor": 0.4})
    assert resolve_group_class([motor_vote_lamp]) == "LampuMotor", "vote motor lebih tinggi harus LampuMotor"
    car_vote_lamp = build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.5, {"LampuMobil": 0.4, "LampuMotor": 0.3})
    assert resolve_group_class([car_vote_lamp]) == "LampuMobil", "vote mobil lebih tinggi harus LampuMobil"
    no_vote_lamp = build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.5, {})
    assert resolve_group_class([no_vote_lamp]) == "LampuMotor", "tanpa vote harus default LampuMotor"


def check_class_signal_hierarchy(namespace: dict) -> None:
    resolve_group_class = namespace["resolve_group_class"]
    group_signals_agree = namespace["group_signals_agree"]
    two_lamp_group = [
        build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.6, {}),
        build_test_lamp(LAMP_BOX_NEAR_RIGHT, 0.5, {}),
    ]
    motor_preferring_lamp = build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.5, {"LampuMobil": 0.3, "LampuMotor": 0.4})
    car_preferring_lamp = build_test_lamp(LAMP_BOX_NEAR_LEFT, 0.5, {"LampuMobil": 0.4, "LampuMotor": 0.3})
    assert resolve_group_class([motor_preferring_lamp], "LampuMobil") == "LampuMobil", "vote kendaraan harus menang atas skor DINO"
    assert resolve_group_class(two_lamp_group, "LampuMotor") == "LampuMobil", "geometri grup harus tetap prioritas"
    assert resolve_group_class([car_preferring_lamp], None) == "LampuMobil", "tanpa vote harus fallback skor DINO"
    assert not group_signals_agree(two_lamp_group, "LampuMotor", "LampuMobil"), "grup 2 lampu + vote motor harus konflik"
    assert group_signals_agree([motor_preferring_lamp], "LampuMotor", "LampuMotor"), "sinyal seragam harus sepakat"
    assert group_signals_agree([car_preferring_lamp], None, "LampuMobil"), "tanpa vote tidak boleh dianggap konflik"


def check_vehicle_coverage(namespace: dict) -> None:
    box_coverage = namespace["box_coverage"]
    vehicle_vote_for_box = namespace["vehicle_vote_for_box"]
    coverage_value = box_coverage([0.0, 0.0, 100.0, 100.0], [50.0, 50.0, 150.0, 150.0])
    assert abs(coverage_value - 0.25) < 1e-9, f"coverage 25% salah: {coverage_value}"
    overlapping_detections = [
        {"class_name": "LampuMotor", "score": 0.8, "box": [8.0, 8.0, 32.0, 32.0]},
        {"class_name": "LampuMobil", "score": 0.6, "box": [0.0, 0.0, 100.0, 100.0]},
    ]
    vote = vehicle_vote_for_box(overlapping_detections, [10.0, 10.0, 30.0, 30.0])
    assert vote == "LampuMotor", f"vote harus dari box coverage tertinggi: {vote}"
    assert vehicle_vote_for_box(overlapping_detections, [200.0, 200.0, 220.0, 220.0]) is None, "box lampu tanpa kendaraan harus None"


def check_group_detections(namespace: dict) -> None:
    group_detections = namespace["build_group_detections"](
        [
            {"class_name": "LampuMobil", "score": 0.6, "box": LAMP_BOX_NEAR_LEFT},
            {"class_name": "LampuMobil", "score": 0.5, "box": LAMP_BOX_NEAR_RIGHT},
        ]
    )
    assert len(group_detections) == 1, "dua lampu satu kendaraan harus jadi satu deteksi"
    assert group_detections[0]["class_name"] == "LampuMobil", "kelas deteksi grup salah"
    assert group_detections[0]["lamp_count"] == 2, "jumlah lampu grup salah"
    assert group_detections[0]["box"] == [0.0, 0.0, 40.0, 10.0], f"box union grup salah: {group_detections[0]['box']}"


def main() -> None:
    nb02_builder_module = load_module_from_path(NB02_BUILDER_PATH)
    namespace = build_nb02_namespace(nb02_builder_module)
    with tempfile.TemporaryDirectory() as temporary_directory:
        frame_path = Path(temporary_directory) / "frame_test.png"
        write_white_frame(frame_path)
        check_polygon_crop(namespace, frame_path)
        check_rectangle_crop(namespace, frame_path)
        check_full_frame_crop(namespace, frame_path)
        check_polygon_precedence(namespace, frame_path)
    check_intersection_over_union(namespace)
    check_merge_overlapping_lamps(namespace)
    check_grouping(namespace)
    check_class_rule(namespace)
    check_class_signal_hierarchy(namespace)
    check_vehicle_coverage(namespace)
    check_group_detections(namespace)
    print("Self-check NB02 OK (crop polygon/rect/full, IoU, merge, grouping, aturan kelas, klasifikasi bertingkat, coverage kendaraan, group detections).")


if __name__ == "__main__":
    main()
