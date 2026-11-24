"""Sumber tunggal helper ROI, peta split, daftar kelas, kontrak hitung, dan penyambungan track."""

import json
import math
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image, ImageDraw

CLASS_NAMES = [
    "MobilStandar",
    "MobilTidakStandar",
    "MotorStandar",
    "MotorTidakStandar",
    "PantulanCahaya",
]
# Status anotasi sementara saja; bukan kelas model dan tidak boleh diekspor sebagai label.
PENDING_CLASS_NAME = "BelumDitentukan"
# Model vocabulary is exactly the reviewed annotation vocabulary. Reviewer workflow
# status is metadata and must never become a model class.
TRAINING_CLASS_NAMES = CLASS_NAMES.copy()
SPLIT_NAME_BY_CAMERA = {
    "bogor-kab-bappenda1": "train",
    "bogor-kab-bappenda2": "test",
    "bogor-kab-bappenda3": "train",
    "bogor-kab-pdam2": "test",
    "bogor-kab-pemda2": "val",
}
MAX_CROP_SIDE = 640
DEDUP_HASH_IMAGE_SIZE = (9, 8)
DUPLICATE_WINDOW_FRAMES = 1
DUPLICATE_HASH_DISTANCE = 4
COUNT_CONTRACT = {
    "jumlah_kendaraan_mobil_tidak_standar_melintasi_garis": ["MobilTidakStandar"],
    "jumlah_kendaraan_motor_tidak_standar_melintasi_garis": ["MotorTidakStandar"],
}
BOX_COUNT_CONTRACT = {
    "jumlah_bbox_mobil_standar": ["MobilStandar"],
    "jumlah_bbox_mobil_tidak_standar": ["MobilTidakStandar"],
    "jumlah_bbox_motor_standar": ["MotorStandar"],
    "jumlah_bbox_motor_tidak_standar": ["MotorTidakStandar"],
    "jumlah_bbox_pantulan_cahaya": ["PantulanCahaya"],
}
# BOX_COUNT_CONTRACT is box-level; COUNT_CONTRACT above is track-crossing-level.
# Reflection remains a trained class but is excluded from both vehicle count contracts.
FRAGMENT_MAX_GAP_FRAMES = 50
FRAGMENT_MIN_IOU = 0.50
FRAGMENT_DISTANCE_RATIO = 0.35


def load_roi_config(roi_directory: Path) -> dict:
    with (roi_directory / "roi_config.json").open(encoding="utf-8") as roi_file:
        return json.load(roi_file)


def filter_store_frames(frames_payload: dict, active_frame_stems: set[str]) -> dict:
    """Buang entry anotasi yang frame-nya tidak ada di manifest aktif."""
    return {
        frame_stem: entry
        for frame_stem, entry in frames_payload.items()
        if frame_stem in active_frame_stems
    }


def resolve_roi_polygon(roi_entry: dict | None, frame_width: int, frame_height: int) -> list[list[int]] | None:
    if not roi_entry:
        return None
    polygon_points = roi_entry.get("polygon")
    if not polygon_points or len(polygon_points) < 3:
        return None
    return [
        [min(max(int(point[0]), 0), frame_width), min(max(int(point[1]), 0), frame_height)]
        for point in polygon_points
    ]


def resolve_roi_bounds(roi_entry: dict | None, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    if not roi_entry:
        return None
    roi = roi_entry.get("roi")
    if not roi:
        return None
    crop_x1 = max(0, roi["x"])
    crop_y1 = max(0, roi["y"])
    crop_x2 = min(frame_width, roi["x"] + roi["width"])
    crop_y2 = min(frame_height, roi["y"] + roi["height"])
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return None
    return (crop_x1, crop_y1, crop_x2, crop_y2)


def crop_to_polygon(frame_image: Image.Image, polygon_points: list[list[int]]) -> Image.Image:
    polygon_xs = [point[0] for point in polygon_points]
    polygon_ys = [point[1] for point in polygon_points]
    crop_x1 = max(0, min(polygon_xs))
    crop_y1 = max(0, min(polygon_ys))
    crop_x2 = min(frame_image.size[0], max(polygon_xs))
    crop_y2 = min(frame_image.size[1], max(polygon_ys))
    if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
        return frame_image.copy()
    cropped_image = frame_image.crop((crop_x1, crop_y1, crop_x2, crop_y2))
    local_points = [(point[0] - crop_x1, point[1] - crop_y1) for point in polygon_points]
    polygon_mask = Image.new("L", cropped_image.size, 0)
    ImageDraw.Draw(polygon_mask).polygon(local_points, fill=255)
    masked_image = Image.new("RGB", cropped_image.size, (0, 0, 0))
    masked_image.paste(cropped_image, mask=polygon_mask)
    return masked_image


def mask_frame_to_roi(frame_image: Image.Image, roi_entry: dict | None) -> Image.Image:
    """Kembalikan full-frame dengan area di luar ROI dinetralkan menjadi hitam."""
    frame_width, frame_height = frame_image.size
    polygon_points = resolve_roi_polygon(roi_entry, frame_width, frame_height)
    if polygon_points is not None:
        mask = Image.new("L", frame_image.size, 0)
        ImageDraw.Draw(mask).polygon([tuple(point) for point in polygon_points], fill=255)
        masked_frame = Image.new("RGB", frame_image.size, (0, 0, 0))
        masked_frame.paste(frame_image.convert("RGB"), mask=mask)
        return masked_frame
    roi_bounds = resolve_roi_bounds(roi_entry, frame_width, frame_height)
    if roi_bounds is None:
        return frame_image.convert("RGB")
    mask = Image.new("L", frame_image.size, 0)
    ImageDraw.Draw(mask).rectangle(roi_bounds, fill=255)
    masked_frame = Image.new("RGB", frame_image.size, (0, 0, 0))
    masked_frame.paste(frame_image.convert("RGB"), mask=mask)
    return masked_frame


def build_roi_crop(
    frame_image: Image.Image,
    roi_entry: dict | None,
    max_side: int = MAX_CROP_SIDE,
) -> tuple[Image.Image, int, int, float]:
    """Kembalikan crop ROI plus offset dan skala agar koordinat crop bisa dipetakan ke frame asli."""
    frame_width, frame_height = frame_image.size
    polygon_points = resolve_roi_polygon(roi_entry, frame_width, frame_height)
    roi_bounds = resolve_roi_bounds(roi_entry, frame_width, frame_height)
    if polygon_points is not None:
        cropped_image = crop_to_polygon(frame_image, polygon_points)
        offset_x = max(0, min(point[0] for point in polygon_points))
        offset_y = max(0, min(point[1] for point in polygon_points))
    elif roi_bounds is not None:
        cropped_image = frame_image.crop(roi_bounds)
        offset_x, offset_y = roi_bounds[0], roi_bounds[1]
    else:
        cropped_image = frame_image.copy()
        offset_x, offset_y = 0, 0
    if cropped_image.size == frame_image.size:
        offset_x, offset_y = 0, 0
    pre_scale_width = cropped_image.size[0]
    resized_image = cropped_image.copy()
    resized_image.thumbnail((max_side, max_side), Image.LANCZOS)
    scale = resized_image.size[0] / pre_scale_width if pre_scale_width else 1.0
    return resized_image, offset_x, offset_y, scale


def training_class_name(annotation_class_name: str) -> str | None:
    """Return a known five-class label unchanged; reject unknown and pending labels."""
    return annotation_class_name if annotation_class_name in TRAINING_CLASS_NAMES else None


def resolve_project_root(start_path: str | Path) -> Path:
    """Find the repository root by stable marker files, independent of its folder name."""
    start = Path(start_path).resolve()
    candidates = (start, *start.parents)
    for candidate in candidates:
        if (candidate / "pipeline_common.py").is_file() and (candidate / "requirements.txt").is_file():
            return candidate
    raise FileNotFoundError(
        f"Root proyek tidak ditemukan dari {start}; perlu pipeline_common.py dan requirements.txt."
    )


def difference_hash(gray_image, hash_image_size: tuple[int, int] = DEDUP_HASH_IMAGE_SIZE) -> int:
    """Compute the shared dHash value from a grayscale image array."""
    image = Image.fromarray(np.asarray(gray_image, dtype=np.uint8)).convert("L")
    resized = np.asarray(image.resize(hash_image_size, Image.Resampling.LANCZOS), dtype=np.int16)
    differences = resized[:, 1:] >= resized[:, :-1]
    return sum(1 << index for index, bit in enumerate(differences.reshape(-1)) if bit)


def allocate_sample_quota(camera_counts: dict[str, int], total_frames: int) -> dict[str, int]:
    """Allocate an integer sampling target proportionally across cameras."""
    if total_frames < 0 or any(count < 0 for count in camera_counts.values()):
        raise ValueError("jumlah frame dan kandidat kamera tidak boleh negatif")
    eligible_total = sum(camera_counts.values())
    if eligible_total == 0:
        return {camera_id: 0 for camera_id in camera_counts}
    target_total = min(total_frames, eligible_total)
    quotas = {
        camera_id: target_total * camera_count // eligible_total
        for camera_id, camera_count in camera_counts.items()
    }
    remainder = target_total - sum(quotas.values())
    fractional_order = sorted(
        camera_counts,
        key=lambda camera_id: (-(camera_counts[camera_id] % eligible_total), camera_id),
    )
    for camera_id in fractional_order[:remainder]:
        quotas[camera_id] += 1
    return quotas


def allocate_protected_sample_quota(
    candidate_counts: dict[str, int],
    protected_counts: dict[str, int],
    total_frames: int,
) -> dict[str, int]:
    """Keep every protected frame, distributing the remaining budget proportionally."""
    cameras = set(candidate_counts) | set(protected_counts)
    candidates = {camera: int(candidate_counts.get(camera, 0)) for camera in cameras}
    protected = {camera: int(protected_counts.get(camera, 0)) for camera in cameras}
    if any(value < 0 for value in [*candidates.values(), *protected.values()]):
        raise ValueError("jumlah kandidat dan frame terlindungi tidak boleh negatif")
    if any(protected[camera] > candidates[camera] for camera in cameras):
        raise ValueError("jumlah frame terlindungi melebihi kandidat kamera")
    target = min(total_frames, sum(candidates.values()))
    protected_total = sum(protected.values())
    if protected_total > target:
        raise ValueError(
            f"frame terlindungi {protected_total} melebihi target sampling efektif {target}"
        )
    remaining_counts = {
        camera: candidates[camera] - protected[camera]
        for camera in cameras
    }
    extra_quotas = allocate_sample_quota(remaining_counts, target - protected_total)
    return {camera: protected[camera] + extra_quotas[camera] for camera in cameras}


def near_duplicate_flags(
    frame_hashes: list[int], *, window_frames: int = 1, distance_threshold: int = 4, missing_hash: int = -1
) -> list[bool]:
    """Mark hashes near a prior hash in the same video, using one shared Hamming rule."""
    flags = []
    for index, current_hash in enumerate(frame_hashes):
        is_duplicate = False
        if current_hash != missing_hash:
            start = max(0, index - window_frames)
            is_duplicate = any(
                previous != missing_hash and (int(current_hash) ^ int(previous)).bit_count() <= distance_threshold
                for previous in frame_hashes[start:index]
            )
        flags.append(is_duplicate)
    return flags


def class_family(class_name: str) -> str:
    if class_name.startswith("Mobil"):
        return "Mobil"
    if class_name.startswith("Motor"):
        return "Motor"
    return class_name


def box_overlap_iou(box_a: list[float], box_b: list[float]) -> float:
    intersection_left = max(box_a[0], box_b[0])
    intersection_top = max(box_a[1], box_b[1])
    intersection_right = min(box_a[2], box_b[2])
    intersection_bottom = min(box_a[3], box_b[3])
    if intersection_right <= intersection_left or intersection_bottom <= intersection_top:
        return 0.0
    intersection_area = (intersection_right - intersection_left) * (intersection_bottom - intersection_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union_area = area_a + area_b - intersection_area
    return intersection_area / union_area if union_area > 0 else 0.0


def box_center_distance(box_a: list[float], box_b: list[float]) -> float:
    return math.dist(
        ((box_a[0] + box_a[2]) / 2.0, (box_a[1] + box_a[3]) / 2.0),
        ((box_b[0] + box_b[2]) / 2.0, (box_b[1] + box_b[3]) / 2.0),
    )


def box_diagonal(box: list[float]) -> float:
    return math.dist((box[0], box[1]), (box[2], box[3]))


def detect_center_line_crossing(
    observations: list[dict],
    *,
    line_y: float,
    line_x1: float,
    line_x2: float,
    hysteresis: float,
    min_hits: int,
) -> dict | None:
    """Return the first stable bbox-center crossing of a horizontal count segment."""
    if len(observations) < min_hits:
        return None
    stable_side = None
    for row in sorted(observations, key=lambda item: int(item["frame_index"])):
        center_y = (float(row["frame_y1"]) + float(row["frame_y2"])) / 2.0
        center_x = (float(row["frame_x1"]) + float(row["frame_x2"])) / 2.0
        delta = center_y - line_y
        side = -1 if delta < -hysteresis else 1 if delta > hysteresis else 0
        if side == 0:
            continue
        if stable_side is not None and side != stable_side:
            if not line_x1 <= center_x <= line_x2:
                stable_side = side
                continue
            return {
                "row": row,
                "crossing_direction": "down" if stable_side < side else "up",
            }
        stable_side = side
    return None


def qualify_by_class_threshold(detection_frame: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Filter deteksi dengan confidence minimum yang dipilih untuk kelas masing-masing."""
    if detection_frame.empty:
        return detection_frame.copy()
    threshold_series = detection_frame["class_name"].map(thresholds)
    return detection_frame[detection_frame["score"] >= threshold_series].copy()


def resolve_track_class(track_frame: pd.DataFrame) -> tuple[str, float]:
    """Pilih kelas mayoritas per track, lalu pecahkan seri dengan rerata confidence."""
    class_stats = track_frame.groupby("class_name")["score"].agg(["count", "mean"])
    class_stats = class_stats.sort_values(["count", "mean"], ascending=[False, False])
    return str(class_stats.index[0]), float(class_stats.iloc[0]["mean"])


UNIT_SUMMARY_COLUMNS = [
    "video_stem", "unit_id", "camera_id", "class_name", "hits",
    "first_frame", "last_frame", "mean_score",
    "first_x1", "first_y1", "first_x2", "first_y2",
    "last_x1", "last_y1", "last_x2", "last_y2",
]


def summarise_units(detection_frame: pd.DataFrame, grouping_column: str) -> pd.DataFrame:
    """Ringkas observasi per unit ID ke kelas, durasi, confidence, dan bbox awal/akhir."""
    if detection_frame.empty:
        return pd.DataFrame(columns=UNIT_SUMMARY_COLUMNS)
    unit_rows = []
    for (video_stem, unit_id), unit_frame in detection_frame.groupby(["video_stem", grouping_column], sort=True):
        ordered_frame = unit_frame.sort_values("frame_index")
        first_row = ordered_frame.iloc[0]
        last_row = ordered_frame.iloc[-1]
        resolved_class, mean_score = resolve_track_class(ordered_frame)
        unit_rows.append({
            "video_stem": video_stem,
            "unit_id": int(unit_id),
            "camera_id": first_row["camera_id"],
            "class_name": resolved_class,
            "hits": int(len(ordered_frame)),
            "first_frame": int(first_row["frame_index"]),
            "last_frame": int(last_row["frame_index"]),
            "mean_score": round(mean_score, 4),
            "first_x1": float(first_row["frame_x1"]),
            "first_y1": float(first_row["frame_y1"]),
            "first_x2": float(first_row["frame_x2"]),
            "first_y2": float(first_row["frame_y2"]),
            "last_x1": float(last_row["frame_x1"]),
            "last_y1": float(last_row["frame_y1"]),
            "last_x2": float(last_row["frame_x2"]),
            "last_y2": float(last_row["frame_y2"]),
        })
    return pd.DataFrame(unit_rows, columns=UNIT_SUMMARY_COLUMNS)


def detect_line_crossings(
    detection_frame: pd.DataFrame,
    line_by_video: dict[str, dict],
    *,
    min_track_hits: int,
    hysteresis_ratio: float,
) -> pd.DataFrame:
    """Keluarkan satu crossing per track kendaraan tidak standar yang benar-benar melintasi garis."""
    crossing_columns = [
        "video_stem", "camera_id", "vehicle_id", "track_id", "class_name", "score",
        "mean_track_score", "crossing_frame", "crossing_direction", "line_y",
        "frame_x1", "frame_y1", "frame_x2", "frame_y2",
    ]
    if detection_frame.empty:
        return pd.DataFrame(columns=crossing_columns)
    crossing_rows = []
    event_classes = {"MobilTidakStandar", "MotorTidakStandar"}
    for (video_stem, vehicle_id), unit_frame in detection_frame.groupby(["video_stem", "vehicle_id"], sort=True):
        ordered = unit_frame.sort_values("frame_index")
        if len(ordered) < min_track_hits:
            continue
        event_class, mean_score = resolve_track_class(ordered)
        if event_class not in event_classes:
            continue
        job = line_by_video[video_stem]
        crossing = detect_center_line_crossing(
            ordered.to_dict(orient="records"),
            line_y=float(job["count_line_y"]),
            line_x1=float(job["count_line_x1"]),
            line_x2=float(job["count_line_x2"]),
            hysteresis=max(3.0, float(job["height"]) * hysteresis_ratio),
            min_hits=min_track_hits,
        )
        if crossing is None:
            continue
        row = crossing["row"]
        crossing_rows.append({
            "video_stem": video_stem,
            "camera_id": row["camera_id"],
            "vehicle_id": int(vehicle_id),
            "track_id": int(row["track_id"]),
            "class_name": event_class,
            "score": round(float(row["score"]), 4),
            "mean_track_score": round(mean_score, 4),
            "crossing_frame": int(row["frame_index"]),
            "crossing_direction": crossing["crossing_direction"],
            "line_y": int(job["count_line_y"]),
            "frame_x1": float(row["frame_x1"]),
            "frame_y1": float(row["frame_y1"]),
            "frame_x2": float(row["frame_x2"]),
            "frame_y2": float(row["frame_y2"]),
        })
    return pd.DataFrame(crossing_rows, columns=crossing_columns)


def stitch_track_fragments(
    track_frame: pd.DataFrame,
    max_gap_frames: int = FRAGMENT_MAX_GAP_FRAMES,
    min_iou: float = FRAGMENT_MIN_IOU,
    distance_ratio: float = FRAGMENT_DISTANCE_RATIO,
) -> pd.DataFrame:
    """Petakan track_id ke vehicle_id dengan menyambung fragmen sejenis yang berdekatan."""
    assignment_rows = []
    for video_stem, video_tracks in track_frame.groupby("video_stem", sort=True):
        ordered_frame = video_tracks.sort_values(["first_frame", "unit_id"]).reset_index(drop=True)
        open_groups = []
        for track_position in range(len(ordered_frame)):
            current_row = ordered_frame.iloc[track_position]
            open_groups = [
                group
                for group in open_groups
                if group["last_frame"] >= current_row["first_frame"] - max_gap_frames
            ]
            current_first_box = [
                float(current_row["first_x1"]),
                float(current_row["first_y1"]),
                float(current_row["first_x2"]),
                float(current_row["first_y2"]),
            ]
            current_last_box = [
                float(current_row["last_x1"]),
                float(current_row["last_y1"]),
                float(current_row["last_x2"]),
                float(current_row["last_y2"]),
            ]
            current_family = class_family(str(current_row["class_name"]))
            best_group = None
            best_rank = None
            for group in open_groups:
                if group["family"] != current_family:
                    continue
                gap_frames = int(current_row["first_frame"]) - group["last_frame"]
                if gap_frames < 1 or gap_frames > max_gap_frames:
                    continue
                overlap = box_overlap_iou(group["last_box"], current_first_box)
                close_by_position = (
                    box_center_distance(group["last_box"], current_first_box)
                    <= distance_ratio * box_diagonal(current_first_box)
                )
                if overlap < min_iou and not close_by_position:
                    continue
                rank = (-overlap, gap_frames)
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_group = group
            if best_group is None:
                best_group = {
                    "root_id": int(current_row["unit_id"]),
                    "last_frame": int(current_row["last_frame"]),
                    "last_box": current_last_box,
                    "family": current_family,
                }
                open_groups.append(best_group)
            else:
                best_group["last_frame"] = int(current_row["last_frame"])
                best_group["last_box"] = current_last_box
            assignment_rows.append({
                "video_stem": video_stem,
                "track_id": int(current_row["unit_id"]),
                "vehicle_id": best_group["root_id"],
            })
    return pd.DataFrame(assignment_rows, columns=["video_stem", "track_id", "vehicle_id"])


def validate_tracker_thresholds(tracker_settings: dict, selected_thresholds: dict[str, float]) -> None:
    """Tolak tracker yang ambang pembuatan track-nya lebih tinggi dari ambang kelas terendah."""
    lowest_class_threshold = min(selected_thresholds.values())
    if float(tracker_settings["new_track_thresh"]) > lowest_class_threshold:
        raise ValueError(
            f"new_track_thresh tracker ({tracker_settings['new_track_thresh']}) lebih tinggi dari ambang "
            f"kelas terendah ({lowest_class_threshold}). Kelas dengan ambang di bawah itu tidak akan pernah "
            "mendapat track sehingga jumlahnya nol secara palsu. Turunkan new_track_thresh di konfigurasi tracker."
        )
