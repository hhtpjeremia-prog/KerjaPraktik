#!/usr/bin/env python3
"""Picker polygon ROI per kamera untuk NB02 — menulis output/roi/roi_config.json."""

import csv
import json
import os
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from PIL import Image, ImageDraw

RISET_ROOT = Path(__file__).resolve().parent.parent
FRAMES_MANIFEST_PATH = RISET_ROOT / "output" / "manifests" / "frames_manifest.csv"
ROI_CONFIG_PATH = RISET_ROOT / "output" / "roi" / "roi_config.json"

MINIMUM_POLYGON_POINTS = 3
GRID_LINE_STEP = 50
FIGURE_SIZE_INCHES = (12, 9)
OUTSIDE_DIM_FACTOR = 0.35
OUTSIDE_MASK_VALUE = 0
INSIDE_MASK_VALUE = 255
POLYGON_LINE_WIDTH = 2
VERTEX_MARKER_SIZE = 6
GRID_LINE_WIDTH = 0.4
GRID_ALPHA = 0.5

POLYGON_COLOR = "#00ff00"
VERTEX_COLOR = "#ffff00"
GRID_COLOR = "#00ffff"
WARNING_COLOR = "#ff4444"
TITLE_COLOR = "black"

LEFT_MOUSE_BUTTON = 1
RIGHT_MOUSE_BUTTON = 3

UNDO_KEY_NAMES = ("u", "backspace")
ACCEPT_KEY_NAME = "enter"
CANCEL_KEY_NAME = "escape"

NON_INTERACTIVE_BACKENDS = ("agg", "pdf", "ps", "svg", "cairo", "template")
INLINE_BACKEND_MARKER = "inline"
NON_INTERACTIVE_BACKEND_HINT = "module://matplotlib_inline.backend_inline"

PICKING_INSTRUCTION = "klik kiri pertama = polygon baru · klik kanan / u = undo · Enter = simpan · Esc = pertahankan lama"


def load_manifest_rows(manifest_path: Path) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as manifest_file:
        return list(csv.DictReader(manifest_file))


def extract_camera_id(video_stem: str) -> str:
    return video_stem.split("_")[1]


def representative_frame_by_camera(manifest_rows: list[dict]) -> dict[str, Path]:
    frame_by_camera: dict[str, Path] = {}
    for manifest_row in manifest_rows:
        video_stem = Path(manifest_row["video_path"]).stem
        camera_id = extract_camera_id(video_stem)
        frame_by_camera.setdefault(camera_id, Path(manifest_row["frame_path"]))
    return frame_by_camera


def load_roi_config(config_path: Path) -> dict:
    if not config_path.exists():
        return {}
    with config_path.open(encoding="utf-8") as config_file:
        return json.load(config_file)


def save_roi_config(config_path: Path, roi_config: dict) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = config_path.with_name(config_path.name + ".tmp")
    with temporary_path.open("w", encoding="utf-8") as config_file:
        json.dump(roi_config, config_file, ensure_ascii=False, indent=2)
    os.replace(temporary_path, config_path)


def draw_reference_grid(axis: Axes, image_width: int, image_height: int) -> None:
    for vertical_x in range(0, image_width, GRID_LINE_STEP):
        axis.axvline(vertical_x, color=GRID_COLOR, linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)
    for horizontal_y in range(0, image_height, GRID_LINE_STEP):
        axis.axhline(horizontal_y, color=GRID_COLOR, linewidth=GRID_LINE_WIDTH, alpha=GRID_ALPHA)


def build_preview_array(frame_array: np.ndarray, points: list[list[float]]) -> np.ndarray:
    if len(points) < MINIMUM_POLYGON_POINTS:
        return frame_array
    image_height, image_width = frame_array.shape[:2]
    mask_image = Image.new("L", (image_width, image_height), OUTSIDE_MASK_VALUE)
    integer_points = [(int(round(point[0])), int(round(point[1]))) for point in points]
    ImageDraw.Draw(mask_image).polygon(integer_points, fill=INSIDE_MASK_VALUE)
    inside_mask = np.array(mask_image) > OUTSIDE_MASK_VALUE
    dimmed_array = (frame_array * OUTSIDE_DIM_FACTOR).astype(np.uint8)
    return np.where(inside_mask[:, :, np.newaxis], frame_array, dimmed_array)


def draw_picking_state(
    axis: Axes,
    frame_array: np.ndarray,
    camera_id: str,
    points: list[list[float]],
    warning_message: str,
) -> None:
    image_height, image_width = frame_array.shape[:2]
    axis.clear()
    axis.imshow(build_preview_array(frame_array, points))
    draw_reference_grid(axis, image_width, image_height)
    if points:
        closed_points = list(points) + [points[0]]
        axis.plot(
            [point[0] for point in closed_points],
            [point[1] for point in closed_points],
            color=POLYGON_COLOR,
            linewidth=POLYGON_LINE_WIDTH,
        )
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            markersize=VERTEX_MARKER_SIZE,
            color=VERTEX_COLOR,
            linestyle="none",
        )
    if warning_message:
        axis.set_title(f"{camera_id} | {warning_message}", color=WARNING_COLOR)
    else:
        axis.set_title(f"{camera_id} | {len(points)} verteks | {PICKING_INSTRUCTION}", color=TITLE_COLOR)
    axis.figure.canvas.draw_idle()


def disable_default_keymap(figure: Figure) -> None:
    key_press_handler_id = getattr(figure.canvas.manager, "key_press_handler_id", None)
    if key_press_handler_id is not None:
        figure.canvas.mpl_disconnect(key_press_handler_id)


def pick_polygon_for_camera(
    frame_path: Path, camera_id: str, existing_polygon: list[list[int]] | None
) -> tuple[bool, list[list[int]] | None]:
    with Image.open(frame_path) as frame_image:
        frame_array = np.array(frame_image.convert("RGB"))
    points: list[list[float]] = [list(point) for point in existing_polygon] if existing_polygon else []
    warning_message = ""
    accepted = False
    started_new_polygon = not bool(existing_polygon)

    figure, axis = plt.subplots(figsize=FIGURE_SIZE_INCHES)

    def redraw() -> None:
        draw_picking_state(axis, frame_array, camera_id, points, warning_message)

    def handle_button_press(event) -> None:
        nonlocal warning_message, started_new_polygon
        if event.inaxes is not axis or event.xdata is None or event.ydata is None:
            return
        if event.button == LEFT_MOUSE_BUTTON:
            if not started_new_polygon:
                points.clear()
                started_new_polygon = True
            points.append([event.xdata, event.ydata])
        elif event.button == RIGHT_MOUSE_BUTTON and points:
            points.pop()
        else:
            return
        warning_message = ""
        redraw()

    def handle_key_press(event) -> None:
        nonlocal warning_message, accepted
        if event.key in UNDO_KEY_NAMES:
            if points:
                points.pop()
            warning_message = ""
            redraw()
            return
        if event.key == CANCEL_KEY_NAME:
            points.clear()
            plt.close(figure)
            return
        if event.key != ACCEPT_KEY_NAME:
            return
        # Enter selalu menerima dan menyimpan titik yang sedang digambar.
        # Validasi ROI tidak boleh menghalangi proses pinpoint manual.
        accepted = True
        plt.close(figure)

    figure.canvas.mpl_connect("button_press_event", handle_button_press)
    figure.canvas.mpl_connect("key_press_event", handle_key_press)
    disable_default_keymap(figure)
    redraw()
    plt.show(block=True)
    polygon = [[int(round(point[0])), int(round(point[1]))] for point in points]
    if len(polygon) < MINIMUM_POLYGON_POINTS:
        return False, None
    return accepted, polygon


def ensure_interactive_backend() -> None:
    backend_name = matplotlib.get_backend().lower()
    has_inline_backend = INLINE_BACKEND_MARKER in backend_name
    if has_inline_backend or backend_name in NON_INTERACTIVE_BACKENDS:
        raise RuntimeError(
            f"Backend matplotlib '{backend_name}' tidak interaktif — picker butuh backend GUI "
            f"(mis. TkAgg/QtAgg). Jalankan dari terminal tanpa MPLBACKEND=Agg, "
            f"bukan dari dalam notebook ({NON_INTERACTIVE_BACKEND_HINT})."
        )


def main() -> None:
    ensure_interactive_backend()
    manifest_rows = load_manifest_rows(FRAMES_MANIFEST_PATH)
    frame_by_camera = representative_frame_by_camera(manifest_rows)
    roi_config = load_roi_config(ROI_CONFIG_PATH)
    for camera_id in sorted(frame_by_camera):
        existing_entry = roi_config.get(camera_id, {})
        accepted, polygon = pick_polygon_for_camera(
            frame_by_camera[camera_id], camera_id, existing_entry.get("polygon")
        )
        if polygon is None:
            print(f"{camera_id}: dilewati — ROI lama dipertahankan")
            continue
        roi_config[camera_id] = {**existing_entry, "polygon": polygon}
        status = "disimpan" if accepted else "disimpan dari jendela yang ditutup"
        print(f"{camera_id}: polygon {len(polygon)} verteks {status}")
    save_roi_config(ROI_CONFIG_PATH, roi_config)
    print(f"ROI config: {ROI_CONFIG_PATH}")


if __name__ == "__main__":
    main()
