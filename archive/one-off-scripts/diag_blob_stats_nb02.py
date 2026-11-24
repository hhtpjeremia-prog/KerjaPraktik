import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

RISET_ROOT = Path(__file__).resolve().parent.parent
BUILDERS_DIR = RISET_ROOT / "builders"
sys.path.insert(0, str(BUILDERS_DIR))
import selfcheck_nb02 as selfcheck

module = selfcheck.load_module_from_path(selfcheck.NB02_BUILDER_PATH)
constant_names = tuple(selfcheck.EXTRACTED_CONSTANT_NAMES) + ("MAX_RESIZE_SIDE",)
config_constants, _ = selfcheck.extract_named_sources(module.CONFIG_SOURCE, (), constant_names)
_, dino_functions = selfcheck.extract_named_sources(module.DINO_SOURCE, selfcheck.EXTRACTED_FUNCTION_NAMES, ())
namespace = {"np": np, "cv2": cv2, "Image": Image, "ImageDraw": ImageDraw, "Path": Path}
exec("\n\n".join(config_constants + dino_functions), namespace)

crop_and_fit_frame = namespace["crop_and_fit_frame"]
max_resize_side = namespace["MAX_RESIZE_SIDE"]
bright_threshold = namespace["BRIGHT_BLOB_THRESHOLD"]

roi_config = json.loads((RISET_ROOT / "output" / "roi" / "roi_config.json").read_text(encoding="utf-8"))
chunk_paths = sorted((RISET_ROOT / "output" / "nb02" / "chunks").glob("chunk_*.jsonl"))[:4]


def blob_areas(image, box, margin, threshold):
    image_width, image_height = image.size
    x1 = max(0, int(box[0]) - margin)
    y1 = max(0, int(box[1]) - margin)
    x2 = min(image_width, int(box[2]) + margin)
    y2 = min(image_height, int(box[3]) + margin)
    if x2 <= x1 or y2 <= y1:
        return []
    gray = np.array(image.crop((x1, y1, x2, y2)).convert("L"))
    mask = (gray >= threshold).astype(np.uint8)
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    return sorted([int(stats[index, cv2.CC_STAT_AREA]) for index in range(1, component_count)], reverse=True)


def summarize(values):
    if not values:
        return {}
    array = np.array(values, dtype=float)
    return {f"p{percentile}": round(float(np.percentile(array, percentile)), 2) for percentile in (10, 25, 50, 75, 90)}


for margin in (0, 40):
    multi_lamp_stats = {"count_ge4": [], "count_ge60": [], "count_ge150": [], "peer_ratio": [], "largest": []}
    single_lamp_stats = {"count_ge4": [], "count_ge60": [], "count_ge150": [], "peer_ratio": [], "largest": []}
    for chunk_path in chunk_paths:
        with chunk_path.open(encoding="utf-8") as chunk_file:
            for chunk_line in chunk_file:
                chunk_row = json.loads(chunk_line)
                roi_entry = roi_config.get(chunk_row["camera_id"], {})
                fitted_image = crop_and_fit_frame(Path(chunk_row["frame_path"]), roi_entry, max_resize_side)
                for detection in chunk_row["detections"]:
                    areas = blob_areas(fitted_image, detection["box"], margin, bright_threshold)
                    target = multi_lamp_stats if detection["lamp_count"] >= 2 else single_lamp_stats
                    target["count_ge4"].append(len(areas))
                    target["count_ge60"].append(sum(1 for area in areas if area >= 60))
                    target["count_ge150"].append(sum(1 for area in areas if area >= 150))
                    target["largest"].append(areas[0] if areas else 0)
                    peer = areas[1] / areas[0] if len(areas) >= 2 and areas[0] > 0 else 0.0
                    target["peer_ratio"].append(round(peer, 3))
    print(f"=== margin {margin} ===")
    print("grup >=2 lampu (mobil by geometri), n=%d:" % len(multi_lamp_stats["count_ge4"]))
    for key in ("count_ge4", "count_ge60", "count_ge150", "peer_ratio", "largest"):
        print(f"  {key}: {summarize(multi_lamp_stats[key])}")
    print("grup 1 lampu, n=%d:" % len(single_lamp_stats["count_ge4"]))
    for key in ("count_ge4", "count_ge60", "count_ge150", "peer_ratio", "largest"):
        print(f"  {key}: {summarize(single_lamp_stats[key])}")
    strong_peer_multi = sum(1 for area_count, peer in zip(multi_lamp_stats["count_ge60"], multi_lamp_stats["peer_ratio"]) if area_count >= 2 and peer >= 0.35)
    strong_peer_single = sum(1 for area_count, peer in zip(single_lamp_stats["count_ge60"], single_lamp_stats["peer_ratio"]) if area_count >= 2 and peer >= 0.35)
    print(f"  dua blob >=60px dgn peer ratio >=0.35 -> grup>=2 lampu: {strong_peer_multi}/{len(multi_lamp_stats['count_ge4'])}; "
          f"grup 1 lampu: {strong_peer_single}/{len(single_lamp_stats['count_ge4'])}")
