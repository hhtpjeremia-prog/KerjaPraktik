import json
import time
from pathlib import Path

import torch
import torchvision.transforms.functional as vision_functional
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2

CHUNK_DIR = Path(__file__).resolve().parent.parent / "output" / "nb02" / "chunks"
FRAME_SAMPLE_COUNT = 6
BOX_SCORE_THRESHOLD = 0.4
DETECTION_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
CUDA_WARMUP_ITERATIONS = 1

weights = FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1
categories = weights.meta["categories"]
category_index = {name: index for index, name in enumerate(categories)}
print("mapping kategori:", {name: category_index.get(name) for name in ("car", "motorcycle", "bus", "truck")}, flush=True)

first_chunk_path = sorted(CHUNK_DIR.glob("chunk_*.jsonl"))[0]
frame_paths = []
with first_chunk_path.open(encoding="utf-8") as chunk_file:
    for chunk_line in chunk_file:
        frame_paths.append(json.loads(chunk_line)["frame_path"])
        if len(frame_paths) >= FRAME_SAMPLE_COUNT:
            break

sample_images = []
for frame_path in frame_paths:
    frame_image = Image.open(frame_path).convert("RGB")
    frame_image.thumbnail((640, 640))
    sample_images.append(frame_image)
print(f"ukuran sample: {[image.size for image in sample_images]}", flush=True)

image_tensors = [vision_functional.to_tensor(image) for image in sample_images]

cpu_model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=BOX_SCORE_THRESHOLD)
cpu_model.eval()
with torch.inference_mode():
    for tensor in image_tensors[:2]:
        cpu_model([tensor])
    cpu_start = time.perf_counter()
    for tensor in image_tensors:
        cpu_model([tensor])
    cpu_seconds_per_frame = (time.perf_counter() - cpu_start) / len(image_tensors)
print(f"CPU detik/frame: {cpu_seconds_per_frame:.3f}", flush=True)

with torch.inference_mode():
    cpu_predictions = [cpu_model([tensor])[0] for tensor in image_tensors]
for frame_index, prediction in enumerate(cpu_predictions):
    detected_names = [
        categories[int(label)]
        for label, score in zip(prediction["labels"], prediction["scores"])
        if categories[int(label)] in ("car", "motorcycle", "bus", "truck")
    ]
    print(f"CPU frame {frame_index}: kendaraan={detected_names}", flush=True)

if not torch.cuda.is_available():
    print("CUDA tidak tersedia — benchmark GPU dilewati", flush=True)
    raise SystemExit(0)

from transformers.models.grounding_dino import GroundingDinoForObjectDetection, GroundingDinoProcessor

dino_processor = GroundingDinoProcessor.from_pretrained(DETECTION_MODEL_ID)
dino_model = GroundingDinoForObjectDetection.from_pretrained(DETECTION_MODEL_ID).to("cuda").eval()
gpu_model = fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=BOX_SCORE_THRESHOLD).to("cuda").eval()
gpu_tensors = [tensor.to("cuda") for tensor in image_tensors]

torch.cuda.reset_peak_memory_stats()
with torch.inference_mode():
    for _ in range(CUDA_WARMUP_ITERATIONS):
        for image_tensor in gpu_tensors:
            dino_processor(images=[sample_images[0]], text="car headlight.", return_tensors="pt").to("cuda")
            gpu_model([image_tensor])
    torch.cuda.synchronize()
    gpu_start = time.perf_counter()
    for image_tensor in gpu_tensors:
        inputs = dino_processor(images=[sample_images[0]], text="car headlight.", return_tensors="pt").to("cuda")
        dino_model(**inputs)
        gpu_model([image_tensor])
    torch.cuda.synchronize()
    gpu_seconds_per_frame = (time.perf_counter() - gpu_start) / len(gpu_tensors)

weights_bytes = sum(parameter.numel() * parameter.element_size() for parameter in gpu_model.parameters())
dino_bytes = sum(parameter.numel() * parameter.element_size() for parameter in dino_model.parameters())
peak_gib = torch.cuda.max_memory_allocated() / 1024**3
print(f"CUDA detik/frame (DINO 1-pass + FRCNN): {gpu_seconds_per_frame:.3f}", flush=True)
print(f"VRAM peak: {peak_gib:.2f} GiB (FRCNN {weights_bytes / 1024**3:.2f} GiB + DINO {dino_bytes / 1024**3:.2f} GiB weights)", flush=True)

with torch.inference_mode():
    gpu_predictions = [gpu_model([tensor])[0] for tensor in gpu_tensors]
for frame_index, prediction in enumerate(gpu_predictions):
    detected_names = [
        categories[int(label)]
        for label in prediction["labels"]
        if categories[int(label)] in ("car", "motorcycle", "bus", "truck")
    ]
    print(f"CUDA frame {frame_index}: kendaraan={detected_names}", flush=True)
