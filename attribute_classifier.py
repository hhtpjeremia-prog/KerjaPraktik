"""Classifier supervised untuk kondisi kendaraan dan kelas pantulan cahaya."""

from __future__ import annotations

from pathlib import Path
import hashlib
import json

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ATTRIBUTE_NAMES = (
    "MobilStandar",
    "MobilTidakStandar",
    "MotorStandar",
    "MotorTidakStandar",
    "PantulanCahaya",
)
ATTRIBUTE_BY_ANNOTATION_CLASS = {name: index for index, name in enumerate(ATTRIBUTE_NAMES)}
VEHICLE_TYPE_BY_ANNOTATION_CLASS = {
    **{name: "mobil" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Mobil")},
    **{name: "motor" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Motor")},
}
FEATURE_VERSION = "pixel-hsv-v1"
FEATURE_SIZE = 35
MASK_FEATURE_VERSION = "sam2-masked-hsv-shape-v2-empty-zero"
MASK_FEATURE_SIZE = 42
MODEL_NUMERIC_FEATURE_SIZE = FEATURE_SIZE + MASK_FEATURE_SIZE
MASK_SHAPE_FEATURE_OFFSET = FEATURE_SIZE * 2
MASK_SHAPE_FEATURE_INDEX = {
    "area_fraction": 0,
    "width_fraction": 1,
    "height_fraction": 2,
    "centroid_x": 3,
    "centroid_y": 4,
    "compactness": 5,
    "valid_mask": 6,
}
MASK_SHAPE_FEATURE_NAMES = tuple(MASK_SHAPE_FEATURE_INDEX)
MASK_CENTROID_X_FEATURE_INDEX = MASK_SHAPE_FEATURE_OFFSET + MASK_SHAPE_FEATURE_INDEX["centroid_x"]
MASK_VALID_FEATURE_INDEX = MASK_SHAPE_FEATURE_OFFSET + MASK_SHAPE_FEATURE_INDEX["valid_mask"]
EMEANS_EPOCHS = 5
EMEANS_SEED = 20261116
BRIGHT_VALUE_THRESHOLD = 240
COLORED_SATURATION_MIN = 80
COLORED_VALUE_MIN = 80


def attributes_for_label(class_name: str) -> int:
    if class_name not in ATTRIBUTE_BY_ANNOTATION_CLASS:
        raise ValueError(f"Kelas bukan kendaraan beratribut: {class_name}")
    return ATTRIBUTE_BY_ANNOTATION_CLASS[class_name]


def _hsv_from_rgb(rgb: np.ndarray) -> np.ndarray:
    rgb = rgb.astype(np.float32) / 255.0
    maximum = rgb.max(axis=2)
    minimum = rgb.min(axis=2)
    delta = maximum - minimum
    hue = np.zeros_like(maximum)
    nonzero = delta > 1e-6
    red, green, blue = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    red_case = nonzero & (maximum == red)
    green_case = nonzero & (maximum == green)
    blue_case = nonzero & (maximum == blue)
    hue[red_case] = ((green[red_case] - blue[red_case]) / delta[red_case]) % 6
    hue[green_case] = (blue[green_case] - red[green_case]) / delta[green_case] + 2
    hue[blue_case] = (red[blue_case] - green[blue_case]) / delta[blue_case] + 4
    hue /= 6.0
    saturation = np.divide(delta, maximum, out=np.zeros_like(delta), where=maximum > 1e-6)
    return np.stack((hue, saturation, maximum), axis=2)


def _pixel_features_from_rgb_pixels(rgb_pixels: np.ndarray) -> np.ndarray:
    if rgb_pixels.size == 0:
        raise ValueError("fitur pixel membutuhkan setidaknya satu pixel RGB")
    rgb_image = np.asarray(rgb_pixels, dtype=np.uint8).reshape(-1, 1, 3)
    hsv = _hsv_from_rgb(rgb_image)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    bright = (value >= BRIGHT_VALUE_THRESHOLD / 255).astype(np.float32)
    colored = (
        (saturation >= COLORED_SATURATION_MIN / 255)
        & (value >= COLORED_VALUE_MIN / 255)
    ).astype(np.float32)
    hue = hsv[..., 0][colored.astype(bool)]
    hue_counts, _ = np.histogram(hue, bins=16, range=(0, 1))
    hue_histogram = hue_counts / max(len(hue), 1)
    features = np.concatenate([
        hsv.mean(axis=(0, 1)),
        hsv.std(axis=(0, 1)),
        np.quantile(hsv.reshape(-1, 3), [0.1, 0.5, 0.9], axis=0).reshape(-1),
        [bright.mean(), colored.mean(), float(value.max()), float(saturation.max())],
        hue_histogram.astype(np.float32),
    ]).astype(np.float32)
    if len(features) != FEATURE_SIZE:
        raise AssertionError(f"feature size berubah: {len(features)} != {FEATURE_SIZE}")
    return features


def _load_rgb(image_path: Path | Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image_path, np.ndarray):
        image = image_path
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"citra RGB harus berukuran HxWx3; didapat {image.shape}")
        return image.astype(np.uint8, copy=False)
    if isinstance(image_path, Image.Image):
        return np.asarray(image_path.convert("RGB"))
    with Image.open(image_path) as source:
        return np.asarray(source.convert("RGB"))


def _clipped_box(box_xyxy: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return x1, y1, x2, y2


def extract_pixel_features(image_path: Path | Image.Image | np.ndarray, box_xyxy: list[float]) -> np.ndarray:
    image = _load_rgb(image_path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clipped_box(box_xyxy, width, height)
    return _pixel_features_from_rgb_pixels(image[y1:y2, x1:x2].reshape(-1, 3))


def mask_cache_key(frame_stem: str, box_xyxy: list[float]) -> str:
    """Stable identity shared by NB02e training masks and the CNN dataset loader."""
    canonical = json.dumps(
        [str(frame_stem), [round(float(value), 2) for value in box_xyxy]],
        separators=(",", ":"),
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:16]
    return f"{frame_stem}__{digest}"


def extract_mask_aware_features(
    image_path: Path | Image.Image | np.ndarray,
    box_xyxy: list[float],
    mask_source: Path | np.ndarray,
) -> np.ndarray:
    """Combine bbox photometry, SAM2-foreground photometry, and mask shape."""
    image = _load_rgb(image_path)
    height, width = image.shape[:2]
    x1, y1, x2, y2 = _clipped_box(box_xyxy, width, height)
    crop = image[y1:y2, x1:x2]
    if isinstance(mask_source, np.ndarray):
        full_mask = np.asarray(mask_source)
    else:
        with Image.open(mask_source) as mask_image:
            full_mask = np.asarray(mask_image.convert("L"))
    if full_mask.shape != (height, width):
        raise ValueError(
            f"ukuran mask SAM2 {full_mask.shape} tidak sama dengan citra {(height, width)}"
        )
    crop_mask = full_mask[y1:y2, x1:x2] > 0
    valid_mask = bool(crop_mask.any())
    masked_photometry = (
        _pixel_features_from_rgb_pixels(crop[crop_mask])
        if valid_mask
        else np.zeros(FEATURE_SIZE, dtype=np.float32)
    )

    shape_features = np.zeros(MASK_FEATURE_SIZE - FEATURE_SIZE, dtype=np.float32)
    shape_features[MASK_SHAPE_FEATURE_INDEX["area_fraction"]] = float(crop_mask.mean())
    shape_features[MASK_SHAPE_FEATURE_INDEX["valid_mask"]] = float(valid_mask)
    if valid_mask:
        ys, xs = np.nonzero(crop_mask)
        crop_height, crop_width = crop_mask.shape
        min_x, max_x = int(xs.min()), int(xs.max())
        min_y, max_y = int(ys.min()), int(ys.max())
        shape_width = max_x - min_x + 1
        shape_height = max_y - min_y + 1
        shape_features[MASK_SHAPE_FEATURE_INDEX["width_fraction"]] = shape_width / max(crop_width, 1)
        shape_features[MASK_SHAPE_FEATURE_INDEX["height_fraction"]] = shape_height / max(crop_height, 1)
        shape_features[MASK_SHAPE_FEATURE_INDEX["centroid_x"]] = float(xs.mean() / max(crop_width - 1, 1))
        shape_features[MASK_SHAPE_FEATURE_INDEX["centroid_y"]] = float(ys.mean() / max(crop_height - 1, 1))
        shape_features[MASK_SHAPE_FEATURE_INDEX["compactness"]] = float(crop_mask.sum() / max(shape_width * shape_height, 1))
    bbox_photometry = extract_pixel_features(image, [x1, y1, x2, y2])
    features = np.concatenate([bbox_photometry, masked_photometry, shape_features]).astype(np.float32)
    if len(features) != MODEL_NUMERIC_FEATURE_SIZE:
        raise AssertionError(
            f"feature mask-aware berubah: {len(features)} != {MODEL_NUMERIC_FEATURE_SIZE}"
        )
    return features


class AttributeFusionCNN(nn.Module):
    """ResNet18 fused with mask-aware numeric features and e-means cluster encoding."""

    def __init__(
        self,
        cluster_count: int,
        pretrained: bool = False,
        feature_size: int = MODEL_NUMERIC_FEATURE_SIZE,
    ) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        image_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.feature_size = int(feature_size)
        self.cluster_count = int(cluster_count)
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(image_size + self.feature_size + self.cluster_count, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, len(ATTRIBUTE_NAMES)),
        )

    def forward(
        self,
        images: torch.Tensor,
        features: torch.Tensor,
        cluster_features: torch.Tensor,
    ) -> torch.Tensor:
        return self.head(torch.cat((self.backbone(images), features, cluster_features), dim=1))


class ResizeLongestSideAndPad:
    """Keep the full bbox crop and aspect ratio, then pad to a square canvas."""

    def __init__(self, size: int = 224) -> None:
        self.size = size
        # RGB ImageNet mean, so padding becomes zero after Normalize below.
        self.fill = (124, 116, 104)

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"crop has invalid dimensions: {image.size}")
        scale = self.size / max(width, height)
        resized_width = max(1, min(self.size, round(width * scale)))
        resized_height = max(1, min(self.size, round(height * scale)))
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        resized = image.resize((resized_width, resized_height), resampling)
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        offset = ((self.size - resized_width) // 2, (self.size - resized_height) // 2)
        canvas.paste(resized, offset)
        return canvas


IMAGE_TRANSFORM = transforms.Compose([
    ResizeLongestSideAndPad(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def extract_image_crop(image_path: Path | Image.Image, box_xyxy: list[float]) -> Image.Image:
    image = image_path.convert("RGB") if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB")
    width, height = image.size
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    return image.crop((x1, y1, x2, y2))


def save_classifier(
    path: Path,
    model: nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    cluster_centroids: np.ndarray,
    metadata: dict,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(), "mean": mean, "std": std,
        "cluster_centroids": cluster_centroids, "metadata": metadata,
    }, path)


def load_classifier(
    path: Path, device: str = "cpu"
) -> tuple[AttributeFusionCNN, np.ndarray, np.ndarray, np.ndarray, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    model_type = metadata.get("model_type", "mlp")
    if model_type != "resnet18_sam2_emeans_fusion_5class":
        raise ValueError(
            f"Checkpoint classifier tidak kompatibel: model_type={model_type!r}; "
            "latih ulang NB04b dengan fitur mask SAM2 + e-means lima kelas."
        )
    model = AttributeFusionCNN(
        pretrained=False,
        feature_size=int(metadata["numeric_feature_size"]),
        cluster_count=int(metadata["e_means"]["n_clusters"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return (
        model,
        np.asarray(payload["mean"], dtype=np.float32),
        np.asarray(payload["std"], dtype=np.float32),
        np.asarray(payload["cluster_centroids"], dtype=np.float32),
        metadata,
    )


def decode_attributes(logits: np.ndarray) -> dict[str, bool]:
    predicted_index = int(np.argmax(np.asarray(logits, dtype=np.float64)))
    return {name: index == predicted_index for index, name in enumerate(ATTRIBUTE_NAMES)}
