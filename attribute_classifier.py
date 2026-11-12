"""Classifier supervised untuk kondisi kendaraan dan kelas pantulan cahaya."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

ATTRIBUTE_NAMES = ("standar", "tidak_standar", "pantulan_cahaya")
ATTRIBUTE_BY_ANNOTATION_CLASS = {
    "MobilStandar": 0,
    "MobilTidakStandar": 1,
    "MotorStandar": 0,
    "MotorTidakStandar": 1,
    "PantulanCahaya": 2,
}
VEHICLE_TYPE_BY_ANNOTATION_CLASS = {
    **{name: "mobil" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Mobil")},
    **{name: "motor" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Motor")},
}
FEATURE_VERSION = "pixel-hsv-v1"
FEATURE_SIZE = 35


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


def extract_pixel_features(image_path: Path | Image.Image, box_xyxy: list[float]) -> np.ndarray:
    image = np.asarray(image_path.convert("RGB") if isinstance(image_path, Image.Image) else Image.open(image_path).convert("RGB"))
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box_xyxy
    x1 = max(0, min(width - 1, int(round(x1))))
    y1 = max(0, min(height - 1, int(round(y1))))
    x2 = max(x1 + 1, min(width, int(round(x2))))
    y2 = max(y1 + 1, min(height, int(round(y2))))
    crop = image[y1:y2, x1:x2]
    hsv = _hsv_from_rgb(crop)
    value = hsv[..., 2]
    saturation = hsv[..., 1]
    bright = (value >= 240 / 255).astype(np.float32)
    colored = ((saturation >= 80 / 255) & (value >= 80 / 255)).astype(np.float32)
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


class AttributeFusionCNN(nn.Module):
    """ResNet18 image encoder fused with engineered HSV/color statistics."""

    def __init__(self, pretrained: bool = False, feature_size: int = FEATURE_SIZE) -> None:
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = models.resnet18(weights=weights)
        image_size = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.head = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(image_size + feature_size, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.20),
            nn.Linear(128, len(ATTRIBUTE_NAMES)),
        )

    def forward(self, images: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        return self.head(torch.cat((self.backbone(images), features), dim=1))


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


def save_classifier(path: Path, model: nn.Module, mean: np.ndarray, std: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "metadata": metadata}, path)


def load_classifier(path: Path, device: str = "cpu") -> tuple[AttributeFusionCNN, np.ndarray, np.ndarray, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    metadata = payload["metadata"]
    model_type = metadata.get("model_type", "mlp")
    if model_type != "resnet18_hsv_fusion":
        raise ValueError(
            f"Checkpoint classifier tidak kompatibel: model_type={model_type!r}; "
            "latih ulang dengan `python builders/train_attribute_classifier.py`."
        )
    model = AttributeFusionCNN(pretrained=False)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), metadata


def decode_attributes(logits: np.ndarray, thresholds: dict[str, float] | None = None) -> dict[str, bool]:
    del thresholds  # Multiclass output uses one argmax decision, not independent sigmoid cutoffs.
    predicted_index = int(np.argmax(np.asarray(logits, dtype=np.float64)))
    return {name: index == predicted_index for index, name in enumerate(ATTRIBUTE_NAMES)}
