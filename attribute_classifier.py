"""Multi-label classifier untuk atribut cahaya pada crop kendaraan.

Detektor hanya bertugas mencari kendaraan. Modul ini membaca piksel crop kendaraan
dan memprediksi dua atribut yang dapat hadir bersamaan: silau dan berwarna.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn

ATTRIBUTE_NAMES = ("silau", "berwarna")
ATTRIBUTE_BY_ANNOTATION_CLASS = {
    "MobilNormal": (0, 0),
    "MobilSilau": (1, 0),
    "MobilBerwarna": (0, 1),
    "MobilSilauBerwarna": (1, 1),
    "MotorNormal": (0, 0),
    "MotorSilau": (1, 0),
    "MotorBerwarna": (0, 1),
    "MotorSilauBerwarna": (1, 1),
}
VEHICLE_TYPE_BY_ANNOTATION_CLASS = {
    **{name: "mobil" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Mobil")},
    **{name: "motor" for name in ATTRIBUTE_BY_ANNOTATION_CLASS if name.startswith("Motor")},
}
FEATURE_VERSION = "pixel-hsv-v1"
FEATURE_SIZE = 35


def attributes_for_label(class_name: str) -> tuple[int, int]:
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


class AttributeMLP(nn.Module):
    def __init__(self, input_size: int = FEATURE_SIZE) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Dropout(0.20),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, len(ATTRIBUTE_NAMES)),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


def save_classifier(path: Path, model: AttributeMLP, mean: np.ndarray, std: np.ndarray, metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "metadata": metadata}, path)


def load_classifier(path: Path, device: str = "cpu") -> tuple[AttributeMLP, np.ndarray, np.ndarray, dict]:
    payload = torch.load(path, map_location=device, weights_only=False)
    model = AttributeMLP()
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, np.asarray(payload["mean"], dtype=np.float32), np.asarray(payload["std"], dtype=np.float32), payload["metadata"]


def decode_attributes(logits: np.ndarray, thresholds: dict[str, float] | None = None) -> dict[str, bool]:
    thresholds = thresholds or {name: 0.5 for name in ATTRIBUTE_NAMES}
    probabilities = 1.0 / (1.0 + np.exp(-logits))
    return {name: bool(probabilities[index] >= thresholds.get(name, 0.5)) for index, name in enumerate(ATTRIBUTE_NAMES)}
