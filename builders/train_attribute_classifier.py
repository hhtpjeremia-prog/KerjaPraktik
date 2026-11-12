#!/usr/bin/env python3
"""Latih ResNet18 supervised untuk Standar/TidakStandar/PantulanCahaya."""

from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attribute_classifier import (  # noqa: E402
    ATTRIBUTE_BY_ANNOTATION_CLASS,
    ATTRIBUTE_NAMES,
    AttributeFusionCNN,
    FEATURE_SIZE,
    IMAGE_TRANSFORM,
    extract_image_crop,
    extract_pixel_features,
    save_classifier,
)

ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUTPUT = ROOT / "output"
ANNOTATION = OUTPUT / "annotation"
MODEL_DIR = OUTPUT / "attribute_classifier"
HISTORY_PATH = MODEL_DIR / "training_history.csv"
PLOT_PATH = MODEL_DIR / "training_curves.png"
METRICS_PATH = MODEL_DIR / "metrics.json"
SEED = 20261102
EPOCHS = int(os.environ.get("ATTRIBUTE_EPOCHS", "40"))
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WARMUP_HEAD_EPOCHS = 5
MIN_TRAIN_PER_CLASS = 20
MIN_VAL_PER_CLASS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TRAIN_AUGMENT = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.08, hue=0.0),
])
@dataclass(frozen=True)
class Example:
    image_path: Path
    box: tuple[float, float, float, float]
    label: int
    split: str


class CropDataset(Dataset):
    def __init__(self, examples: list[Example], training: bool = False) -> None:
        self.examples = examples
        self.training = training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            crop = extract_image_crop(image, list(example.box))
        # Numerical photometry uses the clean crop in both train and inference.
        # Color jitter augments only the CNN image branch.
        features = extract_pixel_features(crop, [0, 0, crop.width, crop.height])
        if self.training:
            crop = TRAIN_AUGMENT(crop)
        return IMAGE_TRANSFORM(crop), torch.from_numpy(features), torch.tensor(example.label, dtype=torch.long)


def resolve_image_path(path_value: str | Path) -> Path:
    image_path = Path(path_value)
    if image_path.exists():
        return image_path
    if "output" in image_path.parts:
        candidate = ROOT / Path(*image_path.parts[image_path.parts.index("output"):])
        if candidate.exists():
            return candidate
    candidate = ROOT / image_path
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"gambar anotasi tidak ditemukan: {path_value}")


def load_examples() -> list[Example]:
    examples: list[Example] = []
    sources = [
        (ANNOTATION / "sample_manifest.json", ANNOTATION / "annotations.json", False),
        (OUTPUT / "ground_truth" / "sample_manifest.json", OUTPUT / "ground_truth" / "annotations.json", True),
    ]
    for manifest_path, store_path, is_test_gt in sources:
        if not manifest_path.exists() or not store_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = json.loads(store_path.read_text(encoding="utf-8"))
        for record in manifest.get("frames", []):
            entry = store.get("frames", {}).get(record["frame_stem"])
            if not entry or not entry.get("reviewed"):
                continue
            image_path = resolve_image_path(record["image_path"])
            split = "test" if is_test_gt else record.get("split", store.get("split_by_camera", {}).get(record["camera_id"], "unknown"))
            for box in entry.get("boxes", []):
                class_name = box.get("class_name")
                if box.get("deleted") or class_name not in ATTRIBUTE_BY_ANNOTATION_CLASS:
                    continue
                examples.append(Example(image_path, tuple(float(value) for value in box["box"]), int(ATTRIBUTE_BY_ANNOTATION_CLASS[class_name]), split))
    return examples


def set_backbone_trainable(model: AttributeFusionCNN, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: float) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=1)
    truth = labels.reshape(-1).long()
    prediction = probabilities.argmax(dim=1)
    class_precision, class_recall, class_f1, class_support = [], [], [], []
    for class_index in range(len(ATTRIBUTE_NAMES)):
        true_positive = int(((prediction == class_index) & (truth == class_index)).sum())
        false_positive = int(((prediction == class_index) & (truth != class_index)).sum())
        false_negative = int(((prediction != class_index) & (truth == class_index)).sum())
        precision_value = true_positive / max(true_positive + false_positive, 1)
        recall_value = true_positive / max(true_positive + false_negative, 1)
        class_precision.append(precision_value)
        class_recall.append(recall_value)
        class_f1.append(2 * precision_value * recall_value / max(precision_value + recall_value, 1e-9))
        class_support.append(int((truth == class_index).sum()))
    one_hot_truth = torch.nn.functional.one_hot(truth, num_classes=len(ATTRIBUTE_NAMES)).float()
    brier_mse = torch.mean((probabilities - one_hot_truth) ** 2)
    return {
        "loss": float(loss),
        "brier_mse": float(brier_mse),
        "brier_rmse": float(torch.sqrt(brier_mse)),
        "accuracy": float((prediction == truth).float().mean()),
        "precision_macro": float(np.mean(class_precision)),
        "recall_macro": float(np.mean(class_recall)),
        "f1_macro": float(np.mean(class_f1)),
        **{f"precision_{name}": class_precision[index] for index, name in enumerate(ATTRIBUTE_NAMES)},
        **{f"recall_{name}": class_recall[index] for index, name in enumerate(ATTRIBUTE_NAMES)},
        **{f"support_{name}": class_support[index] for index, name in enumerate(ATTRIBUTE_NAMES)},
        "support": len(truth),
    }


def evaluate(model: AttributeFusionCNN, loader: DataLoader | None, loss_function: nn.Module, feature_mean: np.ndarray, feature_std: np.ndarray) -> dict[str, float] | None:
    if loader is None:
        return None
    model.eval()
    logits_rows, label_rows = [], []
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for images, features, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = ((features.numpy() - feature_mean) / feature_std).astype(np.float32)
            logits = model(images, torch.from_numpy(features).to(DEVICE))
            loss = loss_function(logits, labels)
            total_loss += float(loss) * len(labels)
            total_items += len(labels)
            logits_rows.append(logits.cpu())
            label_rows.append(labels.cpu())
    if not logits_rows:
        return None
    return calculate_metrics(torch.cat(logits_rows), torch.cat(label_rows), total_loss / max(total_items, 1))


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    examples = load_examples()
    train_examples = [example for example in examples if example.split == "train"]
    val_examples = [example for example in examples if example.split == "val"]
    test_examples = [example for example in examples if example.split == "test"]
    if not train_examples or not val_examples:
        raise RuntimeError("Classifier membutuhkan contoh train dan val.")
    train_class_counts = {name: sum(example.label == index for example in train_examples) for index, name in enumerate(ATTRIBUTE_NAMES)}
    val_class_counts = {name: sum(example.label == index for example in val_examples) for index, name in enumerate(ATTRIBUTE_NAMES)}
    insufficient_train = {name: count for name, count in train_class_counts.items() if count < MIN_TRAIN_PER_CLASS}
    insufficient_val = {name: count for name, count in val_class_counts.items() if count < MIN_VAL_PER_CLASS}
    if insufficient_train or insufficient_val:
        raise RuntimeError(
            f"Sampel classifier kurang per kelas. Train: {insufficient_train}; validation: {insufficient_val}."
        )

    train_loader = DataLoader(CropDataset(train_examples, training=True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val_examples), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(CropDataset(test_examples), batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if test_examples else None
    raw_features = []
    for example in train_examples:
        with Image.open(example.image_path) as image:
            crop = extract_image_crop(image, list(example.box))
            raw_features.append(extract_pixel_features(crop, [0, 0, crop.width, crop.height]))
    feature_mean = np.mean(raw_features, axis=0).astype(np.float32)
    feature_std = np.maximum(np.std(raw_features, axis=0), 1e-6).astype(np.float32)
    try:
        model = AttributeFusionCNN(pretrained=True).to(DEVICE)
        pretrained_source = "ImageNet"
    except Exception as error:
        print(f"Peringatan: bobot ImageNet tidak tersedia ({error}); memakai ResNet18 tanpa bobot awal.")
        model = AttributeFusionCNN(pretrained=False).to(DEVICE)
        pretrained_source = "random_init_fallback"
    set_backbone_trainable(model, False)
    class_counts = train_class_counts
    class_weights = torch.tensor(
        [len(train_examples) / (len(ATTRIBUTE_NAMES) * max(train_class_counts[name], 1)) for name in ATTRIBUTE_NAMES],
        dtype=torch.float32,
        device=DEVICE,
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    print(json.dumps({
        "model": "ResNet18",
        "pretrained_source": pretrained_source,
        "device": DEVICE,
        "train_class_counts": class_counts,
        "loss": "weighted CrossEntropyLoss",
        "class_weights": {name: float(class_weights[index].item()) for index, name in enumerate(ATTRIBUTE_NAMES)},
    }, indent=2, ensure_ascii=False))
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-3)
    best_state, best_val_loss = None, float("inf")
    history_rows = []
    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_HEAD_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.25, weight_decay=1e-3)
        model.train()
        if epoch <= WARMUP_HEAD_EPOCHS:
            # Freeze pretrained BatchNorm running statistics while only training the head.
            model.backbone.eval()
            model.head.train()
        train_logits, train_labels = [], []
        train_loss_total, train_items = 0.0, 0
        for images, features, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            features = ((features.numpy() - feature_mean) / feature_std).astype(np.float32)
            features = torch.from_numpy(features).to(DEVICE)
            optimizer.zero_grad()
            logits = model(images, features)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += float(loss) * len(labels)
            train_items += len(labels)
            train_logits.append(logits.detach().cpu())
            train_labels.append(labels.cpu())
        train_metrics = calculate_metrics(torch.cat(train_logits), torch.cat(train_labels), train_loss_total / max(train_items, 1))
        val_metrics = evaluate(model, val_loader, loss_function, feature_mean, feature_std)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history_rows.append(row)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Tidak ada checkpoint valid dari training CNN.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, loss_function, feature_mean, feature_std)
    history_frame = pd.DataFrame(history_rows)
    if test_metrics:
        for key, value in test_metrics.items():
            history_frame[f"test_{key}"] = value
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    history_frame.to_csv(HISTORY_PATH, index=False)
    metadata = {
        "model_type": "resnet18_hsv_fusion",
        "numeric_feature_version": "pixel-hsv-v1",
        "numeric_feature_size": FEATURE_SIZE,
        "input_size": [224, 224],
        "crop_resize": "preserve_aspect_ratio_longest_side_then_mean_padding",
        "pixel_normalization": {
            "to_tensor_range": [0.0, 1.0],
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "pretrained_source": pretrained_source,
        "attribute_names": ATTRIBUTE_NAMES,
        "train_examples": len(train_examples),
        "val_examples": len(val_examples),
        "test_examples": len(test_examples),
        "class_counts": class_counts,
        "class_counts_train": train_class_counts,
        "class_counts_val": val_class_counts,
        "loss": "weighted CrossEntropyLoss",
        "best_val_loss": best_val_loss,
        "test_metrics": test_metrics,
        "decision_rule": "softmax_argmax",
        "history_path": str(HISTORY_PATH),
    }
    save_classifier(MODEL_DIR / "best.pt", model, feature_mean, feature_std, metadata)
    METRICS_PATH.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(history_frame["epoch"], history_frame["train_loss"], label="train")
        axes[0].plot(history_frame["epoch"], history_frame["val_loss"], label="val")
        axes[0].set_title("Loss per epoch")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(history_frame["epoch"], history_frame["train_f1_macro"], label="train")
        axes[1].plot(history_frame["epoch"], history_frame["val_f1_macro"], label="val")
        axes[1].set_title("Macro F1 per epoch")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(PLOT_PATH, dpi=140)
        plt.close(figure)
    except Exception as error:
        print(f"Peringatan: plot CNN gagal dibuat: {error}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
