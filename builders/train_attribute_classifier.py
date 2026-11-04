#!/usr/bin/env python3
"""Latih classifier multi-label berbasis fitur piksel HSV dari gold annotations."""

import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attribute_classifier import (
    ATTRIBUTE_NAMES,
    ATTRIBUTE_BY_ANNOTATION_CLASS,
    FEATURE_SIZE,
    FEATURE_VERSION,
    AttributeMLP,
    extract_pixel_features,
    save_classifier,
)

ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUTPUT = ROOT / "output"
ANNOTATION = OUTPUT / "annotation"
MODEL_DIR = OUTPUT / "attribute_classifier"
SEED = 20261102
EPOCHS = 150
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
MIN_TRAIN_PER_ATTRIBUTE = 40
MIN_VAL_PER_ATTRIBUTE = 10


def load_examples() -> tuple[np.ndarray, np.ndarray, list[str]]:
    manifest = json.loads((ANNOTATION / "sample_manifest.json").read_text(encoding="utf-8"))
    store = json.loads((ANNOTATION / "annotations.json").read_text(encoding="utf-8"))
    features, labels, splits = [], [], []
    for record in manifest["frames"]:
        entry = store["frames"].get(record["frame_stem"])
        if not entry or not entry.get("reviewed"):
            continue
        image_path = Path(record["image_path"])
        if not image_path.exists():
            candidate = ROOT / Path(*image_path.parts[image_path.parts.index("output"):]) if "output" in image_path.parts else ROOT / image_path
            if candidate.exists():
                image_path = candidate
            else:
                raise FileNotFoundError(
                    f"{image_path} tidak ada. Jalankan ulang NB02b agar sample_manifest memuat path yang benar."
                )
        split = record.get("split", store.get("split_by_camera", {}).get(record["camera_id"], "unknown"))
        for box in entry["boxes"]:
            if box.get("deleted") or box["class_name"] not in ATTRIBUTE_BY_ANNOTATION_CLASS:
                continue
            features.append(extract_pixel_features(image_path, box["box"]))
            labels.append(ATTRIBUTE_BY_ANNOTATION_CLASS[box["class_name"]])
            splits.append(split)
    return np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.float32), splits


def main() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    features, labels, splits = load_examples()
    train_mask = np.asarray([split == "train" for split in splits])
    val_mask = np.asarray([split == "val" for split in splits])
    if not train_mask.any() or not val_mask.any():
        raise RuntimeError("Classifier membutuhkan gold box pada train dan val; val saat ini kosong.")
    train_counts = labels[train_mask].sum(axis=0)
    val_counts = labels[val_mask].sum(axis=0)
    if np.any(train_counts < MIN_TRAIN_PER_ATTRIBUTE) or np.any(val_counts < MIN_VAL_PER_ATTRIBUTE):
        raise RuntimeError(f"Gold atribut terlalu tipis: train={train_counts.tolist()} val={val_counts.tolist()}")
    mean = features[train_mask].mean(axis=0)
    std = features[train_mask].std(axis=0)
    std[std < 1e-6] = 1.0
    train_x = torch.from_numpy((features[train_mask] - mean) / std)
    train_y = torch.from_numpy(labels[train_mask])
    val_x = torch.from_numpy((features[val_mask] - mean) / std)
    val_y = torch.from_numpy(labels[val_mask])
    model = AttributeMLP(FEATURE_SIZE)
    positive = train_y.sum(dim=0)
    negative = len(train_y) - positive
    loss_function = nn.BCEWithLogitsLoss(pos_weight=negative / torch.clamp(positive, min=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-3)
    loader = DataLoader(TensorDataset(train_x, train_y), batch_size=BATCH_SIZE, shuffle=True, generator=torch.Generator().manual_seed(SEED))
    best_state, best_loss = None, float("inf")
    for _ in range(EPOCHS):
        model.train()
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            loss = loss_function(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = float(loss_function(model(val_x), val_y))
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    model.load_state_dict(best_state)
    with torch.no_grad():
        probabilities = torch.sigmoid(model(val_x)).numpy()
    metrics = {}
    thresholds = {}
    for index, name in enumerate(ATTRIBUTE_NAMES):
        truth = val_y.numpy()[:, index] == 1
        candidates = np.arange(0.20, 0.81, 0.05)
        threshold = max(
            candidates,
            key=lambda candidate: (
                2 * ((probabilities[:, index] >= candidate) & truth).sum()
                / max((probabilities[:, index] >= candidate).sum() + truth.sum(), 1),
                -candidate,
            ),
        )
        thresholds[name] = float(threshold)
        prediction = probabilities[:, index] >= threshold
        metrics[name] = {
            "support": int(truth.sum()),
            "threshold": float(threshold),
            "precision": float((prediction & truth).sum() / max(prediction.sum(), 1)),
            "recall": float((prediction & truth).sum() / max(truth.sum(), 1)),
        }
    metadata = {
        "feature_version": FEATURE_VERSION,
        "attribute_names": ATTRIBUTE_NAMES,
        "train_examples": int(train_mask.sum()),
        "val_examples": int(val_mask.sum()),
        "train_positive_counts": train_counts.tolist(),
        "val_positive_counts": val_counts.tolist(),
        "best_val_loss": best_loss,
        "metrics": metrics,
        "thresholds": thresholds,
    }
    save_classifier(MODEL_DIR / "best.pt", model, mean, std, metadata)
    (MODEL_DIR / "metrics.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
