#!/usr/bin/env python3
"""Train the five-class CNN with SAM2 mask features and train-only e-means."""

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
from PIL import ImageOps
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from attribute_classifier import (  # noqa: E402
    ATTRIBUTE_BY_ANNOTATION_CLASS,
    ATTRIBUTE_NAMES,
    AttributeFusionCNN,
    EMEANS_EPOCHS,
    EMEANS_SEED,
    IMAGE_TRANSFORM,
    MASK_CENTROID_X_FEATURE_INDEX,
    MASK_FEATURE_VERSION,
    MASK_VALID_FEATURE_INDEX,
    MODEL_NUMERIC_FEATURE_SIZE,
    extract_image_crop,
    extract_mask_aware_features,
    mask_cache_key,
    save_classifier,
)
from feature_clustering import (  # noqa: E402
    assign_e_means_clusters,
    cluster_one_hot,
    elbow_k_from_objectives,
    fit_e_means,
    mean_squared_quantization_error,
)

ROOT = Path(os.environ.get("RISET_PROJECT_ROOT", Path(__file__).resolve().parent.parent)).resolve()
OUTPUT = ROOT / "output"
ANNOTATION = OUTPUT / "annotation"
MODEL_DIR = OUTPUT / "attribute_classifier"
SEED = 20261102
EPOCHS = int(os.environ.get("ATTRIBUTE_EPOCHS", "40"))
BATCH_SIZE = 32
LEARNING_RATE = 2e-4
WARMUP_HEAD_EPOCHS = 5
MIN_TRAIN_PER_CLASS = 20
MIN_VAL_PER_CLASS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EMEANS_K_MIN = 2
EMEANS_K_SEARCH_MAX = 12
EMEANS_SELECTION_RESTARTS = 3

TRAIN_AUGMENT = transforms.Compose([
    transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.08, hue=0.0),
])
MASK_CENTROID_X_FEATURE = MASK_CENTROID_X_FEATURE_INDEX
EMPTY_MASK_MAX_FRACTION = 0.01


@dataclass(frozen=True)
class Example:
    image_path: Path
    box: tuple[float, float, float, float]
    label: int
    split: str
    mask_id: str
    mask_path: Path
    source: str


class CropDataset(Dataset):
    def __init__(
        self,
        examples: list[Example],
        features_by_mask: dict[str, np.ndarray],
        clusters_by_mask: dict[str, np.ndarray],
        cluster_centroids: np.ndarray,
        mirror_feature_offset: float,
        training: bool = False,
    ) -> None:
        self.examples = examples
        self.features_by_mask = features_by_mask
        self.clusters_by_mask = clusters_by_mask
        self.cluster_centroids = cluster_centroids
        self.mirror_feature_offset = float(mirror_feature_offset)
        self.training = training

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, ...]:
        example = self.examples[index]
        with Image.open(example.image_path) as image:
            crop = extract_image_crop(image, list(example.box))
        features = self.features_by_mask[example.mask_id]
        cluster_features = self.clusters_by_mask[example.mask_id]
        if self.training and random.random() < 0.5:
            crop = ImageOps.mirror(crop)
            features = features.copy()
            features[MASK_CENTROID_X_FEATURE] = self.mirror_feature_offset - features[MASK_CENTROID_X_FEATURE]
            cluster_id = assign_e_means_clusters(features[None, :], self.cluster_centroids)
            cluster_features = cluster_one_hot(cluster_id, len(self.cluster_centroids))[0]
        if self.training:
            crop = TRAIN_AUGMENT(crop)
        return (
            IMAGE_TRANSFORM(crop),
            torch.from_numpy(features),
            torch.from_numpy(cluster_features),
            torch.tensor(example.label, dtype=torch.long),
        )


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        resolved = path.resolve()
    else:
        resolved = (ROOT / path).resolve()
    try:
        resolved.relative_to(ROOT)
    except ValueError as error:
        raise ValueError(f"path manifest keluar dari root proyek: {value}") from error
    if not resolved.exists():
        raise FileNotFoundError(f"file input tidak ditemukan: {resolved}")
    return resolved


def load_examples(*, include_test: bool = True) -> tuple[list[Example], set[str], dict]:
    mask_manifest_path = OUTPUT / "nb02e_feature_masks" / "mask_manifest.json"
    if not mask_manifest_path.exists():
        raise FileNotFoundError(f"Manifest mask belum ada: {mask_manifest_path}. Jalankan NB02e sampai complete.")
    mask_payload = json.loads(mask_manifest_path.read_text(encoding="utf-8"))
    if mask_payload.get("schema_version") != 2 or mask_payload.get("status") != "complete":
        raise RuntimeError("NB04b memerlukan mask manifest skema 2 berstatus complete; output preview tidak cukup.")
    mask_items = mask_payload.get("items", {})
    examples: list[Example] = []
    reviewed_stems: set[str] = set()
    sources = [
        (ANNOTATION / "sample_manifest.json", ANNOTATION / "annotations.json", None, "human_annotation"),
        (OUTPUT / "ground_truth" / "sample_manifest.json", OUTPUT / "ground_truth" / "annotations.json", "test", "gold_test"),
    ]
    for manifest_path, store_path, split_override, source_name in sources:
        if split_override == "test" and not include_test:
            continue
        if not manifest_path.exists() or not store_path.exists():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        store = json.loads(store_path.read_text(encoding="utf-8"))
        for record in manifest.get("frames", []):
            frame_stem = record["frame_stem"]
            entry = store.get("frames", {}).get(frame_stem)
            if not entry or not entry.get("reviewed"):
                continue
            reviewed_stems.add(frame_stem)  # Includes reviewed-empty images; suppress pseudo labels too.
            split = split_override or record.get(
                "split", store.get("split_by_camera", {}).get(record.get("camera_id"), "unknown")
            )
            if split_override is None and split not in {"train", "val"}:
                raise ValueError(
                    f"Frame anotasi manual {frame_stem} memiliki split {split!r}; "
                    "NB04b hanya menerima manual train/val dan GT dari store terpisah."
                )
            for box_entry in entry.get("boxes", []):
                class_name = box_entry.get("class_name")
                if box_entry.get("deleted") or class_name not in ATTRIBUTE_BY_ANNOTATION_CLASS:
                    continue
                box = tuple(float(v) for v in box_entry["box"])
                mask_id = mask_cache_key(frame_stem, list(box))
                mask_record = mask_items.get(mask_id)
                if mask_record is None:
                    raise FileNotFoundError(f"Mask SAM2 tidak ada: {frame_stem}, bbox={box}")
                examples.append(Example(
                    image_path=resolve_path(mask_record["image_path"]), box=box,
                    label=int(ATTRIBUTE_BY_ANNOTATION_CLASS[class_name]), split=split,
                    mask_id=mask_id, mask_path=resolve_path(mask_record["mask_path"]), source=source_name,
                ))

    proposal_path = OUTPUT / "nb02d_fiveclass" / "base" / "proposal_manifest.json"
    if proposal_path.exists():
        proposals = json.loads(proposal_path.read_text(encoding="utf-8"))
        if proposals.get("status") != "complete":
            raise RuntimeError("Proposal NB02d belum complete; NB04b menolak pseudo-label parsial.")
        for record in proposals.get("frames", []):
            if record.get("split") != "train" or record["frame_stem"] in reviewed_stems:
                continue
            for detection in record.get("detections", []):
                class_name = detection.get("class_name")
                if class_name not in ATTRIBUTE_BY_ANNOTATION_CLASS:
                    raise ValueError(f"Kelas pseudo-label tidak dikenal: {record['frame_stem']}: {class_name}")
                box = tuple(float(v) for v in detection["box"])
                mask_id = mask_cache_key(record["frame_stem"], list(box))
                mask_record = mask_items.get(mask_id)
                if mask_record is None:
                    raise FileNotFoundError(f"Mask pseudo train tidak ada: {record['frame_stem']}, bbox={box}")
                examples.append(Example(
                    image_path=resolve_path(mask_record["image_path"]), box=box,
                    label=int(ATTRIBUTE_BY_ANNOTATION_CLASS[class_name]), split="train",
                    mask_id=mask_id, mask_path=resolve_path(mask_record["mask_path"]),
                    source="grounding_dino_pseudo",
                ))
    return examples, reviewed_stems, mask_payload


def set_backbone_trainable(model: AttributeFusionCNN, trainable: bool) -> None:
    for parameter in model.backbone.parameters():
        parameter.requires_grad = trainable
    for parameter in model.head.parameters():
        parameter.requires_grad = True


def calculate_metrics(logits: torch.Tensor, labels: torch.Tensor, loss: float) -> dict[str, float]:
    probabilities = torch.softmax(logits, dim=1)
    truth = labels.reshape(-1).long()
    prediction = probabilities.argmax(dim=1)
    precision, recall, f1, support = [], [], [], []
    for class_index in range(len(ATTRIBUTE_NAMES)):
        tp = int(((prediction == class_index) & (truth == class_index)).sum())
        fp = int(((prediction == class_index) & (truth != class_index)).sum())
        fn = int(((prediction != class_index) & (truth == class_index)).sum())
        p = tp / max(tp + fp, 1)
        r = tp / max(tp + fn, 1)
        precision.append(p)
        recall.append(r)
        f1.append(2 * p * r / max(p + r, 1e-9))
        support.append(int((truth == class_index).sum()))
    one_hot_truth = torch.nn.functional.one_hot(truth, num_classes=len(ATTRIBUTE_NAMES)).float()
    brier = torch.mean((probabilities - one_hot_truth) ** 2)
    return {
        "loss": float(loss), "brier_mse": float(brier), "brier_rmse": float(torch.sqrt(brier)),
        "accuracy": float((prediction == truth).float().mean()),
        "precision_macro": float(np.mean(precision)), "recall_macro": float(np.mean(recall)),
        "f1_macro": float(np.mean(f1)),
        **{f"precision_{name}": precision[i] for i, name in enumerate(ATTRIBUTE_NAMES)},
        **{f"recall_{name}": recall[i] for i, name in enumerate(ATTRIBUTE_NAMES)},
        **{f"support_{name}": support[i] for i, name in enumerate(ATTRIBUTE_NAMES)},
        "support": len(truth),
    }


def evaluate(model, loader, loss_function) -> dict[str, float] | None:
    if loader is None:
        return None
    model.eval()
    logits_rows, label_rows = [], []
    total_loss = total_items = 0
    with torch.no_grad():
        for images, features, clusters, labels in loader:
            images, features = images.to(DEVICE), features.to(DEVICE)
            clusters, labels = clusters.to(DEVICE), labels.to(DEVICE)
            logits = model(images, features, clusters)
            loss = loss_function(logits, labels)
            total_loss += float(loss) * len(labels)
            total_items += len(labels)
            logits_rows.append(logits.cpu())
            label_rows.append(labels.cpu())
    if not logits_rows:
        return None
    return calculate_metrics(torch.cat(logits_rows), torch.cat(label_rows), total_loss / max(total_items, 1))


def prepare_feature_encodings(examples: list[Example], model_dir: Path):
    raw_by_mask: dict[str, np.ndarray] = {}
    for example in examples:
        if example.mask_id in raw_by_mask:
            continue
        with Image.open(example.image_path) as image:
            raw_by_mask[example.mask_id] = extract_mask_aware_features(
                image, list(example.box), example.mask_path
            )
    if any(len(values) != MODEL_NUMERIC_FEATURE_SIZE for values in raw_by_mask.values()):
        raise ValueError("Dimensi fitur mask-aware berubah; periksa versi feature extractor.")
    empty_mask_ids = [
        mask_id for mask_id, values in raw_by_mask.items()
        if values[MASK_VALID_FEATURE_INDEX] < 0.5
    ]
    empty_mask_fraction = len(empty_mask_ids) / max(len(raw_by_mask), 1)
    mask_quality = {
        "unique_masks": len(raw_by_mask),
        "empty_masks": len(empty_mask_ids),
        "empty_mask_fraction": empty_mask_fraction,
        "maximum_allowed_fraction": EMPTY_MASK_MAX_FRACTION,
        "empty_mask_photometry_policy": "zero vector; mask-validity feature is 0",
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "mask_quality.json").write_text(
        json.dumps(mask_quality, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if empty_mask_fraction > EMPTY_MASK_MAX_FRACTION:
        raise RuntimeError(
            f"Mask kosong {len(empty_mask_ids)}/{len(raw_by_mask)} ({empty_mask_fraction:.2%}) "
            f"melebihi batas {EMPTY_MASK_MAX_FRACTION:.0%}; rincian: {model_dir / 'mask_quality.json'}"
        )
    train_mask_ids = list(dict.fromkeys(example.mask_id for example in examples if example.split == "train"))
    train_matrix = np.stack([raw_by_mask[key] for key in train_mask_ids]).astype(np.float32)
    mean = train_matrix.mean(axis=0).astype(np.float32)
    std = np.maximum(train_matrix.std(axis=0), 1e-6).astype(np.float32)
    normalized = {key: ((value - mean) / std).astype(np.float32) for key, value in raw_by_mask.items()}
    train_normalized = np.stack([normalized[key] for key in train_mask_ids])
    max_k = min(EMEANS_K_SEARCH_MAX, len(train_mask_ids) - 1)
    candidate_ks = list(range(EMEANS_K_MIN, max_k + 1))
    if len(candidate_ks) < 3:
        raise RuntimeError(
            f"Elbow e-means memerlukan minimal tiga kandidat k; train hanya memiliki {len(train_mask_ids)} mask unik."
        )
    selection_rows = []
    model_dir.mkdir(parents=True, exist_ok=True)
    selection_path = model_dir / "emeans_k_selection.csv"
    pd.DataFrame(columns=[
        "k", "mean_train_objective", "std_train_objective", "selection_restarts"
    ]).to_csv(selection_path, index=False)
    for k_value in candidate_ks:
        seed_objectives = []
        for restart in range(EMEANS_SELECTION_RESTARTS):
            candidate_centroids, _, _ = fit_e_means(
                train_normalized,
                n_clusters=k_value,
                epochs=EMEANS_EPOCHS,
                seed=EMEANS_SEED + restart,
            )
            seed_objectives.append(mean_squared_quantization_error(train_normalized, candidate_centroids))
        selection_rows.append({
            "k": k_value,
            "mean_train_objective": float(np.mean(seed_objectives)),
            "std_train_objective": float(np.std(seed_objectives)),
            "selection_restarts": EMEANS_SELECTION_RESTARTS,
        })
        pd.DataFrame(selection_rows).to_csv(selection_path, index=False)
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame["bic_delta_to_linear"] = np.nan
    selection_frame["selected"] = False
    selected_k, elbow_scores = elbow_k_from_objectives(
        [row["k"] for row in selection_rows],
        [row["mean_train_objective"] for row in selection_rows],
    )
    for row, score in zip(selection_rows, elbow_scores):
        row["bic_delta_to_linear"] = score
        row["selected"] = row["k"] == selected_k
    selection_frame = pd.DataFrame(selection_rows)
    selection_frame.to_csv(selection_path, index=False)
    try:
        import matplotlib.pyplot as plt
        figure, axis = plt.subplots(figsize=(8, 5))
        axis.errorbar(
            selection_frame["k"], selection_frame["mean_train_objective"],
            yerr=selection_frame["std_train_objective"], marker="o", capsize=3,
        )
        axis.axvline(selected_k, color="tab:red", linestyle="--", label=f"elbow k={selected_k}")
        axis.set(title="Pemilihan jumlah cluster e-means (train only)", xlabel="Jumlah cluster (k)", ylabel="Mean squared distance")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(model_dir / "emeans_k_selection.png", dpi=140)
        plt.close(figure)
    except Exception as error:
        print(f"Peringatan: gagal membuat plot pemilihan k e-means: {error}")
    centroids, _, cluster_metadata = fit_e_means(
        train_normalized,
        n_clusters=selected_k,
        epochs=EMEANS_EPOCHS,
        seed=EMEANS_SEED,
    )
    train_labels = assign_e_means_clusters(train_normalized, centroids)
    mask_ids = list(normalized)
    normalized_matrix = np.stack([normalized[mask_id] for mask_id in mask_ids])
    labels_by_mask = assign_e_means_clusters(normalized_matrix, centroids)
    onehots = cluster_one_hot(labels_by_mask, len(centroids))
    cluster_by_mask = dict(zip(normalized.keys(), onehots))
    split_by_mask = {}
    for example in examples:
        previous_split = split_by_mask.setdefault(example.mask_id, example.split)
        if previous_split != example.split:
            raise ValueError(f"Mask {example.mask_id} muncul di lebih dari satu split.")
    try:
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        mask_index_by_id = {mask_id: index for index, mask_id in enumerate(mask_ids)}
        train_indices = [mask_index_by_id[mask_id] for mask_id in train_mask_ids]
        train_features = normalized_matrix[train_indices]
        pca_center = train_features.mean(axis=0)
        centered_train = train_features - pca_center
        covariance = centered_train.T @ centered_train
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        order = np.argsort(eigenvalues)[::-1]
        components = eigenvectors[:, order[:2]]
        projected = (normalized_matrix - pca_center) @ components
        total_variance = max(float(eigenvalues.sum()), 1e-12)
        explained = np.maximum(eigenvalues[order[:2]], 0) / total_variance * 100
        projection_frame = pd.DataFrame({
            "mask_id": mask_ids,
            "split": [split_by_mask[mask_id] for mask_id in mask_ids],
            "cluster_id": labels_by_mask.astype(int),
            "pc1": projected[:, 0],
            "pc2": projected[:, 1],
        })
        projection_csv = model_dir / "emeans_pca_projection.csv"
        projection_png = model_dir / "emeans_pca_scatter.png"
        projection_frame.to_csv(projection_csv, index=False)

        figure, axis = plt.subplots(figsize=(10, 7))
        palette = plt.get_cmap("tab20", len(centroids))
        split_markers = {"train": "o", "val": "^", "test": "s"}
        for cluster_id in range(len(centroids)):
            for split_name, marker in split_markers.items():
                selected = projection_frame[
                    (projection_frame["cluster_id"] == cluster_id)
                    & (projection_frame["split"] == split_name)
                ]
                if selected.empty:
                    continue
                axis.scatter(
                    selected["pc1"], selected["pc2"],
                    s=11 if split_name == "train" else 22,
                    alpha=0.32 if split_name == "train" else 0.78,
                    marker=marker, color=palette(cluster_id), linewidths=0.35,
                )
        cluster_handles = [
            Line2D([0], [0], marker="o", linestyle="", color=palette(index),
                   label=f"Cluster {index}", markersize=6)
            for index in range(len(centroids))
        ]
        split_handles = [
            Line2D([0], [0], marker=marker, linestyle="", color="#444444",
                   label=split_name, markersize=6)
            for split_name, marker in split_markers.items()
            if split_name in set(projection_frame["split"])
        ]
        cluster_legend = axis.legend(handles=cluster_handles, title="e-means", loc="upper right", ncol=2)
        axis.add_artist(cluster_legend)
        axis.legend(handles=split_handles, title="Split", loc="lower right")
        axis.set_title(
            "Proyeksi PCA fitur e-means (PCA di-fit hanya pada train)\n"
            "Proyeksi 2D untuk inspeksi; e-means tetap bekerja pada 77 fitur"
        )
        axis.set_xlabel(f"PC1 ({explained[0]:.1f}% variansi train)")
        axis.set_ylabel(f"PC2 ({explained[1]:.1f}% variansi train)")
        axis.grid(alpha=0.18)
        figure.tight_layout()
        figure.savefig(projection_png, dpi=150)
        plt.close(figure)
        cluster_metadata["pca_preview"] = {
            "plot": projection_png.resolve().relative_to(ROOT).as_posix(),
            "projection_csv": projection_csv.resolve().relative_to(ROOT).as_posix(),
            "fit_split": "train_only",
            "interpretation": "2D visualization only; clustering uses original 77-dimensional features",
        }
    except Exception as error:
        print(f"Peringatan: gagal membuat scatter PCA e-means: {error}")
    cluster_metadata["train_cluster_counts"] = {
        str(index): int((train_labels == index).sum()) for index in range(len(centroids))
    }
    cluster_metadata["fit_split"] = "train_only"
    cluster_metadata["selection_method"] = "BIC comparison: linear objective vs interior segmented-linear elbow"
    cluster_metadata["candidate_k"] = candidate_ks
    cluster_metadata["selected_k"] = selected_k
    cluster_metadata["selection_restarts_per_k"] = EMEANS_SELECTION_RESTARTS
    cluster_metadata["selection_curve_csv"] = selection_path.resolve().relative_to(ROOT).as_posix()
    cluster_metadata["mask_quality"] = mask_quality
    return mean, std, centroids, cluster_metadata, normalized, cluster_by_mask


def main(*, output_dir: Path | None = None, include_test_eval: bool = True) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    examples, _, mask_payload = load_examples(include_test=include_test_eval)
    train = [example for example in examples if example.split == "train"]
    val = [example for example in examples if example.split == "val"]
    test = [example for example in examples if example.split == "test"] if include_test_eval else []
    if not train or not val:
        raise RuntimeError("CNN membutuhkan split train dan validation yang berisi box berlabel.")
    train_counts = {name: sum(item.label == i for item in train) for i, name in enumerate(ATTRIBUTE_NAMES)}
    val_counts = {name: sum(item.label == i for item in val) for i, name in enumerate(ATTRIBUTE_NAMES)}
    low_train = {name: count for name, count in train_counts.items() if count < MIN_TRAIN_PER_CLASS}
    low_val = {name: count for name, count in val_counts.items() if count < MIN_VAL_PER_CLASS}
    if low_train or low_val:
        raise RuntimeError(f"Sampel kelas kurang. Train={low_train}; validation={low_val}.")

    model_dir = Path(output_dir) if output_dir is not None else MODEL_DIR
    feature_mean, feature_std, centroids, cluster_metadata, features_by_mask, clusters_by_mask = prepare_feature_encodings(examples, model_dir)
    mirror_offset = (1.0 - 2.0 * feature_mean[MASK_CENTROID_X_FEATURE]) / feature_std[MASK_CENTROID_X_FEATURE]
    train_loader = DataLoader(CropDataset(train, features_by_mask, clusters_by_mask, centroids, mirror_offset, True), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(CropDataset(val, features_by_mask, clusters_by_mask, centroids, mirror_offset), batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(CropDataset(test, features_by_mask, clusters_by_mask, centroids, mirror_offset), batch_size=BATCH_SIZE, shuffle=False, num_workers=0) if test else None
    try:
        model = AttributeFusionCNN(pretrained=True, cluster_count=len(centroids)).to(DEVICE)
        pretrained_source = "ImageNet"
    except Exception as error:
        print(f"Peringatan: bobot ImageNet tidak tersedia ({error}); memakai ResNet18 tanpa bobot awal.")
        model = AttributeFusionCNN(pretrained=False, cluster_count=len(centroids)).to(DEVICE)
        pretrained_source = "random_init_fallback"
    class_weights = torch.tensor(
        [len(train) / (len(ATTRIBUTE_NAMES) * max(train_counts[name], 1)) for name in ATTRIBUTE_NAMES],
        dtype=torch.float32, device=DEVICE,
    )
    loss_function = nn.CrossEntropyLoss(weight=class_weights)
    print(json.dumps({
        "model": "ResNet18 + SAM2 mask-aware features + e-means",
        "pretrained_source": pretrained_source, "device": DEVICE,
        "train_class_counts": train_counts, "loss": "weighted CrossEntropyLoss",
        "class_weights": {name: float(class_weights[i]) for i, name in enumerate(ATTRIBUTE_NAMES)},
        "numeric_feature_size": MODEL_NUMERIC_FEATURE_SIZE, "e_means": cluster_metadata,
    }, indent=2, ensure_ascii=False))
    set_backbone_trainable(model, False)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE, weight_decay=1e-3)
    best_state, best_val_loss, history_rows = None, float("inf"), []
    for epoch in range(1, EPOCHS + 1):
        if epoch == WARMUP_HEAD_EPOCHS + 1:
            set_backbone_trainable(model, True)
            optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE * 0.25, weight_decay=1e-3)
        model.train()
        if epoch <= WARMUP_HEAD_EPOCHS:
            model.backbone.eval()
            model.head.train()
        logits_rows, labels_rows = [], []
        loss_total = item_total = 0
        for images, features, clusters, labels in train_loader:
            images, features = images.to(DEVICE), features.to(DEVICE)
            clusters, labels = clusters.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images, features, clusters)
            loss = loss_function(logits, labels)
            loss.backward()
            optimizer.step()
            loss_total += float(loss) * len(labels)
            item_total += len(labels)
            logits_rows.append(logits.detach().cpu())
            labels_rows.append(labels.cpu())
        train_metrics = calculate_metrics(torch.cat(logits_rows), torch.cat(labels_rows), loss_total / max(item_total, 1))
        val_metrics = evaluate(model, val_loader, loss_function)
        row = {"epoch": epoch, "learning_rate": optimizer.param_groups[0]["lr"]}
        row.update({f"train_{key}": value for key, value in train_metrics.items()})
        row.update({f"val_{key}": value for key, value in val_metrics.items()})
        history_rows.append(row)
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("Training tidak menghasilkan checkpoint validation yang valid.")
    model.load_state_dict(best_state)
    test_metrics = evaluate(model, test_loader, loss_function)
    history = pd.DataFrame(history_rows)
    model_dir.mkdir(parents=True, exist_ok=True)
    history_path = model_dir / "training_history.csv"
    plot_path = model_dir / "training_curves.png"
    history.to_csv(history_path, index=False)
    metadata = {
        "model_type": "resnet18_sam2_emeans_fusion_5class",
        "numeric_feature_version": MASK_FEATURE_VERSION,
        "numeric_feature_size": MODEL_NUMERIC_FEATURE_SIZE,
        "numeric_feature_parts": {"bbox_photometry": 35, "sam2_foreground_photometry": 35, "mask_geometry": 7},
        "augmentation": "horizontal flip with mask-centroid x mirrored consistently; ColorJitter on RGB branch",
        "e_means": cluster_metadata,
        "mask_source_signature": mask_payload.get("source_signature"),
        "input_size": [224, 224],
        "crop_resize": "preserve_aspect_ratio_longest_side_then_mean_padding",
        "pixel_normalization": {"to_tensor_range": [0.0, 1.0], "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]},
        "pretrained_source": pretrained_source,
        "attribute_names": ATTRIBUTE_NAMES,
        "train_examples": len(train), "val_examples": len(val), "test_examples": len(test),
        "class_counts_train": train_counts, "class_counts_val": val_counts,
        "loss": "weighted CrossEntropyLoss", "best_val_loss": best_val_loss,
        "test_metrics": test_metrics, "decision_rule": "softmax_argmax",
        "feature_scaler_fit": "train_only", "e_means_fit": "train_only",
        "history_path": str(history_path), "test_evaluation_enabled": bool(include_test_eval),
    }
    save_classifier(model_dir / "best.pt", model, feature_mean, feature_std, centroids, metadata)
    (model_dir / "metrics.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt
        figure, axes = plt.subplots(1, 2, figsize=(13, 4))
        axes[0].plot(history["epoch"], history["train_loss"], label="train")
        axes[0].plot(history["epoch"], history["val_loss"], label="validation")
        axes[0].set_title("Loss per epoch")
        axes[0].set_xlabel("epoch")
        axes[0].legend()
        axes[1].plot(history["epoch"], history["train_f1_macro"], label="train")
        axes[1].plot(history["epoch"], history["val_f1_macro"], label="validation")
        axes[1].set_title("Macro F1 per epoch")
        axes[1].set_xlabel("epoch")
        axes[1].legend()
        figure.tight_layout()
        figure.savefig(plot_path, dpi=140)
        plt.close(figure)
    except Exception as error:
        print(f"Peringatan: gagal membuat plot CNN: {error}")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
