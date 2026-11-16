"""Fit and apply the official Meta FAIR e-means implementation on CNN features."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(PROJECT_ROOT))


def fit_e_means(
    train_features: np.ndarray,
    *,
    n_clusters: int,
    epochs: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit centroids with native PyTorch assignment, usable on Windows CPU/CUDA."""
    try:
        from vendor.faiss_e_means import Emeans
    except ImportError as error:
        raise RuntimeError(
            "Implementasi resmi e-means belum terpasang. Jalankan sekali "
            "`powershell -ExecutionPolicy Bypass -File builders/install_emeans.ps1`, "
            "lalu ulangi NB04b."
        ) from error
    matrix = np.ascontiguousarray(train_features, dtype=np.float32)
    if matrix.ndim != 2 or len(matrix) < n_clusters:
        raise ValueError(
            f"e-means memerlukan matriks 2D dengan minimal {n_clusters} baris; "
            f"didapat {matrix.shape}"
        )
    clusterer = Emeans(
        d=matrix.shape[1],
        k=n_clusters,
        nepochs=epochs,
        seed=seed,
        assignment_backend="native",
        verbose=True,
    )
    tensor = torch.from_numpy(matrix)
    centroids, labels = clusterer.train(tensor)
    objective_history = getattr(clusterer, "obj", None)
    if objective_history is None:
        objective_history = []
    elif isinstance(objective_history, torch.Tensor):
        objective_history = objective_history.detach().cpu().reshape(-1).tolist()
    else:
        objective_history = np.asarray(objective_history).reshape(-1).tolist()
    metadata = {
        "algorithm": "Meta FAIR e-means (official PyTorch implementation)",
        "implementation_tag": "faiss-v1.16.0",
        "assignment_backend": "native-pytorch",
        "n_train": int(len(matrix)),
        "feature_dimension": int(matrix.shape[1]),
        "n_clusters": int(n_clusters),
        "epochs": float(epochs),
        "seed": int(seed),
        "recipe": dict(clusterer.recipe),
        "training_objective_by_epoch": [float(value) for value in objective_history],
    }
    return (
        centroids.detach().cpu().numpy().astype(np.float32, copy=True),
        labels.detach().cpu().numpy().astype(np.int64, copy=True),
        metadata,
    )


def assign_e_means_clusters(features: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign rows to their nearest saved centroid, returning integer IDs."""
    matrix = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)
    if matrix.ndim != 2 or centers.ndim != 2 or matrix.shape[1] != centers.shape[1]:
        raise ValueError(f"dimensi cluster tidak cocok: data={matrix.shape}, centroid={centers.shape}")
    labels = np.empty(len(matrix), dtype=np.int64)
    chunk_size = 2048
    center_norms = np.sum(centers * centers, axis=1)
    for start in range(0, len(matrix), chunk_size):
        stop = min(start + chunk_size, len(matrix))
        block = matrix[start:stop]
        distances = (
            np.sum(block * block, axis=1, keepdims=True)
            + center_norms[None, :]
            - 2.0 * block @ centers.T
        )
        labels[start:stop] = np.argmin(distances, axis=1)
    return labels


def mean_squared_quantization_error(features: np.ndarray, centroids: np.ndarray) -> float:
    """Mean squared distance to the nearest centroid, on the supplied data."""
    matrix = np.asarray(features, dtype=np.float32)
    centers = np.asarray(centroids, dtype=np.float32)
    labels = assign_e_means_clusters(matrix, centers)
    residual = matrix - centers[labels]
    return float(np.mean(np.sum(residual * residual, axis=1)))


def elbow_k_from_objectives(
    k_values: list[int], objectives: list[float]
) -> tuple[int, list[float]]:
    """Select an interior elbow when a 2-segment model beats a linear trend by BIC."""
    ks = np.asarray(k_values, dtype=np.float64)
    losses = np.asarray(objectives, dtype=np.float64)
    if ks.ndim != 1 or losses.shape != ks.shape or len(ks) < 3:
        raise ValueError("elbow memerlukan minimal tiga pasangan k/objective")
    if not np.all(np.diff(ks) > 0) or not np.all(np.isfinite(losses)):
        raise ValueError("k harus naik berurutan dan objective harus finite")
    loss_range = float(losses[0] - losses[-1])
    if loss_range <= 0:
        raise ValueError("objective tidak menurun pada rentang k; elbow tidak dapat dihitung")
    x = (ks - ks[0]) / (ks[-1] - ks[0])
    y = (losses - losses[-1]) / loss_range

    def bic(design: np.ndarray) -> float:
        coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
        residual = y - design @ coefficients
        rss_floor = np.finfo(np.float64).eps * max(float(np.sum(y * y)), 1.0)
        rss = max(float(np.sum(residual * residual)), rss_floor)
        return float(len(y) * np.log(rss / len(y)) + design.shape[1] * np.log(len(y)))

    linear_bic = bic(np.column_stack((np.ones_like(x), x)))
    bic_delta_by_index = [float("nan")] * len(ks)
    best_index, best_bic = None, float("inf")
    for knot_index in range(1, len(ks) - 1):
        hinge = np.maximum(0.0, x - x[knot_index])
        segmented_bic = bic(np.column_stack((np.ones_like(x), x, hinge)))
        bic_delta_by_index[knot_index] = segmented_bic - linear_bic
        if segmented_bic < best_bic:
            best_index, best_bic = knot_index, segmented_bic
    if best_index is None or best_bic >= linear_bic:
        raise ValueError("kurva objective tidak mendukung elbow interior menurut BIC")
    return int(ks[best_index]), bic_delta_by_index


def cluster_one_hot(labels: np.ndarray, n_clusters: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if n_clusters < 1:
        raise ValueError("n_clusters harus positif")
    if np.any(labels < 0) or np.any(labels >= n_clusters):
        raise ValueError("label cluster berada di luar rentang centroid tersimpan")
    return np.eye(n_clusters, dtype=np.float32)[labels]
