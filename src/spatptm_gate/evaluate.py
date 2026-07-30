from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from .common import seed_everything, sha256_array, sha256_file, write_json
from .model import FocalLoss, SpatPTM


EXPERIMENT = "Main_SpatPTM_GATE43"
METRICS = ("AUC", "AUPR", "Accuracy", "F1", "MCC", "Sensitivity", "Specificity")


class WindowDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        self.features = torch.from_numpy(features).float()
        self.labels = torch.from_numpy(labels).float()

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        return self.features[index].permute(1, 0), self.labels[index]


def load_site_data(data_dir: str | Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    data_dir = Path(data_dir).resolve()
    positive_path = data_dir / "gate_c.npy"
    negative_path = data_dir / "gate_n.npy"
    meta_positive_path = data_dir / "meta_c.csv"
    meta_negative_path = data_dir / "meta_n.csv"

    positive = np.load(positive_path).astype(np.float32)
    negative = np.load(negative_path).astype(np.float32)
    if positive.shape[1:] != (43, 128) or negative.shape[1:] != (43, 128):
        raise ValueError(f"Expected GATE arrays N x 43 x 128, got {positive.shape} and {negative.shape}")

    meta_positive = pd.read_csv(meta_positive_path)
    meta_negative = pd.read_csv(meta_negative_path)
    if len(meta_positive) != len(positive) or len(meta_negative) != len(negative):
        raise ValueError("Metadata row counts do not match the GATE arrays")

    meta_positive = meta_positive.copy()
    meta_negative = meta_negative.copy()
    meta_positive["class_source"] = "c"
    meta_negative["class_source"] = "n"
    meta_positive["within_class_index"] = np.arange(len(meta_positive))
    meta_negative["within_class_index"] = np.arange(len(meta_negative))
    metadata = pd.concat([meta_positive, meta_negative], ignore_index=True)
    metadata["global_index"] = np.arange(len(metadata))

    features = np.concatenate([positive, negative], axis=0)
    labels = np.concatenate([
        np.ones(len(positive), dtype=np.int64),
        np.zeros(len(negative), dtype=np.int64),
    ])
    audit = {
        "data_dir": str(data_dir),
        "feature": "PTM-centred GATE window",
        "positive_shape": list(positive.shape),
        "negative_shape": list(negative.shape),
        "class_counts": {"positive": int(len(positive)), "negative": int(len(negative))},
        "inputs": {
            "gate_c.npy": {"sha256_file": sha256_file(positive_path), "sha256_array": sha256_array(positive)},
            "gate_n.npy": {"sha256_file": sha256_file(negative_path), "sha256_array": sha256_array(negative)},
            "meta_c.csv": {"sha256_file": sha256_file(meta_positive_path), "rows": int(len(meta_positive))},
            "meta_n.csv": {"sha256_file": sha256_file(meta_negative_path), "rows": int(len(meta_negative))},
        },
    }
    return features, labels, metadata, audit


def site_stratified_splits(labels: np.ndarray, folds: int, seed: int):
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return splitter.split(np.zeros(len(labels)), labels)


def scale_fold(features: np.ndarray, train_index: np.ndarray, validation_index: np.ndarray):
    sequence_length, dimension = features.shape[1:]
    scaler = StandardScaler()
    train = scaler.fit_transform(features[train_index].reshape(-1, dimension))
    validation = scaler.transform(features[validation_index].reshape(-1, dimension))
    return (
        train.reshape(-1, sequence_length, dimension).astype(np.float32),
        validation.reshape(-1, sequence_length, dimension).astype(np.float32),
    )


def metrics_from_probabilities(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, float]:
    predictions = (probabilities > 0.5).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "AUC": float(roc_auc_score(labels, probabilities)),
        "AUPR": float(average_precision_score(labels, probabilities)),
        "Accuracy": float(accuracy_score(labels, predictions)),
        "F1": float(f1_score(labels, predictions)),
        "MCC": float(matthews_corrcoef(labels, predictions)),
        "Sensitivity": float(tp / (tp + fn + 1e-8)),
        "Specificity": float(tn / (tn + fp + 1e-8)),
    }


def _historical_mixup(x: torch.Tensor, y: torch.Tensor, alpha: float, device: torch.device):
    coefficient = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    # Keep the original CPU randperm followed by a device transfer for exact reruns.
    permutation = torch.randperm(x.size(0)).to(device)
    return coefficient * x + (1.0 - coefficient) * x[permutation], y, y[permutation], coefficient


def train_fold(
    features: np.ndarray,
    labels: np.ndarray,
    train_index: np.ndarray,
    validation_index: np.ndarray,
    seed: int,
    fold: int,
    device: torch.device,
    epochs: int,
    patience: int,
    batch_size: int,
    learning_rate: float,
    mixup_alpha: float,
    num_workers: int,
) -> tuple[dict[str, object], pd.DataFrame]:
    seed_everything(seed)
    train_features, validation_features = scale_fold(features, train_index, validation_index)
    train_loader = DataLoader(
        WindowDataset(train_features, labels[train_index]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    validation_loader = DataLoader(
        WindowDataset(validation_features, labels[validation_index]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = SpatPTM().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = FocalLoss()
    best_auc = -np.inf
    best_epoch = 0
    best_probabilities: np.ndarray | None = None
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        model.train()
        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()
            mixed, labels_a, labels_b, coefficient = _historical_mixup(
                batch_features, batch_labels, mixup_alpha, device
            )
            logits = model(mixed)
            loss = coefficient * criterion(logits, labels_a) + (1.0 - coefficient) * criterion(logits, labels_b)
            loss.backward()
            optimizer.step()

        model.eval()
        fold_probabilities = []
        with torch.no_grad():
            for validation_features_batch, _ in validation_loader:
                logits = model(validation_features_batch.to(device))
                fold_probabilities.append(torch.sigmoid(logits).cpu().numpy())
        probabilities = np.concatenate(fold_probabilities)
        auc = roc_auc_score(labels[validation_index], probabilities)
        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_probabilities = probabilities
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break

    if best_probabilities is None:
        raise RuntimeError(f"seed={seed} fold={fold}: no validation predictions")

    row: dict[str, object] = metrics_from_probabilities(labels[validation_index], best_probabilities)
    row.update({
        "Experiment": EXPERIMENT,
        "Group": "main",
        "Seed": seed,
        "Fold": fold,
        "BestEpoch": best_epoch,
        "NTrain": int(len(train_index)),
        "NVal": int(len(validation_index)),
        "Feature": "gate",
        "Window": 43,
        "ScalerFit": "train_fold_only",
        "ScalerMeanDim": 128,
    })
    predictions = pd.DataFrame({
        "Experiment": EXPERIMENT,
        "Group": "main",
        "Seed": seed,
        "Fold": fold,
        "index": validation_index,
        "y_true": labels[validation_index],
        "y_prob": best_probabilities,
        "y_pred": (best_probabilities > 0.5).astype(int),
    })
    return row, predictions


def summarize_metrics(metrics: pd.DataFrame) -> dict[str, object]:
    summary: dict[str, object] = {
        "Experiment": EXPERIMENT,
        "Feature": "gate",
        "Window": 43,
        "NFolds": int(len(metrics)),
    }
    for metric in METRICS:
        mean = float(metrics[metric].mean())
        standard_deviation = float(metrics[metric].std(ddof=0))
        summary[f"{metric}_mean"] = mean
        summary[f"{metric}_std"] = standard_deviation
        summary[metric] = f"{mean:.4f} +/- {standard_deviation:.4f}"
    return summary


def _append_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8-sig")


def evaluate_site_cv(
    data_dir: str | Path,
    output_dir: str | Path,
    seeds: list[int] | None = None,
    folds: int = 10,
    epochs: int = 50,
    patience: int = 5,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    mixup_alpha: float = 0.4,
    device_name: str | None = None,
    num_workers: int = 0,
    resume: bool = False,
    max_folds: int | None = None,
) -> dict[str, object]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = seeds or [42, 2024, 888]
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    features, labels, metadata, audit = load_site_data(data_dir)
    write_json(output_dir / "data_audit.json", audit)
    metadata.to_csv(output_dir / "sample_manifest.csv", index=False, encoding="utf-8-sig")

    config = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "experiment": EXPERIMENT,
        "protocol": "site-level StratifiedKFold; fold-local StandardScaler; early stopping on validation AUC",
        "seeds": seeds,
        "folds": folds,
        "epochs": epochs,
        "patience": patience,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mixup_alpha": mixup_alpha,
        "device": str(device),
        "num_workers": num_workers,
        "max_folds": max_folds,
    }
    write_json(output_dir / "run_config.json", config)

    metrics_path = output_dir / "fold_metrics.csv"
    predictions_path = output_dir / "fold_predictions.csv"
    completed: set[tuple[int, int]] = set()
    if resume and metrics_path.exists():
        existing = pd.read_csv(metrics_path)
        completed = {(int(row.Seed), int(row.Fold)) for row in existing.itertuples(index=False)}
    elif not resume:
        for path in (metrics_path, predictions_path):
            if path.exists():
                raise FileExistsError(f"{path} already exists; choose another output directory or pass --resume")

    split_rows = []
    split_dir = output_dir / "splits"
    split_dir.mkdir(exist_ok=True)
    for seed in seeds:
        assignments = np.full(len(labels), -1, dtype=np.int64)
        for fold, (train_index, validation_index) in enumerate(site_stratified_splits(labels, folds, seed), start=1):
            assignments[validation_index] = fold
            np.savez_compressed(
                split_dir / f"seed_{seed}_fold_{fold}.npz",
                train_idx=train_index,
                val_idx=validation_index,
            )
            split_rows.append({
                "Seed": seed,
                "Fold": fold,
                "NTrain": int(len(train_index)),
                "NVal": int(len(validation_index)),
                "ValPositive": int(labels[validation_index].sum()),
                "ValNegative": int((1 - labels[validation_index]).sum()),
            })
            if max_folds is not None and fold > max_folds:
                continue
            if (seed, fold) in completed:
                print(f"[evaluate] seed={seed} fold={fold:02d}: reused", flush=True)
                continue
            row, predictions = train_fold(
                features,
                labels,
                train_index,
                validation_index,
                seed,
                fold,
                device,
                epochs,
                patience,
                batch_size,
                learning_rate,
                mixup_alpha,
                num_workers,
            )
            _append_csv(metrics_path, pd.DataFrame([row]))
            _append_csv(predictions_path, predictions)
            completed.add((seed, fold))
            print(f"[evaluate] seed={seed} fold={fold:02d}: AUC={row['AUC']:.4f}", flush=True)
        pd.DataFrame({"index": np.arange(len(labels)), "y": labels, f"fold_seed_{seed}": assignments}).to_csv(
            split_dir / f"fold_assignment_seed_{seed}.csv", index=False
        )

    pd.DataFrame(split_rows).to_csv(output_dir / "cv_split_summary.csv", index=False, encoding="utf-8-sig")
    if not metrics_path.exists():
        raise RuntimeError("No folds were evaluated")
    metrics = pd.read_csv(metrics_path)
    summary = summarize_metrics(metrics)
    pd.DataFrame([summary]).to_csv(output_dir / "summary.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "summary.json", summary)
    return summary
