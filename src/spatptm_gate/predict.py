from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .common import sha256_array, sha256_file, write_json
from .model import SpatPTM


def predict_windows(
    windows_path: str | Path,
    sites_path: str | Path,
    model_dir: str | Path,
    output_path: str | Path,
    threshold: float = 0.5,
    device_name: str | None = None,
) -> pd.DataFrame:
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("Threshold must be between 0 and 1")
    windows_path = Path(windows_path).resolve()
    sites_path = Path(sites_path).resolve()
    model_dir = Path(model_dir).resolve()
    output_path = Path(output_path).resolve()
    windows = np.load(windows_path).astype(np.float32)
    sites = pd.read_csv(sites_path)
    if windows.shape != (len(sites), 43, 128):
        raise ValueError(f"Windows/sites mismatch: {windows.shape} vs {len(sites)} rows")
    scaler = np.load(model_dir / "scaler.npz")
    standardized = ((windows - scaler["mean"][None, None, :]) / scaler["scale"][None, None, :]).astype(np.float32)
    tensor = torch.from_numpy(standardized).permute(0, 2, 1)
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    probabilities: dict[int, np.ndarray] = {}
    checkpoints = sorted(model_dir.glob("model_seed_*.pt"))
    if not checkpoints:
        raise RuntimeError(f"No model_seed_*.pt checkpoints in {model_dir}")
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model = SpatPTM().to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        with torch.inference_mode():
            probabilities[int(checkpoint["seed"])] = torch.sigmoid(model(tensor.to(device))).cpu().numpy()
    if set(probabilities) != {42, 2024, 888}:
        raise RuntimeError(f"Expected frozen seeds 42, 2024, and 888; found {sorted(probabilities)}")
    stacked = np.stack([probabilities[seed] for seed in sorted(probabilities)], axis=1)
    result = sites.copy()
    for column, seed in enumerate(sorted(probabilities)):
        result[f"probability_seed_{seed}"] = stacked[:, column]
    result["cancer_probability"] = stacked.mean(axis=1)
    result["ensemble_std"] = stacked.std(axis=1, ddof=0)
    result["predicted_cancer_related"] = result["cancer_probability"] > threshold
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8-sig")
    write_json(output_path.with_suffix(".audit.json"), {
        "feature_pipeline": "PSSM + SPOT contact graph -> GATE -> 43 x 128 window; no ESM2 classifier input",
        "threshold_rule": f"cancer_probability > {threshold}",
        "sites": len(result),
        "positive_predictions": int(result["predicted_cancer_related"].sum()),
        "predicted_positive_fraction": float(result["predicted_cancer_related"].mean()),
        "windows_sha256": sha256_array(windows),
        "scaler_sha256": sha256_file(model_dir / "scaler.npz"),
        "checkpoints": {path.name: sha256_file(path) for path in checkpoints},
    })
    return result
