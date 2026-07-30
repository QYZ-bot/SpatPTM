from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from .common import seed_everything, sha256_array, sha256_file, write_json
from .model import FocalLoss, SpatPTM, mixup


DEFAULT_EPOCHS = {42: 5, 2024: 5, 888: 8}
EPOCH_SELECTION_AUDIT = {
    42: {"fold_best_epochs": [5, 15, 2, 2, 4, 6, 5, 11, 4, 5], "median": 5.0, "selected_epoch": 5},
    2024: {"fold_best_epochs": [4, 8, 5, 3, 5, 10, 4, 1, 10, 3], "median": 4.5, "selected_epoch": 5},
    888: {"fold_best_epochs": [8, 3, 12, 1, 7, 11, 11, 3, 1, 8], "median": 7.5, "selected_epoch": 8},
}


def freeze_models(
    data_dir: str | Path,
    output_dir: str | Path,
    epochs_by_seed: dict[int, int] | None = None,
    batch_size: int = 64,
    learning_rate: float = 3e-4,
    mixup_alpha: float = 0.4,
    device_name: str | None = None,
) -> dict[str, object]:
    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    positive_path = data_dir / "gate_c.npy"
    negative_path = data_dir / "gate_n.npy"
    positive = np.load(positive_path).astype(np.float32)
    negative = np.load(negative_path).astype(np.float32)
    if positive.shape[1:] != (43, 128) or negative.shape[1:] != (43, 128):
        raise ValueError(f"Expected GATE arrays N x 43 x 128, got {positive.shape} and {negative.shape}")
    x = np.concatenate([positive, negative])
    y = np.concatenate([np.ones(len(positive), dtype=np.float32), np.zeros(len(negative), dtype=np.float32)])
    scaler = StandardScaler().fit(x.reshape(-1, 128))
    standardized = scaler.transform(x.reshape(-1, 128)).reshape(x.shape).astype(np.float32)
    np.savez(output_dir / "scaler.npz", mean=scaler.mean_, scale=scaler.scale_, var=scaler.var_, n_samples_seen=np.asarray(scaler.n_samples_seen_))
    device = torch.device(device_name or ("cuda" if torch.cuda.is_available() else "cpu"))
    epochs_by_seed = epochs_by_seed or DEFAULT_EPOCHS
    checkpoint_rows: list[dict[str, object]] = []

    for seed in sorted(epochs_by_seed):
        seed_everything(seed)
        features = torch.from_numpy(standardized).permute(0, 2, 1)
        labels = torch.from_numpy(y)
        generator = torch.Generator().manual_seed(seed)
        loader = DataLoader(TensorDataset(features, labels), batch_size=batch_size, shuffle=True, generator=generator)
        model = SpatPTM().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        criterion = FocalLoss()
        losses: list[float] = []
        for epoch in range(int(epochs_by_seed[seed])):
            model.train()
            running = 0.0
            seen = 0
            for batch_x, batch_y in loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                optimizer.zero_grad(set_to_none=True)
                mixed, y_a, y_b, coefficient = mixup(batch_x, batch_y, mixup_alpha, device)
                logits = model(mixed)
                loss = coefficient * criterion(logits, y_a) + (1.0 - coefficient) * criterion(logits, y_b)
                loss.backward()
                optimizer.step()
                running += float(loss.detach().cpu()) * len(batch_x)
                seen += len(batch_x)
            losses.append(running / seen)
            print(f"[freeze] seed={seed} epoch={epoch + 1}/{epochs_by_seed[seed]} loss={losses[-1]:.6f}", flush=True)
        checkpoint = output_dir / f"model_seed_{seed}.pt"
        torch.save({
            "state_dict": model.state_dict(),
            "seed": seed,
            "epochs": int(epochs_by_seed[seed]),
            "architecture": "SpatPTM_GATE43_input128_branches3and5_transformer2_attention_pool",
            "training_losses": losses,
        }, checkpoint)
        checkpoint_rows.append({"seed": seed, "epochs": int(epochs_by_seed[seed]), "sha256": sha256_file(checkpoint), "losses": losses})

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "new full-internal-data deployment checkpoints for original GATE-SpatPTM; historical repository had CV results but no deployment checkpoint",
        "input_feature": "PTM-centred 43 x 128 GATE window",
        "architecture": "two CNN branches k=3,5 (128 channels each) -> 1x1 projection to 256 -> two 8-head Transformer blocks (FFN 1024) -> attention pooling -> 256-64-1 classifier",
        "parameter_count": 1804418,
        "input_shape": list(x.shape),
        "class_counts": {"positive": len(positive), "negative": len(negative)},
        "device": str(device),
        "optimizer": "AdamW",
        "learning_rate": learning_rate,
        "loss": "binary focal loss alpha=1 gamma=2 with mixup",
        "mixup_alpha": mixup_alpha,
        "batch_size": batch_size,
        "epoch_selection": "rounded medians from prior internal protein-group CV: seed42=5, seed2024=4.5->5, seed888=7.5->8",
        "epoch_selection_audit": {str(seed): EPOCH_SELECTION_AUDIT.get(seed, {"selected_epoch": epochs_by_seed[seed]}) for seed in sorted(epochs_by_seed)},
        "inputs": {
            "gate_c.npy": {"sha256_file": sha256_file(positive_path), "sha256_array": sha256_array(positive)},
            "gate_n.npy": {"sha256_file": sha256_file(negative_path), "sha256_array": sha256_array(negative)},
        },
        "scaler": {"file": "scaler.npz", "sha256": sha256_file(output_dir / "scaler.npz"), "fit_rows": int(x.shape[0] * x.shape[1])},
        "checkpoints": checkpoint_rows,
    }
    write_json(output_dir / "freeze_manifest.json", manifest)
    return manifest
