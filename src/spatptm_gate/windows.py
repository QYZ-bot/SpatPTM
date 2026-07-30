from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256_array, write_json


def crop_window(full: np.ndarray, position: int, window: int = 43) -> np.ndarray:
    if window % 2 != 1:
        raise ValueError("Window length must be odd")
    if full.ndim != 2:
        raise ValueError(f"Expected L x F array, got {full.shape}")
    if position < 1 or position > full.shape[0]:
        raise ValueError(f"1-based position {position} outside protein length {full.shape[0]}")
    half = window // 2
    center = position - 1
    lo, hi = center - half, center + half + 1
    src_lo, src_hi = max(0, lo), min(full.shape[0], hi)
    result = np.zeros((window, full.shape[1]), dtype=np.float32)
    dst_lo = src_lo - lo
    result[dst_lo:dst_lo + (src_hi - src_lo)] = full[src_lo:src_hi]
    return result


def build_windows(sites_path: str | Path, gate_dir: str | Path, output_dir: str | Path, window: int = 43) -> tuple[np.ndarray, pd.DataFrame]:
    sites_path = Path(sites_path).resolve()
    gate_dir = Path(gate_dir).resolve()
    output_dir = Path(output_dir).resolve()
    sites = pd.read_csv(sites_path)
    required = {"accession", "position"}
    if not required.issubset(sites.columns):
        raise ValueError(f"Sites table requires columns {sorted(required)}")
    output_dir.mkdir(parents=True, exist_ok=True)
    arrays: list[np.ndarray] = []
    rows: list[dict[str, object]] = []
    cache: dict[str, np.ndarray] = {}
    for index, row in sites.reset_index(drop=True).iterrows():
        accession = str(row["accession"]).strip().upper()
        if accession not in cache:
            path = gate_dir / f"{accession}_gate.npy"
            if not path.exists():
                raise FileNotFoundError(path)
            cache[accession] = np.load(path).astype(np.float32)
        gate = cache[accession]
        position = int(row["position"])
        arrays.append(crop_window(gate, position, window))
        payload = row.to_dict()
        payload.update({"array_index": index, "accession": accession, "position": position, "protein_length": gate.shape[0]})
        rows.append(payload)
    matrix = np.stack(arrays).astype(np.float32)
    manifest = pd.DataFrame(rows)
    np.save(output_dir / "gate_windows.npy", matrix)
    manifest.to_csv(output_dir / "window_manifest.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "window_audit.json", {
        "window": window,
        "shape": list(matrix.shape),
        "sites_sha256": __import__("hashlib").sha256(sites_path.read_bytes()).hexdigest(),
        "windows_sha256": sha256_array(matrix),
        "unique_proteins": int(manifest["accession"].nunique()),
    })
    return matrix, manifest
