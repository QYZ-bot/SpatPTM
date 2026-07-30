from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .common import sha256_array, write_json


def top3l_graph(probability: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(probability, dtype=np.float32)
    if matrix.ndim == 3 and matrix.shape[0] == 1:
        matrix = matrix[0]
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"Expected square contact matrix, got {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Contact matrix contains non-finite values")
    length = matrix.shape[0]
    symmetric = (matrix + matrix.T) / 2.0
    rows, cols = np.triu_indices(length, k=1)
    edge_count = 3 * length
    if edge_count > len(rows):
        raise ValueError(f"Protein length {length} has fewer than 3L residue pairs")
    values = symmetric[rows, cols]
    selected = np.argsort(-values, kind="mergesort")[:edge_count]
    adjacency = np.zeros((length, length), dtype=np.int8)
    adjacency[rows[selected], cols[selected]] = 1
    adjacency[cols[selected], rows[selected]] = 1
    return adjacency, symmetric


def build_graphs(probability_dir: str | Path, output_dir: str | Path, overwrite: bool = False) -> list[dict[str, object]]:
    probability_dir = Path(probability_dir).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        path for path in sorted(probability_dir.glob("*.npy"))
        if not path.stem.rsplit("_", 1)[-1].isdigit()
    ]
    if not sources:
        raise RuntimeError(f"No ensemble SPOT contact maps found in {probability_dir}")
    records: list[dict[str, object]] = []
    for source in sources:
        output = output_dir / f"{source.stem}_bin.npy"
        if output.exists() and not overwrite:
            adjacency = np.load(output)
            symmetric = (np.asarray(np.load(source)).squeeze() + np.asarray(np.load(source)).squeeze().T) / 2.0
            status = "reused"
        else:
            adjacency, symmetric = top3l_graph(np.load(source))
            np.save(output, adjacency)
            status = "generated"
        length = int(adjacency.shape[0])
        upper_edges = int(np.count_nonzero(np.triu(adjacency, k=1)))
        if upper_edges != 3 * length or not np.array_equal(adjacency, adjacency.T) or np.count_nonzero(np.diag(adjacency)):
            raise AssertionError(f"Graph audit failed for {source.name}")
        records.append({
            "accession": source.stem,
            "status": status,
            "length": length,
            "upper_edges": upper_edges,
            "probability_sha256": sha256_array(np.load(source)),
            "symmetric_probability_sha256": sha256_array(symmetric),
            "adjacency_sha256": sha256_array(adjacency),
            "output": str(output),
        })
        print(f"[graph] {source.stem}: L={length}, edges={upper_edges}", flush=True)
    with (output_dir / "contact_graph_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output_dir / "contact_graph_audit.json", {
        "rule": "symmetrize probability; global stable Top-3L strict upper triangle; mirror; zero diagonal",
        "records": records,
    })
    return records

