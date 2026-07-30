from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow._api.v2.compat.v1 as tf

tf.disable_eager_execution()


def array_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).view(np.uint8)).hexdigest()


def parse_pssm(path: Path) -> tuple[str, np.ndarray]:
    residues = []
    rows = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 22:
                continue
            try:
                int(parts[0])
                scores = [float(value) for value in parts[2:22]]
            except ValueError:
                continue
            residues.append(parts[1].upper())
            rows.append(scores)
    if not rows:
        raise ValueError(f"No PSSM rows parsed: {path}")
    return "".join(residues), np.asarray(rows, dtype=np.float32)


def sparse_tuple(adjacency: np.ndarray):
    graph = nx.from_numpy_array(adjacency)
    matrix = nx.adjacency_matrix(graph).astype(np.float32) + sp.eye(adjacency.shape[0], dtype=np.float32)
    coo = matrix.tocoo()
    indices = np.vstack((coo.col, coo.row)).T
    return (indices, coo.data, coo.shape), coo.row, coo.col


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-protein GATE embeddings from PSSM nodes and SPOT Top-3L graphs.")
    parser.add_argument("--pssm-dir", required=True)
    parser.add_argument("--graph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--external-code-root", required=True, help="PMiSLocMF-main/code directory")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--lambda-structure", type=float, default=1.0)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    code_root = Path(args.external_code_root).resolve()
    sys.path.insert(0, str(code_root))
    trainer_module = importlib.import_module("feature_extraction.miRNA_disease_feature_extraction.gate_trainer")
    GATETrainer = trainer_module.GATETrainer

    pssm_dir = Path(args.pssm_dir).resolve()
    graph_dir = Path(args.graph_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    pssms = sorted(pssm_dir.glob("*.pssm"))
    if not pssms:
        raise RuntimeError(f"No PSSM files in {pssm_dir}")
    records = []
    for count, pssm_path in enumerate(pssms, start=1):
        accession = pssm_path.stem
        graph_path = graph_dir / f"{accession}_bin.npy"
        output_path = output_dir / f"{accession}_gate.npy"
        if not graph_path.exists():
            raise FileNotFoundError(graph_path)
        sequence, features = parse_pssm(pssm_path)
        adjacency = np.load(graph_path)
        if adjacency.shape != (len(sequence), len(sequence)):
            raise ValueError(f"{accession}: graph {adjacency.shape} != PSSM length {len(sequence)}")
        if output_path.exists() and not args.overwrite:
            embedding = np.load(output_path)
            status = "reused"
        else:
            tf.reset_default_graph()
            random.seed(args.seed)
            np.random.seed(args.seed)
            tf.set_random_seed(args.seed)
            graph, sources, receivers = sparse_tuple(adjacency)
            trainer_args = SimpleNamespace(
                lr=args.learning_rate,
                n_epochs=args.epochs,
                hidden_dims=[20, 256, 128],
                lambda_=args.lambda_structure,
                dropout=args.dropout,
                gradient_clipping=5.0,
            )
            trainer = GATETrainer(trainer_args)
            trainer(graph, features, sources, receivers)
            embedding, _ = trainer.infer(graph, features, sources, receivers)
            trainer.session.close()
            embedding = np.asarray(embedding, dtype=np.float32)
            if embedding.shape != (len(sequence), 128):
                raise ValueError(f"{accession}: unexpected GATE shape {embedding.shape}")
            np.save(output_path, embedding)
            status = "generated"
        records.append({
            "accession": accession,
            "status": status,
            "length": len(sequence),
            "pssm_sha256": hashlib.sha256(pssm_path.read_bytes()).hexdigest(),
            "graph_sha256": array_hash(adjacency),
            "gate_sha256": array_hash(embedding),
            "output": str(output_path),
        })
        print(f"[gate] {count}/{len(pssms)} {accession} {embedding.shape}", flush=True)

    with (output_dir / "gate_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "per-protein graph attention autoencoder; PSSM Lx20 nodes + SPOT Top-3L graph -> Lx128",
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "lambda_structure": args.lambda_structure,
        "dropout": args.dropout,
        "hidden_dimensions": [20, 256, 128],
        "seed": args.seed,
        "seed_declaration": "Historical generator did not record a seed. This deployment fixes the seed for reproducibility.",
        "external_code_root": str(code_root),
        "external_source_files": {
            str(path.relative_to(code_root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                code_root / "feature_extraction" / "miRNA_disease_feature_extraction" / "gate.py",
                code_root / "feature_extraction" / "miRNA_disease_feature_extraction" / "gate_trainer.py",
            )
        },
        "tensorflow_version": tf.__version__,
        "records": records,
    }
    (output_dir / "gate_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
