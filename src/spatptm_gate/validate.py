from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .common import sha256_array, sha256_file, write_json
from .pssm import parse_ascii_pssm


def _read_fasta(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(">"))


def validate_run(run_dir: str | Path, require_predictions: bool = False, max_length: int = 1000) -> dict[str, object]:
    run_dir = Path(run_dir).resolve()
    sites_path = run_dir / "sites.csv"
    fasta_dir = run_dir / "fasta"
    if not sites_path.exists() or not fasta_dir.is_dir():
        raise FileNotFoundError("Run directory must contain sites.csv and fasta/")
    sites = pd.read_csv(sites_path)
    if sites.duplicated(["accession", "position"]).any():
        raise ValueError("sites.csv contains duplicate accession-position pairs")
    records: list[dict[str, object]] = []
    for accession in sorted(sites["accession"].astype(str).unique()):
        fasta = fasta_dir / f"{accession}.fasta"
        if not fasta.exists():
            raise FileNotFoundError(fasta)
        sequence = _read_fasta(fasta).upper()
        if len(sequence) > max_length:
            raise ValueError(f"{accession}: protein length {len(sequence)} exceeds limit {max_length}")
        record: dict[str, object] = {
            "accession": accession,
            "length": len(sequence),
            "fasta_sha256": sha256_file(fasta),
        }
        pssm = run_dir / "pssm" / f"{accession}.pssm"
        if pssm.exists():
            residues, scores = parse_ascii_pssm(pssm)
            if "".join(residues) != sequence or len(scores) != len(sequence):
                raise ValueError(f"{accession}: PSSM/FASTA mismatch")
            record["pssm_sha256"] = sha256_file(pssm)
        spot1d = run_dir / "spot1d" / f"{accession}.csv"
        if spot1d.exists():
            with spot1d.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = sum(1 for _ in csv.reader(handle)) - 1
            if rows != len(sequence):
                raise ValueError(f"{accession}: SPOT-1D length {rows} != {len(sequence)}")
            record["spot1d_sha256"] = sha256_file(spot1d)
        contact = run_dir / "spot_contact" / f"{accession}.npy"
        if contact.exists():
            array = np.asarray(np.load(contact)).squeeze()
            if array.shape != (len(sequence), len(sequence)) or not np.isfinite(array).all():
                raise ValueError(f"{accession}: invalid contact shape/content {array.shape}")
            record["contact_sha256"] = sha256_array(array)
        graph = run_dir / "contact_graphs" / f"{accession}_bin.npy"
        if graph.exists():
            adjacency = np.load(graph)
            edges = int(np.count_nonzero(np.triu(adjacency, k=1)))
            if adjacency.shape != (len(sequence), len(sequence)) or edges != 3 * len(sequence):
                raise ValueError(f"{accession}: invalid Top-3L graph")
            if not np.array_equal(adjacency, adjacency.T) or np.count_nonzero(np.diag(adjacency)):
                raise ValueError(f"{accession}: graph is not symmetric with a zero diagonal")
            record["graph_sha256"] = sha256_array(adjacency)
        gate = run_dir / "gate" / f"{accession}_gate.npy"
        if gate.exists():
            embedding = np.load(gate)
            if embedding.shape != (len(sequence), 128) or not np.isfinite(embedding).all():
                raise ValueError(f"{accession}: invalid GATE embedding {embedding.shape}")
            record["gate_sha256"] = sha256_array(embedding)
        records.append(record)

    for row in sites.itertuples(index=False):
        accession = str(row.accession)
        sequence = _read_fasta(fasta_dir / f"{accession}.fasta").upper()
        position = int(row.position)
        if not 1 <= position <= len(sequence):
            raise ValueError(f"{accession}:{position} is outside sequence")
        if hasattr(row, "residue") and str(row.residue).strip() and sequence[position - 1] != str(row.residue).strip().upper():
            raise ValueError(f"{accession}:{position} residue mismatch")

    windows = run_dir / "windows" / "gate_windows.npy"
    if windows.exists():
        matrix = np.load(windows)
        if matrix.shape != (len(sites), 43, 128) or not np.isfinite(matrix).all():
            raise ValueError(f"Invalid windows array {matrix.shape}")
    predictions = run_dir / "predictions.csv"
    if require_predictions and not predictions.exists():
        raise FileNotFoundError(predictions)
    if predictions.exists() and len(pd.read_csv(predictions)) != len(sites):
        raise ValueError("Prediction row count differs from sites.csv")

    if require_predictions:
        required_stage_paths = [
            run_dir / "pssm", run_dir / "spot1d", run_dir / "spot_contact",
            run_dir / "contact_graphs", run_dir / "gate", windows, predictions,
        ]
        missing_stages = [str(path) for path in required_stage_paths if not path.exists()]
        if missing_stages:
            raise FileNotFoundError(f"Final validation is missing stages: {missing_stages}")
        for record in records:
            required_keys = {"pssm_sha256", "spot1d_sha256", "contact_sha256", "graph_sha256", "gate_sha256"}
            if not required_keys.issubset(record):
                raise ValueError(f"{record['accession']}: incomplete final-stage evidence")

    report = {
        "status": "passed",
        "run_dir": str(run_dir),
        "sites": len(sites),
        "proteins": int(sites["accession"].nunique()),
        "maximum_protein_length": max_length,
        "available_stages": {
            name: (run_dir / path).exists()
            for name, path in {
                "pssm": "pssm", "spot1d": "spot1d", "spot_contact": "spot_contact",
                "contact_graphs": "contact_graphs", "gate": "gate", "windows": "windows/gate_windows.npy",
                "predictions": "predictions.csv",
            }.items()
        },
        "records": records,
    }
    write_json(run_dir / "validation_audit.json", report)
    return report
