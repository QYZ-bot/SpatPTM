from __future__ import annotations

import csv
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from .common import sha256_file, sorted_files, write_json


def parse_ascii_pssm(path: str | Path) -> tuple[list[str], list[list[float]]]:
    residues: list[str] = []
    scores: list[list[float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            parts = line.split()
            if len(parts) < 22:
                continue
            try:
                int(parts[0])
                row = [float(value) for value in parts[2:22]]
            except ValueError:
                continue
            residues.append(parts[1].upper())
            scores.append(row)
    if not scores:
        raise ValueError(f"No residue rows parsed from ASCII PSSM: {path}")
    return residues, scores


def _run_one(
    fasta: Path,
    output_dir: Path,
    psiblast: Path,
    database: Path,
    iterations: int,
    evalue: float,
    threads: int,
    overwrite: bool,
) -> dict[str, object]:
    output = output_dir / f"{fasta.stem}.pssm"
    sparse_tail_bytes_removed = 0
    if output.exists() and not overwrite:
        status = "reused"
    else:
        temporary_output = output_dir / f".{fasta.stem}.pssm.tmp"
        temporary_output.unlink(missing_ok=True)
        command = [
            str(psiblast),
            "-query", str(fasta),
            "-db", str(database),
            "-evalue", str(evalue),
            "-num_iterations", str(iterations),
            "-num_threads", str(threads),
            "-out_ascii_pssm", str(temporary_output),
        ]
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            temporary_output.unlink(missing_ok=True)
            raise RuntimeError(
                f"PSI-BLAST failed for {fasta.name} (exit {completed.returncode}):\n{completed.stderr[-4000:]}"
            )
        raw_output = temporary_output.read_bytes()
        first_nul = raw_output.find(b"\x00")
        if first_nul >= 0:
            sparse_tail_bytes_removed = len(raw_output) - first_nul
            temporary_output.write_bytes(raw_output[:first_nul])
        temporary_output.replace(output)
        status = "generated"
    raw_output = output.read_bytes()
    first_nul = raw_output.find(b"\x00")
    if first_nul >= 0:
        sparse_tail_bytes_removed += len(raw_output) - first_nul
        output.write_bytes(raw_output[:first_nul])
    residues, scores = parse_ascii_pssm(output)
    sequence = "".join(line.strip() for line in fasta.read_text(encoding="utf-8").splitlines() if not line.startswith(">"))
    if len(scores) != len(sequence):
        raise ValueError(f"{fasta.stem}: PSSM length {len(scores)} != FASTA length {len(sequence)}")
    if "".join(residues) != sequence.upper():
        raise ValueError(f"{fasta.stem}: ASCII PSSM residue order differs from FASTA")
    return {
        "accession": fasta.stem,
        "status": status,
        "sequence_length": len(sequence),
        "fasta": str(fasta.resolve()),
        "fasta_sha256": sha256_file(fasta),
        "pssm": str(output.resolve()),
        "pssm_sha256": sha256_file(output),
        "sparse_tail_bytes_removed": sparse_tail_bytes_removed,
    }


def generate_pssms(
    fasta_dir: str | Path,
    output_dir: str | Path,
    psiblast: str | Path,
    database: str | Path,
    database_acquired_at: str,
    iterations: int = 3,
    evalue: float = 0.001,
    workers: int = 1,
    threads_per_query: int = 1,
    overwrite: bool = False,
) -> list[dict[str, object]]:
    fasta_dir = Path(fasta_dir).resolve()
    output_dir = Path(output_dir).resolve()
    psiblast = Path(psiblast).resolve()
    database = Path(database).resolve()
    if not psiblast.is_file():
        raise FileNotFoundError(psiblast)
    if not any(database.parent.glob(database.name + ".*")):
        raise FileNotFoundError(f"BLAST database prefix not found: {database}")
    fastas = sorted_files(fasta_dir, ".fasta")
    if not fastas:
        raise RuntimeError(f"No FASTA files found in {fasta_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not database_acquired_at or database_acquired_at.lower() in {"unknown", "yyyy-mm-dd"}:
        raise ValueError("A real Swiss-Prot database acquisition/release date is required")
    version = subprocess.run([str(psiblast), "-version"], capture_output=True, text=True, check=False)
    metadata_path = Path(str(database) + ".pjs")
    database_metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else None

    prior_removed: dict[str, int] = {}
    prior_audit_path = output_dir / "pssm_audit.json"
    if prior_audit_path.exists():
        try:
            prior_audit = json.loads(prior_audit_path.read_text(encoding="utf-8"))
            prior_removed = {
                str(record["accession"]): int(record.get("sparse_tail_bytes_removed", 0))
                for record in prior_audit.get("records", [])
            }
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            prior_removed = {}

    records: list[dict[str, object]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        futures = {
            pool.submit(
                _run_one, fasta, output_dir, psiblast, database, iterations, evalue,
                threads_per_query, overwrite,
            ): fasta
            for fasta in fastas
        }
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            print(f"[pssm] {record['accession']}: {record['status']}", flush=True)
    records.sort(key=lambda row: str(row["accession"]))
    for record in records:
        if int(record["sparse_tail_bytes_removed"]) == 0:
            record["sparse_tail_bytes_removed"] = prior_removed.get(str(record["accession"]), 0)

    with (output_dir / "pssm_audit.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output_dir / "pssm_audit.json", {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": "PSI-BLAST ASCII PSSM",
        "psiblast": str(psiblast),
        "psiblast_version": (version.stdout or version.stderr).strip(),
        "psiblast_sha256": sha256_file(psiblast),
        "database_prefix": str(database),
        "database_acquired_at": database_acquired_at,
        "database_metadata": database_metadata,
        "database_files": [
            {"file": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}
            for path in sorted(database.parent.glob(database.name + ".*")) if path.is_file()
        ],
        "iterations": iterations,
        "evalue": evalue,
        "threads_per_query": threads_per_query,
        "ascii_sparse_tail_policy": "if NUL bytes occur after a complete report, remove bytes from the first NUL onward; full 1..L residue identity validation remains mandatory",
        "records": records,
    })
    return records
