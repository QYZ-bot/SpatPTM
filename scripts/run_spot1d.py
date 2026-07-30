from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run upstream SPOT-1D-Single with explicit paths and validate outputs.")
    parser.add_argument("--python", required=True, help="Python executable in the SPOT environment")
    parser.add_argument("--spot1d-root", required=True)
    parser.add_argument("--fasta-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    root = Path(args.spot1d_root).resolve()
    script = root / "spot1d_single.py"
    if not script.exists():
        raise FileNotFoundError(script)
    fasta_dir = Path(args.fasta_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    fastas = sorted(fasta_dir.glob("*.fasta"))
    if not fastas:
        raise RuntimeError(f"No FASTA files in {fasta_dir}")
    pending = [path for path in fastas if args.overwrite or not (output_dir / f"{path.stem}.csv").exists()]
    if pending:
        file_list = output_dir / "spot1d_input_files.txt"
        file_list.write_text("\n".join(path.as_posix() for path in pending) + "\n", encoding="utf-8")
        command = [args.python, str(script), "--file_list", str(file_list), "--save_path", str(output_dir), "--device", args.device]
        subprocess.run(command, cwd=root, check=True)

    records = []
    for fasta in fastas:
        sequence = "".join(line.strip() for line in fasta.read_text(encoding="utf-8").splitlines() if not line.startswith(">"))
        result = output_dir / f"{fasta.stem}.csv"
        if not result.exists():
            raise FileNotFoundError(result)
        with result.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = sum(1 for _ in csv.reader(handle)) - 1
        if rows != len(sequence):
            raise ValueError(f"{fasta.stem}: SPOT-1D rows {rows} != sequence length {len(sequence)}")
        records.append((fasta.stem, len(sequence), str(result), "generated" if fasta in pending else "reused"))
    with (output_dir / "spot1d_audit.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["accession", "sequence_length", "output", "status"])
        writer.writerows(records)
    suffix = "gpu" if args.device.startswith("cuda") else "cpu"
    weight_paths = sorted((root / "jits").glob(f"*_{suffix}.pth"))
    support_paths = [script, root / "main.py", root / "means_single.pkl", root / "stds_single.pkl"]
    (output_dir / "spot1d_audit.json").write_text(json.dumps({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": "SPOT-1D-Single",
        "device_variant": suffix,
        "files": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": hash_file(path)}
            for path in [*support_paths, *weight_paths]
        },
        "records": [dict(zip(("accession", "sequence_length", "output", "status"), row)) for row in records],
    }, indent=2), encoding="utf-8")
    print(f"[spot1d] validated {len(records)} proteins")


if __name__ == "__main__":
    main()
