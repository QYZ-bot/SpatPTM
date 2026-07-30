from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from spatptm_gate.dependencies import check_config


STAGES = ["pssm", "spot1d", "spot_contact", "graphs", "gate", "windows", "predict", "validate"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete original PSSM + SPOT + GATE + SpatPTM workflow.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True, help="Prepared directory containing sites.csv and fasta/")
    parser.add_argument("--model-dir", required=True, help="Frozen original-GATE SpatPTM deployment ensemble")
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--to-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    config_path = Path(args.config).resolve()
    check_config(config_path)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    run_dir = Path(args.run_dir).resolve()
    sites = run_dir / "sites.csv"
    fasta = run_dir / "fasta"
    if not sites.exists() or not fasta.is_dir():
        raise FileNotFoundError("Run directory must first be prepared and contain sites.csv plus fasta/")
    start, stop = STAGES.index(args.from_stage), STAGES.index(args.to_stage)
    if start > stop:
        raise ValueError("from-stage must not occur after to-stage")

    runtimes, blast, spot, gate, model = (config[key] for key in ("runtimes", "blast", "spot", "gate", "spatptm"))
    model_python = str(Path(runtimes["model_python"]).resolve())
    spot_python = str(Path(runtimes["spot_python"]).resolve())
    gate_python = str(Path(runtimes["gate_python"]).resolve())
    overwrite = ["--overwrite"] if args.overwrite else []
    commands: dict[str, list[str]] = {
        "pssm": [model_python, "-m", "spatptm_gate.cli", "pssm", "--fasta-dir", str(fasta), "--output-dir", str(run_dir / "pssm"), "--psiblast", blast["psiblast"], "--database", blast["database"], "--database-acquired-at", blast["database_acquired_at"], "--iterations", str(blast.get("iterations", 3)), "--evalue", str(blast.get("evalue", 0.001)), "--workers", str(blast.get("workers", 1)), "--threads-per-query", str(blast.get("threads_per_query", 1)), *overwrite],
        "spot1d": [model_python, str(repo / "scripts" / "run_spot1d.py"), "--python", spot_python, "--spot1d-root", spot["spot1d_root"], "--fasta-dir", str(fasta), "--output-dir", str(run_dir / "spot1d"), "--device", spot.get("device", "cpu"), *overwrite],
        "spot_contact": [spot_python, str(repo / "scripts" / "run_spot_contact.py"), "--fasta-dir", str(fasta), "--spot1d-dir", str(run_dir / "spot1d"), "--model-root", spot["spot_contact_root"], "--output-dir", str(run_dir / "spot_contact"), "--device", spot.get("device", "cpu"), "--esm-device", spot.get("esm_device", spot.get("device", "cpu")), *overwrite],
        "graphs": [model_python, "-m", "spatptm_gate.cli", "graphs", "--probability-dir", str(run_dir / "spot_contact"), "--output-dir", str(run_dir / "contact_graphs"), *overwrite],
        "gate": [gate_python, str(repo / "scripts" / "run_gate.py"), "--pssm-dir", str(run_dir / "pssm"), "--graph-dir", str(run_dir / "contact_graphs"), "--output-dir", str(run_dir / "gate"), "--external-code-root", gate["external_code_root"], "--epochs", str(gate.get("epochs", 100)), "--learning-rate", str(gate.get("learning_rate", 0.01)), "--lambda-structure", str(gate.get("lambda", 1.0)), "--dropout", str(gate.get("dropout", 0.5)), "--seed", str(gate.get("seed", 42)), *overwrite],
        "windows": [model_python, "-m", "spatptm_gate.cli", "windows", "--sites", str(sites), "--gate-dir", str(run_dir / "gate"), "--output-dir", str(run_dir / "windows"), "--window", str(model.get("window", 43))],
        "predict": [model_python, "-m", "spatptm_gate.cli", "predict", "--windows", str(run_dir / "windows" / "gate_windows.npy"), "--sites", str(run_dir / "windows" / "window_manifest.csv"), "--model-dir", str(Path(args.model_dir).resolve()), "--output", str(run_dir / "predictions.csv"), "--threshold", str(model.get("threshold", 0.5))],
        "validate": [model_python, "-m", "spatptm_gate.cli", "validate", "--run-dir", str(run_dir), "--require-predictions"],
    }
    selected = STAGES[start:stop + 1]
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "run_dir": str(run_dir),
        "model_dir": str(Path(args.model_dir).resolve()),
        "selected_stages": selected,
        "commands": commands,
    }
    (run_dir / "pipeline_commands.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    environment = os.environ.copy()
    if spot.get("torch_home"):
        environment["TORCH_HOME"] = str(Path(spot["torch_home"]).resolve())
    subprocess.run(
        [model_python, "-m", "spatptm_gate.cli", "validate", "--run-dir", str(run_dir), "--max-length", "1000"],
        cwd=repo, env=environment, check=True,
    )
    for stage in selected:
        print(f"[pipeline] starting {stage}", flush=True)
        subprocess.run(commands[stage], cwd=repo, env=environment, check=True)
        print(f"[pipeline] completed {stage}", flush=True)


if __name__ == "__main__":
    main()
