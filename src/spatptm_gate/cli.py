from __future__ import annotations

import argparse
import json
from pathlib import Path

from .contact_graph import build_graphs
from .dependencies import check_config
from .evaluate import evaluate_site_cv
from .freeze import freeze_models
from .predict import predict_windows
from .prepare import prepare_sites
from .pssm import generate_pssms
from .validate import validate_run
from .windows import build_windows


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(prog="spatptm-gate")
    commands = root.add_subparsers(dest="command", required=True)

    config = commands.add_parser("check-config", help="Validate all configured runtimes, databases, tools, and weights")
    config.add_argument("--config", required=True)
    config.add_argument("--output", default=None)

    pssm = commands.add_parser("pssm", help="Generate full-protein ASCII PSSMs with PSI-BLAST")
    pssm.add_argument("--fasta-dir", required=True)
    pssm.add_argument("--output-dir", required=True)
    pssm.add_argument("--psiblast", required=True)
    pssm.add_argument("--database", required=True)
    pssm.add_argument("--database-acquired-at", required=True, help="Swiss-Prot acquisition/release date, e.g. 2026-07-28")
    pssm.add_argument("--iterations", type=int, default=3)
    pssm.add_argument("--evalue", type=float, default=0.001)
    pssm.add_argument("--workers", type=int, default=1)
    pssm.add_argument("--threads-per-query", type=int, default=1)
    pssm.add_argument("--overwrite", action="store_true")

    prepare = commands.add_parser("prepare", help="Validate sites, source sequences, filter >1000 aa, and write FASTA")
    prepare.add_argument("--input", required=True)
    prepare.add_argument("--output-dir", required=True)
    prepare.add_argument("--accession-column", default="accession")
    prepare.add_argument("--position-column", default="position")
    prepare.add_argument("--residue-column", default="residue")
    prepare.add_argument("--sequence-column", default="sequence")
    prepare.add_argument("--sequence-source", choices=["auto", "provided", "uniprot"], default="auto")
    prepare.add_argument("--max-length", type=int, default=1000)
    prepare.add_argument("--fetch-workers", type=int, default=8)

    graphs = commands.add_parser("graphs", help="Convert SPOT probability maps to audited Top-3L graphs")
    graphs.add_argument("--probability-dir", required=True)
    graphs.add_argument("--output-dir", required=True)
    graphs.add_argument("--overwrite", action="store_true")

    windows = commands.add_parser("windows", help="Crop 43-residue PTM windows from full GATE embeddings")
    windows.add_argument("--sites", required=True)
    windows.add_argument("--gate-dir", required=True)
    windows.add_argument("--output-dir", required=True)
    windows.add_argument("--window", type=int, default=43)

    freeze = commands.add_parser("freeze", help="Train full-data deployment checkpoints on internal GATE windows")
    freeze.add_argument("--data-dir", required=True)
    freeze.add_argument("--output-dir", required=True)
    freeze.add_argument("--epochs", default='{"42":5,"2024":5,"888":8}')
    freeze.add_argument("--batch-size", type=int, default=64)
    freeze.add_argument("--learning-rate", type=float, default=3e-4)
    freeze.add_argument("--mixup-alpha", type=float, default=0.4)
    freeze.add_argument("--device", default=None)

    evaluate = commands.add_parser("evaluate", help="Reproduce the formal site-level 3-seed x 10-fold experiment")
    evaluate.add_argument("--data-dir", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--seeds", nargs="+", type=int, default=[42, 2024, 888])
    evaluate.add_argument("--folds", type=int, default=10)
    evaluate.add_argument("--epochs", type=int, default=50)
    evaluate.add_argument("--patience", type=int, default=5)
    evaluate.add_argument("--batch-size", type=int, default=64)
    evaluate.add_argument("--learning-rate", type=float, default=3e-4)
    evaluate.add_argument("--mixup-alpha", type=float, default=0.4)
    evaluate.add_argument("--device", default=None)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--resume", action="store_true")
    evaluate.add_argument("--max-folds", type=int, default=None, help="Debug only; full reproduction leaves this unset")

    predict = commands.add_parser("predict", help="Predict cancer association from GATE windows")
    predict.add_argument("--windows", required=True)
    predict.add_argument("--sites", required=True)
    predict.add_argument("--model-dir", required=True)
    predict.add_argument("--output", required=True)
    predict.add_argument("--threshold", type=float, default=0.5)
    predict.add_argument("--device", default=None)

    validate = commands.add_parser("validate", help="Audit sequence and array consistency across a run directory")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--require-predictions", action="store_true")
    validate.add_argument("--max-length", type=int, default=1000)
    return root


def main() -> None:
    args = parser().parse_args()
    if args.command == "check-config":
        report = check_config(args.config, args.output)
        print(json.dumps(report, ensure_ascii=False, indent=2))
    elif args.command == "prepare":
        prepare_sites(args.input, args.output_dir, args.accession_column, args.position_column, args.residue_column, args.sequence_column, args.sequence_source, args.max_length, args.fetch_workers)
    elif args.command == "pssm":
        generate_pssms(args.fasta_dir, args.output_dir, args.psiblast, args.database, args.database_acquired_at, args.iterations, args.evalue, args.workers, args.threads_per_query, args.overwrite)
    elif args.command == "graphs":
        build_graphs(args.probability_dir, args.output_dir, args.overwrite)
    elif args.command == "windows":
        build_windows(args.sites, args.gate_dir, args.output_dir, args.window)
    elif args.command == "freeze":
        epochs = {int(key): int(value) for key, value in json.loads(args.epochs).items()}
        freeze_models(args.data_dir, args.output_dir, epochs, args.batch_size, args.learning_rate, args.mixup_alpha, args.device)
    elif args.command == "evaluate":
        summary = evaluate_site_cv(
            args.data_dir,
            args.output_dir,
            args.seeds,
            args.folds,
            args.epochs,
            args.patience,
            args.batch_size,
            args.learning_rate,
            args.mixup_alpha,
            args.device,
            args.num_workers,
            args.resume,
            args.max_folds,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        predict_windows(args.windows, args.sites, args.model_dir, args.output, args.threshold, args.device)
    elif args.command == "validate":
        validate_run(args.run_dir, args.require_predictions, args.max_length)


if __name__ == "__main__":
    main()
