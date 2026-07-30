from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import esm
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


BASES = "ARNDCQEGHILKMFPSTWYV"
ASA_MAX = dict(zip("ACDEFGHIKLMNPQRSTVWY-X", (115, 135, 150, 190, 210, 75, 195, 175, 200, 170, 185, 160, 145, 180, 225, 115, 140, 155, 255, 230, 1, 1)))


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sequence(path: Path) -> str:
    return "".join(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if not line.startswith(">"))


def one_hot(sequence: str) -> np.ndarray:
    bases = np.array(list(BASES))
    return np.concatenate([[(bases == residue.upper()).astype(int)] if residue.upper() in BASES else np.array([[-1] * 20]) for residue in sequence])


def pairwise(features: np.ndarray) -> np.ndarray:
    tiled = np.tile(features[None, :, :], (features.shape[0], 1, 1))
    return np.concatenate([tiled, np.transpose(tiled, (1, 0, 2))], axis=2)


def angle_features(values: np.ndarray) -> np.ndarray:
    radians = np.deg2rad(values[:, None])
    return (np.concatenate([np.sin(radians), np.cos(radians)], axis=1) + 1.0) / 2.0


def spot1d_features(path: Path, sequence: str) -> np.ndarray:
    data = pd.read_csv(path)
    ss3 = data[["P3C", "P3E", "P3H"]].to_numpy(dtype=np.float32)
    ss8 = data[["P8C", "P8S", "P8T", "P8H", "P8G", "P8I", "P8E", "P8B"]].to_numpy(dtype=np.float32)
    asa_denominator = np.asarray([ASA_MAX[residue] for residue in sequence], dtype=np.float32)[:, None]
    asa = np.clip(data["ASA"].to_numpy(dtype=np.float32)[:, None] / asa_denominator, 0, 1)
    scalar = data[["HseU", "HseD", "CN"]].to_numpy(dtype=np.float32)
    angles = np.concatenate([angle_features(data[column].to_numpy(dtype=np.float32)) for column in ("Psi", "Phi", "Theta", "Tau")], axis=1)
    result = np.concatenate([ss3, ss8, asa, scalar, angles], axis=1)
    if len(result) != len(sequence):
        raise ValueError(f"SPOT-1D length {len(result)} != sequence length {len(sequence)}")
    return result


def symmetrize(x: torch.Tensor) -> torch.Tensor:
    return x + x.transpose(-1, -2)


def apc(x: torch.Tensor) -> torch.Tensor:
    row = x.sum(-1, keepdim=True)
    column = x.sum(-2, keepdim=True)
    total = x.sum((-1, -2), keepdim=True)
    return x - (row * column / total)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the six-model SPOT-Contact-Single ensemble with one ESM pass per protein.")
    parser.add_argument("--fasta-dir", required=True)
    parser.add_argument("--spot1d-dir", required=True)
    parser.add_argument("--model-root", required=True, help="SPOT-Contact-Single directory containing contact_jits")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--esm-device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_device = torch.device(args.device)
    esm_device = torch.device(args.esm_device or args.device)
    root = Path(args.model_root).resolve()
    suffix = "gpu" if args.device.startswith("cuda") else "cpu"
    names = [
        f"atten_single_contact_{suffix}.pth", f"atten_only_contact_{suffix}.pth", f"atten_all_contact_{suffix}.pth",
        f"atten_sgl_dist_{suffix}.pth", f"atten_only_dist_{suffix}.pth", f"atten_all_dist_{suffix}.pth",
    ]
    model_paths = [root / "contact_jits" / name for name in names]
    for path in model_paths:
        if not path.exists():
            raise FileNotFoundError(path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    fastas = sorted(Path(args.fasta_dir).resolve().glob("*.fasta"))
    if not fastas:
        raise RuntimeError("No FASTA files found")
    pending = [
        fasta for fasta in fastas
        if args.overwrite or not all((output_dir / f"{fasta.stem}{suffix_name}.npy").exists() for suffix_name in ("", "_0", "_1", "_2", "_3", "_4", "_5"))
    ]
    language_model = alphabet = batch_converter = None
    contact_models = []
    if pending:
        print("[spot-contact] loading ESM-1b", flush=True)
        language_model, alphabet = esm.pretrained.esm1b_t33_650M_UR50S()
        language_model = language_model.to(esm_device).eval()
        batch_converter = alphabet.get_batch_converter()
        contact_models = [torch.jit.load(str(path), map_location=run_device).to(run_device).eval() for path in model_paths]

    for count, fasta in enumerate(fastas, start=1):
        sequence = read_sequence(fasta)
        accession = fasta.stem
        if fasta not in pending:
            ensemble = np.load(output_dir / f"{accession}.npy")
            if np.asarray(ensemble).squeeze().shape != (len(sequence), len(sequence)):
                raise ValueError(f"{accession}: cached contact map has wrong shape {ensemble.shape}")
            records.append({"accession": accession, "status": "reused", "length": len(sequence), "shape": list(ensemble.shape), "sha256": hashlib.sha256(np.ascontiguousarray(ensemble).view(np.uint8)).hexdigest()})
            print(f"[spot-contact] {count}/{len(fastas)} {accession} reused", flush=True)
            continue
        spot_csv = Path(args.spot1d_dir).resolve() / f"{accession}.csv"
        seq_pair = torch.from_numpy(pairwise(one_hot(sequence))).to(run_device, dtype=torch.float32)[None]
        spot_pair = torch.from_numpy(pairwise(spot1d_features(spot_csv, sequence))).to(run_device, dtype=torch.float32)[None]
        _, _, tokens = batch_converter([(accession, sequence)])
        with torch.no_grad():
            result = language_model(tokens.to(esm_device), repr_layers=[33], return_contacts=True)
        attention = result["attentions"][:, :, :, :-1, :-1][:, :, :, 1:, 1:]
        last_features = apc(symmetrize(attention[:, -1])).permute(0, 2, 3, 1).to(run_device, dtype=torch.float32)
        batch, layers, heads, length, _ = attention.shape
        all_features = apc(symmetrize(attention.reshape(batch, layers * heads, length, length))).permute(0, 2, 3, 1).to(run_device, dtype=torch.float32)
        del attention, result
        predictions = []
        generated_indices = []
        reused_indices = []
        for index, model in enumerate(contact_models):
            branch_path = output_dir / f"{accession}_{index}.npy"
            if branch_path.exists() and not args.overwrite:
                array = np.load(branch_path)
                if np.asarray(array).squeeze().shape != (len(sequence), len(sequence)):
                    raise ValueError(f"{accession}: cached branch {index} has wrong shape {array.shape}")
                predictions.append(array)
                reused_indices.append(index)
                continue
            base = last_features if index in (0, 3) else all_features
            features = torch.cat([base, seq_pair, spot_pair], dim=3) if index in (2, 5) else base
            with torch.no_grad():
                if index < 3:
                    prediction = model(features).squeeze(-1)
                else:
                    distance, _, _, _ = model(features)
                    prediction = F.softmax(distance, dim=3)[:, :, :, 1:13].sum(dim=3)
            array = prediction.cpu().numpy()
            np.save(branch_path, array)
            predictions.append(array)
            generated_indices.append(index)
        ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
        np.save(output_dir / f"{accession}.npy", ensemble)
        records.append({
            "accession": accession,
            "status": "generated",
            "length": len(sequence),
            "shape": list(ensemble.shape),
            "sha256": hashlib.sha256(np.ascontiguousarray(ensemble).view(np.uint8)).hexdigest(),
            "generated_branches": generated_indices,
            "reused_branches": reused_indices,
        })
        print(f"[spot-contact] {count}/{len(fastas)} {accession} L={len(sequence)}", flush=True)
    audit = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "upstream": "SPOT-Contact-Single six-model ensemble",
        "esm_dependency": "esm1b_t33_650M_UR50S used only to derive SPOT contact attention features",
        "classifier_node_feature": "not ESM; downstream GATE node feature is PSI-BLAST PSSM",
        "optimization": "one ESM attention forward per protein; mathematically identical attention transforms for six JIT models",
        "model_files": {path.name: hash_file(path) for path in model_paths},
        "esm_files": {
            path.name: {"path": str(path), "bytes": path.stat().st_size, "sha256": hash_file(path)}
            for path in (
                Path(torch.hub.get_dir()) / "checkpoints" / "esm1b_t33_650M_UR50S.pt",
                Path(torch.hub.get_dir()) / "checkpoints" / "esm1b_t33_650M_UR50S-contact-regression.pt",
            )
        },
        "records": records,
    }
    (output_dir / "spot_contact_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
