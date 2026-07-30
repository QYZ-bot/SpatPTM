from __future__ import annotations

import json
from pathlib import Path

from .common import write_json


def check_config(config_path: str | Path, output_path: str | Path | None = None) -> dict[str, object]:
    config_path = Path(config_path).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    runtimes, blast, spot, gate = (config[key] for key in ("runtimes", "blast", "spot", "gate"))
    checks: list[dict[str, object]] = []

    def require(label: str, path: str | Path) -> Path:
        target = Path(path).resolve()
        exists = target.exists()
        checks.append({"label": label, "path": str(target), "exists": exists})
        if not exists:
            raise FileNotFoundError(f"{label}: {target}")
        return target

    require("model Python", runtimes["model_python"])
    require("SPOT Python", runtimes["spot_python"])
    require("GATE Python", runtimes["gate_python"])
    require("PSI-BLAST", blast["psiblast"])
    database = Path(blast["database"]).resolve()
    database_files = sorted(database.parent.glob(database.name + ".p*"))
    if not database_files:
        raise FileNotFoundError(f"Swiss-Prot database prefix: {database}")
    checks.append({"label": "Swiss-Prot database", "path": str(database), "exists": True, "components": len(database_files)})

    suffix = "gpu" if str(spot.get("device", "cpu")).startswith("cuda") else "cpu"
    spot1d_root = require("SPOT-1D-Single root", spot["spot1d_root"])
    require("SPOT-1D-Single entry point", spot1d_root / "spot1d_single.py")
    for name in ("model1_class", "model2_class", "model3_class", "model1_reg", "model2_reg", "model3_reg"):
        require(f"SPOT-1D weight {name}", spot1d_root / "jits" / f"{name}_{suffix}.pth")

    contact_root = require("SPOT-Contact-Single root", spot["spot_contact_root"])
    for name in ("atten_single_contact", "atten_only_contact", "atten_all_contact", "atten_sgl_dist", "atten_only_dist", "atten_all_dist"):
        require(f"SPOT contact weight {name}", contact_root / "contact_jits" / f"{name}_{suffix}.pth")
    torch_cache = Path(spot["torch_home"]).resolve() / "hub" / "checkpoints"
    require("ESM-1b checkpoint", torch_cache / "esm1b_t33_650M_UR50S.pt")
    require("ESM-1b contact regression", torch_cache / "esm1b_t33_650M_UR50S-contact-regression.pt")

    gate_root = require("GATE external code root", gate["external_code_root"])
    require("GATE trainer", gate_root / "feature_extraction" / "miRNA_disease_feature_extraction" / "gate_trainer.py")
    require("GATE model", gate_root / "feature_extraction" / "miRNA_disease_feature_extraction" / "gate.py")
    report = {"status": "passed", "config": str(config_path), "checks": checks}
    if output_path:
        write_json(output_path, report)
    return report
