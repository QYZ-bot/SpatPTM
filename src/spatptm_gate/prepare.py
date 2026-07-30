from __future__ import annotations

import hashlib
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .common import write_json


ACCESSION_PATTERN = re.compile(r"^[A-Z0-9]+(?:-[0-9]+)?$")
PIPELINE_RESIDUES = set("ACDEFGHIKLMNPQRSTVWYX")
VALID_UNIPROT_RESIDUES = PIPELINE_RESIDUES | set("UOBZ")


def _clean_sequence(value: object) -> str:
    return re.sub(r"[^A-Za-z]", "", str(value)).upper()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, dtype=str, keep_default_na=False)
    if suffix in {".xlsx", ".xlsm"}:
        return pd.read_excel(path, dtype=str, keep_default_na=False)
    raise ValueError(f"Unsupported sites format {suffix}; use CSV or XLSX")


def _fetch_uniprot(accession: str, retries: int = 3) -> tuple[str, str, str]:
    url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "SpatPTM-GATE-Pipeline/0.1 (sequence provenance audit)"},
    )
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                text = response.read().decode("utf-8")
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            if not lines or not lines[0].startswith(">"):
                raise ValueError(f"UniProt did not return FASTA for {accession}")
            sequence = _clean_sequence("".join(lines[1:]))
            if not sequence:
                raise ValueError(f"UniProt returned an empty sequence for {accession}")
            return sequence, lines[0], url
        except (urllib.error.URLError, TimeoutError, ValueError) as error:
            last_error = error
            if attempt + 1 < retries:
                time.sleep(2**attempt)
    raise RuntimeError(f"Unable to retrieve UniProt sequence for {accession}: {last_error}")


def prepare_sites(
    input_path: str | Path,
    output_dir: str | Path,
    accession_column: str = "accession",
    position_column: str = "position",
    residue_column: str | None = "residue",
    sequence_column: str | None = "sequence",
    sequence_source: str = "auto",
    max_length: int = 1000,
    fetch_workers: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    if sequence_source not in {"auto", "provided", "uniprot"}:
        raise ValueError("sequence_source must be auto, provided, or uniprot")
    source = _read_table(input_path)
    missing = {accession_column, position_column} - set(source.columns)
    if missing:
        raise ValueError(f"Input table is missing columns: {sorted(missing)}")
    if sequence_source == "provided" and (not sequence_column or sequence_column not in source.columns):
        raise ValueError("Provided sequence mode requires an existing sequence column")

    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_dir = output_dir / "fasta"
    fasta_dir.mkdir(parents=True, exist_ok=True)
    retrieved_at = datetime.now(timezone.utc).isoformat()
    sequence_cache: dict[str, dict[str, object]] = {}
    fetched_sequences: dict[str, tuple[str, str, str]] = {}
    accepted: list[dict[str, object]] = []
    excluded: list[dict[str, object]] = []
    seen_sites: set[tuple[str, int]] = set()

    needs_uniprot: set[str] = set()
    if sequence_source != "provided":
        for _, row in source.iterrows():
            accession = str(row[accession_column]).strip().upper()
            if not ACCESSION_PATTERN.fullmatch(accession):
                continue
            provided_sequence = ""
            if sequence_column and sequence_column in source.columns:
                provided_sequence = _clean_sequence(row[sequence_column])
            if sequence_source == "uniprot" or not provided_sequence:
                needs_uniprot.add(accession)
    if needs_uniprot:
        failures: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, fetch_workers)) as executor:
            futures = {executor.submit(_fetch_uniprot, accession): accession for accession in sorted(needs_uniprot)}
            for future in as_completed(futures):
                accession = futures[future]
                try:
                    fetched_sequences[accession] = future.result()
                except Exception as error:
                    failures[accession] = str(error)
        if failures:
            details = "; ".join(f"{accession}: {message}" for accession, message in sorted(failures.items()))
            raise RuntimeError(f"Unable to retrieve {len(failures)} UniProt sequences: {details}")

    for source_index, row in source.reset_index(drop=True).iterrows():
        payload = row.to_dict()
        payload["source_row"] = source_index + 2
        accession = str(row[accession_column]).strip().upper()
        try:
            numeric_position = float(str(row[position_column]).strip())
            if not numeric_position.is_integer():
                raise ValueError
            position = int(numeric_position)
        except ValueError:
            payload["exclusion_reason"] = "invalid_position"
            excluded.append(payload)
            continue
        if not ACCESSION_PATTERN.fullmatch(accession):
            payload["exclusion_reason"] = "invalid_accession"
            excluded.append(payload)
            continue

        provided_sequence = ""
        if sequence_column and sequence_column in source.columns:
            provided_sequence = _clean_sequence(row[sequence_column])
        if accession not in sequence_cache:
            use_provided = sequence_source == "provided" or (sequence_source == "auto" and bool(provided_sequence))
            if use_provided:
                sequence, header, url, origin = provided_sequence, f">{accession} provided_sequence", "", "input_table"
                if not sequence:
                    raise ValueError(f"No provided sequence for {accession}")
            else:
                sequence, header, url = fetched_sequences[accession]
                origin = "UniProtKB REST canonical/isoform FASTA"
            invalid = sorted(set(sequence) - VALID_UNIPROT_RESIDUES)
            if invalid:
                raise ValueError(f"{accession}: unsupported residues in sequence: {invalid}")
            canonical_sequence = sequence
            normalization = [
                {"position": index + 1, "canonical_residue": residue, "pipeline_residue": "X"}
                for index, residue in enumerate(canonical_sequence)
                if residue not in PIPELINE_RESIDUES
            ]
            sequence = "".join(residue if residue in PIPELINE_RESIDUES else "X" for residue in canonical_sequence)
            sequence_cache[accession] = {
                "accession": accession,
                "sequence": sequence,
                "canonical_sequence": canonical_sequence,
                "length": len(sequence),
                "source": origin,
                "source_url": url,
                "fasta_header": header,
                "retrieved_at_utc": retrieved_at if url else "not_applicable_provided_by_user",
                "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
                "canonical_sequence_sha256": hashlib.sha256(canonical_sequence.encode("ascii")).hexdigest(),
                "pipeline_normalization": normalization,
            }
        record = sequence_cache[accession]
        sequence = str(record["sequence"])
        canonical_sequence = str(record["canonical_sequence"])
        if provided_sequence and provided_sequence != canonical_sequence:
            raise ValueError(f"Conflicting sequences supplied for {accession}")

        reason = ""
        if len(sequence) > max_length:
            reason = f"protein_length_gt_{max_length}"
        elif position < 1 or position > len(sequence):
            reason = "position_out_of_range"
        expected = ""
        if residue_column and residue_column in source.columns:
            expected = str(row[residue_column]).strip().upper()[:1]
        observed = canonical_sequence[position - 1] if 1 <= position <= len(canonical_sequence) else ""
        if not reason and expected and expected != observed:
            reason = "residue_mismatch"
        site_key = (accession, position)
        if not reason and site_key in seen_sites:
            reason = "duplicate_accession_position"

        normalized = payload | {
            "accession": accession,
            "position": position,
            "residue": expected or observed,
            "observed_residue": observed,
            "protein_length": len(sequence),
            "sequence_source": record["source"],
            "sequence_sha256": record["sequence_sha256"],
            "canonical_sequence_sha256": record["canonical_sequence_sha256"],
            "pipeline_normalization_count": len(record["pipeline_normalization"]),
        }
        if reason:
            normalized["exclusion_reason"] = reason
            excluded.append(normalized)
            continue
        seen_sites.add(site_key)
        accepted.append(normalized)

    if not accepted:
        raise RuntimeError("No eligible PTM sites remained after validation")
    accepted_frame = pd.DataFrame(accepted)
    excluded_frame = pd.DataFrame(excluded)
    if excluded_frame.empty:
        excluded_frame = pd.DataFrame(columns=[*accepted_frame.columns, "exclusion_reason"])
    accepted_frame.to_csv(output_dir / "sites.csv", index=False, encoding="utf-8-sig")
    excluded_frame.to_csv(output_dir / "excluded_sites.csv", index=False, encoding="utf-8-sig")

    eligible_accessions = sorted(set(accepted_frame["accession"]))
    provenance_rows = []
    for accession in eligible_accessions:
        record = sequence_cache[accession]
        fasta_path = fasta_dir / f"{accession}.fasta"
        sequence = str(record["sequence"])
        wrapped = "\n".join(sequence[index:index + 80] for index in range(0, len(sequence), 80))
        fasta_path.write_text(f">{accession}\n{wrapped}\n", encoding="ascii")
        provenance_rows.append({key: value for key, value in record.items() if key not in {"sequence", "canonical_sequence"}} | {"fasta": str(fasta_path)})
    pd.DataFrame(provenance_rows).to_csv(output_dir / "sequence_provenance.csv", index=False, encoding="utf-8-sig")
    write_json(output_dir / "prepare_audit.json", {
        "created_at_utc": retrieved_at,
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "coordinate_system": "1-based protein residue position",
        "sequence_policy": sequence_source,
        "uniprot_fetch_workers": max(1, fetch_workers),
        "nonstandard_residue_policy": "retain canonical UniProt sequence for coordinate/residue audit; map U/O/B/Z to X only in length-preserving pipeline FASTA for PSI-BLAST and SPOT compatibility",
        "length_rule": f"exclude protein length > {max_length} aa; length == {max_length} is retained",
        "deduplication": "accession + position; first valid occurrence retained",
        "eligible_sites": len(accepted_frame),
        "excluded_sites": len(excluded_frame),
        "eligible_proteins": len(eligible_accessions),
        "sequence_records": provenance_rows,
    })
    return accepted_frame, excluded_frame
