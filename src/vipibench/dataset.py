from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from vipibench.dataio import load_jsonl, sha256_file, write_json
from vipibench.schema import DatasetRecord
from vipibench.splits import audit_split_directory, split_records, write_splits
from vipibench.validation import validate_path


def _records_hash(records: list[DatasetRecord]) -> str:
    payload = "\n".join(
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False, sort_keys=True)
        for record in sorted(records, key=lambda item: item.sample_id)
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def freeze_dataset(
    dataset_path: Path,
    output_dir: Path,
    *,
    seed: int = 17,
    require_research_gates: bool = True,
    agreement_report_path: Path | None = None,
) -> dict[str, object]:
    validation = validate_path(
        dataset_path,
        require_research_gates=require_research_gates,
        agreement_report_path=agreement_report_path,
    )
    if validation.status != "PASS":
        raise ValueError(f"dataset validation failed: {validation.errors}")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"refusing to overwrite non-empty frozen split directory: {output_dir}"
        )
    records = load_jsonl(dataset_path)
    split_map = split_records(records, seed=seed)
    write_splits(output_dir, split_map)
    audit = audit_split_directory(output_dir)
    if audit.status != "PASS":
        raise ValueError(f"split audit failed: {audit.errors}")
    native_test = [
        record
        for record in split_map["test"]
        if record.language_form.value == "native_vi" and not record.fixture_only
    ]
    manifest = {
        "schema_version": "1.0.0",
        "status": "PASS",
        "frozen": True,
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256_file(dataset_path),
        "split_seed": seed,
        "group_keys": ["seed_id", "template_id", "semantic_cluster", "matched_pair_id"],
        "split_hashes": audit.split_hashes,
        "split_counts": audit.counts,
        "native_test_count": len(native_test),
        "native_test_sha256": _records_hash(native_test),
        "fixture_only": all(record.fixture_only for record in records),
        "agreement_report_sha256": (
            sha256_file(agreement_report_path) if agreement_report_path else None
        ),
    }
    write_json(output_dir / "split_manifest.json", manifest)
    return manifest


def role_label_leakage_report(dataset_path: Path, output_path: Path) -> dict[str, object]:
    records = [record for record in load_jsonl(dataset_path) if not record.fixture_only]
    cells = Counter((record.source_role.value, record.label.value) for record in records)
    author_cells = Counter((record.author_id, record.label.value) for record in records)
    roles = sorted({role for role, _ in cells})
    total = len(records)
    label_totals = Counter(record.label.value for record in records)
    role_totals = Counter(record.source_role.value for record in records)
    chi_square = 0.0
    if total:
        for role in roles:
            for label in ("benign", "injection"):
                expected = role_totals[role] * label_totals[label] / total
                if expected:
                    chi_square += (cells[(role, label)] - expected) ** 2 / expected
    cramers_v = (chi_square / total) ** 0.5 if total else None
    core_ratios: dict[str, float | None] = {}
    for role in ("user", "retrieved"):
        low = min(cells[(role, "benign")], cells[(role, "injection")])
        high = max(cells[(role, "benign")], cells[(role, "injection")])
        core_ratios[role] = low / high if high else None
    authors = sorted({record.author_id for record in records})
    author_totals = Counter(record.author_id for record in records)
    author_ratios: dict[str, float | None] = {}
    author_chi_square = 0.0
    for author in authors:
        low = min(author_cells[(author, "benign")], author_cells[(author, "injection")])
        high = max(author_cells[(author, "benign")], author_cells[(author, "injection")])
        author_ratios[author] = low / high if high else None
        for label in ("benign", "injection"):
            expected = author_totals[author] * label_totals[label] / total if total else 0
            if expected:
                author_chi_square += (author_cells[(author, label)] - expected) ** 2 / expected
    cramers_v_author = (author_chi_square / total) ** 0.5 if total else None
    role_gate = records and all((value or 0) >= 0.8 for value in core_ratios.values())
    author_gate = len(authors) >= 3 and all((value or 0) >= 0.8 for value in author_ratios.values())
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if role_gate and author_gate else "FAIL",
        "research_record_count": total,
        "dataset_sha256": sha256_file(dataset_path),
        "role_label_counts": {
            f"{role}:{label}": count for (role, label), count in sorted(cells.items())
        },
        "core_minority_majority_ratio": core_ratios,
        "cramers_v_role_label": cramers_v,
        "author_count": len(authors),
        "author_label_counts": {
            f"{author}:{label}": count for (author, label), count in sorted(author_cells.items())
        },
        "author_minority_majority_ratio": author_ratios,
        "cramers_v_author_label": cramers_v_author,
        "gate_minimum_balance_ratio": 0.8,
        "gate_minimum_author_count": 3,
        "interpretation": (
            "Distribution-only role/author diagnostic; trained role-only and author-only "
            "baselines are still required before scientific claims."
        ),
    }
    write_json(output_path, result)
    return result
