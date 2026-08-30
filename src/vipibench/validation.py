from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from vipibench.dataio import load_jsonl, sha256_file
from vipibench.schema import DatasetRecord, Label, SourceRole


@dataclass(frozen=True)
class ValidationResult:
    status: str
    path: str
    sha256: str
    record_count: int
    research_record_count: int
    fixture_record_count: int
    errors: list[str]
    warnings: list[str]
    counts: dict[str, dict[str, int]]
    research_gates: dict[str, object]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _counts(records: list[DatasetRecord]) -> dict[str, dict[str, int]]:
    return {
        "label": dict(Counter(record.label.value for record in records)),
        "source_role": dict(Counter(record.source_role.value for record in records)),
        "language_form": dict(Counter(record.language_form.value for record in records)),
        "role_label": dict(
            Counter(f"{record.source_role.value}:{record.label.value}" for record in records)
        ),
        "license_decision": dict(Counter(record.license_decision for record in records)),
    }


def validate_records(
    records: list[DatasetRecord],
    *,
    require_research_gates: bool = False,
) -> tuple[list[str], list[str], dict[str, object]]:
    errors: list[str] = []
    warnings: list[str] = []
    duplicate_ids = sorted(
        key
        for key, count in Counter(record.sample_id for record in records).items()
        if count > 1
    )
    if duplicate_ids:
        errors.append(f"duplicate sample_id values: {duplicate_ids[:10]}")

    research = [record for record in records if not record.fixture_only]
    role_label = Counter((record.source_role.value, record.label.value) for record in research)
    native_count = sum(record.language_form.value == "native_vi" for record in research)
    hard_negative_count = sum(record.hard_negative for record in research)
    pair_members: dict[str, list[DatasetRecord]] = {}
    for record in research:
        if record.matched_pair_id:
            pair_members.setdefault(record.matched_pair_id, []).append(record)
    valid_pairs = {
        pair_id
        for pair_id, members in pair_members.items()
        if len(members) == 2
        and {member.label for member in members} == {Label.BENIGN, Label.INJECTION}
        and len({member.source_role for member in members}) == 1
        and len({member.semantic_cluster for member in members}) == 1
    }
    invalid_pairs = sorted(set(pair_members) - valid_pairs)
    if invalid_pairs:
        errors.append(f"invalid matched pairs: {invalid_pairs[:10]}")

    gates: dict[str, object] = {
        "applicable": bool(research),
        "exact_target_total_2400": len(research) == 2400,
        "exact_template_families_80": len({record.source_family for record in research}) == 80,
        "minimum_native_fraction_0_5": bool(research) and native_count / len(research) >= 0.5,
        "minimum_hard_negatives_600": hard_negative_count >= 600,
        "minimum_matched_pairs_200": len(valid_pairs) >= 200,
        "all_licenses_resolved": bool(research)
        and all(
            record.license_decision in {"allowed", "internal_only"} for record in research
        ),
        "all_labels_have_non_llm_basis": bool(research)
        and all(
            record.label_basis.value in {"construction", "executable_oracle"}
            for record in research
        ),
        "minimum_core_cell_250": all(
            role_label[(role.value, label.value)] >= 250
            for role in (SourceRole.USER, SourceRole.RETRIEVED)
            for label in (Label.INJECTION, Label.BENIGN)
        ),
        "native_record_count": native_count,
        "hard_negative_count": hard_negative_count,
        "matched_pair_count": len(valid_pairs),
        "invalid_matched_pair_count": len(invalid_pairs),
    }
    if require_research_gates:
        if not research:
            errors.append("research gates requested but the input contains no research records")
        for key, value in gates.items():
            if key != "applicable" and isinstance(value, bool) and not value:
                errors.append(f"research gate failed: {key}")
    elif not research:
        warnings.append("fixture-only input: research dataset gates are not applicable")
    return errors, warnings, gates


def validate_path(path: Path, *, require_research_gates: bool = False) -> ValidationResult:
    records = load_jsonl(path)
    errors, warnings, gates = validate_records(
        records, require_research_gates=require_research_gates
    )
    research_count = sum(not record.fixture_only for record in records)
    return ValidationResult(
        status="PASS" if not errors else "FAIL",
        path=str(path),
        sha256=sha256_file(path),
        record_count=len(records),
        research_record_count=research_count,
        fixture_record_count=len(records) - research_count,
        errors=errors,
        warnings=warnings,
        counts=_counts(records),
        research_gates=gates,
    )
