from __future__ import annotations

from collections import Counter
from pathlib import Path

import yaml

from vipibench.dataio import sha256_file, write_json

REQUIRED_REQUIREMENT_IDS = {
    "SOURCE-001",
    "DATA-001",
    "DATA-002",
    "DATA-003",
    "DATA-004",
    "DATA-005",
    "DATA-006",
    "DATA-007",
    "DATA-008",
    "DATA-009",
    "MODEL-001",
    "MODEL-002",
    "MODEL-003",
    "MODEL-004",
    "EXPERIMENT-001",
    "EXPERIMENT-002",
    "EXPERIMENT-003",
    "EXPERIMENT-004",
    "GUARDRAIL-001",
    "GUARDRAIL-002",
    "GUARDRAIL-003",
    "GUARDRAIL-004",
    "COMPUTE-001",
    "GOVERNANCE-001",
    "REPORT-001",
}
REQUIRED_EXTERNAL_GATES = {
    "resource_measurement",
}
ALLOWED_STATUSES = {
    "PASS_ENGINEERING",
    "PENDING_IMPLEMENTATION",
    "PENDING_EXTERNAL",
    "DEFERRED_POST_RUN",
}
ALLOWED_PHASES = {"pre_run", "post_run"}
ALLOWED_PROPOSAL_SECTIONS = {
    "5.1",
    "6",
    "7",
    "8",
    "9",
    "11",
    "12",
    "14",
    "15",
    "19",
}


def _load_mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _relative_project_path(project_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    path = Path(value)
    if path.is_absolute():
        return None
    resolved = (project_root / path).resolve()
    try:
        resolved.relative_to(project_root.parent.resolve())
    except ValueError:
        return None
    return resolved


def audit_proposal_coverage(
    project_root: Path,
    manifest_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    errors: list[str] = []
    if not manifest_path.is_file():
        result = {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "errors": [f"missing proposal coverage manifest: {manifest_path}"],
        }
        if output_path:
            write_json(output_path, result)
        return result

    manifest = _load_mapping(manifest_path)
    raw_requirements = manifest.get("requirements")
    requirements = raw_requirements if isinstance(raw_requirements, list) else []
    requirement_ids = [
        str(item.get("id")) for item in requirements if isinstance(item, dict) and item.get("id")
    ]
    duplicate_ids = sorted(
        requirement_id for requirement_id, count in Counter(requirement_ids).items() if count > 1
    )
    missing_ids = sorted(REQUIRED_REQUIREMENT_IDS - set(requirement_ids))
    unknown_ids = sorted(set(requirement_ids) - REQUIRED_REQUIREMENT_IDS)
    if duplicate_ids:
        errors.append(f"duplicate requirement IDs: {duplicate_ids}")
    if missing_ids:
        errors.append(f"missing required proposal coverage IDs: {missing_ids}")
    if unknown_ids:
        errors.append(f"unknown proposal coverage IDs: {unknown_ids}")

    status_counts: Counter[str] = Counter()
    covered_external_gates: set[str] = set()
    missing_expected_artifacts: dict[str, list[str]] = {}
    for item in requirements:
        if not isinstance(item, dict):
            errors.append("proposal coverage requirements must be mappings")
            continue
        requirement_id = str(item.get("id", "<missing>"))
        status = str(item.get("status", ""))
        phase = str(item.get("phase", ""))
        section = str(item.get("proposal_section", ""))
        statement = item.get("requirement")
        verifier = item.get("verifier")
        if status not in ALLOWED_STATUSES:
            errors.append(f"invalid status for {requirement_id}: {status}")
        else:
            status_counts[status] += 1
        if phase not in ALLOWED_PHASES:
            errors.append(f"invalid phase for {requirement_id}: {phase}")
        if section not in ALLOWED_PROPOSAL_SECTIONS:
            errors.append(f"invalid proposal section for {requirement_id}: {section}")
        if not isinstance(statement, str) or not statement.strip():
            errors.append(f"missing requirement statement: {requirement_id}")
        if not isinstance(verifier, str) or not verifier.strip():
            errors.append(f"missing verifier: {requirement_id}")

        evidence = item.get("implemented_evidence")
        evidence_paths = evidence if isinstance(evidence, list) else []
        if not evidence_paths:
            errors.append(f"missing implemented evidence list: {requirement_id}")
        for evidence_value in evidence_paths:
            evidence_path = _relative_project_path(root, evidence_value)
            if evidence_path is None or not evidence_path.is_file():
                errors.append(
                    f"implemented evidence missing or outside workspace: "
                    f"{requirement_id}:{evidence_value}"
                )

        blocked_by = item.get("blocked_by")
        blocked_gates = set(blocked_by) if isinstance(blocked_by, list) else set()
        if not all(isinstance(gate, str) for gate in blocked_gates):
            errors.append(f"blocked_by must contain strings: {requirement_id}")
            blocked_gates = set()
        unknown_gates = blocked_gates - REQUIRED_EXTERNAL_GATES
        if unknown_gates:
            errors.append(f"unknown blocker gate for {requirement_id}: {sorted(unknown_gates)}")
        covered_external_gates.update(blocked_gates & REQUIRED_EXTERNAL_GATES)
        if status == "PENDING_EXTERNAL" and (phase != "pre_run" or not blocked_gates):
            errors.append(
                f"PENDING_EXTERNAL requires pre_run phase and blocked_by: {requirement_id}"
            )
        if status == "PASS_ENGINEERING" and (phase != "pre_run" or blocked_gates):
            errors.append(
                f"PASS_ENGINEERING requires pre_run phase and no blocked_by: {requirement_id}"
            )
        if status == "PENDING_IMPLEMENTATION" and (phase != "pre_run" or blocked_gates):
            errors.append(
                f"PENDING_IMPLEMENTATION requires pre_run phase and no blocked_by: "
                f"{requirement_id}"
            )
        if status == "DEFERRED_POST_RUN" and (phase != "post_run" or blocked_gates):
            errors.append(
                f"DEFERRED_POST_RUN requires post_run phase and no blocked_by: {requirement_id}"
            )

        expected = item.get("required_runtime_artifacts")
        expected_paths = expected if isinstance(expected, list) else []
        missing_runtime: list[str] = []
        for expected_value in expected_paths:
            expected_path = _relative_project_path(root, expected_value)
            if expected_path is None:
                errors.append(
                    f"runtime artifact path outside workspace: {requirement_id}:{expected_value}"
                )
            elif not expected_path.is_file():
                missing_runtime.append(str(expected_value))
        if missing_runtime:
            missing_expected_artifacts[requirement_id] = missing_runtime
        if status == "PASS_ENGINEERING" and missing_runtime:
            errors.append(f"PASS_ENGINEERING requirement lacks runtime artifacts: {requirement_id}")

    missing_gate_owners = sorted(REQUIRED_EXTERNAL_GATES - covered_external_gates)
    if missing_gate_owners:
        errors.append(f"external preflight gates lack proposal owners: {missing_gate_owners}")

    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "required_requirement_count": len(REQUIRED_REQUIREMENT_IDS),
        "observed_requirement_count": len(requirement_ids),
        "status_counts": dict(sorted(status_counts.items())),
        "covered_external_gates": sorted(covered_external_gates),
        "missing_expected_runtime_artifacts": missing_expected_artifacts,
        "coverage_complete": not missing_ids and not unknown_ids and not duplicate_ids,
        "readiness_inferred": False,
        "note": (
            "Coverage PASS proves that locked requirements have owners, evidence, and verifiers. "
            "It does not convert pending external or post-run evidence into readiness."
        ),
    }
    if output_path:
        write_json(output_path, result)
    return result
