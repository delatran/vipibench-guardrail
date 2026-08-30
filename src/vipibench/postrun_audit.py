"""Post-run evidence auditor that cannot convert local fixtures into thesis claims."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path

import jsonschema

from vipibench.dataio import canonical_json, sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS
from vipibench.runtime_telemetry import strict_capacity_receipt_sha256, verify_telemetry_ledger

SCHEMA_VERSION = "1.0.0"
REQUIRED_PROTOCOL_AMENDMENT = "a100_80gb-response-truncation-guard-v8-2026-08-12"
REQUIRED_DURABLE_LINEAGE = "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12"
RUN_CONTEXT_FILENAME = "run_manifest.pre_audit.json"
POSTRUN_AUDIT_FILENAME = "postrun_audit.json"
FINAL_CLAIM_AUDIT_FILENAME = "final_claim_audit.json"
RAW_PREDICTIONS_MANIFEST = "raw_predictions_manifest.json"
RAW_TRAJECTORIES_MANIFEST = "raw_trajectories_manifest.json"

LOCKED_RUN_IDS = tuple(
    f"mdeberta-{mode}-s{seed}" for mode in LOCKED_INPUT_MODES for seed in LOCKED_SEEDS
)
RAW_PREDICTION_PATHS = tuple(
    f"mdeberta/{run_id}/test_predictions.jsonl" for run_id in LOCKED_RUN_IDS
)
RAW_TRAJECTORY_PATHS = (
    "core_target_trajectories.jsonl",
    "attack_target_trajectories.jsonl",
)
REQUIRED_ARTIFACTS = {
    "launch_authorization": "launch_authorization.json",
    "strict_capacity_receipt": "strict_capacity_receipt.json",
    "encoder_matrix": "mdeberta/matrix_manifest.json",
    "model_selection": "mdeberta/model_selection.json",
    "encoder_ablation": "encoder_ablation_analysis.json",
    "static_analysis": "static_analysis.json",
    "rq2_analysis": "rq2_analysis.json",
    "h3_analysis": "h3_analysis.json",
    "adaptive_validity": "attack_candidates.validity.json",
    "adaptive_analysis": "adaptive_analysis.json",
    "runtime_telemetry": "runtime_telemetry.json",
    "fallback_ledger": "final_holdout_format_fallback_ledger.json",
    "failure_ledger": "failure_ledger.json",
}
REQUIRED_PROJECT_BINDINGS = (
    "requirements-experiment.lock",
    "configs/experiments/exec_system.yaml",
    "configs/experiments/confirmatory_analysis.yaml",
    "configs/models/mdeberta_core.yaml",
    "configs/models/target_agent.yaml",
    "configs/resources/confirmatory_stage_plan.json",
    "configs/generation/adaptive_generator.yaml",
    "data/splits/confirmatory_final/manifest.json",
    "data/splits/confirmatory_final/test.jsonl",
    "src/vipibench/outcome_contract.py",
    "src/vipibench/metrics.py",
    "src/vipibench/transformer_runner.py",
    "src/vipibench/ablation_analysis.py",
    "src/vipibench/rq2_analysis.py",
    "src/vipibench/h3_contract.py",
    "src/vipibench/h3_analysis.py",
    "src/vipibench/system_runner.py",
    "src/vipibench/system_analysis.py",
    "src/vipibench/adaptive_runner.py",
    "src/vipibench/adaptive_analysis.py",
    "src/vipibench/postrun_audit.py",
)


def audit_postrun(
    project_root: Path,
    output_root: Path,
    *,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Audit a retained run directory and write no final claim audit unless eligible."""

    root = project_root.resolve()
    run_root = output_root.resolve()
    result_path = output_path or run_root / POSTRUN_AUDIT_FILENAME
    errors: list[str] = []
    if not root.is_dir():
        raise ValueError(f"project root missing: {root}")
    if not run_root.is_dir():
        errors.append("run_root_missing")
        result = audit_postrun_records(
            project_root=root,
            run_root=run_root,
            run_context={},
            artifacts={},
            strict_receipt=None,
            runtime_telemetry=None,
            input_errors=errors,
        )
        validate_postrun_audit_schema(root, result)
        write_json(result_path, result)
        return result

    context_path = run_root / RUN_CONTEXT_FILENAME
    if (run_root / FINAL_CLAIM_AUDIT_FILENAME).exists():
        errors.append("final_claim_audit_must_be_generated_by_current_postrun_audit")
    run_context = _load_json(context_path, "run context", errors)
    artifacts = {
        name: _load_json(run_root / relative_path, name, errors)
        for name, relative_path in REQUIRED_ARTIFACTS.items()
    }
    strict_receipt = artifacts.get("strict_capacity_receipt")
    runtime_telemetry = artifacts.get("runtime_telemetry")
    _validate_run_context(root, run_context, errors)
    _validate_required_artifacts(artifacts, errors)
    _validate_run_artifact_bindings(run_root, run_context, errors)
    _validate_launch_authorization(artifacts.get("launch_authorization"), errors)
    _validate_encoder_runs(run_root, artifacts.get("encoder_matrix"), errors)
    _validate_fallback_and_failure_ledgers(artifacts, errors)
    raw_prediction_manifest = _build_raw_manifest_if_complete(
        run_root,
        RAW_PREDICTION_PATHS,
        run_root / RAW_PREDICTIONS_MANIFEST,
        kind="raw_predictions",
        expected_row_counts=_expected_raw_row_counts(run_context, RAW_PREDICTION_PATHS, errors),
        errors=errors,
    )
    raw_trajectory_manifest = _build_raw_manifest_if_complete(
        run_root,
        RAW_TRAJECTORY_PATHS,
        run_root / RAW_TRAJECTORIES_MANIFEST,
        kind="raw_trajectories",
        expected_row_counts=_expected_raw_row_counts(run_context, RAW_TRAJECTORY_PATHS, errors),
        errors=errors,
    )
    result = audit_postrun_records(
        project_root=root,
        run_root=run_root,
        run_context=run_context,
        artifacts=artifacts,
        strict_receipt=strict_receipt,
        runtime_telemetry=runtime_telemetry,
        raw_prediction_manifest=raw_prediction_manifest,
        raw_trajectory_manifest=raw_trajectory_manifest,
        input_errors=errors,
    )
    if result["research_evidence_eligible"] is True:
        preimage_sha256 = final_claim_audit_preimage_sha256(result)
        final_claim_audit = _build_final_claim_audit(result, preimage_sha256)
        write_json(run_root / FINAL_CLAIM_AUDIT_FILENAME, final_claim_audit)
        result["final_claim_audit_created"] = True
        result["final_claim_audit_sha256"] = sha256_file(run_root / FINAL_CLAIM_AUDIT_FILENAME)
        result["final_claim_audit_preimage_sha256"] = preimage_sha256
    validate_postrun_audit_schema(root, result)
    write_json(result_path, result)
    return result


def build_postrun_raw_manifests(
    *,
    project_root: Path,
    output_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    """Build retained raw-artifact manifests before the final post-run audit."""

    root = project_root.resolve()
    run_root = output_root.resolve()
    result_path = output_path or run_root / "postrun_raw_manifests_stage.json"
    errors: list[str] = []
    context = _load_json(run_root / RUN_CONTEXT_FILENAME, "run context", errors)
    _validate_run_context(root, context, errors)
    expected_rows = _expected_raw_row_counts(
        context,
        (*RAW_PREDICTION_PATHS, *RAW_TRAJECTORY_PATHS),
        errors,
    )
    predictions = _build_raw_manifest_if_complete(
        run_root,
        RAW_PREDICTION_PATHS,
        run_root / RAW_PREDICTIONS_MANIFEST,
        kind="raw_predictions",
        expected_row_counts=expected_rows,
        errors=errors,
    )
    trajectories = _build_raw_manifest_if_complete(
        run_root,
        RAW_TRAJECTORY_PATHS,
        run_root / RAW_TRAJECTORIES_MANIFEST,
        kind="raw_trajectories",
        expected_row_counts=expected_rows,
        errors=errors,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS"
        if not errors and predictions is not None and trajectories is not None
        else "FAIL",
        "errors": sorted(set(errors)),
        "raw_predictions_manifest_sha256": (
            sha256_file(run_root / RAW_PREDICTIONS_MANIFEST)
            if (run_root / RAW_PREDICTIONS_MANIFEST).is_file()
            else None
        ),
        "raw_trajectories_manifest_sha256": (
            sha256_file(run_root / RAW_TRAJECTORIES_MANIFEST)
            if (run_root / RAW_TRAJECTORIES_MANIFEST).is_file()
            else None
        ),
        "claim_boundary": (
            "Raw manifests retain evidence only; the final audit decides completion "
            "and eligibility."
        ),
    }
    write_json(result_path, result)
    return result


def audit_postrun_records(
    *,
    project_root: Path,
    run_root: Path,
    run_context: Mapping[str, object],
    artifacts: Mapping[str, Mapping[str, object]],
    strict_receipt: Mapping[str, object] | None,
    runtime_telemetry: Mapping[str, object] | None,
    raw_prediction_manifest: Mapping[str, object] | None = None,
    raw_trajectory_manifest: Mapping[str, object] | None = None,
    input_errors: Sequence[str] = (),
    observed_hardware_override: bool | None = None,
    telemetry_live_override: bool | None = None,
) -> dict[str, object]:
    """Pure audit core used by fixture tests; wrapper owns all filesystem loading."""

    errors = list(input_errors)
    checks: list[dict[str, object]] = []
    _record_check(
        checks,
        "run_context",
        bool(run_context) and run_context.get("status") == "PASS",
        run_context,
    )
    observed_hardware = (
        observed_hardware_override
        if observed_hardware_override is not None
        else _validate_observed_hardware(strict_receipt, errors)
    )
    if observed_hardware is not True:
        errors.append("strict_capacity_receipt_not_observed")
    _record_check(checks, "observed_a100_80gb_receipt", observed_hardware, strict_receipt or {})
    telemetry_live = (
        telemetry_live_override
        if telemetry_live_override is not None
        else _validate_live_telemetry(runtime_telemetry, strict_receipt, errors)
    )
    if telemetry_live is not True:
        errors.append("runtime_telemetry_not_observed_live")
    _record_check(checks, "runtime_telemetry_live", telemetry_live, runtime_telemetry or {})
    for name in REQUIRED_ARTIFACTS:
        artifact = artifacts.get(name)
        valid = _artifact_is_pass(name, artifact)
        _record_check(checks, f"artifact:{name}", valid, artifact or {})
        if not valid:
            errors.append(f"required_artifact_invalid:{name}")
    for name, manifest in (
        ("raw_predictions_manifest", raw_prediction_manifest),
        ("raw_trajectories_manifest", raw_trajectory_manifest),
    ):
        valid = _raw_manifest_is_valid(manifest)
        _record_check(checks, name, valid, manifest or {})
        if not valid:
            errors.append(f"raw_manifest_invalid:{name}")

    fixture_only = _contains_fixture_marker(run_context) or any(
        _contains_fixture_marker(artifact) for artifact in artifacts.values()
    )
    if raw_prediction_manifest is not None:
        fixture_only = fixture_only or _contains_fixture_marker(raw_prediction_manifest)
    if raw_trajectory_manifest is not None:
        fixture_only = fixture_only or _contains_fixture_marker(raw_trajectory_manifest)
    analyses = {
        "RQ1": artifacts.get("encoder_ablation", {}),
        "H2": artifacts.get("rq2_analysis", {}),
        "RQ2": artifacts.get("rq2_analysis", {}),
        "RQ3": artifacts.get("static_analysis", {}),
        "H3": artifacts.get("h3_analysis", {}),
        "H4": artifacts.get("adaptive_analysis", {}),
    }
    structural_complete = not errors
    dispositions = {
        "engineering_completeness": (
            "COMPLETE_FIXTURE_ONLY"
            if structural_complete and fixture_only
            else "COMPLETE_LIVE"
            if structural_complete
            else "INCOMPLETE"
        ),
        "RUN_COMPLETE": "PASS" if structural_complete else "FAIL",
    }
    for claim, artifact in analyses.items():
        dispositions[claim] = _claim_disposition(
            claim,
            artifact,
            structural_complete=structural_complete,
            fixture_only=fixture_only,
        )
    research_eligible = (
        structural_complete
        and not fixture_only
        and all(dispositions[claim] == "ELIGIBLE_FOR_REPORTING" for claim in analyses)
    )
    dispositions["final_E3_eligibility"] = (
        "ELIGIBLE_FOR_REPORTING" if research_eligible else "NOT_ELIGIBLE"
    )
    dispositions["RESEARCH_EVIDENCE_ELIGIBLE"] = research_eligible
    source_hashes = {
        name: sha256_file(run_root / relative_path)
        for name, relative_path in REQUIRED_ARTIFACTS.items()
        if (run_root / relative_path).is_file()
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if structural_complete else "FAIL",
        "errors": sorted(set(errors)),
        "project_root": str(project_root),
        "output_root": str(run_root),
        "runtime_source_fingerprint": runtime_source_fingerprint(project_root),
        "checks": checks,
        "raw_artifact_manifests": {
            "raw_predictions_manifest": raw_prediction_manifest,
            "raw_trajectories_manifest": raw_trajectory_manifest,
        },
        "artifact_sha256": source_hashes,
        "fixture_only": fixture_only,
        "dispositions": dispositions,
        "research_evidence_eligible": research_eligible,
        "final_claim_audit_created": False,
        "final_claim_audit_sha256": None,
        "final_claim_audit_preimage_sha256": None,
        "claim_boundary": (
            "A structurally complete fixture remains fixture-ineligible. Negative, null, or "
            "underpowered outcomes may be reportable only when all raw live evidence is complete; "
            "no component can promote another claim."
        ),
    }
    return result


def validate_postrun_audit_schema(project_root: Path, result: Mapping[str, object]) -> None:
    schema_path = project_root / "outputs" / "postrun_audit.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator(schema).validate(dict(result))


def validate_postrun_audit_for_finalization(
    *,
    project_root: Path,
    output_root: Path,
    postrun_audit_path: Path,
) -> tuple[dict[str, object], list[str]]:
    """Revalidate a terminal audit against the retained current-run evidence.

    The terminal manifest is a claim boundary, so it must not trust a caller
    supplied JSON object.  This verifier binds the canonical in-run audit to
    the current source fingerprint, the pre-audit context, every required
    artifact, both raw manifests, and (when eligible) its non-cyclic final
    claim-audit preimage.
    """

    root = project_root.resolve()
    run_root = output_root.resolve()
    audit_path = postrun_audit_path.resolve()
    errors: list[str] = []
    expected_audit_path = (run_root / POSTRUN_AUDIT_FILENAME).resolve()
    if audit_path != expected_audit_path:
        errors.append("postrun_audit_path_must_be_current_run_artifact")
    if postrun_audit_path.is_symlink() or not audit_path.is_file():
        errors.append("postrun_audit_missing_or_unsafe")
        return {}, sorted(set(errors))
    audit = _load_json(audit_path, "postrun_audit", errors)
    try:
        validate_postrun_audit_schema(root, audit)
    except (jsonschema.ValidationError, OSError, UnicodeError, json.JSONDecodeError) as exc:
        errors.append(f"postrun_audit_schema_invalid:{type(exc).__name__}")
    _validate_terminal_audit_fields(root, run_root, audit, errors)
    _validate_terminal_audit_bindings(root, run_root, audit, errors)
    _validate_terminal_final_claim_audit(run_root, audit, errors)
    return audit, sorted(set(errors))


def _validate_run_context(
    project_root: Path,
    run_context: Mapping[str, object],
    errors: list[str],
) -> None:
    if run_context.get("status") != "PASS":
        errors.append("run_context_not_pass")
    if run_context.get("runtime_source_fingerprint") != runtime_source_fingerprint(project_root):
        errors.append("run_context_runtime_fingerprint_mismatch")
    bindings = run_context.get("project_bindings")
    if not isinstance(bindings, dict):
        errors.append("run_context_project_bindings_missing")
        return
    for relative_path in REQUIRED_PROJECT_BINDINGS:
        expected = bindings.get(relative_path)
        path = project_root / relative_path
        if not path.is_file() or not _is_sha256(expected) or sha256_file(path) != expected:
            errors.append(f"run_context_binding_mismatch:{relative_path}")


def _validate_required_artifacts(
    artifacts: Mapping[str, Mapping[str, object]], errors: list[str]
) -> None:
    for name in REQUIRED_ARTIFACTS:
        artifact = artifacts.get(name)
        if not isinstance(artifact, Mapping):
            errors.append(f"required_artifact_missing:{name}")
            continue
        if _contains_nonfinite(artifact):
            errors.append(f"required_artifact_nonfinite:{name}")
        if not _artifact_is_pass(name, artifact):
            errors.append(f"required_artifact_status:{name}")


def _validate_run_artifact_bindings(
    run_root: Path,
    run_context: Mapping[str, object],
    errors: list[str],
) -> None:
    bindings = run_context.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        errors.append("run_context_artifact_bindings_missing")
        return
    if set(bindings) != set(REQUIRED_ARTIFACTS):
        errors.append("run_context_artifact_binding_names_mismatch")
        return
    for name, relative_path in REQUIRED_ARTIFACTS.items():
        expected = bindings.get(name)
        path = run_root / relative_path
        if not path.is_file() or not _is_sha256(expected) or sha256_file(path) != expected:
            errors.append(f"run_context_artifact_binding_mismatch:{name}")


def _validate_launch_authorization(
    authorization: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if not isinstance(authorization, Mapping):
        errors.append("launch_authorization_missing")
        return
    if authorization.get("confirmatory_execution_approved") is not True:
        errors.append("launch_authorization_confirmatory_scope_missing")
    if authorization.get("paid_compute_approved") is not True:
        errors.append("launch_authorization_paid_compute_scope_missing")
    if authorization.get("protocol_amendment") != REQUIRED_PROTOCOL_AMENDMENT:
        errors.append("launch_authorization_protocol_amendment_mismatch")
    if authorization.get("durable_lineage") != REQUIRED_DURABLE_LINEAGE:
        errors.append("launch_authorization_durable_lineage_mismatch")


def _validate_encoder_runs(
    run_root: Path,
    matrix: Mapping[str, object] | None,
    errors: list[str],
) -> None:
    if not isinstance(matrix, Mapping):
        errors.append("encoder_matrix_missing")
        return
    if matrix.get("required_run_count") != len(LOCKED_RUN_IDS):
        errors.append("encoder_matrix_run_count_mismatch")
    completed = matrix.get("completed_runs")
    if not isinstance(completed, list) or set(completed) != set(LOCKED_RUN_IDS):
        errors.append("encoder_matrix_completed_runs_mismatch")
    selection_path = run_root / REQUIRED_ARTIFACTS["model_selection"]
    selection = _load_json(selection_path, "model selection", errors)
    if selection.get("status") != "PASS":
        errors.append("model_selection_not_pass")
    for run_id in LOCKED_RUN_IDS:
        for filename in ("training_decision.json", "test_manifest.json"):
            artifact = _load_json(run_root / "mdeberta" / run_id / filename, filename, errors)
            if artifact.get("status") != "PASS" or artifact.get("run_id") != run_id:
                errors.append(f"encoder_run_decision_invalid:{run_id}:{filename}")


def _validate_fallback_and_failure_ledgers(
    artifacts: Mapping[str, Mapping[str, object]], errors: list[str]
) -> None:
    fallback = artifacts.get("fallback_ledger", {})
    records = fallback.get("fallback_records") if isinstance(fallback, Mapping) else None
    if not isinstance(records, dict):
        errors.append("fallback_ledger_records_missing")
    elif records and fallback.get("status") != "PASS":
        errors.append("fallback_ledger_invalid")
    failure = artifacts.get("failure_ledger", {})
    if not isinstance(failure.get("events") if isinstance(failure, Mapping) else None, list):
        errors.append("failure_ledger_events_missing")
    repair_observations = (
        failure.get("repair_observations") if isinstance(failure, Mapping) else None
    )
    if not isinstance(repair_observations, list):
        errors.append("failure_ledger_repair_observations_missing")
    elif any(
        not isinstance(item, Mapping)
        or item.get("kind") != "bounded_schema_repair"
        or not isinstance(item.get("stage"), str)
        or isinstance(item.get("count"), bool)
        or not isinstance(item.get("count"), int)
        or int(item.get("count", 0)) <= 0
        for item in repair_observations
    ):
        errors.append("failure_ledger_repair_observations_invalid")


def _build_raw_manifest_if_complete(
    run_root: Path,
    allowlist: Sequence[str],
    output_path: Path,
    *,
    kind: str,
    expected_row_counts: Mapping[str, int] | None,
    errors: list[str],
) -> dict[str, object] | None:
    rows: list[dict[str, object]] = []
    for relative_path in allowlist:
        path = _safe_member(run_root, relative_path)
        if path is None or not path.is_file() or path.is_symlink():
            errors.append(f"raw_artifact_missing_or_unsafe:{kind}:{relative_path}")
            return None
        count, fixture_row_count = _jsonl_summary(path)
        if count <= 0:
            errors.append(f"raw_artifact_empty_or_invalid:{kind}:{relative_path}")
            return None
        if expected_row_counts is None or expected_row_counts.get(relative_path) != count:
            errors.append(f"raw_artifact_row_count_mismatch:{kind}:{relative_path}")
            return None
        rows.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "row_count": count,
                "fixture_row_count": fixture_row_count,
            }
        )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "kind": kind,
        "status": "PASS",
        "allowlist": list(allowlist),
        "artifacts": rows,
        "fixture_only": any(int(row["fixture_row_count"]) > 0 for row in rows),
    }
    manifest["manifest_sha256"] = _sha256_payload(manifest)
    write_json(output_path, manifest)
    return manifest


def _validate_observed_hardware(
    receipt: Mapping[str, object] | None,
    errors: list[str],
) -> bool:
    if receipt is None:
        errors.append("strict_capacity_receipt_missing")
        return False
    try:
        strict_capacity_receipt_sha256(receipt)
    except ValueError as exc:
        errors.append(f"strict_capacity_receipt_invalid:{exc}")
        return False
    return True


def _validate_live_telemetry(
    telemetry: Mapping[str, object] | None,
    receipt: Mapping[str, object] | None,
    errors: list[str],
) -> bool:
    if telemetry is None:
        errors.append("runtime_telemetry_missing")
        return False
    try:
        validated = verify_telemetry_ledger(telemetry, strict_capacity_receipt=receipt)
    except ValueError as exc:
        errors.append(f"runtime_telemetry_invalid:{exc}")
        return False
    if validated["local_only"] is True or validated["hardware_observed"] is not True:
        errors.append("runtime_telemetry_not_observed_live")
        return False
    return True


def _claim_disposition(
    claim: str,
    artifact: Mapping[str, object],
    *,
    structural_complete: bool,
    fixture_only: bool,
) -> str:
    if not structural_complete:
        return "NOT_ELIGIBLE_INCOMPLETE_EVIDENCE"
    if fixture_only:
        return "NOT_ELIGIBLE_FIXTURE_ONLY"
    if claim == "RQ1":
        if artifact.get("research_claim_eligible") is True:
            return "ELIGIBLE_FOR_REPORTING"
        return "INCONCLUSIVE_OR_INVALID"
    if claim == "H2":
        identity = artifact.get("h2_counterfactual_identity")
        if isinstance(identity, Mapping) and identity.get("status") == "PASS":
            return "ELIGIBLE_FOR_REPORTING"
        return "INCONCLUSIVE_OR_INVALID"
    if artifact.get("research_claim_eligible") is True:
        return "ELIGIBLE_FOR_REPORTING"
    return "INCONCLUSIVE_OR_INVALID"


def final_claim_audit_preimage_sha256(postrun_audit: Mapping[str, object]) -> str:
    """Return the stable audit preimage bound by an eligible claim audit.

    A final audit cannot byte-hash the terminal post-run audit directly because
    the terminal audit records the final audit's own hash.  Normalising the
    three lifecycle fields breaks that cycle while still binding every other
    audit field deterministically.
    """

    preimage = dict(postrun_audit)
    preimage["final_claim_audit_created"] = False
    preimage["final_claim_audit_sha256"] = None
    preimage["final_claim_audit_preimage_sha256"] = None
    return hashlib.sha256(canonical_json(preimage).encode("utf-8")).hexdigest().upper()


def _build_final_claim_audit(
    postrun_audit: Mapping[str, object],
    postrun_audit_preimage_sha256: str,
) -> dict[str, object]:
    raw_manifests = postrun_audit["raw_artifact_manifests"]
    if not isinstance(raw_manifests, Mapping):
        raise ValueError("postrun audit raw manifest bindings missing")
    predictions = raw_manifests.get("raw_predictions_manifest")
    trajectories = raw_manifests.get("raw_trajectories_manifest")
    if not isinstance(predictions, Mapping) or not isinstance(trajectories, Mapping):
        raise ValueError("postrun audit raw manifests invalid")
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "EVIDENCE_RECONCILED",
        "postrun_audit_preimage_sha256": postrun_audit_preimage_sha256,
        "runtime_source_fingerprint": postrun_audit["runtime_source_fingerprint"],
        "raw_predictions_manifest_sha256": predictions.get("manifest_sha256"),
        "raw_trajectories_manifest_sha256": trajectories.get("manifest_sha256"),
        "dispositions": postrun_audit["dispositions"],
        "artifact_sha256": postrun_audit["artifact_sha256"],
        "claim_boundary": "Generated only from a schema-valid eligible post-run audit.",
    }


def _validate_terminal_audit_fields(
    project_root: Path,
    run_root: Path,
    audit: Mapping[str, object],
    errors: list[str],
) -> None:
    if audit.get("status") != "PASS" or audit.get("errors") != []:
        errors.append("postrun_audit_not_terminal_pass")
    if audit.get("project_root") != str(project_root):
        errors.append("postrun_audit_project_root_mismatch")
    if audit.get("output_root") != str(run_root):
        errors.append("postrun_audit_output_root_mismatch")
    if audit.get("runtime_source_fingerprint") != runtime_source_fingerprint(project_root):
        errors.append("postrun_audit_runtime_fingerprint_mismatch")
    if audit.get("fixture_only") is not False:
        errors.append("postrun_audit_fixture_only")
    dispositions = audit.get("dispositions")
    if not isinstance(dispositions, Mapping):
        errors.append("postrun_audit_dispositions_missing")
        return
    required = {
        "engineering_completeness",
        "RUN_COMPLETE",
        "RQ1",
        "H2",
        "RQ2",
        "RQ3",
        "H3",
        "H4",
        "final_E3_eligibility",
        "RESEARCH_EVIDENCE_ELIGIBLE",
    }
    if set(dispositions) != required:
        errors.append("postrun_audit_disposition_names_mismatch")
    if dispositions.get("engineering_completeness") != "COMPLETE_LIVE":
        errors.append("postrun_audit_not_complete_live")
    if dispositions.get("RUN_COMPLETE") != "PASS":
        errors.append("postrun_audit_run_complete_not_pass")
    research_eligible = audit.get("research_evidence_eligible")
    if not isinstance(research_eligible, bool):
        errors.append("postrun_audit_research_eligibility_missing")
        return
    if dispositions.get("RESEARCH_EVIDENCE_ELIGIBLE") is not research_eligible:
        errors.append("postrun_audit_research_eligibility_inconsistent")
    expected_e3 = "ELIGIBLE_FOR_REPORTING" if research_eligible else "NOT_ELIGIBLE"
    if dispositions.get("final_E3_eligibility") != expected_e3:
        errors.append("postrun_audit_final_e3_disposition_inconsistent")
    if research_eligible and any(
        dispositions.get(name) != "ELIGIBLE_FOR_REPORTING"
        for name in ("RQ1", "H2", "RQ2", "RQ3", "H3", "H4")
    ):
        errors.append("postrun_audit_eligible_claim_disposition_invalid")


def _validate_terminal_audit_bindings(
    project_root: Path,
    run_root: Path,
    audit: Mapping[str, object],
    errors: list[str],
) -> None:
    run_context = _load_run_json(run_root, RUN_CONTEXT_FILENAME, "run_context", errors)
    _validate_run_context(project_root, run_context, errors)
    _validate_run_artifact_bindings(run_root, run_context, errors)

    artifacts = {
        name: _load_run_json(run_root, relative_path, name, errors)
        for name, relative_path in REQUIRED_ARTIFACTS.items()
    }
    _validate_required_artifacts(artifacts, errors)
    _validate_launch_authorization(artifacts.get("launch_authorization"), errors)
    _validate_encoder_runs(run_root, artifacts.get("encoder_matrix"), errors)
    _validate_fallback_and_failure_ledgers(artifacts, errors)
    observed_hardware = _validate_observed_hardware(
        artifacts.get("strict_capacity_receipt"), errors
    )
    telemetry_live = _validate_live_telemetry(
        artifacts.get("runtime_telemetry"), artifacts.get("strict_capacity_receipt"), errors
    )
    if observed_hardware is not True:
        errors.append("terminal_audit_hardware_not_observed")
    if telemetry_live is not True:
        errors.append("terminal_audit_telemetry_not_live")

    observed_hashes = audit.get("artifact_sha256")
    if not isinstance(observed_hashes, Mapping) or set(observed_hashes) != set(REQUIRED_ARTIFACTS):
        errors.append("postrun_audit_artifact_hash_names_mismatch")
    else:
        for name, relative_path in REQUIRED_ARTIFACTS.items():
            path = run_root / relative_path
            if (
                path.is_symlink()
                or not path.is_file()
                or observed_hashes.get(name) != sha256_file(path)
            ):
                errors.append(f"postrun_audit_artifact_hash_mismatch:{name}")

    raw_bindings = audit.get("raw_artifact_manifests")
    if not isinstance(raw_bindings, Mapping):
        errors.append("postrun_audit_raw_manifests_missing")
        return
    for name, filename in (
        ("raw_predictions_manifest", RAW_PREDICTIONS_MANIFEST),
        ("raw_trajectories_manifest", RAW_TRAJECTORIES_MANIFEST),
    ):
        actual = _load_run_json(run_root, filename, name, errors)
        recorded = raw_bindings.get(name)
        if not isinstance(recorded, Mapping) or canonical_json(recorded) != canonical_json(actual):
            errors.append(f"postrun_audit_raw_manifest_binding_mismatch:{name}")
        if not _raw_manifest_is_valid(actual):
            errors.append(f"postrun_audit_raw_manifest_invalid:{name}")


def _validate_terminal_final_claim_audit(
    run_root: Path, audit: Mapping[str, object], errors: list[str]
) -> None:
    research_eligible = audit.get("research_evidence_eligible") is True
    created = audit.get("final_claim_audit_created")
    final_hash = audit.get("final_claim_audit_sha256")
    preimage_hash = audit.get("final_claim_audit_preimage_sha256")
    final_path = run_root / FINAL_CLAIM_AUDIT_FILENAME
    if not research_eligible:
        if created is not False or final_hash is not None or preimage_hash is not None:
            errors.append("ineligible_postrun_audit_final_claim_lifecycle_invalid")
        if final_path.exists():
            errors.append("ineligible_postrun_audit_final_claim_present")
        return
    if created is not True or not _is_sha256(final_hash) or not _is_sha256(preimage_hash):
        errors.append("eligible_postrun_audit_final_claim_lifecycle_invalid")
        return
    if final_path.is_symlink() or not final_path.is_file() or sha256_file(final_path) != final_hash:
        errors.append("eligible_postrun_audit_final_claim_hash_mismatch")
        return
    final_claim = _load_run_json(run_root, FINAL_CLAIM_AUDIT_FILENAME, "final_claim_audit", errors)
    expected_fields = {
        "schema_version",
        "status",
        "postrun_audit_preimage_sha256",
        "runtime_source_fingerprint",
        "raw_predictions_manifest_sha256",
        "raw_trajectories_manifest_sha256",
        "dispositions",
        "artifact_sha256",
        "claim_boundary",
    }
    if set(final_claim) != expected_fields:
        errors.append("final_claim_audit_fields_invalid")
        return
    raw_manifests = audit.get("raw_artifact_manifests")
    expected_preimage = final_claim_audit_preimage_sha256(audit)
    if (
        preimage_hash != expected_preimage
        or final_claim.get("postrun_audit_preimage_sha256") != preimage_hash
    ):
        errors.append("final_claim_audit_preimage_mismatch")
    if (
        final_claim.get("schema_version") != SCHEMA_VERSION
        or final_claim.get("status") != "EVIDENCE_RECONCILED"
        or final_claim.get("runtime_source_fingerprint") != audit.get("runtime_source_fingerprint")
        or final_claim.get("dispositions") != audit.get("dispositions")
        or final_claim.get("artifact_sha256") != audit.get("artifact_sha256")
        or not isinstance(raw_manifests, Mapping)
    ):
        errors.append("final_claim_audit_binding_mismatch")
        return
    predictions = raw_manifests.get("raw_predictions_manifest")
    trajectories = raw_manifests.get("raw_trajectories_manifest")
    if (
        not isinstance(predictions, Mapping)
        or not isinstance(trajectories, Mapping)
        or final_claim.get("raw_predictions_manifest_sha256") != predictions.get("manifest_sha256")
        or final_claim.get("raw_trajectories_manifest_sha256")
        != trajectories.get("manifest_sha256")
    ):
        errors.append("final_claim_audit_raw_manifest_mismatch")


def _load_run_json(
    run_root: Path, relative_path: str, label: str, errors: list[str]
) -> dict[str, object]:
    path = _safe_member(run_root, relative_path)
    if path is None or path.is_symlink() or not path.is_file():
        errors.append(f"run_artifact_missing_or_unsafe:{label}")
        return {}
    return _load_json(path, label, errors)


def _raw_manifest_is_valid(manifest: Mapping[str, object] | None) -> bool:
    if not isinstance(manifest, Mapping) or manifest.get("status") != "PASS":
        return False
    payload = dict(manifest)
    observed_hash = payload.pop("manifest_sha256", None)
    if not _is_sha256(observed_hash) or observed_hash != _sha256_payload(payload):
        return False
    artifacts = manifest.get("artifacts")
    return (
        isinstance(artifacts, list)
        and bool(artifacts)
        and all(
            isinstance(item, Mapping)
            and _is_sha256(item.get("sha256"))
            and isinstance(item.get("row_count"), int)
            and int(item["row_count"]) > 0
            for item in artifacts
        )
    )


def _artifact_is_pass(
    name: str, artifact: Mapping[str, object] | None
) -> bool:
    if not isinstance(artifact, Mapping):
        return False
    if name == "runtime_telemetry":
        # Runtime telemetry is a domain ledger, not a generic command envelope.
        # Its cryptographic and live-hardware semantics are checked separately by
        # ``verify_telemetry_ledger`` in ``_validate_live_telemetry``.
        return artifact.get("validation_status") == "PASS"
    return artifact.get("status") == "PASS" and artifact.get("errors") in (None, [])


def _contains_fixture_marker(value: object) -> bool:
    if isinstance(value, Mapping):
        return any(
            (key in {"synthetic_fixture", "fixture_only"} and item is True)
            or _contains_fixture_marker(item)
            for key, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_fixture_marker(item) for item in value)
    return False


def _contains_nonfinite(value: object) -> bool:
    if isinstance(value, float):
        return not math.isfinite(value)
    if isinstance(value, Mapping):
        return any(_contains_nonfinite(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_nonfinite(item) for item in value)
    return False


def _record_check(
    checks: list[dict[str, object]], name: str, passed: bool, evidence: object
) -> None:
    checks.append({"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence})


def _load_json(path: Path, label: str, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        errors.append(f"artifact_missing:{label}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"artifact_invalid_json:{label}:{type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"artifact_not_object:{label}")
        return {}
    return value


def _safe_member(root: Path, relative_path: str) -> Path | None:
    candidate = Path(relative_path)
    if candidate.is_absolute() or ".." in candidate.parts:
        return None
    resolved = (root / candidate).resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _expected_raw_row_counts(
    run_context: Mapping[str, object],
    allowlist: Sequence[str],
    errors: list[str],
) -> dict[str, int] | None:
    raw = run_context.get("raw_row_counts")
    required_paths = set(RAW_PREDICTION_PATHS).union(RAW_TRAJECTORY_PATHS)
    if not isinstance(raw, Mapping) or set(raw) != required_paths:
        errors.append("run_context_raw_row_counts_mismatch")
        return None
    counts: dict[str, int] = {}
    for path in allowlist:
        value = raw.get(path)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            errors.append(f"run_context_raw_row_count_invalid:{path}")
            return None
        counts[path] = value
    return counts


def _jsonl_summary(path: Path) -> tuple[int, int]:
    count = 0
    fixture_count = 0
    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            if not raw_line.strip():
                continue
            value = json.loads(raw_line)
            if not isinstance(value, dict) or _contains_nonfinite(value):
                return 0, 0
            if _contains_fixture_marker(value):
                fixture_count += 1
            count += 1
    except (OSError, UnicodeError, json.JSONDecodeError):
        return 0, 0
    return count, fixture_count


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _sha256_payload(value: Mapping[str, object]) -> str:
    import hashlib

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()
