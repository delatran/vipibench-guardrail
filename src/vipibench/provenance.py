from __future__ import annotations

import json
import re
from pathlib import Path

import yaml

from vipibench.dataio import sha256_file, write_json
from vipibench.security import scan_secrets

SHA256_RE = re.compile(r"^[A-F0-9]{64}$")
PROVENANCE_BINDINGS = {
    "benchmark",
    "template_manifest",
    "split_manifest",
    "catalog",
    "compiler",
    "oracle",
    "episode_schema",
    "provenance_contrast",
    "provenance_contrast_config",
    "provenance_contrast_manifest",
    "confirmatory_holdout_manifest",
    "confirmatory_holdout_test",
    "confirmatory_holdout_templates",
}
AUTHORIZATION_BINDINGS = {
    "executable_benchmark",
    "split_manifest",
    "template_manifest",
    "provenance_ledger",
    "oracle_verification",
    "data_card",
    "secret_scan",
    "experiment_protocol",
    "provenance_contrast",
    "provenance_contrast_audit",
    "provenance_contrast_manifest",
    "confirmatory_holdout_manifest",
    "confirmatory_holdout_test",
    "confirmatory_holdout_templates",
    "confirmatory_analysis",
}


def _load_mapping(path: Path) -> dict[str, object]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected YAML mapping: {path}")
    return value


def _mapping(value: object) -> dict[str, object]:
    return value if isinstance(value, dict) else {}


def _resolve_project_path(project_root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        return None
    resolved = (project_root / value).resolve()
    try:
        resolved.relative_to(project_root)
    except ValueError:
        return None
    return resolved


def _verify_bindings(
    project_root: Path,
    raw_bindings: object,
    expected_names: set[str],
) -> tuple[dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    bindings = _mapping(raw_bindings)
    if set(bindings) != expected_names:
        errors.append("bindings_must_equal_required_set")
    results: dict[str, dict[str, object]] = {}
    for name in sorted(expected_names):
        binding = _mapping(bindings.get(name))
        path = _resolve_project_path(project_root, binding.get("path"))
        expected_hash = str(binding.get("sha256", ""))
        observed_hash = sha256_file(path) if path is not None and path.is_file() else None
        matched = bool(SHA256_RE.fullmatch(expected_hash)) and observed_hash == expected_hash
        results[name] = {
            "path": binding.get("path"),
            "expected_sha256": expected_hash,
            "observed_sha256": observed_hash,
            "matched": matched,
        }
        if not matched:
            errors.append(f"binding_mismatch:{name}")
    return results, errors


def verify_provenance(
    project_root: Path,
    ledger_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    ledger_file = ledger_path or root / "data/provenance_ledger.yaml"
    ledger = _load_mapping(ledger_file)
    errors: list[str] = []
    if ledger.get("schema_version") != "3.0.0":
        errors.append("provenance_schema_version_must_equal_3_0_0")
    if ledger.get("status") != "complete_for_frozen_benchmarks_and_confirmatory_holdout":
        errors.append("provenance_status_not_complete_for_confirmatory_scope")

    raw_sources = ledger.get("sources")
    sources = raw_sources if isinstance(raw_sources, list) else []
    if not sources:
        errors.append("provenance_sources_missing")
    unresolved_sources: list[str] = []
    for source in sources:
        if not isinstance(source, dict):
            errors.append("provenance_source_must_be_mapping")
            continue
        if source.get("unresolved") is not False:
            unresolved_sources.append(str(source.get("source_family", "<missing>")))
    if unresolved_sources:
        errors.append(f"unresolved_sources:{','.join(sorted(unresolved_sources))}")

    binding_results, binding_errors = _verify_bindings(
        root, ledger.get("artifact_bindings"), PROVENANCE_BINDINGS
    )
    errors.extend(binding_errors)

    contract = _mapping(ledger.get("expected_episode_contract"))
    benchmark_binding = _mapping(_mapping(ledger.get("artifact_bindings")).get("benchmark"))
    benchmark_path = _resolve_project_path(root, benchmark_binding.get("path"))
    episode_count = context_count = external_source_count = 0
    if benchmark_path is None or not benchmark_path.is_file():
        errors.append("benchmark_missing_for_provenance_inspection")
    else:
        expected_prefix = str(contract.get("source_uri_prefix", ""))
        expected_version = str(contract.get("source_version", ""))
        expected_license = str(contract.get("license_id", ""))
        with benchmark_path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                if not raw_line.strip():
                    continue
                episode_count += 1
                try:
                    episode = json.loads(raw_line)
                except json.JSONDecodeError:
                    errors.append(f"benchmark_invalid_json_line:{line_number}")
                    continue
                contexts = episode.get("context") if isinstance(episode, dict) else None
                if not isinstance(contexts, list):
                    errors.append(f"benchmark_context_missing_line:{line_number}")
                    continue
                context_count += len(contexts)
                for context in contexts:
                    if not isinstance(context, dict):
                        errors.append(f"benchmark_context_invalid_line:{line_number}")
                        continue
                    source_uri = str(context.get("source_uri", ""))
                    if not source_uri.startswith(expected_prefix):
                        external_source_count += 1
                    if context.get("source_version") != expected_version:
                        errors.append(f"source_version_mismatch_line:{line_number}")
                    if context.get("license_id") != expected_license:
                        errors.append(f"license_id_mismatch_line:{line_number}")

    expected_counts = (
        contract.get("episode_count"),
        contract.get("context_count"),
        contract.get("external_source_count"),
    )
    if (episode_count, context_count, external_source_count) != expected_counts:
        errors.append("observed_provenance_counts_do_not_match_contract")
    if expected_counts != (2400, 4800, 0):
        errors.append("provenance_contract_must_equal_2400_4800_0")

    confirmatory_contract = _mapping(ledger.get("expected_confirmatory_contract"))
    confirmatory_manifest_binding = _mapping(
        _mapping(ledger.get("artifact_bindings")).get("confirmatory_holdout_manifest")
    )
    confirmatory_manifest_path = _resolve_project_path(
        root, confirmatory_manifest_binding.get("path")
    )
    confirmatory_manifest: dict[str, object] = {}
    if confirmatory_manifest_path is None or not confirmatory_manifest_path.is_file():
        errors.append("confirmatory_holdout_manifest_missing")
    else:
        try:
            raw_manifest = json.loads(confirmatory_manifest_path.read_text(encoding="utf-8"))
            confirmatory_manifest = _mapping(raw_manifest)
        except json.JSONDecodeError:
            errors.append("confirmatory_holdout_manifest_invalid_json")
    expected_confirmatory = {
        "test_episode_count": 480,
        "benign_count": 240,
        "injection_count": 240,
        "episode_id_overlap_with_frozen": 0,
        "content_hash_overlap_with_frozen": 0,
        "family_overlap_with_train_or_dev": 0,
    }
    if confirmatory_contract != expected_confirmatory:
        errors.append("confirmatory_contract_must_equal_locked_counts")
    episode_counts = _mapping(confirmatory_manifest.get("episode_counts"))
    label_counts = _mapping(confirmatory_manifest.get("test_label_counts"))
    overlap_counts = _mapping(confirmatory_manifest.get("overlap_counts"))
    observed_confirmatory = {
        "test_episode_count": episode_counts.get("test"),
        "benign_count": label_counts.get("benign"),
        "injection_count": label_counts.get("injection"),
        "episode_id_overlap_with_frozen": overlap_counts.get("episode_id_with_frozen"),
        "content_hash_overlap_with_frozen": overlap_counts.get("content_hash_with_frozen"),
        "family_overlap_with_train_or_dev": overlap_counts.get("family_with_train_or_dev"),
    }
    if (
        confirmatory_manifest.get("status") != "PASS"
        or confirmatory_manifest.get("sealed") is not True
    ):
        errors.append("confirmatory_holdout_manifest_not_sealed_pass")
    if observed_confirmatory != expected_confirmatory:
        errors.append("observed_confirmatory_counts_do_not_match_contract")

    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "ledger_path": str(ledger_file),
        "ledger_sha256": sha256_file(ledger_file),
        "resolved_source_count": len(sources) - len(unresolved_sources),
        "unresolved_sources": unresolved_sources,
        "observed_episode_count": episode_count,
        "observed_context_count": context_count,
        "observed_external_source_count": external_source_count,
        "observed_confirmatory_contract": observed_confirmatory,
        "binding_results": binding_results,
        "claim_boundary": (
            "PASS covers the current frozen internal benchmarks and the sealed final surface-"
            "realization holdout. It does not prove semantic independence from the synthetic "
            "renderer; future generated contexts and external data require new provenance records."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def verify_training_authorization(
    project_root: Path,
    decision_path: Path | None = None,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    decision_file = decision_path or root / "data/release_decision.yaml"
    decision = _load_mapping(decision_file)
    errors: list[str] = []
    if decision.get("schema_version") != "3.0.0":
        errors.append("authorization_schema_version_must_equal_3_0_0")
    if decision.get("decision") != "APPROVED_INTERNAL_PREPARATION":
        errors.append("decision_must_equal_approved_internal_preparation")
    if decision.get("status") != "approved_internal_preparation":
        errors.append("authorization_status_mismatch")
    if decision.get("authorization_scope") != "internal_research_preparation_only":
        errors.append("authorization_scope_must_be_internal_preparation_only")
    if decision.get("training_use_authorized") is not True:
        errors.append("internal_training_use_not_authorized")
    denied_capabilities = {
        "public_release_authorized",
        "upload_authorized",
        "paid_compute_authorized",
        "publication_authorized",
        "production_deployment_authorized",
    }
    for capability in sorted(denied_capabilities):
        if decision.get(capability) is not False:
            errors.append(f"external_capability_must_remain_false:{capability}")

    binding_results, binding_errors = _verify_bindings(
        root, decision.get("bindings"), AUTHORIZATION_BINDINGS
    )
    errors.extend(binding_errors)
    provenance_result = verify_provenance(root)
    if provenance_result["status"] != "PASS":
        errors.append("provenance_verification_not_pass")

    bindings = _mapping(decision.get("bindings"))
    required_pass_artifacts = {
        "oracle_verification",
        "secret_scan",
        "experiment_protocol",
        "provenance_contrast_audit",
        "confirmatory_holdout_manifest",
        "confirmatory_analysis",
    }
    observed_artifact_statuses: dict[str, object] = {}
    for name in sorted(required_pass_artifacts):
        binding = _mapping(bindings.get(name))
        path = _resolve_project_path(root, binding.get("path"))
        status: object = None
        if path is not None and path.is_file():
            try:
                artifact = json.loads(path.read_text(encoding="utf-8"))
                status = artifact.get("status") if isinstance(artifact, dict) else None
            except json.JSONDecodeError:
                status = "INVALID_JSON"
        observed_artifact_statuses[name] = status
        if status != "PASS":
            errors.append(f"required_artifact_not_pass:{name}")

    secret_binding = _mapping(bindings.get("secret_scan"))
    secret_path = _resolve_project_path(root, secret_binding.get("path"))
    recorded_secret_fingerprint: object = None
    if secret_path is not None and secret_path.is_file():
        try:
            secret_artifact = json.loads(secret_path.read_text(encoding="utf-8"))
            if isinstance(secret_artifact, dict):
                recorded_secret_fingerprint = secret_artifact.get("scanned_file_set_sha256")
        except json.JSONDecodeError:
            pass
    current_secret_scan = scan_secrets(root)
    current_secret_fingerprint = current_secret_scan.get("scanned_file_set_sha256")
    secret_scan_current = (
        current_secret_scan.get("status") == "PASS"
        and isinstance(recorded_secret_fingerprint, str)
        and recorded_secret_fingerprint == current_secret_fingerprint
    )
    if not secret_scan_current:
        errors.append("secret_scan_not_current")

    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "decision_path": str(decision_file),
        "decision_sha256": sha256_file(decision_file),
        "authorization_scope": decision.get("authorization_scope"),
        "internal_training_use_authorized": decision.get("training_use_authorized"),
        "paid_compute_authorized": decision.get("paid_compute_authorized"),
        "external_mutations_authorized": False,
        "binding_results": binding_results,
        "required_artifact_statuses": observed_artifact_statuses,
        "secret_scan_current": secret_scan_current,
        "recorded_secret_scan_fingerprint": recorded_secret_fingerprint,
        "current_secret_scan_fingerprint": current_secret_fingerprint,
        "provenance_status": provenance_result["status"],
        "claim_boundary": (
            "PASS authorizes internal preparation only. It does not authorize paid compute, "
            "upload, release, publication, or production deployment."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
