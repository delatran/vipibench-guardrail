import hashlib
from pathlib import Path

from vipibench.dataio import canonical_json
from vipibench.postrun_audit import (
    RAW_PREDICTION_PATHS,
    RAW_TRAJECTORY_PATHS,
    REQUIRED_ARTIFACTS,
    _build_raw_manifest_if_complete,
    _expected_raw_row_counts,
    _validate_required_artifacts,
    audit_postrun,
    audit_postrun_records,
    final_claim_audit_preimage_sha256,
    validate_postrun_audit_schema,
)
from vipibench.runtime_telemetry import build_telemetry_ledger, record_stage_interval


def _sha(value: object) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest().upper()


def _raw_manifest(kind: str) -> dict[str, object]:
    manifest: dict[str, object] = {
        "schema_version": "1.0.0",
        "kind": kind,
        "status": "PASS",
        "allowlist": [f"{kind}.jsonl"],
        "artifacts": [
            {
                "path": f"{kind}.jsonl",
                "sha256": "A" * 64,
                "size_bytes": 1,
                "row_count": 1,
            }
        ],
    }
    manifest["manifest_sha256"] = _sha(manifest)
    return manifest


def _runtime_telemetry() -> dict[str, object]:
    record = record_stage_interval(
        stage_id="fixture-validation",
        interval_id="fixture-validation-interval",
        run_id="fixture-validation-run",
        attempt_id="fixture-validation-attempt",
        start_monotonic_seconds=1.0,
        end_monotonic_seconds=2.0,
        status="completed",
        accelerator_stage=False,
        observed_device_receipt_sha256=None,
        input_artifact_hashes={"input": "A" * 64},
        output_artifact_hashes={"output": "B" * 64},
        resumed_from_interval_ids=(),
    )
    return build_telemetry_ledger([record], local_only=True)


def _artifacts(*, fixture: bool) -> dict[str, dict[str, object]]:
    artifacts = {name: {"status": "PASS", "errors": []} for name in REQUIRED_ARTIFACTS}
    artifacts["runtime_telemetry"] = _runtime_telemetry()
    artifacts["encoder_ablation"] = {
        "status": "PASS",
        "errors": [],
        "research_claim_eligible": True,
    }
    artifacts["rq2_analysis"] = {
        "status": "PASS",
        "errors": [],
        "research_claim_eligible": True,
        "h2_counterfactual_identity": {"status": "PASS"},
    }
    artifacts["static_analysis"] = {
        "status": "PASS",
        "errors": [],
        "research_claim_eligible": True,
        "synthetic_fixture": fixture,
    }
    artifacts["h3_analysis"] = {
        "status": "PASS",
        "errors": [],
        "research_claim_eligible": True,
    }
    artifacts["adaptive_analysis"] = {
        "status": "PASS",
        "errors": [],
        "research_claim_eligible": False,
    }
    artifacts["fallback_ledger"] = {
        "status": "PASS",
        "errors": [],
        "fallback_records": {},
    }
    artifacts["failure_ledger"] = {
        "status": "PASS",
        "errors": [],
        "events": [],
        "repair_observations": [],
    }
    return artifacts


def test_production_runtime_telemetry_schema_passes_both_artifact_gates() -> None:
    artifacts = _artifacts(fixture=True)
    telemetry = artifacts["runtime_telemetry"]
    errors: list[str] = []

    _validate_required_artifacts(artifacts, errors)
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "fixture",
        run_context={"status": "PASS", "fixture_only": True},
        artifacts=artifacts,
        strict_receipt={"status": "PASS"},
        runtime_telemetry=telemetry,
        raw_prediction_manifest=_raw_manifest("raw_predictions"),
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
        observed_hardware_override=True,
        telemetry_live_override=True,
    )

    assert telemetry["validation_status"] == "PASS"
    assert "status" not in telemetry
    assert errors == []
    assert "required_artifact_invalid:runtime_telemetry" not in result["errors"]
    assert result["status"] == "PASS", result["errors"]


def test_runtime_telemetry_transport_status_cannot_replace_domain_validation() -> None:
    artifacts = _artifacts(fixture=True)
    artifacts["runtime_telemetry"] = {"status": "PASS", "errors": []}
    errors: list[str] = []

    _validate_required_artifacts(artifacts, errors)
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "fixture",
        run_context={"status": "PASS", "fixture_only": True},
        artifacts=artifacts,
        strict_receipt={"status": "PASS"},
        runtime_telemetry=artifacts["runtime_telemetry"],
        raw_prediction_manifest=_raw_manifest("raw_predictions"),
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
        observed_hardware_override=True,
        telemetry_live_override=True,
    )

    assert errors == ["required_artifact_status:runtime_telemetry"]
    assert "required_artifact_invalid:runtime_telemetry" in result["errors"]
    assert result["status"] == "FAIL"


def test_full_raw_count_context_is_safely_sliced_for_each_manifest() -> None:
    all_paths = (*RAW_PREDICTION_PATHS, *RAW_TRAJECTORY_PATHS)
    run_context = {"raw_row_counts": {path: index + 1 for index, path in enumerate(all_paths)}}
    errors: list[str] = []

    prediction_counts = _expected_raw_row_counts(
        run_context, RAW_PREDICTION_PATHS, errors
    )
    trajectory_counts = _expected_raw_row_counts(
        run_context, RAW_TRAJECTORY_PATHS, errors
    )

    assert errors == []
    assert prediction_counts == {
        path: run_context["raw_row_counts"][path] for path in RAW_PREDICTION_PATHS
    }
    assert trajectory_counts == {
        path: run_context["raw_row_counts"][path] for path in RAW_TRAJECTORY_PATHS
    }


def test_raw_count_context_rejects_missing_or_extra_paths() -> None:
    all_paths = (*RAW_PREDICTION_PATHS, *RAW_TRAJECTORY_PATHS)
    complete = {path: 1 for path in all_paths}

    for invalid in (
        {path: count for path, count in complete.items() if path != all_paths[-1]},
        {**complete, "unexpected.jsonl": 1},
    ):
        errors: list[str] = []
        assert (
            _expected_raw_row_counts(
                {"raw_row_counts": invalid}, RAW_PREDICTION_PATHS, errors
            )
            is None
        )
        assert errors == ["run_context_raw_row_counts_mismatch"]


def test_complete_synthetic_structure_is_schema_valid_but_never_research_eligible() -> None:
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "fixture",
        run_context={"status": "PASS", "fixture_only": True},
        artifacts=_artifacts(fixture=True),
        strict_receipt={"status": "PASS"},
        runtime_telemetry={"status": "PASS"},
        raw_prediction_manifest=_raw_manifest("raw_predictions"),
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
        observed_hardware_override=True,
        telemetry_live_override=True,
    )

    validate_postrun_audit_schema(Path.cwd(), result)
    assert result["status"] == "PASS", result["errors"]
    assert result["dispositions"]["engineering_completeness"] == "COMPLETE_FIXTURE_ONLY"
    assert result["dispositions"]["RQ1"] == "NOT_ELIGIBLE_FIXTURE_ONLY"
    assert result["dispositions"]["H4"] == "NOT_ELIGIBLE_FIXTURE_ONLY"
    assert result["research_evidence_eligible"] is False
    assert result["final_claim_audit_created"] is False


def test_missing_or_unobserved_receipts_fail_closed() -> None:
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "fixture",
        run_context={"status": "PASS"},
        artifacts=_artifacts(fixture=False),
        strict_receipt=None,
        runtime_telemetry=None,
        raw_prediction_manifest=_raw_manifest("raw_predictions"),
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
    )

    assert result["status"] == "FAIL"
    assert "strict_capacity_receipt_missing" in result["errors"]
    assert "runtime_telemetry_missing" in result["errors"]
    assert result["dispositions"]["final_E3_eligibility"] == "NOT_ELIGIBLE"


def test_missing_run_directory_writes_schema_valid_not_eligible_audit_without_final_claim(
    tmp_path: Path,
) -> None:
    missing_root = tmp_path / "missing-run"
    output = tmp_path / "postrun_audit.json"

    result = audit_postrun(Path.cwd(), missing_root, output_path=output)

    assert result["status"] == "FAIL"
    assert output.is_file()
    assert not (missing_root / "final_claim_audit.json").exists()
    validate_postrun_audit_schema(Path.cwd(), result)


def test_renamed_template_cannot_be_accepted_as_a_final_claim_audit(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    (run_root / "final_claim_audit.json").write_text(
        '{"status":"PENDING_OBSERVED_RESULTS"}\n', encoding="utf-8"
    )

    result = audit_postrun(Path.cwd(), run_root, output_path=tmp_path / "audit.json")

    assert result["status"] == "FAIL"
    assert "final_claim_audit_must_be_generated_by_current_postrun_audit" in result["errors"]


def test_tampered_raw_manifest_hash_fails_closed() -> None:
    raw = _raw_manifest("raw_predictions")
    raw["manifest_sha256"] = "0" * 64
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "fixture",
        run_context={"status": "PASS", "fixture_only": True},
        artifacts=_artifacts(fixture=True),
        strict_receipt={"status": "PASS"},
        runtime_telemetry={"status": "PASS"},
        raw_prediction_manifest=raw,
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
        observed_hardware_override=True,
        telemetry_live_override=True,
    )

    assert result["status"] == "FAIL"
    assert "raw_manifest_invalid:raw_predictions_manifest" in result["errors"]


def test_final_claim_preimage_is_noncyclic_and_binds_the_complete_audit() -> None:
    artifacts = _artifacts(fixture=False)
    artifacts["adaptive_analysis"]["research_claim_eligible"] = True
    result = audit_postrun_records(
        project_root=Path.cwd(),
        run_root=Path.cwd() / "outputs" / "live-like",
        run_context={"status": "PASS"},
        artifacts=artifacts,
        strict_receipt={"status": "PASS"},
        runtime_telemetry={"status": "PASS"},
        raw_prediction_manifest=_raw_manifest("raw_predictions"),
        raw_trajectory_manifest=_raw_manifest("raw_trajectories"),
        observed_hardware_override=True,
        telemetry_live_override=True,
    )

    assert result["research_evidence_eligible"] is True
    preimage = final_claim_audit_preimage_sha256(result)
    result["final_claim_audit_created"] = True
    result["final_claim_audit_sha256"] = "A" * 64
    result["final_claim_audit_preimage_sha256"] = preimage

    assert final_claim_audit_preimage_sha256(result) == preimage
    validate_postrun_audit_schema(Path.cwd(), result)


def test_raw_manifest_uses_only_safe_nonempty_allowlisted_jsonl_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "prediction.jsonl"
    raw.parent.mkdir(parents=True)
    raw.write_text('{"episode_id":"fixture"}\n', encoding="utf-8")
    errors: list[str] = []

    manifest = _build_raw_manifest_if_complete(
        tmp_path,
        ("raw/prediction.jsonl",),
        tmp_path / "raw_predictions_manifest.json",
        kind="raw_predictions",
        expected_row_counts={"raw/prediction.jsonl": 1},
        errors=errors,
    )

    assert errors == []
    assert manifest is not None
    assert manifest["artifacts"][0]["row_count"] == 1

    traversal_errors: list[str] = []
    invalid = _build_raw_manifest_if_complete(
        tmp_path,
        ("../outside.jsonl",),
        tmp_path / "invalid.json",
        kind="raw_predictions",
        expected_row_counts={"../outside.jsonl": 1},
        errors=traversal_errors,
    )
    assert invalid is None
    assert traversal_errors == ["raw_artifact_missing_or_unsafe:raw_predictions:../outside.jsonl"]
