import json
from pathlib import Path

from vipibench.modeling import load_yaml
from vipibench.postrun_audit import (
    RAW_PREDICTION_PATHS,
    RAW_TRAJECTORY_PATHS,
    REQUIRED_ARTIFACTS,
    build_postrun_raw_manifests,
)
from vipibench.postrun_preparation import (
    finalize_confirmatory_run,
    prepare_postrun_supporting_evidence,
    write_postrun_run_context,
)
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS
from vipibench.runtime_capacity import RuntimeProbe, check_runtime_profile
from vipibench.runtime_telemetry import build_strict_capacity_receipt


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _strict_receipt() -> dict[str, object]:
    profile = load_yaml(Path("configs/profiles/accelerator_80gb.yaml"))
    assert isinstance(profile, dict)
    runtime = check_runtime_profile(
        profile,
        RuntimeProbe(
            compute_available=True,
            device_type="cuda",
            device_name="NVIDIA A100-SXM4-80GB",
            device_index=0,
            device_memory_gib=79.2,
            bf16_supported=True,
            tensor_probe_passed=True,
            system_ram_gib=53.0,
            disk_free_gib=120.0,
            compute_capability="8.0",
            evidence_kind="observed",
        ),
    )
    return build_strict_capacity_receipt(runtime, project_root=Path.cwd())


def _populate_raw_artifacts(run_root: Path) -> None:
    for relative_path in (*RAW_PREDICTION_PATHS, *RAW_TRAJECTORY_PATHS):
        path = run_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"record": "retained"}\n', encoding="utf-8")


def _populate_required_artifacts(run_root: Path) -> None:
    for relative_path in REQUIRED_ARTIFACTS.values():
        _write_json(run_root / relative_path, {"status": "PASS", "errors": []})


def _populate_matrix_and_target_runs(
    run_root: Path,
    *,
    fallback: bool = False,
    repair: bool = False,
    within_budget: bool = False,
) -> None:
    for mode in LOCKED_INPUT_MODES:
        for seed in LOCKED_SEEDS:
            run_id = f"mdeberta-{mode}-s{seed}"
            _write_json(
                run_root / "mdeberta" / run_id / "training_decision.json",
                {"status": "PASS", "run_id": run_id, "test_accessed": False},
            )
            _write_json(
                run_root / "mdeberta" / run_id / "test_manifest.json",
                {"status": "PASS", "run_id": run_id},
            )
    for label in ("core", "attack"):
        fallback_ids = ["episode-1"] if fallback and label == "core" else []
        repair_ids = ["episode-2"] if repair and label == "core" else []
        payload: dict[str, object] = {
            "status": "PASS",
            "format_fallback_episode_ids": fallback_ids,
            "format_repair_episode_ids": repair_ids,
            "parse_failure_episode_ids": sorted(fallback_ids + repair_ids),
            "unresolved_parse_failure_episode_ids": fallback_ids,
        }
        if within_budget and fallback_ids:
            payload["format_failure_summary"] = {
                "status": "PASS",
                "unusable_observation_budget_exceeded": False,
                "unusable_observation_policy": (
                    "bounded_unusable_observation_two_sided_bounds_v1"
                ),
            }
        _write_json(run_root / f"{label}_target_trajectories.run.json", payload)


def test_supporting_evidence_preserves_target_fallback_as_a_failure(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    receipt_path = run_root / "strict_capacity_receipt.json"
    _write_json(receipt_path, _strict_receipt())
    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        {
            "status": "AUTHORIZED",
            "scopes": {"drive_upload": True, "paid_compute": True},
            "protocol_amendment": (
                "a100_80gb-response-truncation-guard-v8-2026-08-12"
            ),
            "durable_lineage": (
                "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12"
            ),
        },
    )
    _populate_matrix_and_target_runs(run_root, fallback=False)

    passed = prepare_postrun_supporting_evidence(
        project_root=Path.cwd(),
        output_root=run_root,
        launch_authorization_source=authorization,
        strict_capacity_receipt_path=receipt_path,
    )

    assert passed["status"] == "PASS", passed["errors"]
    assert (
        json.loads((run_root / "final_holdout_format_fallback_ledger.json").read_text())[
            "fallback_records"
        ]
        == {}
    )

    _populate_matrix_and_target_runs(run_root, repair=True)
    repaired = prepare_postrun_supporting_evidence(
        project_root=Path.cwd(),
        output_root=run_root,
        launch_authorization_source=authorization,
        strict_capacity_receipt_path=receipt_path,
    )
    assert repaired["status"] == "PASS", repaired["errors"]
    failure_ledger = json.loads((run_root / "failure_ledger.json").read_text())
    assert failure_ledger["events"] == []
    assert failure_ledger["repair_observations"] == [
        {"stage": "core", "kind": "bounded_schema_repair", "count": 1}
    ]

    _populate_matrix_and_target_runs(run_root, fallback=True)
    failed = prepare_postrun_supporting_evidence(
        project_root=Path.cwd(),
        output_root=run_root,
        launch_authorization_source=authorization,
        strict_capacity_receipt_path=receipt_path,
    )

    assert failed["status"] == "FAIL"
    assert "final_holdout_format_fallback_observed" in failed["errors"]
    assert json.loads((run_root / "failure_ledger.json").read_text())["status"] == "FAIL"

    _populate_matrix_and_target_runs(run_root, fallback=True, within_budget=True)
    bounded = prepare_postrun_supporting_evidence(
        project_root=Path.cwd(),
        output_root=run_root,
        launch_authorization_source=authorization,
        strict_capacity_receipt_path=receipt_path,
    )

    assert bounded["status"] == "PASS", bounded["errors"]
    fallback_ledger = json.loads(
        (run_root / "final_holdout_format_fallback_ledger.json").read_text()
    )
    assert fallback_ledger["status"] == "PASS"
    assert fallback_ledger["fallback_records"] == {"core": ["episode-1"]}
    failure_ledger = json.loads((run_root / "failure_ledger.json").read_text())
    assert failure_ledger["status"] == "PASS"
    assert failure_ledger["events"] == []


def test_supporting_evidence_rejects_a_preamendment_lineage(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    receipt_path = run_root / "strict_capacity_receipt.json"
    _write_json(receipt_path, _strict_receipt())
    authorization = tmp_path / "authorization.json"
    _write_json(
        authorization,
        {
            "status": "AUTHORIZED",
            "scopes": {"drive_upload": True, "paid_compute": True},
            "protocol_amendment": "a100-80gb-hardware-only-2026-08-08",
            "durable_lineage": "fp32-finite-v3",
        },
    )
    _populate_matrix_and_target_runs(run_root)

    result = prepare_postrun_supporting_evidence(
        project_root=Path.cwd(),
        output_root=run_root,
        launch_authorization_source=authorization,
        strict_capacity_receipt_path=receipt_path,
    )

    assert result["status"] == "FAIL"
    assert "launch_authorization_source_invalid" in result["errors"]
    assert json.loads((run_root / "launch_authorization.json").read_text())["status"] == "FAIL"


def test_context_and_raw_manifest_stages_bind_retained_allowlisted_rows(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    _populate_required_artifacts(run_root)
    _populate_raw_artifacts(run_root)

    context = write_postrun_run_context(project_root=Path.cwd(), output_root=run_root)
    raw = build_postrun_raw_manifests(project_root=Path.cwd(), output_root=run_root)

    assert context["status"] == "PASS", context["errors"]
    assert raw["status"] == "PASS", raw["errors"]
    assert (run_root / "raw_predictions_manifest.json").is_file()
    assert (run_root / "raw_trajectories_manifest.json").is_file()


def test_final_manifest_rejects_a_forged_minimal_eligible_audit(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    audit_path = run_root / "postrun_audit.json"
    _write_json(
        audit_path,
        {
            "status": "PASS",
            "dispositions": {"RUN_COMPLETE": "PASS"},
            "research_evidence_eligible": True,
            "final_claim_audit_created": False,
            "final_claim_audit_sha256": None,
        },
    )

    manifest = finalize_confirmatory_run(
        project_root=Path.cwd(), output_root=run_root, postrun_audit_path=audit_path
    )

    assert manifest["status"] == "FAIL"
    assert manifest["RUN_COMPLETE"] == "FAIL"
    assert manifest["RESEARCH_EVIDENCE_ELIGIBLE"] is False
    assert "postrun_audit_schema_invalid:ValidationError" in manifest["errors"]
    assert "postrun_audit_did_not_confirm_run_complete" in manifest["errors"]
