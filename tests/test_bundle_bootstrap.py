import ensurepip
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from vipibench.postrun_audit import (
    REQUIRED_DURABLE_LINEAGE as AUDIT_DURABLE_LINEAGE,
)
from vipibench.postrun_audit import (
    REQUIRED_PROTOCOL_AMENDMENT as AUDIT_PROTOCOL_AMENDMENT,
)
from vipibench.postrun_preparation import (
    REQUIRED_DURABLE_LINEAGE as PREPARATION_DURABLE_LINEAGE,
)
from vipibench.postrun_preparation import (
    REQUIRED_PROTOCOL_AMENDMENT as PREPARATION_PROTOCOL_AMENDMENT,
)
from vipibench.security import (
    EXCLUDED_DIRECTORY_SUFFIXES,
    EXCLUDED_PARTS,
    EXCLUDED_RELATIVE_PATHS,
    TEXT_SUFFIXES,
)

BOOTSTRAP_PATH = Path(__file__).parents[2] / "bootstrap.py"
SPEC = importlib.util.spec_from_file_location("vipibench_bundle_bootstrap", BOOTSTRAP_PATH)
assert SPEC is not None and SPEC.loader is not None
BOOTSTRAP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BOOTSTRAP)
BUNDLE_ROOT = Path(__file__).parents[2]


def _args(**updates: object) -> Namespace:
    values = {
        "mode": "confirmatory",
        "confirm_upload_authorized": True,
        "confirm_paid_compute_authorized": True,
    }
    values.update(updates)
    return Namespace(**values)


def _artifact_entry(project: Path, path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(project).as_posix(),
        "sha256": BOOTSTRAP._sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _write_secret_scan_receipt(project: Path) -> Path:
    scanned_files = BOOTSTRAP._secret_scan_file_set(project)
    fingerprint = BOOTSTRAP.hashlib.sha256(
        BOOTSTRAP._canonical_json(scanned_files).encode("utf-8")
    ).hexdigest().upper()
    path = project / BOOTSTRAP.SECRET_SCAN_RECEIPT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema_version": BOOTSTRAP.SECRET_SCAN_SCHEMA_VERSION,
                "status": "PASS",
                "scanned_file_count": len(scanned_files),
                "scanned_file_set_sha256": fingerprint,
                "scan_policy": BOOTSTRAP._secret_scan_policy(),
                "findings": [],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_complete_acceptance_fixture(
    bundle: Path, stage_ids: list[str]
) -> tuple[Path, Path]:
    status_path = BOOTSTRAP.controller_status_path(bundle)
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "COMPLETE",
                "execution_mode": "sequential_single_a100_80gb_runtime",
                "parallel_gpu_processes": 1,
                "durable_lineage": BOOTSTRAP.DURABLE_LINEAGE,
                "stages": [
                    {
                        "id": stage_id,
                        "state": "COMPLETE",
                        "prerequisite": stage_ids[index - 1] if index else None,
                        "verification_errors": [],
                    }
                    for index, stage_id in enumerate(stage_ids)
                ],
                "last_attempt": {
                    "selected_stage": "finalize",
                    "result": "PASS",
                    "error": None,
                },
            }
        ),
        encoding="utf-8",
    )
    output_root = bundle / BOOTSTRAP.RUN_OUTPUT_DIRECTORY_NAME
    output_root.mkdir()
    (output_root / BOOTSTRAP.RESOURCE_MEASUREMENT_FILENAME).write_text(
        "{}\n", encoding="utf-8"
    )
    postrun_audit_path = output_root / BOOTSTRAP.POSTRUN_AUDIT_FILENAME
    postrun_audit_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "errors": [],
                "fixture_only": False,
                "dispositions": {
                    "engineering_completeness": "COMPLETE_LIVE",
                    "RUN_COMPLETE": "PASS",
                },
                "research_evidence_eligible": False,
                "final_claim_audit_created": False,
                "final_claim_audit_sha256": None,
                "final_claim_audit_preimage_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    run_manifest_path = output_root / BOOTSTRAP.RUN_MANIFEST_FILENAME
    run_manifest_path.write_text(
        json.dumps(
            {
                "schema_version": "1.0.0",
                "status": "PASS",
                "errors": [],
                "RUN_COMPLETE": "PASS",
                "postrun_audit_sha256": BOOTSTRAP._sha256(postrun_audit_path),
                "RESEARCH_EVIDENCE_ELIGIBLE": False,
                "final_claim_audit_created": False,
                "final_claim_audit_sha256": None,
            }
        ),
        encoding="utf-8",
    )
    return run_manifest_path, postrun_audit_path


def test_confirmatory_bootstrap_requires_both_external_authorizations() -> None:
    with pytest.raises(PermissionError, match="confirm-upload-authorized"):
        BOOTSTRAP.validate_confirmatory_authorization(_args(confirm_upload_authorized=False))
    with pytest.raises(PermissionError, match="confirm-paid-compute-authorized"):
        BOOTSTRAP.validate_confirmatory_authorization(_args(confirm_paid_compute_authorized=False))


def test_confirmatory_bootstrap_accepts_explicit_external_authorization() -> None:
    BOOTSTRAP.validate_confirmatory_authorization(_args())


def test_smoke_mode_does_not_claim_external_authorization() -> None:
    BOOTSTRAP.validate_confirmatory_authorization(
        _args(
            mode="smoke",
            confirm_upload_authorized=False,
            confirm_paid_compute_authorized=False,
        )
    )


@pytest.mark.parametrize(
    "stage",
    [
        "preflight",
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
        "analysis",
        "finalize",
    ],
)
def test_accelerator_dependency_profile_is_required_for_every_public_stage(stage: str) -> None:
    assert (
        BOOTSTRAP.validate_dependency_profile(
            mode="confirmatory",
            selected_stage=stage,
            dependency_profile=BOOTSTRAP.ACCELERATOR_DEPENDENCY_PROFILE,
        )
        == BOOTSTRAP.ACCELERATOR_DEPENDENCY_PROFILE
    )


@pytest.mark.parametrize("stage", ["analysis", "finalize", "encoder"])
def test_cpu_dependency_profile_is_rejected_for_every_public_stage(stage: str) -> None:
    with pytest.raises(ValueError, match="requires the accelerator dependency profile"):
        BOOTSTRAP.validate_dependency_profile(
            mode="confirmatory",
            selected_stage=stage,
            dependency_profile="analysis-cpu",
        )


def test_smoke_mode_rejects_cpu_dependency_profile() -> None:
    with pytest.raises(ValueError, match="requires the accelerator dependency profile"):
        BOOTSTRAP.validate_dependency_profile(
            mode="smoke",
            selected_stage="preflight",
            dependency_profile="analysis-cpu",
        )


def test_bootstrap_exposes_no_operator_selected_training_budget() -> None:
    parser = BOOTSTRAP.build_parser()
    option_strings = {option for action in parser._actions for option in action.option_strings}
    assert "--max-compute-hours" not in option_strings
    assert "--snapshot-interval-seconds" not in option_strings
    assert "--stage" in option_strings
    assert "--status" in option_strings
    assert "--assert-complete" in option_strings
    dependency_action = next(
        action for action in parser._actions if "--dependency-profile" in action.option_strings
    )
    assert dependency_action.choices == ["accelerator"]


def test_bootstrap_stage_selection_uses_the_locked_sequential_plan() -> None:
    _, stage_ids = BOOTSTRAP.load_public_stage_plan(BUNDLE_ROOT / BOOTSTRAP.PROJECT_NAME)

    assert stage_ids == [
        "preflight",
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
        "analysis",
        "finalize",
    ]


def test_bootstrap_accepts_only_complete_hash_bound_terminal_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stage_ids = BOOTSTRAP.load_public_stage_plan(BUNDLE_ROOT / BOOTSTRAP.PROJECT_NAME)
    run_manifest_path, _ = _write_complete_acceptance_fixture(tmp_path, stage_ids)
    monkeypatch.setattr(
        BOOTSTRAP,
        "verify_terminal_resource_measurement",
        lambda bundle, expected_stages: {
            "status": "PASS",
            "hardware_observed": True,
            "completed_public_stages": expected_stages,
            "missing_public_stages": [],
            "final_holdout_feedback_used": False,
        },
    )

    result = BOOTSTRAP.assert_confirmatory_complete(tmp_path, stage_ids)

    assert result["status"] == "PASS"
    assert result["one_click_run_all_complete"] is True
    assert result["run_manifest_sha256"] == BOOTSTRAP._sha256(run_manifest_path)
    assert result["resource_measurement_sha256"] == BOOTSTRAP._sha256(
        tmp_path
        / BOOTSTRAP.RUN_OUTPUT_DIRECTORY_NAME
        / BOOTSTRAP.RESOURCE_MEASUREMENT_FILENAME
    )
    assert result["research_evidence_eligible"] is False


def test_bootstrap_rejects_tampered_terminal_results(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, stage_ids = BOOTSTRAP.load_public_stage_plan(BUNDLE_ROOT / BOOTSTRAP.PROJECT_NAME)
    _, postrun_audit_path = _write_complete_acceptance_fixture(tmp_path, stage_ids)
    monkeypatch.setattr(
        BOOTSTRAP,
        "verify_terminal_resource_measurement",
        lambda bundle, expected_stages: {"status": "PASS"},
    )
    postrun_audit = json.loads(postrun_audit_path.read_text(encoding="utf-8"))
    postrun_audit["status"] = "FAIL"
    postrun_audit_path.write_text(json.dumps(postrun_audit), encoding="utf-8")

    with pytest.raises(
        RuntimeError,
        match="run_manifest_postrun_audit_hash_mismatch",
    ):
        BOOTSTRAP.assert_confirmatory_complete(tmp_path, stage_ids)
    assert BOOTSTRAP.select_confirmatory_stage("confirmatory", "ENCODER", stage_ids) == (
        "encoder"
    )
    with pytest.raises(ValueError, match="independent resource observation"):
        BOOTSTRAP.select_confirmatory_stage("confirmatory", None, stage_ids)
    with pytest.raises(ValueError, match="Unknown confirmatory stage"):
        BOOTSTRAP.select_confirmatory_stage("confirmatory", "parallel", stage_ids)
    with pytest.raises(ValueError, match="supported only"):
        BOOTSTRAP.select_confirmatory_stage("smoke", "data", stage_ids)


def test_lineage_launch_authorization_is_created_once_and_reused_immutably(
    tmp_path: Path,
) -> None:
    path = tmp_path / "launch_authorization.json"
    template = {
        "schema_version": "1.0.0",
        "status": "AUTHORIZED",
        "binding": "A" * 64,
    }
    first, reused_first = BOOTSTRAP.create_or_reuse_launch_authorization(
        path,
        template,
        {"status": "SKIP", "reason": "durable_snapshot_absent"},
    )
    first_hash = BOOTSTRAP._sha256(path)

    second, reused_second = BOOTSTRAP.create_or_reuse_launch_authorization(
        path,
        template,
        {"status": "PASS", "restored": True},
    )

    assert reused_first is False
    assert reused_second is True
    assert second == first
    assert BOOTSTRAP._sha256(path) == first_hash
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["binding"] = "B" * 64
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="binding mismatch"):
        BOOTSTRAP.create_or_reuse_launch_authorization(path, template, {})


def test_lineage_launch_authorization_rebinds_maintenance_manifest_fields(
    tmp_path: Path,
) -> None:
    path = tmp_path / "launch_authorization.json"
    template = {
        "schema_version": "1.0.0",
        "status": "AUTHORIZED",
        "protocol_amendment": "a100_80gb-response-truncation-guard-v8-2026-08-12",
        "durable_lineage": "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12",
        "stage_plan_sha256": "A" * 64,
        "bundle_manifest_sha256": "B" * 64,
        "project_artifact_manifest_sha256": "C" * 64,
        "bootstrap_sha256": "D" * 64,
    }
    BOOTSTRAP.create_or_reuse_launch_authorization(path, template, {"status": "PASS"})
    updated = {
        **template,
        "bundle_manifest_sha256": "E" * 64,
        "project_artifact_manifest_sha256": "F" * 64,
        "bootstrap_sha256": "1" * 64,
    }

    authorization, reused = BOOTSTRAP.create_or_reuse_launch_authorization(
        path,
        updated,
        {"status": "PASS", "restored": True},
    )

    assert reused is True
    assert authorization["bundle_manifest_sha256"] == "E" * 64
    assert json.loads(path.read_text(encoding="utf-8"))["bootstrap_sha256"] == "1" * 64
    with pytest.raises(ValueError, match="binding mismatch"):
        BOOTSTRAP.create_or_reuse_launch_authorization(
            path,
            {**updated, "protocol_amendment": "different-amendment"},
            {},
        )


def test_lineage_launch_authorization_rejects_unsafe_existing_target(tmp_path: Path) -> None:
    path = tmp_path / "launch_authorization.json"
    path.mkdir()

    with pytest.raises(ValueError, match="path is unsafe"):
        BOOTSTRAP.create_or_reuse_launch_authorization(
            path,
            {"schema_version": "1.0.0", "status": "AUTHORIZED"},
            {},
        )


def test_bootstrap_starts_a_new_staged_durable_lineage() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert (
        BOOTSTRAP.PROTOCOL_AMENDMENT
        == "a100_80gb-response-truncation-guard-v8-2026-08-12"
    )
    assert (
        BOOTSTRAP.DURABLE_LINEAGE
        == "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12"
    )
    assert BOOTSTRAP.DURABLE_LINEAGE not in {
        "fp32-finite-v3",
        "fp32-content-store-candidate-validity-staged-2026-08-02-v3",
    }
    assert 'bundle / ".vipibench-durable" / DURABLE_LINEAGE' in source
    assert "PeriodicContentAddressedSnapshot" in source
    assert "restore_verified_content_snapshot" in source
    assert "run-output.snapshot-store" in source
    assert "run-output.tar.gz" not in source
    assert '"protocol_amendment": PROTOCOL_AMENDMENT' in source
    assert '"durable_lineage": DURABLE_LINEAGE' in source
    assert "create_runtime_storage_plan" in source
    assert "runtime_storage_environment" in source
    assert "StageResourceObserver" in source
    assert "build_resource_measurement" in source
    assert "VIPIBENCH_RUNTIME_STORAGE_PLAN_FILE_SHA256" in source
    assert "session_capacity_check_path" in source


def test_launch_authorization_requires_a100_for_all_public_stages(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    project = bundle / BOOTSTRAP.PROJECT_NAME
    project.mkdir(parents=True)
    (bundle / BOOTSTRAP.BUNDLE_MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
    (project / BOOTSTRAP.PROJECT_MANIFEST_FILENAME).write_text("{}\n", encoding="utf-8")
    policy_path = project / "policy.json"
    policy_path.write_text("{}\n", encoding="utf-8")
    stage_plan_path = project / "stage-plan.json"
    stage_plan_path.write_text("{}\n", encoding="utf-8")
    policy = type(
        "Policy",
        (),
        {
            "hard_ceiling_hours": 72.0,
            "decision_split": "dev",
            "decision_metric": "dev_auprc",
            "final_holdout_feedback_allowed": False,
        },
    )()

    authorization = BOOTSTRAP.launch_authorization_template(
        bundle=bundle,
        autonomous_policy_path=policy_path,
        autonomous_policy=policy,
        stage_plan_path=stage_plan_path,
    )

    target = authorization["exact_runtime_target"]
    assert target["accelerator_memory_range_gib"] == [70, 82]
    assert target["accelerator_stages"] == [
        "preflight",
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
        "analysis",
        "finalize",
    ]
    assert target["dependency_profile"] == "accelerator"
    assert target["single_runtime_required"] is True
    assert target["runtime_transition_allowed"] is False
    assert target["allow_non_a100_accelerator_substitution"] is False


def test_outer_and_postrun_layers_share_the_same_amendment_contract() -> None:
    assert BOOTSTRAP.PROTOCOL_AMENDMENT == PREPARATION_PROTOCOL_AMENDMENT
    assert BOOTSTRAP.PROTOCOL_AMENDMENT == AUDIT_PROTOCOL_AMENDMENT
    assert BOOTSTRAP.DURABLE_LINEAGE == PREPARATION_DURABLE_LINEAGE
    assert BOOTSTRAP.DURABLE_LINEAGE == AUDIT_DURABLE_LINEAGE


def test_bootstrap_bundle_requires_only_the_executable_project(tmp_path: Path) -> None:
    project = tmp_path / BOOTSTRAP.PROJECT_NAME
    project.mkdir()

    assert BOOTSTRAP.resolve_project_source(tmp_path) == project
    assert "VIPIBENCH_PROPOSAL_PATH" not in BOOTSTRAP_PATH.read_text(encoding="utf-8")


def test_bundle_integrity_manifest_binds_immutable_launch_inputs_only(
    tmp_path: Path,
) -> None:
    for relative_path in BOOTSTRAP.BUNDLE_MANIFEST_PATHS:
        path = tmp_path / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative_path.as_posix(), encoding="utf-8")
    notebook = tmp_path / "RUN_EXPERIMENT.ipynb"
    notebook.write_text('{"cells": []}\n', encoding="utf-8")

    manifest = BOOTSTRAP.build_bundle_manifest(tmp_path)

    assert manifest["status"] == "PASS"
    assert BOOTSTRAP.verify_bundle_manifest(tmp_path)["durable_lineage"] == (
        "fp32-a100_80gb-response-truncation-guard-v8-2026-08-12"
    )
    manifest_path = tmp_path / BOOTSTRAP.BUNDLE_MANIFEST_FILENAME
    malformed = json.loads(manifest_path.read_text(encoding="utf-8"))
    malformed["unexpected"] = True
    manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
    with pytest.raises(ValueError, match="fields are invalid"):
        BOOTSTRAP.verify_bundle_manifest(tmp_path)
    BOOTSTRAP.build_bundle_manifest(tmp_path)
    notebook.write_text('{"cells": [], "metadata": {"colab": {}}}\n', encoding="utf-8")
    assert BOOTSTRAP.verify_bundle_manifest(tmp_path)["status"] == "PASS"
    (tmp_path / "bootstrap.py").write_text("drift", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        BOOTSTRAP.verify_bundle_manifest(tmp_path)


def test_upload_inventory_rejects_unbound_generated_state(tmp_path: Path) -> None:
    project = tmp_path / BOOTSTRAP.PROJECT_NAME
    project.mkdir()
    source = project / "configs" / "nested" / "source.txt"
    source.parent.mkdir(parents=True)
    source.write_text("bound\n", encoding="utf-8")
    secret_scan = _write_secret_scan_receipt(project)
    project_manifest = {
        "schema_version": "1.0.0",
        "project": BOOTSTRAP.PROJECT_NAME,
        "generated_at": "2026-07-29T00:00:00+00:00",
        "status": "PASS",
        "artifacts": [
            _artifact_entry(project, source),
            _artifact_entry(project, secret_scan),
        ],
    }
    (project / "artifact_manifest.json").write_text(json.dumps(project_manifest), encoding="utf-8")
    (tmp_path / "bootstrap.py").write_text("bootstrap\n", encoding="utf-8")
    (tmp_path / "RUN_EXPERIMENT.ipynb").write_text("{}\n", encoding="utf-8")
    BOOTSTRAP.build_bundle_manifest(tmp_path)

    assert BOOTSTRAP.verify_upload_inventory(tmp_path)["status"] == "PASS"
    (project / "unbound-generated.txt").write_text("unbound\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Project upload inventory mismatch"):
        BOOTSTRAP.verify_upload_inventory(tmp_path)


def test_upload_inventory_rejects_stale_secret_scan_receipt(tmp_path: Path) -> None:
    project = tmp_path / BOOTSTRAP.PROJECT_NAME
    project.mkdir()
    source = project / "source.txt"
    source.write_text("bound\n", encoding="utf-8")
    secret_scan = _write_secret_scan_receipt(project)
    project_manifest = {
        "schema_version": "1.0.0",
        "project": BOOTSTRAP.PROJECT_NAME,
        "generated_at": "2026-08-02T00:00:00+00:00",
        "status": "PASS",
        "artifacts": [
            _artifact_entry(project, source),
            _artifact_entry(project, secret_scan),
        ],
    }
    project_manifest_path = project / "artifact_manifest.json"
    project_manifest_path.write_text(json.dumps(project_manifest), encoding="utf-8")
    (tmp_path / "bootstrap.py").write_text("bootstrap\n", encoding="utf-8")
    (tmp_path / "RUN_EXPERIMENT.ipynb").write_text("{}\n", encoding="utf-8")
    BOOTSTRAP.build_bundle_manifest(tmp_path)

    assert BOOTSTRAP.verify_upload_inventory(tmp_path)["status"] == "PASS"

    source.write_text("drift after scan\n", encoding="utf-8")
    project_manifest["artifacts"] = [
        _artifact_entry(project, source),
        _artifact_entry(project, secret_scan),
    ]
    project_manifest_path.write_text(json.dumps(project_manifest), encoding="utf-8")
    BOOTSTRAP.build_bundle_manifest(tmp_path)

    with pytest.raises(ValueError, match="secret-scan receipt is stale"):
        BOOTSTRAP.verify_upload_inventory(tmp_path)


def _remove_upload_polluting_caches(root: Path) -> None:
    for cache_name in ("__pycache__", ".pytest_cache"):
        for cache_dir in root.rglob(cache_name):
            if cache_dir.is_dir():
                shutil.rmtree(cache_dir)


def test_upload_inventory_includes_training_authorization_on_real_bundle() -> None:
    _remove_upload_polluting_caches(BUNDLE_ROOT)
    inventory = BOOTSTRAP.verify_upload_inventory(BUNDLE_ROOT)

    assert inventory["status"] == "PASS"
    assert isinstance(inventory.get("training_authorization_decision_sha256"), str)
    assert len(inventory["training_authorization_decision_sha256"]) == 64


def test_verify_upload_inventory_leaves_no_import_pollutants() -> None:
    _remove_upload_polluting_caches(BUNDLE_ROOT)
    inventory = BOOTSTRAP.verify_upload_inventory(BUNDLE_ROOT)
    assert inventory["status"] == "PASS"
    cache_dir = BUNDLE_ROOT / BOOTSTRAP.PROJECT_NAME / "src" / "vipibench" / "__pycache__"
    assert not cache_dir.exists()


def test_prune_upload_pollutants_removes_generated_caches_on_real_bundle(tmp_path: Path) -> None:
    project = BUNDLE_ROOT / BOOTSTRAP.PROJECT_NAME
    cache_dir = project / "src" / "vipibench" / "__pycache__"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "probe.cpython-312.pyc"
    cache_file.write_bytes(b"pollutant")

    try:
        result = BOOTSTRAP.prune_upload_pollutants(BUNDLE_ROOT)
        assert result["status"] == "PASS"
        assert result["removed_directory_count"] >= 1
        assert not cache_dir.exists()
        inventory = BOOTSTRAP.verify_upload_inventory(BUNDLE_ROOT)
        assert inventory["status"] == "PASS"
    finally:
        if cache_dir.exists():
            shutil.rmtree(cache_dir)


def test_stage_upload_payload_materializes_only_manifest_bound_files(tmp_path: Path) -> None:
    source_bundle = tmp_path / "source_bundle"
    project = source_bundle / BOOTSTRAP.PROJECT_NAME
    project.mkdir(parents=True)
    source = project / "source.txt"
    source.write_text("bound\n", encoding="utf-8")
    secret_scan = _write_secret_scan_receipt(project)
    project_manifest = {
        "schema_version": "1.0.0",
        "project": BOOTSTRAP.PROJECT_NAME,
        "generated_at": "2026-07-29T00:00:00+00:00",
        "status": "PASS",
        "artifacts": [
            _artifact_entry(project, source),
            _artifact_entry(project, secret_scan),
        ],
    }
    (project / "artifact_manifest.json").write_text(
        json.dumps(project_manifest), encoding="utf-8"
    )
    (project / ".pytest_cache").mkdir()
    (source_bundle / "bootstrap.py").write_text("bootstrap\n", encoding="utf-8")
    (source_bundle / "RUN_EXPERIMENT.ipynb").write_text("{}\n", encoding="utf-8")
    BOOTSTRAP.build_bundle_manifest(source_bundle)

    staged_bundle = tmp_path / "upload_ready"
    result = BOOTSTRAP.stage_upload_payload(source_bundle, staged_bundle)

    assert result["status"] == "PASS"
    assert BOOTSTRAP.verify_upload_inventory(staged_bundle)["status"] == "PASS"
    assert (source_bundle / BOOTSTRAP.PROJECT_NAME / ".pytest_cache").is_dir()
    assert not (staged_bundle / BOOTSTRAP.PROJECT_NAME / ".pytest_cache").exists()


def test_single_launch_notebook_has_no_training_decision_knobs() -> None:
    notebook = json.loads((BUNDLE_ROOT / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    assert "MAX_COMPUTE_HOURS" not in source
    assert "UPLOAD_AUTHORIZED =" not in source
    assert "PAID_COMPUTE_AUTHORIZED =" not in source
    assert "--confirm-upload-authorized" in source
    assert "--confirm-paid-compute-authorized" in source
    assert "bundle_manifest.json" in source
    assert "--verify-bundle-manifest" in source
    assert source.index("--verify-bundle-manifest") < source.index(
        "sys.executable, '-m', 'venv', '--without-pip'"
    )


def test_operator_notebook_exposes_ordered_sequential_public_stage_cells() -> None:
    notebook = json.loads((BUNDLE_ROOT / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8"))
    stage_cells = [
        cell
        for cell in notebook["cells"]
        if "vipibench-stage" in cell.get("metadata", {}).get("tags", [])
    ]
    observed = [
        "".join(cell["source"]).split("_run_confirmatory_stage('", maxsplit=1)[1].split(
            "')", maxsplit=1
        )[0]
        for cell in stage_cells
    ]

    assert observed == [
        "preflight",
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
        "analysis",
        "finalize",
    ]
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in stage_cells
    )
    markdown_cells = [
        cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    ]
    assert len(markdown_cells) == 1
    assert "".join(markdown_cells[0]["source"]) == (
        "# NGHIÊN CỨU VÀ ĐÁNH GIÁ MÔ HÌNH PHÁT HIỆN TẤN CÔNG PROMPT "
        "INJECTION TIẾNG VIỆT THEO NGỮ CẢNH CHO ỨNG DỤNG LLM VÀ RAG"
    )
    assert all(
        not line.lstrip().startswith("#")
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
        for line in cell.get("source", [])
    )


def test_operator_notebook_has_presentation_safe_title_and_labels() -> None:
    notebook = json.loads((BUNDLE_ROOT / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8"))
    visible_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert re.search(r"(?i)\b(?:rq|rg|h)\s*[-_]?\d+\b", visible_source) is None
    assert re.search(r"(?i)\bv\d+(?:\.\d+)*\b", visible_source) is None


def test_single_launch_notebook_prepares_dependencies_outside_live_kernel() -> None:
    notebook = json.loads((BUNDLE_ROOT / "RUN_EXPERIMENT.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "scripts/prepare_colab.py" in source
    assert "RUNTIME_DEPENDENCY_PROFILE = 'accelerator'" in source
    assert "LOCK_NAME = 'requirements-experiment.lock'" in source
    assert "requirements-analysis.lock" not in source
    assert "analysis-cpu" not in source
    assert "Khong duoc fallback sang CPU" in source
    assert "'--project-root', PROJECT" in source
    assert "sys.executable, '-m', 'pip', 'install'" not in source
    assert "from transformers import" not in source
    assert "import torch" not in source
    assert "/content/vipibench_runtime" in source
    assert "sys.executable, '-m', 'venv', '--without-pip'" in source
    assert "'-m', 'ensurepip'" not in source
    assert "PIP_BOOTSTRAP_REQUIREMENT = 'pip==25.3'" in source
    assert "PIP_BOOTSTRAP_REQUIREMENT.encode()" in source
    assert "sys.executable, '-m', 'pip', '--python', RUNTIME_PYTHON" in source
    assert "'pip_bootstrap'" in source
    assert "'pip_bootstrap_probe'" in source
    assert "stderr=subprocess.STDOUT" in source
    assert 'RUNTIME_PYTHON, f"{BUNDLE}/bootstrap.py"' in source
    base_manifest_verifier = 'sys.executable, f"{BUNDLE}/bootstrap.py"'
    assert source.count(base_manifest_verifier) == 1
    assert source.index(base_manifest_verifier) < source.index(
        "sys.executable, '-m', 'venv', '--without-pip'"
    )
    assert "--verify-bundle-manifest" in source
    assert "_run_visible(f'confirmatory_{stage}'" in source
    assert "'--stage', stage" in source
    assert source.count("_run_confirmatory_stage('") == 9
    assert "_show_stage_status()" in source
    assert "_assert_run_all_complete()" in source
    assert "'--assert-complete', '--bundle', BUNDLE" in source
    final_acceptance_cell = "".join(notebook["cells"][13]["source"])
    assert final_acceptance_cell.index("_show_stage_status()") < final_acceptance_cell.index(
        "_assert_run_all_complete()"
    )
    assert 'subprocess.check_call([\n    RUNTIME_PYTHON, f"{BUNDLE}/bootstrap.py"' not in source


def test_host_pip_can_seed_runtime_created_without_target_ensurepip(
    tmp_path: Path,
) -> None:
    runtime_root = tmp_path / "runtime"
    subprocess.run(
        [sys.executable, "-m", "venv", "--without-pip", str(runtime_root)],
        check=True,
    )
    runtime_python = runtime_root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")

    missing_pip = subprocess.run(
        [str(runtime_python), "-m", "pip", "--version"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert missing_pip.returncode != 0

    bundled_dir = Path(ensurepip.__file__).resolve().parent / "_bundled"
    pip_wheels = sorted(bundled_dir.glob("pip-*.whl"))
    assert pip_wheels, f"No offline pip wheel available under {bundled_dir}"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "--python",
            str(runtime_python),
            "install",
            "--no-index",
            "--no-deps",
            "--disable-pip-version-check",
            str(pip_wheels[-1]),
        ],
        check=True,
    )
    installed_pip = subprocess.run(
        [str(runtime_python), "-m", "pip", "--version"],
        check=True,
        capture_output=True,
        text=True,
    )
    assert str(runtime_root.resolve()).casefold() in installed_pip.stdout.casefold()


def test_bootstrap_fails_closed_on_prepared_runtime_probe() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert "verify_prepared_runtime(project, dependency_profile)" in source
    assert '"scripts" / "prepare_colab.py"' in source
    assert '"--probe-only"' in source


def test_bootstrap_checks_secret_scan_before_prepared_runtime_probe() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    main_body = source.split("def main(argv: list[str] | None = None) -> int:", maxsplit=1)[1]

    assert main_body.index("verify_project_secret_scan_freshness(project)") < main_body.index(
        "verify_prepared_runtime(project, dependency_profile)"
    )
    assert main_body.index("verify_project_training_authorization(project)") < main_body.index(
        "verify_prepared_runtime(project, dependency_profile)"
    )


def test_bootstrap_secret_scan_policy_matches_runtime_scanner() -> None:
    assert frozenset(TEXT_SUFFIXES) == BOOTSTRAP.SECRET_SCAN_TEXT_SUFFIXES
    assert frozenset(EXCLUDED_PARTS) == BOOTSTRAP.SECRET_SCAN_EXCLUDED_DIRECTORY_NAMES
    assert (
        frozenset(EXCLUDED_DIRECTORY_SUFFIXES)
        == BOOTSTRAP.SECRET_SCAN_EXCLUDED_DIRECTORY_SUFFIXES
    )
    assert (
        frozenset(EXCLUDED_RELATIVE_PATHS)
        == BOOTSTRAP.SECRET_SCAN_EXCLUDED_RELATIVE_PATHS
    )


def test_bootstrap_verifies_outer_bundle_integrity_before_staging_project() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    main_body = source.split("def main(argv: list[str] | None = None) -> int:", maxsplit=1)[1]
    first_verification = main_body.index("verify_bundle_manifest(bundle)")
    normal_flow_verification = main_body.index(
        "verify_bundle_manifest(bundle)", first_verification + 1
    )
    assert normal_flow_verification < main_body.index("resolve_project_source(bundle)")


def test_bootstrap_configures_utf8_output_before_printing_paths() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    main_body = source.split("def main(argv: list[str] | None = None) -> int:", maxsplit=1)[1]
    assert main_body.index("configure_utf8_stdio()") < main_body.index(
        "build_parser().parse_args(argv)"
    )


def test_bootstrap_puts_isolated_python_first_on_kernel_path(tmp_path: Path) -> None:
    environment = BOOTSTRAP.isolated_runtime_environment(tmp_path)

    first_path = environment["PATH"].split(os.pathsep)[0]
    assert Path(first_path) == Path(BOOTSTRAP.sys.executable).absolute().parent
    assert Path(environment["VIRTUAL_ENV"]) == Path(BOOTSTRAP.sys.prefix).absolute()
    assert environment["PYTHONUTF8"] == "1"
    assert environment["PYTHONIOENCODING"] == "utf-8"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert Path(environment["IPYTHONDIR"]) == tmp_path / "ipython"
    assert Path(environment["JUPYTER_CONFIG_DIR"]) == tmp_path / "jupyter-config"


def test_bootstrap_replaces_colab_kernel_configuration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("IPYTHONDIR", "/root/.ipython")
    monkeypatch.setenv("JUPYTER_CONFIG_DIR", "/root/.jupyter")
    monkeypatch.setenv("PYTHONSTARTUP", "/root/.pythonrc.py")

    environment = BOOTSTRAP.isolated_runtime_environment(tmp_path)

    assert environment["IPYTHONDIR"] != "/root/.ipython"
    assert environment["JUPYTER_CONFIG_DIR"] != "/root/.jupyter"
    assert "PYTHONSTARTUP" not in environment
    assert (tmp_path / "ipython").is_dir()
    assert (tmp_path / "jupyter-config").is_dir()


def test_bootstrap_creates_explicit_isolated_kernel_spec(tmp_path: Path) -> None:
    data_root = BOOTSTRAP.create_isolated_kernel_spec(tmp_path)
    kernel_path = data_root / "kernels" / BOOTSTRAP.ISOLATED_KERNEL_NAME / "kernel.json"
    kernel = json.loads(kernel_path.read_text(encoding="utf-8"))

    expected_argv = [
        str(Path(BOOTSTRAP.sys.executable).absolute()),
        "-m",
        "ipykernel_launcher",
        "--IPKernelApp.kernel_class=ipykernel.ipkernel.IPythonKernel",
        *BOOTSTRAP.ISOLATED_KERNEL_TRAIT_OVERRIDES,
        "-f",
        "{connection_file}",
    ]
    assert kernel["argv"] == expected_argv
    assert kernel["display_name"] == "ViPIBench isolated runtime"
    assert kernel["env"] == {
        "IPYTHONDIR": str(tmp_path / "ipython"),
        "JUPYTER_CONFIG_DIR": str(tmp_path / "jupyter-config"),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSTARTUP": "",
        "PYTHONUTF8": "1",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
    }


def test_bootstrap_forces_nbconvert_to_use_isolated_kernel() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")
    notebook = json.loads(
        (BUNDLE_ROOT / "vipibench-guardrail" / "notebooks" / "confirmatory_run.ipynb").read_text(
            encoding="utf-8"
        )
    )
    notebook_source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "create_isolated_kernel_spec(work)" in source
    assert "--ExecutePreprocessor.kernel_name={ISOLATED_KERNEL_NAME}" in source
    assert 'env["VIPIBENCH_RUNTIME_PYTHON"]' in source
    assert 'env["PYTHONPATH"] = (' in source
    assert 'staged_source_root = str(project / "src")' in source
    assert "notebook_kernel_interpreter_mismatch" in notebook_source


def test_bootstrap_executes_a_hash_checked_transient_notebook_copy() -> None:
    source = BOOTSTRAP_PATH.read_text(encoding="utf-8")

    assert 'execution_notebook = work / "controller-notebook" / notebook.name' in source
    assert "shutil.copy2(notebook, execution_notebook)" in source
    assert "notebook_execution_copy_hash_mismatch" in source
    assert "str(execution_notebook)," in source


def test_confirmatory_notebook_surfaces_preflight_receipt_failures() -> None:
    notebook = json.loads(
        (BUNDLE_ROOT / "vipibench-guardrail" / "notebooks" / "confirmatory_run.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert "PREFLIGHT_PROCESS = subprocess.run(" in source
    assert "check=False" in source
    assert '"failed_check_evidence"' in source
    assert 'RuntimeError({"preflight_failed": diagnostic})' in source


def test_confirmatory_notebook_surfaces_encoder_root_cause_line_by_line() -> None:
    notebook = json.loads(
        (BUNDLE_ROOT / "vipibench-guardrail" / "notebooks" / "confirmatory_run.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'encoder_root / "capacity_plan.json"' in source
    assert 'encoder_root.glob("*/numerical_failure.json")' in source
    assert '"core_target_trajectories.run.json"' in source
    assert '"attack_target_trajectories.run.json"' in source
    assert '"related_receipts": related_receipts' in source
    assert "print_stage_failure_tail(stage_id, output_tail, related_receipts)" in source
    assert "Lần chạy này chưa đủ điều kiện để rút ra kết luận nghiên cứu" in source
    assert 'json.dumps({"stage_failure": summary}' not in source


def test_confirmatory_notebook_requires_capacity_numerics_canary() -> None:
    notebook = json.loads(
        (BUNDLE_ROOT / "vipibench-guardrail" / "notebooks" / "confirmatory_run.ipynb").read_text(
            encoding="utf-8"
        )
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'capacity_numerics_canary.get("status") != "PASS"' in source
    assert 'capacity_numerics_canary.get("candidate_id")' in source
    assert '"capacity_numerics_canary": capacity_numerics_canary' in source


def test_isolated_kernel_ignores_host_ipython_extensions(tmp_path: Path) -> None:
    nbformat = pytest.importorskip("nbformat")
    pytest.importorskip("ipykernel")
    pytest.importorskip("nbconvert")

    host_config = tmp_path / "host-ipython"
    host_config.mkdir()
    (host_config / "ipython_config.py").write_text(
        "\n".join(
            [
                "c = get_config()",
                "c.IPKernelApp.extensions = ['vipibench_missing_host_extension']",
                "c.IPKernelApp.reraise_ipython_extension_failures = True",
                "c.IPKernelApp.matplotlib = 'inline'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    startup_hook = tmp_path / "startup-hook"
    startup_hook.mkdir()
    (startup_hook / "sitecustomize.py").write_text(
        "\n".join(
            [
                "import os",
                "from IPython.core import application",
                "host_config = os.environ['VIPIBENCH_TEST_HOST_IPYTHON_CONFIG']",
                "application.SYSTEM_CONFIG_DIRS.insert(0, host_config)",
                "",
            ]
        ),
        encoding="utf-8",
    )

    notebook_path = tmp_path / "kernel_probe.ipynb"
    notebook = nbformat.v4.new_notebook(
        cells=[nbformat.v4.new_code_cell("print('VIPIBENCH_KERNEL_READY')")],
        metadata={
            "kernelspec": {
                "display_name": "ViPIBench isolated runtime",
                "language": "python",
                "name": BOOTSTRAP.ISOLATED_KERNEL_NAME,
            }
        },
    )
    nbformat.write(notebook, notebook_path)

    work = tmp_path / "work"
    environment = BOOTSTRAP.isolated_runtime_environment(work)
    data_root = BOOTSTRAP.create_isolated_kernel_spec(work)
    environment["JUPYTER_PATH"] = str(data_root)
    environment["PYTHONPATH"] = str(startup_hook)
    environment["VIPIBENCH_TEST_HOST_IPYTHON_CONFIG"] = str(host_config)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            "--ExecutePreprocessor.timeout=60",
            f"--ExecutePreprocessor.kernel_name={BOOTSTRAP.ISOLATED_KERNEL_NAME}",
            str(notebook_path),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=90,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    executed = nbformat.read(notebook_path, as_version=4)
    output_text = "".join(
        output.get("text", "")
        for output in executed.cells[0].get("outputs", [])
        if output.get("output_type") == "stream"
    )
    assert "VIPIBENCH_KERNEL_READY" in output_text
