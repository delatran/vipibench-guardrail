from __future__ import annotations

import json
from pathlib import Path

import pytest

from vipibench.checkpoint import StageLedger
from vipibench.dataio import write_json
from vipibench.stage_orchestration import (
    build_stage_status,
    checkpoint_metadata,
    complete_stage_group,
    load_stage_plan,
    public_stage_ids,
    require_stage_prerequisite,
    stage_enabled,
    validate_stage_selection,
    verify_stage_group,
)

PLAN_PATH = Path("configs/resources/confirmatory_stage_plan.json")


def _binding() -> dict[str, object]:
    return {
        "runtime_source_fingerprint": "A" * 64,
        "launch_hashes": {"artifact_manifest": "B" * 64},
        "launch_authorization_sha256": "C" * 64,
        "stage_plan_sha256": "D" * 64,
        "protocol_amendment": "a100-80gb-hardware-only-2026-08-08",
        "durable_lineage": "staged-test-lineage",
    }


def _ledgers(output_root: Path) -> tuple[StageLedger, StageLedger]:
    return (
        StageLedger(output_root / "orchestration_ledger", artifact_root=output_root),
        StageLedger(output_root / "stage_group_ledger", artifact_root=output_root),
    )


def _complete_member(
    ledger: StageLedger,
    output_root: Path,
    stage_id: str,
    binding: dict[str, object],
) -> Path:
    output = output_root / f"{stage_id}.json"
    write_json(output, {"status": "PASS", "stage": stage_id})
    ledger.complete(stage_id, [output], checkpoint_metadata(["run", stage_id], binding))
    return output


def test_locked_stage_plan_is_sequential_and_selectable() -> None:
    plan = load_stage_plan(PLAN_PATH)
    ids = public_stage_ids(plan)

    assert ids == [
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
    stages = {stage["id"]: stage for stage in plan["stages"]}
    assert stages["encoder"]["member_stage_ids"] == ["encoder-matrix"]
    assert stages["analysis"]["member_stage_ids"][:2] == [
        "encoder-test-analysis",
        "encoder-ablation-analysis",
    ]
    assert stages["finalize"]["member_stage_ids"][-1] == "materialize-report-assets"
    assert validate_stage_selection(plan, "ATTACK-GENERATE") == "attack-generate"
    assert validate_stage_selection(plan, "all") == "all"
    assert stage_enabled("all", "encoder") is True
    assert stage_enabled("data", "encoder") is False
    with pytest.raises(ValueError, match="unknown confirmatory stage"):
        validate_stage_selection(plan, "parallel-gpu")


def test_group_completion_verifies_member_and_direct_output_hashes(tmp_path: Path) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)
    binding = _binding()
    for member in ("compile-provenance-contrast", "audit-provenance-contrast"):
        _complete_member(stage_ledger, tmp_path, member, binding)
    direct = tmp_path / "data-direct.json"
    write_json(direct, {"status": "PASS"})

    complete_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
        direct_outputs=[direct],
    )

    assert verify_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )["status"] == "PASS"
    direct.write_text("tampered\n", encoding="utf-8")
    result = verify_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )
    assert result["status"] == "FAIL"
    assert "group_direct_output_hash_mismatch:data-direct.json" in result["errors"]


def test_group_accepts_manifest_maintenance_drift_within_lineage(tmp_path: Path) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)
    binding = _binding()
    for member in ("compile-provenance-contrast", "audit-provenance-contrast"):
        _complete_member(stage_ledger, tmp_path, member, binding)
    complete_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )
    changed = {
        **binding,
        "runtime_source_fingerprint": "E" * 64,
        "launch_authorization_sha256": "F" * 64,
        "launch_hashes": {
            **binding["launch_hashes"],
            "artifact_manifest": "D" * 64,
        },
    }

    result = verify_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=changed,
    )

    assert result["status"] == "PASS"


def test_group_rejects_lineage_identity_drift(tmp_path: Path) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)
    binding = _binding()
    for member in ("compile-provenance-contrast", "audit-provenance-contrast"):
        _complete_member(stage_ledger, tmp_path, member, binding)
    complete_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )
    changed = {**binding, "protocol_amendment": "different-amendment"}

    result = verify_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=changed,
    )

    assert result["status"] == "FAIL"
    assert "group_receipt_contract_mismatch" in result["errors"]


def test_out_of_order_stage_is_rejected_before_workload(tmp_path: Path) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)

    with pytest.raises(RuntimeError, match="public_stage_prerequisite_not_verified"):
        require_stage_prerequisite(
            plan=plan,
            stage_id="encoder",
            stage_ledger=stage_ledger,
            group_ledger=group_ledger,
            output_root=tmp_path,
            run_binding=_binding(),
        )


def test_stage_status_exposes_only_next_ready_group(tmp_path: Path) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)
    binding = _binding()
    preflight_receipt = tmp_path / "session_evidence" / "preflight.json"
    write_json(preflight_receipt, {"status": "PASS"})
    complete_stage_group(
        plan=plan,
        stage_id="preflight",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
        direct_outputs=[preflight_receipt],
    )

    status = build_stage_status(
        plan=plan,
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )

    states = {item["id"]: item["state"] for item in status["stages"]}
    assert states["preflight"] == "COMPLETE"
    assert states["data"] == "READY"
    assert states["baselines"] == "BLOCKED"


def test_stage_status_does_not_claim_downstream_completion_after_prerequisite_drift(
    tmp_path: Path,
) -> None:
    plan = load_stage_plan(PLAN_PATH)
    stage_ledger, group_ledger = _ledgers(tmp_path)
    binding = _binding()
    preflight_receipt = tmp_path / "session_evidence" / "preflight.json"
    write_json(preflight_receipt, {"status": "PASS"})
    complete_stage_group(
        plan=plan,
        stage_id="preflight",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
        direct_outputs=[preflight_receipt],
    )
    for member in ("compile-provenance-contrast", "audit-provenance-contrast"):
        _complete_member(stage_ledger, tmp_path, member, binding)
    complete_stage_group(
        plan=plan,
        stage_id="data",
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )
    preflight_receipt.write_text("tampered\n", encoding="utf-8")

    status = build_stage_status(
        plan=plan,
        stage_ledger=stage_ledger,
        group_ledger=group_ledger,
        output_root=tmp_path,
        run_binding=binding,
    )

    states = {item["id"]: item["state"] for item in status["stages"]}
    errors = {item["id"]: item["verification_errors"] for item in status["stages"]}
    assert states["preflight"] == "READY"
    assert states["data"] == "BLOCKED"
    assert "prerequisite_chain_not_verified:preflight" in errors["data"]


def test_stage_plan_rejects_non_sequential_prerequisite(tmp_path: Path) -> None:
    plan = json.loads(PLAN_PATH.read_text(encoding="utf-8"))
    plan["stages"][2]["prerequisite"] = "preflight"
    path = tmp_path / "stage-plan.json"
    path.write_text(json.dumps(plan), encoding="utf-8")

    with pytest.raises(ValueError, match="prerequisite chain is invalid"):
        load_stage_plan(path)
