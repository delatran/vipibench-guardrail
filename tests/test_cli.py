import json

import pytest
from typer.testing import CliRunner

from vipibench.cli import app

runner = CliRunner()


def test_version_command() -> None:
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"


def test_encoder_gpu_and_cpu_substage_commands_are_exposed() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "run-encoder-accelerator-matrix" in result.stdout
    assert "run-encoder-test-predictions" in result.stdout
    assert "run-encoder-test-analysis" in result.stdout
    assert "validate-precision-scout" in result.stdout
    assert "run-precision-scout" in result.stdout
    assert "evaluate-precision-scout" in result.stdout
    assert "materialize-report-assets" in result.stdout


@pytest.mark.private_integration
def test_doctor_passes_revised_contract() -> None:
    result = runner.invoke(app, ["doctor", "--project-root", "."])
    assert result.exit_code == 0, result.stdout
    assert '"status": "PASS"' in result.stdout


def test_removed_human_annotation_command_is_not_exposed() -> None:
    result = runner.invoke(app, ["annotation-package"])
    assert result.exit_code != 0
    assert "No such command" in result.output


def test_verify_exec_oracle_command_passes(tmp_path) -> None:
    output = tmp_path / "oracle.json"
    result = runner.invoke(app, ["verify-exec-oracle", "--output", str(output)])
    assert result.exit_code == 0, result.stdout
    assert '"exact_contract_agreement": 1.0' in result.stdout
    assert output.is_file()


def test_export_exec_schema_command_passes(tmp_path) -> None:
    output = tmp_path / "episode.schema.json"
    result = runner.invoke(app, ["export-exec-schema", "--output", str(output)])
    assert result.exit_code == 0, result.stdout
    assert output.is_file()


def test_verify_artifact_manifest_fails_when_manifest_is_missing(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "verify-artifact-manifest",
            "--project-root",
            ".",
            "--manifest",
            str(tmp_path / "missing.json"),
        ],
    )
    assert result.exit_code != 0


def test_check_accelerator_fails_closed_without_accelerator_and_writes_evidence(
    monkeypatch, tmp_path
) -> None:
    from vipibench.runtime_capacity import RuntimeProbe

    unavailable_probe = RuntimeProbe(
        compute_available=False,
        device_type=None,
        device_name=None,
        device_index=None,
        device_memory_gib=0.0,
        bf16_supported=False,
        tensor_probe_passed=False,
        system_ram_gib=64.0,
        disk_free_gib=100.0,
        compute_capability=None,
        evidence_kind="mock_test",
    )
    monkeypatch.setattr(
        "vipibench.runtime_capacity.observe_runtime",
        lambda _path: unavailable_probe,
    )

    output = tmp_path / "accelerator.json"
    result = runner.invoke(
        app,
        [
            "check-accelerator",
            "--profile",
            "configs/profiles/accelerator_80gb.yaml",
            "--disk-path",
            ".",
            "--output",
            str(output),
        ],
    )

    assert result.exit_code != 0
    assert '"status": "FAIL"' in result.stdout
    assert "compute_device_unavailable" in result.stdout
    assert output.is_file()
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["status"] == "FAIL"
    assert "compute_device_unavailable" in evidence["errors"]
    assert evidence["hardware_observed"] is False


def test_audit_postrun_fails_closed_for_missing_raw_evidence(tmp_path) -> None:
    output = tmp_path / "postrun_audit.json"
    result = runner.invoke(
        app,
        [
            "audit-postrun",
            "--project-root",
            ".",
            "--output-root",
            str(tmp_path / "missing-run"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 1
    assert '"status": "FAIL"' in result.stdout
    assert output.is_file()


def test_run_target_agent_command_exits_nonzero_for_retained_format_failure(
    monkeypatch, tmp_path
) -> None:
    failure = {
        "schema_version": "2.3.0",
        "status": "FAIL",
        "errors": ["target_run_failures_observed"],
        "format_failure_summary": {
            "status": "FAIL",
            "errors": ["target_run_failures_observed"],
            "format_fallback_count": 0,
            "format_fallback_episode_ids": [],
            "parse_failure_count": 1,
            "parse_failure_episode_ids": ["episode-repaired"],
            "parse_error_class_counts": {"json_decode_error": 1},
            "raw_response_included": False,
        },
    }
    monkeypatch.setattr("vipibench.cli.run_target_agent", lambda **_: failure)

    result = runner.invoke(
        app,
        [
            "run-target-agent",
            "--dataset",
            str(tmp_path / "dataset.jsonl"),
            "--output",
            str(tmp_path / "trajectories.jsonl"),
            "--checkpoint-dir",
            str(tmp_path / "checkpoints"),
        ],
    )

    assert result.exit_code == 1
    emitted = json.loads(result.stdout)
    assert emitted == failure


def test_consolidate_runtime_telemetry_maps_domain_pass_to_cli_success(
    monkeypatch, tmp_path
) -> None:
    output = tmp_path / "runtime_telemetry.json"
    ledger = {
        "schema_version": "1.0.0",
        "kind": "runtime_telemetry_ledger",
        "validation_status": "PASS",
        "execution_status": "completed",
    }

    def fake_consolidate(*args, output_path, **kwargs):
        del args, kwargs
        output_path.write_text(json.dumps(ledger), encoding="utf-8")
        return ledger

    monkeypatch.setattr(
        "vipibench.cli.consolidate_live_telemetry_ledgers",
        fake_consolidate,
    )
    result = runner.invoke(
        app,
        [
            "consolidate-runtime-telemetry",
            "--telemetry",
            str(tmp_path / "core.telemetry.json"),
            "--strict-capacity-receipt",
            str(tmp_path / "strict_capacity_receipt.json"),
            "--output",
            str(output),
        ],
    )

    assert result.exit_code == 0, result.stdout
    emitted = json.loads(result.stdout)
    assert emitted["status"] == "PASS"
    assert emitted["errors"] == []
    assert emitted["output"] == str(output)
    assert emitted["runtime_telemetry"] == ledger
    assert json.loads(output.read_text(encoding="utf-8")) == ledger
