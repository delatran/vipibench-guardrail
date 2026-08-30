import json
import os
from pathlib import Path

import jsonschema
import pytest

import vipibench.preflight as preflight
from vipibench.modeling import load_yaml
from vipibench.preflight import (
    LOCAL_MILESTONE,
    MILESTONE,
    _execution_profile_contract,
    _profile_contract,
    evaluate_confirmatory_launch_readiness,
)


def test_exact_accelerator_profile_satisfies_launch_contract() -> None:
    profile = load_yaml(Path("configs/profiles/accelerator_80gb.yaml"))

    passed, errors = _profile_contract(profile)

    assert passed is True
    assert errors == []


def test_standard_compute_profile_is_explicitly_16_to_24_gib_class() -> None:
    profile = load_yaml(Path("configs/profiles/standard_compute.yaml"))

    passed, errors = _execution_profile_contract("standard_compute", profile)

    assert passed is True
    assert errors == []


def test_accelerator_profile_rejects_wrong_capability_and_memory() -> None:
    profile = load_yaml(Path("configs/profiles/accelerator_80gb.yaml"))
    profile.update(
        {
            "required_compute_capability": "8.9",
            "minimum_device_memory_gib": 69,
            "minimum_system_ram_gib": "invalid",
        }
    )

    passed, errors = _profile_contract(profile)

    assert passed is False
    assert "required_compute_capability_mismatch" in errors
    assert "minimum_device_memory_gib_below_70" in errors
    assert "minimum_system_ram_gib_invalid" in errors


@pytest.mark.private_integration
def test_live_preflight_is_schema_valid_and_claim_bounded(tmp_path: Path) -> None:
    output = tmp_path / "prelaunch_readiness.json"
    bootstrapping = any(
        os.environ.get(name) == "1"
        for name in (
            "VIPIBENCH_CLEAN_ENV_BOOTSTRAP",
            "VIPIBENCH_CURRENT_WHEEL_BOOTSTRAP",
        )
    )

    result = evaluate_confirmatory_launch_readiness(
        Path.cwd(),
        verify_hash=True,
        output_path=output,
        require_clean_environment=not bootstrapping,
    )

    if result["status"] == "PASS":
        assert result["failed_checks"] == []
        assert result["milestone"] == (LOCAL_MILESTONE if bootstrapping else MILESTONE)
    else:
        allowed_host_only_failures = {
            "confirmatory_readiness",
            "launch_artifact_manifest_binding",
            "clean_environment",
            "current_wheel",
        }
        assert set(result["failed_checks"]).issubset(allowed_host_only_failures)
        assert result["milestone"] == "NOT_READY"
    assert result["hardware_observed"] is False
    assert result["paid_compute_authorized"] is False
    assert result["external_actions_performed"] == []
    assert all(
        isinstance(value, str) and len(value) == 64 for value in result["launch_hashes"].values()
    )
    assert "proposal" not in result["launch_hashes"]
    schema = json.loads(Path("outputs/prelaunch_readiness.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    assert output.is_file()


def test_preflight_uses_session_environment_and_defers_stale_host_cleanliness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session_environment = tmp_path / "session_environment.json"
    session_environment.write_text('{"status": "PASS"}', encoding="utf-8")
    captured: dict[str, object] = {}

    def fake_readiness(*args: object, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {"status": "PASS", "failed_checks": []}

    monkeypatch.setattr(preflight, "evaluate_launch_readiness", fake_readiness)
    result = evaluate_confirmatory_launch_readiness(
        Path.cwd(),
        verify_hash=True,
        output_path=tmp_path / "preflight.json",
        runtime_environment_compatibility_path=session_environment,
        active_isolated_runtime=True,
    )

    assert captured["require_clean_environment"] is False
    overrides = captured["artifact_overrides"]
    assert overrides == {"environment_compatibility": session_environment}
    names = [str(check["name"]) for check in result["checks"]]
    assert "clean_environment" not in names
    assert "current_wheel" not in names
    assert "bootstrap_evidence_boundary" in names


@pytest.mark.private_integration
def test_active_runtime_preflight_accepts_current_session_evidence(tmp_path: Path) -> None:
    session_environment = tmp_path / "session_environment.json"
    session_environment.write_text(
        json.dumps(
            {
                "status": "PASS",
                "runtime_source_fingerprint": preflight.runtime_source_fingerprint(Path.cwd()),
            }
        ),
        encoding="utf-8",
    )

    result = evaluate_confirmatory_launch_readiness(
        Path.cwd(),
        verify_hash=True,
        output_path=tmp_path / "preflight.json",
        runtime_environment_compatibility_path=session_environment,
        active_isolated_runtime=True,
    )

    assert result["status"] == "PASS"
    assert result["milestone"] == MILESTONE
    assert result["failed_checks"] == []
