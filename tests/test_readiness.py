import json
import os
from pathlib import Path

import jsonschema
import pytest

from vipibench.readiness import MILESTONE, _artifact_check, evaluate_launch_readiness


def test_missing_artifact_fails_closed() -> None:
    result = _artifact_check(Path.cwd(), "missing", "outputs/does-not-exist.json")
    assert result["status"] == "FAIL"
    assert result["evidence"]["exists"] is False


@pytest.mark.private_integration
def test_live_readiness_is_schema_valid_and_bounded(tmp_path: Path) -> None:
    output = tmp_path / "readiness.json"
    bootstrapping_clean_environment = os.environ.get("VIPIBENCH_CLEAN_ENV_BOOTSTRAP") == "1"
    result = evaluate_launch_readiness(
        Path.cwd(),
        output_path=output,
        require_clean_environment=not bootstrapping_clean_environment,
    )
    if result["status"] == "PASS":
        assert result["failed_checks"] == []
        if bootstrapping_clean_environment:
            assert result["milestone"] == "LOCAL_SMOKE_CONTRACT_PASS"
        else:
            assert result["milestone"] == MILESTONE
    else:
        # A developer host is not the locked Colab runtime.  The readiness
        # checker must expose that mismatch rather than relabel it as a pass.
        allowed_host_only_failures = {
            "clean_environment",
            "environment_compatibility",
            "training_authorization_live",
            "artifact_manifest_live",
        }
        assert set(result["failed_checks"]).issubset(allowed_host_only_failures)
        # The clean-environment bootstrap intentionally refreshes its
        # compatibility receipt before this test executes.  It can therefore
        # be current in either a clean bootstrap or a later developer-host
        # test run.  A stale or malformed receipt is independently represented
        # by the bounded ``environment_compatibility`` failure above; this test
        # must not require that failure when a current isolated receipt exists.
        assert result["milestone"] == "NOT_READY"
    assert result["hardware_observed"] is False
    assert result["paid_compute_authorized"] is False
    assert result["external_actions_performed"] == []
    schema = json.loads(Path("outputs/readiness_report.schema.json").read_text(encoding="utf-8"))
    jsonschema.validate(result, schema)
    assert output.is_file()
