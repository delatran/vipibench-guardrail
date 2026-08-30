from copy import deepcopy
from pathlib import Path

import yaml

from vipibench.analysis_protocol import validate_confirmatory_analysis_protocol

CONFIG = Path("configs/experiments/confirmatory_analysis.yaml")


def _load() -> dict[str, object]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert isinstance(value, dict)
    return value


def test_live_confirmatory_analysis_protocol_passes(tmp_path: Path) -> None:
    output = tmp_path / "analysis.json"
    result = validate_confirmatory_analysis_protocol(Path.cwd(), CONFIG, output)
    assert result["status"] == "PASS", result["errors"]
    assert result["observed_holdout_counts"] == {
        "total": 480,
        "injection": 240,
        "benign": 240,
    }
    assert result["pretest_exposure_hits"] == []
    assert (
        result["mde_summary"]["h3_locked_margins_power_adequate_under_worst_case_variance"] is False
    )
    assert output.is_file()


def test_analysis_protocol_rejects_posthoc_power_claim(tmp_path: Path) -> None:
    config = deepcopy(_load())
    mde = config["mde_and_power"]
    assert isinstance(mde, dict)
    mde["dev_observed_variance_used"] = True
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = validate_confirmatory_analysis_protocol(Path.cwd(), path)
    assert result["status"] == "FAIL"
    assert "unobserved_dev_variance_must_not_be_claimed" in result["errors"]


def test_analysis_protocol_rejects_holdout_hash_tampering(tmp_path: Path) -> None:
    config = deepcopy(_load())
    holdout = config["confirmatory_holdout"]
    assert isinstance(holdout, dict)
    holdout["sha256"] = "0" * 64
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = validate_confirmatory_analysis_protocol(Path.cwd(), path)
    assert result["status"] == "FAIL"
    assert "binding_mismatch:confirmatory_holdout" in result["errors"]


def test_analysis_protocol_rejects_bootstrap_engine_drift(tmp_path: Path) -> None:
    config = deepcopy(_load())
    common = config["common_inference"]
    assert isinstance(common, dict)
    common["grouped_bootstrap_engine"] = "legacy_sequential_rng"
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_confirmatory_analysis_protocol(Path.cwd(), path)

    assert result["status"] == "FAIL"
    assert "common_inference_contract_mismatch" in result["errors"]


def test_analysis_protocol_rejects_outer_amendment_drift(tmp_path: Path) -> None:
    config = deepcopy(_load())
    config["protocol_amendment"] = "historical-amendment"
    path = tmp_path / "analysis.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_confirmatory_analysis_protocol(Path.cwd(), path)

    assert result["status"] == "FAIL"
    assert "analysis_protocol_amendment_mismatch" in result["errors"]
