import json
from pathlib import Path

import pytest
import yaml

from vipibench.dataio import write_json
from vipibench.precision_scout import (
    evaluate_precision_scout,
    run_precision_scout,
    validate_precision_scout_protocol,
)

CONFIG = Path("configs/models/mdeberta_bf16_scout.yaml")


def test_live_precision_scout_protocol_passes() -> None:
    result = validate_precision_scout_protocol(Path.cwd(), CONFIG)

    assert result["status"] == "PASS", result["errors"]
    assert result["run_count_per_arm"] == 3
    assert result["test_access_allowed"] is False
    assert result["automatic_promotion_allowed"] is False


def test_precision_scout_rejects_nonprecision_hyperparameter_drift(tmp_path: Path) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["learning_rate"] = 3.0e-5
    path = tmp_path / "scout.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_precision_scout_protocol(Path.cwd(), path)

    assert result["status"] == "FAIL"
    assert "precision_scout_nonprecision_drift:learning_rate" in result["errors"]


def test_precision_scout_requires_explicit_live_authorization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("VIPIBENCH_PRECISION_SCOUT_APPROVED", raising=False)

    with pytest.raises(PermissionError, match="VIPIBENCH_PRECISION_SCOUT_APPROVED=YES"):
        run_precision_scout(
            Path.cwd(),
            CONFIG,
            Path("data/splits/confirmatory_final"),
            tmp_path,
        )


def _arm_summary(samples_per_second: float) -> dict[str, object]:
    return {
        "status": "PASS",
        "test_accessed": False,
        "loaded_partitions": ["train", "dev"],
        "capacity_plan": {
            "status": "PASS",
            "selected": {"samples_per_second": samples_per_second},
        },
        "runtime_check": {
            "status": "PASS",
            "hardware_observed": True,
            "probe": {
                "device_name": "NVIDIA A100-SXM4-80GB",
                "device_index": 0,
                "compute_capability": "8.0",
                "device_memory_gib": 79.15,
            },
        },
    }


def _write_development_metric(
    path: Path,
    *,
    score: float,
    wall_seconds: float,
) -> None:
    write_json(
        path,
        {
            "status": "PASS",
            "dev_auprc": score,
            "training_wall_seconds": wall_seconds,
            "test_accessed": False,
        },
    )


def test_precision_scout_evaluation_can_pass_gates_without_promoting_bf16(
    tmp_path: Path,
) -> None:
    control_root = tmp_path / "fp32-control"
    candidate_root = tmp_path / "bf16-candidate"
    write_json(control_root / "arm_summary.json", _arm_summary(100.0))
    write_json(candidate_root / "arm_summary.json", _arm_summary(120.0))
    protocol = validate_precision_scout_protocol(Path.cwd(), CONFIG)
    for entry in protocol["run_matrix"]:
        assert isinstance(entry, dict)
        _write_development_metric(
            control_root / str(entry["control_run_id"]) / "development_metrics.json",
            score=0.900,
            wall_seconds=100.0,
        )
        _write_development_metric(
            candidate_root / str(entry["run_id"]) / "development_metrics.json",
            score=0.899,
            wall_seconds=80.0,
        )

    output = tmp_path / "evaluation.json"
    result = evaluate_precision_scout(Path.cwd(), CONFIG, tmp_path, output)

    assert result["status"] == "PASS", result["errors"]
    assert result["quality_gate_passed"] is True
    assert result["future_protocol_review_eligible"] is True
    assert result["automatic_promotion_authorized"] is False
    assert result["research_claim_eligible"] is False
    assert result["final_holdout_accessed"] is False
    assert output.is_file()


def test_precision_scout_evaluation_rejects_test_artifact(tmp_path: Path) -> None:
    (tmp_path / "bf16-candidate").mkdir(parents=True)
    (tmp_path / "bf16-candidate" / "test_predictions.jsonl").write_text(
        json.dumps({"score": 0.5}) + "\n",
        encoding="utf-8",
    )

    result = evaluate_precision_scout(Path.cwd(), CONFIG, tmp_path)

    assert result["status"] == "FAIL"
    assert "scout_test_or_evaluation_artifact_present" in result["errors"]
    assert result["automatic_promotion_authorized"] is False
