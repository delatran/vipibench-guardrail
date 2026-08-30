import json
from pathlib import Path

import pytest

from vipibench.artifact_binding import directory_fingerprint
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS
from vipibench.transformer_runner import _selected_capacity_batch, select_encoder_run


def test_model_selection_uses_only_development_metric(tmp_path: Path) -> None:
    for mode in LOCKED_INPUT_MODES:
        for seed in LOCKED_SEEDS:
            run_id = f"mdeberta-{mode}-s{seed}"
            run_dir = tmp_path / run_id
            run_dir.mkdir(parents=True)
            model_dir = run_dir / "model"
            model_dir.mkdir()
            (model_dir / "weights.bin").write_bytes(f"{mode}-{seed}".encode())
            model_version = directory_fingerprint(model_dir)
            score = 0.5
            if mode == "text_role":
                score = {17: 0.81, 29: 0.84, 43: 0.82}[seed]
            (run_dir / "development_metrics.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "run_id": run_id,
                        "seed": seed,
                        "input_mode": mode,
                        "dev_auprc": score,
                        "test_accessed": False,
                    }
                ),
                encoding="utf-8",
            )
            (run_dir / "model_binding.json").write_text(
                json.dumps(
                    {
                        "status": "PASS",
                        "model_artifact_version": model_version,
                    }
                ),
                encoding="utf-8",
            )
    result = select_encoder_run(Path("configs/models/mdeberta_core.yaml"), tmp_path)
    assert result["status"] == "PASS"
    assert result["selected"]["run_id"] == "mdeberta-text_role-s29"
    assert result["selected"]["model_artifact_version"] == directory_fingerprint(
        tmp_path / "mdeberta-text_role-s29" / "model"
    )
    assert result["test_accessed"] is False
    assert not any(tmp_path.glob("*/test_predictions.jsonl"))


def test_test_evaluation_reuses_autonomously_selected_capacity() -> None:
    capacity = {
        "status": "PASS",
        "selected": {"candidate_id": "batch-32-checkpoint-on", "batch_size": 32},
    }
    assert _selected_capacity_batch(capacity, effective_batch_size=64) == 32


def test_capacity_selection_cannot_break_effective_batch_contract() -> None:
    capacity = {
        "status": "PASS",
        "selected": {"candidate_id": "batch-128-checkpoint-on", "batch_size": 128},
    }
    with pytest.raises(ValueError, match="must divide"):
        _selected_capacity_batch(capacity, effective_batch_size=64)
