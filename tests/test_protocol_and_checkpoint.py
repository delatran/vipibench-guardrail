from pathlib import Path

import pytest
import yaml

from vipibench.checkpoint import StageLedger
from vipibench.run_protocol import validate_encoder_protocol, validate_public_detector_protocol
from vipibench.sample_size import validate_sample_size_protocol


def test_locked_model_protocols_pass() -> None:
    assert validate_encoder_protocol(Path("configs/models/mdeberta_core.yaml"))["status"] == "PASS"
    assert (
        validate_public_detector_protocol(Path("configs/models/public_detector.yaml"))["status"]
        == "PASS"
    )


def test_encoder_protocol_rejects_zero_early_stopping_threshold(tmp_path: Path) -> None:
    source = Path("configs/models/mdeberta_core.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["early_stopping_threshold"] = 0
    path = tmp_path / "mdeberta.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    result = validate_encoder_protocol(path)
    assert result["status"] == "FAIL"
    assert "early_stopping_threshold_outside_locked_range" in result["errors"]


def test_encoder_protocol_rejects_bf16_after_numerics_amendment(tmp_path: Path) -> None:
    source = Path("configs/models/mdeberta_core.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["mixed_precision"] = "bf16"
    path = tmp_path / "mdeberta.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_encoder_protocol(path)

    assert result["status"] == "FAIL"
    assert "mixed_precision_must_equal_fp32" in result["errors"]


def test_encoder_protocol_rejects_reentrant_gradient_checkpointing(tmp_path: Path) -> None:
    source = Path("configs/models/mdeberta_core.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["gradient_checkpointing_use_reentrant"] = True
    path = tmp_path / "mdeberta.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_encoder_protocol(path)

    assert result["status"] == "FAIL"
    assert "gradient_checkpointing_must_use_non_reentrant" in result["errors"]


def test_encoder_protocol_rejects_single_step_numerics_canary(tmp_path: Path) -> None:
    source = Path("configs/models/mdeberta_core.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config["numerics_canary_optimizer_steps"] = 1
    path = tmp_path / "mdeberta.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_encoder_protocol(path)

    assert result["status"] == "FAIL"
    assert "numerics_canary_optimizer_steps_must_equal_2" in result["errors"]


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        (
            "capacity_probe_input_mode",
            "role_only",
            "capacity_probe_input_mode_must_equal_text_role",
        ),
        (
            "capacity_warmup_optimizer_steps",
            1,
            "capacity_warmup_optimizer_steps_must_equal_2",
        ),
        (
            "capacity_measurement_optimizer_steps",
            1,
            "capacity_measurement_optimizer_steps_must_equal_5",
        ),
        (
            "dataloader_worker_candidates",
            [2, 8],
            "dataloader_worker_candidates_must_equal_2_4_8",
        ),
        (
            "dataloader_worker_warmup_batches",
            1,
            "dataloader_worker_warmup_batches_must_equal_2",
        ),
        (
            "dataloader_worker_measurement_batches",
            4,
            "dataloader_worker_measurement_batches_must_equal_8",
        ),
        (
            "dataloader_worker_repeats",
            1,
            "dataloader_worker_repeats_must_equal_2",
        ),
        (
            "dataloader_persistent_workers",
            False,
            "dataloader_persistent_workers_must_be_enabled",
        ),
        (
            "dataloader_prefetch_factor",
            1,
            "dataloader_prefetch_factor_must_equal_2",
        ),
        (
            "final_holdout_feedback_allowed",
            True,
            "final_holdout_feedback_must_remain_disabled",
        ),
    ],
)
def test_encoder_protocol_rejects_capacity_benchmark_drift(
    tmp_path: Path,
    key: str,
    value: object,
    expected_error: str,
) -> None:
    source = Path("configs/models/mdeberta_core.yaml")
    config = yaml.safe_load(source.read_text(encoding="utf-8"))
    config[key] = value
    path = tmp_path / "mdeberta.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_encoder_protocol(path)

    assert result["status"] == "FAIL"
    assert expected_error in result["errors"]


@pytest.mark.parametrize(
    ("key", "value", "expected_error"),
    [
        (
            "batch_candidates",
            [32, 64],
            "public_detector_batch_candidates_must_equal_32_64_128_256",
        ),
        (
            "capacity_warmup_batches",
            0,
            "public_detector_capacity_warmup_batches_must_equal_1",
        ),
        (
            "capacity_measurement_batches",
            1,
            "public_detector_capacity_measurement_batches_must_equal_2",
        ),
        (
            "capacity_repeats",
            1,
            "public_detector_capacity_repeats_must_equal_2",
        ),
    ],
)
def test_public_detector_protocol_rejects_capacity_scout_drift(
    tmp_path: Path,
    key: str,
    value: object,
    expected_error: str,
) -> None:
    config = yaml.safe_load(
        Path("configs/models/public_detector.yaml").read_text(encoding="utf-8")
    )
    config[key] = value
    path = tmp_path / "public-detector.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = validate_public_detector_protocol(path)

    assert result["status"] == "FAIL"
    assert expected_error in result["errors"]


def test_revised_sample_size_protocol_passes() -> None:
    result = validate_sample_size_protocol(Path("configs/experiments/sample_size_scaling.yaml"))
    assert result["status"] == "PASS", result["errors"]


def test_stage_ledger_requires_matching_hashes(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path / "ledger")
    output = tmp_path / "result.json"
    output.write_text("{}\n", encoding="utf-8")
    ledger.complete("stage", [output], {"source_sha256": "A" * 64})
    assert ledger.verified_complete("stage")
    assert ledger.verified_complete("stage", {"source_sha256": "A" * 64})
    assert not ledger.verified_complete("stage", {"source_sha256": "B" * 64})
    output.write_text('{"changed": true}\n', encoding="utf-8")
    assert not ledger.verified_complete("stage")


def test_stage_ledger_replaces_stale_metadata_after_new_outputs(tmp_path: Path) -> None:
    ledger = StageLedger(tmp_path / "ledger")
    output = tmp_path / "result.json"
    output.write_text('{"version": 1}\n', encoding="utf-8")
    ledger.complete("stage", [output], {"source_sha256": "A" * 64})
    output.write_text('{"version": 2}\n', encoding="utf-8")
    metadata = {"source_sha256": "B" * 64}
    ledger.complete("stage", [output], metadata)
    assert ledger.verified_complete("stage", metadata)


def test_stage_ledger_marker_survives_output_root_relocation(tmp_path: Path) -> None:
    first_root = tmp_path / "session-a" / "run-output"
    first_root.mkdir(parents=True)
    output = first_root / "stage" / "result.json"
    output.parent.mkdir()
    output.write_text('{"ok": true}\n', encoding="utf-8")
    ledger = StageLedger(first_root / "ledger")
    marker = ledger.complete("stage", [output], {"source": "locked"})

    second_root = tmp_path / "session-b" / "run-output"
    second_root.mkdir(parents=True)
    relocated_output = second_root / "stage" / "result.json"
    relocated_output.parent.mkdir()
    relocated_output.write_bytes(output.read_bytes())
    relocated_ledger_root = second_root / "ledger"
    relocated_ledger_root.mkdir()
    (relocated_ledger_root / marker.name).write_bytes(marker.read_bytes())

    assert StageLedger(relocated_ledger_root).verified_complete("stage", {"source": "locked"})


def test_stage_ledger_rejects_output_outside_artifact_root(tmp_path: Path) -> None:
    output = tmp_path / "outside.json"
    output.write_text("{}\n", encoding="utf-8")
    ledger = StageLedger(tmp_path / "run-output" / "ledger")
    with pytest.raises(ValueError, match="within artifact_root"):
        ledger.complete("stage", [output], {})
