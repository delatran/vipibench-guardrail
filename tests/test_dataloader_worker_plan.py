import json
from collections import UserDict
from pathlib import Path
from types import SimpleNamespace

import pytest

import vipibench.transformer_runner as transformer_runner
from vipibench.dataio import write_json
from vipibench.modeling import load_yaml
from vipibench.transformer_runner import (
    _load_locked_dataloader_worker_plan,
    _load_or_measure_dataloader_workers,
    _select_dataloader_worker_count,
)


def _measurement(workers: int, throughput: float) -> dict[str, object]:
    return {
        "num_workers": workers,
        "status": "PASS",
        "repeat_samples_per_second": [throughput, throughput],
        "repeat_elapsed_seconds": [1.0, 1.0],
        "repeat_sample_counts": [64, 64],
        "median_samples_per_second": throughput,
        "error_type": None,
        "error_message": None,
    }


def test_worker_selection_prefers_throughput_then_fewer_workers() -> None:
    measurements = [
        _measurement(2, 100.0),
        _measurement(4, 150.0),
        _measurement(8, 150.0),
    ]

    assert _select_dataloader_worker_count(measurements) == 4


def test_worker_scout_accepts_non_dict_mapping_batch_with_labels() -> None:
    selected_batch_size = 4
    config = {
        "capacity_probe_input_mode": "text_role",
        "max_length": 512,
        "dataloader_worker_warmup_batches": 2,
        "dataloader_worker_measurement_batches": 8,
        "dataloader_worker_repeats": 2,
    }

    class FakeDataLoader:
        def __init__(self, *args: object, **kwargs: object) -> None:
            assert kwargs["batch_size"] == selected_batch_size
            assert kwargs["num_workers"] == 2
            assert kwargs["persistent_workers"] is True

        def __iter__(self):
            return iter(
                [
                    UserDict({"labels": [0] * selected_batch_size})
                    for _ in range(10)
                ]
            )

    fake_torch = SimpleNamespace(
        utils=SimpleNamespace(data=SimpleNamespace(DataLoader=FakeDataLoader))
    )

    measurement = transformer_runner._measure_dataloader_worker_candidate(
        config,
        [object()],
        selected_batch_size,
        2,
        fake_torch,
        object(),
        object(),
    )

    assert measurement["status"] == "PASS"
    assert measurement["repeat_sample_counts"] == [32, 32]
    assert measurement["error_type"] is None
    assert measurement["error_message"] is None


def test_runner_consumes_the_locked_worker_plan_for_train_and_test() -> None:
    source = Path("src/vipibench/transformer_runner.py").read_text(encoding="utf-8")

    assert "dataloader_num_workers=2" not in source
    assert "dataloader_num_workers=dataloader_num_workers" in source
    assert 'dataloader_num_workers=context["dataloader_num_workers"]' in source
    assert source.count("dataloader_persistent_workers=") >= 2
    assert source.count("dataloader_prefetch_factor=") >= 2
    assert source.count("dataloader_worker_plan_sha256") >= 5


def test_train_only_worker_scout_writes_and_reuses_bound_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/mdeberta_core.yaml")
    config = load_yaml(config_path)
    capacity_path = tmp_path / "capacity_plan.json"
    write_json(capacity_path, {"status": "PASS", "selected": {"batch_size": 16}})
    output_path = tmp_path / "dataloader_worker_plan.json"
    records = [SimpleNamespace(content_sha256="A" * 64)]
    throughputs = {2: 100.0, 4: 160.0, 8: 140.0}

    monkeypatch.setattr(transformer_runner.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        transformer_runner,
        "_measure_dataloader_worker_candidate",
        lambda config, records, batch, workers, torch, tokenizer, collator: _measurement(
            workers, throughputs[workers]
        ),
    )

    created = _load_or_measure_dataloader_workers(
        output_path,
        config_path,
        config,
        records,
        capacity_path,
        16,
        object(),
        object(),
        object(),
    )
    reused = _load_locked_dataloader_worker_plan(
        output_path,
        config_path,
        config,
        records,
        capacity_path,
        16,
    )

    assert created == reused
    assert created["selected_num_workers"] == 4
    assert created["probe_partition"] == "train"
    assert created["test_accessed"] is False
    assert created["final_holdout_feedback_allowed"] is False


def test_worker_plan_rejects_tampered_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/mdeberta_core.yaml")
    config = load_yaml(config_path)
    capacity_path = tmp_path / "capacity_plan.json"
    write_json(capacity_path, {"status": "PASS", "selected": {"batch_size": 16}})
    output_path = tmp_path / "dataloader_worker_plan.json"
    records = [SimpleNamespace(content_sha256="B" * 64)]
    monkeypatch.setattr(transformer_runner.os, "cpu_count", lambda: 8)
    monkeypatch.setattr(
        transformer_runner,
        "_measure_dataloader_worker_candidate",
        lambda config, records, batch, workers, torch, tokenizer, collator: _measurement(
            workers, float(workers)
        ),
    )
    _load_or_measure_dataloader_workers(
        output_path,
        config_path,
        config,
        records,
        capacity_path,
        16,
        object(),
        object(),
        object(),
    )
    tampered = json.loads(output_path.read_text(encoding="utf-8"))
    tampered["selected_num_workers"] = 2
    write_json(output_path, tampered)

    with pytest.raises(ValueError, match="selection mismatch"):
        _load_locked_dataloader_worker_plan(
            output_path,
            config_path,
            config,
            records,
            capacity_path,
            16,
        )
