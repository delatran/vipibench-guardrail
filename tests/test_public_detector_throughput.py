import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import vipibench.transformer_runner as transformer_runner
from vipibench.dataio import write_json
from vipibench.exec_detector_data import load_executable_episodes
from vipibench.modeling import load_yaml


def test_public_detector_capacity_scout_uses_dev_sync_repeats_and_median(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3

    class FakeMemory:
        def get_memory_info(self):
            return 40 * gib, 80 * gib

        def empty_cache(self):
            return None

        def reset_peak_memory_stats(self):
            return None

        def max_memory_reserved(self):
            return 40 * gib

    class FakeAccelerator:
        def __init__(self) -> None:
            self.memory = FakeMemory()
            self.synchronize_calls = 0

        def synchronize(self) -> None:
            self.synchronize_calls += 1

    accelerator = FakeAccelerator()
    fake_torch = SimpleNamespace(accelerator=accelerator, OutOfMemoryError=RuntimeError)
    config = load_yaml(Path("configs/models/public_detector.yaml"))
    config["batch_candidates"] = [32, 64]
    measured_batches: list[tuple[int, str]] = []

    def fake_score(*args, **kwargs):
        texts = args[4]
        measured_batches.append((len(texts), kwargs["phase"]))
        return [0.5] * len(texts)

    times = iter([0.0, 4.0, 4.0, 6.0, 6.0, 8.0, 8.0, 9.0])
    monkeypatch.setattr(transformer_runner, "_score_public_detector_batch", fake_score)
    monkeypatch.setattr(transformer_runner.time, "perf_counter", lambda: next(times))

    result = transformer_runner._measure_public_detector_capacity(
        fake_torch,
        object(),
        object(),
        object(),
        ["development text"],
        config,
    )

    assert result["status"] == "PASS"
    assert result["selected"]["batch_size"] == 64
    assert result["selection_partition"] == "development_only"
    assert result["test_accessed"] is False
    assert result["repeat_samples_per_second"] == {
        "batch-32": [16.0, 32.0],
        "batch-64": [64.0, 128.0],
    }
    assert [item[0] for item in measured_batches] == [32] * 5 + [64] * 5
    assert all("test" not in phase for _, phase in measured_batches)
    assert accelerator.synchronize_calls == 10


def test_public_detector_benchmark_loads_model_once_and_binds_capacity_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/public_detector.yaml")
    records = load_executable_episodes(Path("data/splits/confirmatory_final/dev.jsonl"))[:2]
    model = object()
    device = SimpleNamespace(type="cuda")
    load_count = 0
    inference_models: list[object] = []

    monkeypatch.setattr(transformer_runner, "_require_confirmatory_authorization", lambda: None)
    monkeypatch.setattr(
        transformer_runner,
        "check_runtime_profile_path",
        lambda *args: {"status": "PASS", "hardware_observed": True, "errors": []},
    )
    monkeypatch.setattr(
        transformer_runner,
        "load_benchmark_partitions",
        lambda *args: ({"dev": records, "test": records}, {"fixture": "A" * 64}),
    )

    def fake_load(config):
        nonlocal load_count
        load_count += 1
        return object(), object(), model, device

    monkeypatch.setattr(transformer_runner, "_load_public_detector", fake_load)
    monkeypatch.setattr(
        transformer_runner,
        "validate_model_device_placement",
        lambda *args, **kwargs: {"status": "PASS", "normalized_placements": ["cuda"]},
    )
    monkeypatch.setattr(
        transformer_runner,
        "_measure_public_detector_capacity",
        lambda *args: {
            "status": "PASS",
            "errors": [],
            "selected": {"candidate_id": "batch-64", "batch_size": 64},
            "selection_partition": "development_only",
            "test_accessed": False,
            "final_holdout_feedback_allowed": False,
        },
    )

    def fake_inference(config, dataset_path, output_path, **kwargs):
        inference_models.append(kwargs["model"])
        output_path.write_text('{"fixture": true}\n', encoding="utf-8")
        return {"status": "PASS", "batch_size": kwargs["batch_size"]}

    monkeypatch.setattr(transformer_runner, "_run_public_detector_loaded", fake_inference)

    def fake_thresholds(predictions, output):
        write_json(output, {"status": "PASS"})
        return {"status": "PASS"}

    def fake_evaluation(predictions, thresholds, output):
        write_json(output, {"status": "PASS", "research_claim_eligible": False})
        return {"status": "PASS", "research_claim_eligible": False}

    monkeypatch.setattr(transformer_runner, "calibrate_thresholds", fake_thresholds)
    monkeypatch.setattr(transformer_runner, "evaluate_predictions", fake_evaluation)

    result = transformer_runner.run_public_detector_benchmark(
        config_path,
        Path("data/splits/confirmatory_final"),
        tmp_path,
    )

    capacity = json.loads((tmp_path / "capacity_plan.json").read_text(encoding="utf-8"))
    assert load_count == 1
    assert inference_models == [model, model]
    assert result["shared_model_load_count"] == 1
    assert result["selected_batch_size"] == 64
    assert capacity["selection_partition"] == "development_only"
    assert capacity["test_accessed"] is False
    assert capacity["development_input_sha256"]
