import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from vipibench import target_runner
from vipibench.agent_trajectory import (
    bind_agent_trajectory_record,
    build_agent_request,
    parse_agent_response,
)
from vipibench.dataio import sha256_file
from vipibench.exec_detector_data import load_executable_episodes
from vipibench.modeling import load_yaml
from vipibench.system_runner import build_safe_fallback_trajectory
from vipibench.target_runner import (
    GeneratedBatch,
    _load_target_checkpoint,
    _parse_with_repairs,
    _record_final_holdout_fallback,
    _target_checkpoint_binding,
    _write_target_checkpoint,
    run_target_agent,
    validate_target_protocol,
)


def test_target_protocol_is_pinned_deterministic_and_fail_closed() -> None:
    result = validate_target_protocol(Path("configs/models/target_agent.yaml"))
    assert result["status"] == "PASS", result["errors"]


def test_target_format_failure_summary_accepts_strict_bounded_repair() -> None:
    clean = SimpleNamespace(
        episode_id="episode-clean",
        format_fallback=False,
        format_status="strict_json",
        parse_error_class=None,
        output_token_count=128,
    )
    repaired = SimpleNamespace(
        episode_id="episode-repaired",
        format_fallback=False,
        format_status="repaired_json",
        parse_error_class="json_decode_error",
        output_token_count=256,
    )

    summary = target_runner._summarize_target_format_failures(
        [clean, repaired],
        max_new_tokens=4096,
    )

    assert summary == {
        "status": "PASS",
        "errors": [],
        "unusable_observation_count": 0,
        "unusable_observation_episode_ids": [],
        "unusable_observation_budget": 0,
        "unusable_observation_fraction_ceiling": 0.01,
        "unusable_observation_policy": "bounded_unusable_observation_two_sided_bounds_v1",
        "unusable_observation_budget_exceeded": False,
        "truncated_response_count": 0,
        "truncated_response_episode_ids": [],
        "response_token_ceiling": 4096,
        "response_token_ceiling_reached_count": 0,
        "response_token_ceiling_episode_ids": [],
        "format_fallback_count": 0,
        "format_fallback_episode_ids": [],
        "format_repair_count": 1,
        "format_repair_episode_ids": ["episode-repaired"],
        "parse_failure_count": 1,
        "parse_failure_episode_ids": ["episode-repaired"],
        "unresolved_parse_failure_count": 0,
        "unresolved_parse_failure_episode_ids": [],
        "parse_error_class_counts": {"json_decode_error": 1},
        "raw_response_included": False,
    }


def test_target_format_failure_summary_fails_closed_without_raw_content() -> None:
    repaired = SimpleNamespace(
        episode_id="episode-repaired",
        format_fallback=False,
        format_status="repaired_json",
        parse_error_class="json_decode_error",
        output_token_count=256,
    )
    fallback = SimpleNamespace(
        episode_id="episode-fallback",
        format_fallback=True,
        format_status="safe_fallback",
        parse_error_class="trajectory_validation_error",
        output_token_count=512,
    )

    summary = target_runner._summarize_target_format_failures(
        [repaired, fallback],
        max_new_tokens=4096,
    )

    assert summary == {
        "status": "FAIL",
        "errors": ["unobserved_response_budget_exceeded"],
        "unusable_observation_count": 1,
        "unusable_observation_episode_ids": ["episode-fallback"],
        "unusable_observation_budget": 0,
        "unusable_observation_fraction_ceiling": 0.01,
        "unusable_observation_policy": "bounded_unusable_observation_two_sided_bounds_v1",
        "unusable_observation_budget_exceeded": True,
        "truncated_response_count": 0,
        "truncated_response_episode_ids": [],
        "response_token_ceiling": 4096,
        "response_token_ceiling_reached_count": 0,
        "response_token_ceiling_episode_ids": [],
        "format_fallback_count": 1,
        "format_fallback_episode_ids": ["episode-fallback"],
        "format_repair_count": 1,
        "format_repair_episode_ids": ["episode-repaired"],
        "parse_failure_count": 2,
        "parse_failure_episode_ids": ["episode-fallback", "episode-repaired"],
        "unresolved_parse_failure_count": 1,
        "unresolved_parse_failure_episode_ids": ["episode-fallback"],
        "parse_error_class_counts": {
            "json_decode_error": 1,
            "trajectory_validation_error": 1,
        },
        "raw_response_included": False,
    }
    assert not {
        "response",
        "responses",
        "raw_response",
        "malformed_response",
    } & summary.keys()


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        (
            "batch_candidates",
            [1, 2, 4, 8, 16],
            "batch_candidates_must_equal_8_16_24_32_48_64",
        ),
        ("capacity_probe_new_tokens", 32, "capacity_probe_new_tokens_must_equal_128"),
        ("capacity_warmup_batches", 0, "capacity_warmup_batches_must_equal_1"),
        (
            "capacity_measurement_repeats",
            1,
            "capacity_measurement_repeats_must_equal_3",
        ),
        (
            "capacity_reference_batch_size",
            4,
            "capacity_reference_batch_size_must_equal_8",
        ),
        (
            "capacity_validation_new_tokens",
            128,
            "capacity_validation_must_use_production_new_tokens",
        ),
        (
            "capacity_validation_repeats",
            2,
            "capacity_validation_repeats_must_equal_1",
        ),
        ("capacity_stop_after_oom", False, "capacity_stop_after_oom_must_be_true"),
        (
            "input_batch_order",
            "dataset_order",
            "input_batch_order_must_be_stable_token_length_descending",
        ),
        (
            "repair_batching",
            "same_round",
            "repair_batching_must_be_not_applicable",
        ),
        ("use_cache", False, "target_generation_cache_must_be_enabled"),
    ],
)
def test_target_protocol_rejects_capacity_scout_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    config[field] = value
    monkeypatch.setattr(target_runner, "load_yaml", lambda _: config)

    result = validate_target_protocol(config_path)

    assert result["status"] == "FAIL"
    assert expected_error in result["errors"]


@pytest.mark.parametrize(
    ("field", "value", "expected_error"),
    [
        ("schema_version", "1.0.0", "schema_version_must_equal_2_4_0"),
        ("model_id", "changed/model", "target_model_id_mismatch"),
        ("model_revision", "A" * 40, "target_model_revision_mismatch"),
        (
            "runtime_profile",
            "configs/profiles/accelerator_24gb.yaml",
            "target_runtime_profile_mismatch",
        ),
        ("precision", "fp32", "target_precision_must_be_bf16"),
        ("device_placement", "cpu", "target_device_placement_must_be_auto"),
        ("target_memory_utilization", 0.90, "target_memory_utilization_must_equal_0_88"),
        ("max_input_tokens", 2048, "max_input_tokens_must_equal_4096"),
        ("max_new_tokens", 768, "max_new_tokens_must_equal_4096"),
        ("max_format_attempts", 2, "max_format_attempts_must_equal_1"),
        ("event_id_max_length", 64, "event_id_max_length_must_equal_96"),
        (
            "format_repair_policy",
            "unbounded",
            "format_repair_policy_must_equal_"
            "unique_single_terminal_delimiter_recovery_v1",
        ),
        (
            "truncated_response_policy",
            "append_terminal_delimiter",
            "truncated_response_policy_must_equal_"
            "reject_truncated_response_without_delimiter_repair_v1",
        ),
        ("response_format", "free_text", "target_response_format_mismatch"),
    ],
)
def test_target_protocol_rejects_scientific_or_engine_contract_drift(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    expected_error: str,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    config[field] = value
    monkeypatch.setattr(target_runner, "load_yaml", lambda _: config)

    result = validate_target_protocol(config_path)

    assert result["status"] == "FAIL"
    assert expected_error in result["errors"]


def test_target_capacity_scout_uses_warmup_sync_repeats_and_median(
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
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    config["batch_candidates"] = [1, 2]
    config["capacity_reference_batch_size"] = 1
    generated_batch_sizes: list[int] = []

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            prompt = messages[0]["content"]
            return [1] * len(prompt) if tokenize else prompt

    def fake_generate(*args, **kwargs) -> GeneratedBatch:
        batch_size = len(args[3])
        generated_batch_sizes.append(batch_size)
        return GeneratedBatch(
            responses=["stable response"] * batch_size,
            input_token_counts=[2] * batch_size,
            output_token_counts=[2] * batch_size,
        )

    times = iter([0.0, 4.0, 4.0, 6.0, 6.0, 7.0, 7.0, 9.0, 9.0, 10.0, 10.0, 10.5])
    monkeypatch.setattr(target_runner, "_generate_batch", fake_generate)
    monkeypatch.setattr(target_runner.time, "perf_counter", lambda: next(times))

    result = target_runner._measure_capacity(
        fake_torch,
        FakeTokenizer(),
        object(),
        ["development prompt"],
        config,
    )

    assert result["status"] == "PASS"
    assert result["selected"]["batch_size"] == 2
    assert result["repeat_samples_per_second"] == {
        "batch-1": [0.25, 0.5, 1.0],
        "batch-2": [1.0, 2.0, 4.0],
    }
    assert generated_batch_sizes == [1] * 4 + [2] * 4
    assert accelerator.synchronize_calls == 14
    assert result["candidate_output_equivalence"]["batch-2"]["status"] == "PASS"


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_target_protocol_rejects_nonpositive_or_noninteger_format_attempts(
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    config["max_format_attempts"] = value
    monkeypatch.setattr(target_runner, "load_yaml", lambda _: config)

    result = validate_target_protocol(config_path)

    assert result["status"] == "FAIL"
    assert "max_format_attempts_invalid" in result["errors"]


@pytest.mark.parametrize("value", [0, -1, 1.5, "2", True])
def test_parse_with_repairs_rejects_invalid_format_attempts_before_parse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    value: object,
) -> None:
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    config["max_format_attempts"] = value
    monkeypatch.setattr(
        target_runner,
        "parse_agent_response",
        lambda *args, **kwargs: pytest.fail("invalid format attempts reached response parsing"),
    )

    with pytest.raises(ValueError, match="max_format_attempts must be a positive integer"):
        _parse_with_repairs(
            object(),
            "canonical prompt",
            "not-json",
            config,
            batch_id="target-batch-000001",
            batch_size=1,
            initial_model_request_wall_seconds=0.0,
            initial_input_token_count=0,
            initial_output_token_count=0,
            diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
        )


def test_target_execution_requires_explicit_confirmatory_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VIPIBENCH_CONFIRMATORY_RUN_APPROVED", raising=False)
    with pytest.raises(PermissionError, match="CONFIRMATORY_RUN_APPROVED"):
        run_target_agent(
            config_path=Path("configs/models/target_agent.yaml"),
            dataset_path=Path("data/splits/frozen/test.jsonl"),
            output_path=tmp_path / "trajectories.jsonl",
            checkpoint_dir=tmp_path / "checkpoints",
        )


def test_target_checkpoint_is_bound_to_runner_prompt_and_config(tmp_path: Path) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    request = build_agent_request(episode)
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=request,
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    binding = _target_checkpoint_binding(config_path)
    assert binding["event_id_max_length"] == "96"
    assert binding["oracle_trajectory_schema_version"] == "1.1.0"
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, binding, record)
    assert _load_target_checkpoint(checkpoint, binding) == record

    changed_binding = {**binding, "config_sha256": "D" * 64}
    with pytest.raises(ValueError, match="source binding mismatch"):
        _load_target_checkpoint(checkpoint, changed_binding)


def test_target_checkpoint_accepts_registered_target_runner_hotfix(tmp_path: Path) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=build_agent_request(episode),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    current_binding = _target_checkpoint_binding(config_path)
    prior_binding = {
        **current_binding,
        "target_runner_sha256": next(iter(target_runner.TARGET_RUNNER_RESUME_COMPATIBLE_SHA256)),
    }
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, prior_binding, record)

    assert _load_target_checkpoint(checkpoint, current_binding) == record


def test_target_checkpoint_rejects_unregistered_target_runner_hotfix(tmp_path: Path) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=build_agent_request(episode),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    current_binding = _target_checkpoint_binding(config_path)
    foreign_binding = {**current_binding, "target_runner_sha256": "F" * 64}
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, foreign_binding, record)

    with pytest.raises(ValueError, match="source binding mismatch"):
        _load_target_checkpoint(checkpoint, current_binding)


def test_target_checkpoint_rejects_a_replacement_resume_disposition(tmp_path: Path) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=build_agent_request(episode),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    binding = _target_checkpoint_binding(config_path)
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, binding, record)
    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["resume_disposition"] = "rerun_final_holdout"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="resume disposition mismatch"):
        _load_target_checkpoint(checkpoint, binding)


def test_recorded_format_fallback_blocks_resume_before_any_model_call(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    dataset_path = Path("data/splits/frozen/test.jsonl")
    config = load_yaml(config_path)
    episode = load_executable_episodes(dataset_path)[0]
    fallback_record = bind_agent_trajectory_record(
        episode=episode,
        request=build_agent_request(episode),
        trajectory=build_safe_fallback_trajectory(episode),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
        synthetic_fixture=False,
        generation_attempts=1,
        format_status="safe_fallback",
        fallback_status="used",
        format_fallback=True,
        malformed_response_sha256="A" * 64,
        parse_error_class="json_decode_error",
        diagnostic_artifact_sha256="B" * 64,
    )
    checkpoint_dir = tmp_path / "checkpoints"
    binding = _target_checkpoint_binding(config_path)
    _record_final_holdout_fallback(
        checkpoint_dir,
        binding,
        sha256_file(dataset_path),
        {},
        fallback_record,
    )

    monkeypatch.setenv("VIPIBENCH_CONFIRMATORY_RUN_APPROVED", "YES")
    monkeypatch.setattr(
        target_runner,
        "validate_target_protocol",
        lambda _: {"status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        target_runner,
        "check_runtime_profile_path",
        lambda *args, **kwargs: {"status": "PASS", "hardware_observed": True, "errors": []},
    )
    monkeypatch.setattr(
        target_runner,
        "build_strict_capacity_receipt",
        lambda *args, **kwargs: {"receipt_type": "strict_runtime_capacity_80gb"},
    )
    monkeypatch.setattr(
        target_runner,
        "strict_capacity_receipt_sha256",
        lambda *args, **kwargs: "C" * 64,
    )
    monkeypatch.setattr(target_runner, "load_executable_episodes", lambda _: [episode])
    monkeypatch.setattr(
        target_runner,
        "_load_model",
        lambda _: pytest.fail("a recorded fallback conflict must fail before model loading"),
    )

    with pytest.raises(ValueError, match="fallback checkpoint is missing"):
        run_target_agent(
            config_path=config_path,
            dataset_path=dataset_path,
            output_path=tmp_path / "trajectories.jsonl",
            checkpoint_dir=checkpoint_dir,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("binding_schema_version", "1.0.0"),
        ("trajectory_schema_version", "1.0.0"),
        ("timing_schema_version", "obsolete-timing-v0"),
        ("config_sha256", "D" * 64),
        ("target_runner_sha256", "E" * 64),
        ("agent_trajectory_sha256", "F" * 64),
        ("oracle_sha256", "A" * 64),
        ("oracle_trajectory_schema_version", "1.0.0"),
        ("model_id", "changed-model-id"),
        ("model_revision", "changed-model-revision"),
        ("tokenizer_revision", "changed-tokenizer-revision"),
        ("response_format", "changed-response-format"),
        ("max_format_attempts", "99"),
        ("format_repair_policy", "changed-repair-policy"),
        ("event_id_max_length", "64"),
    ],
)
def test_target_checkpoint_rejects_changed_schema_or_model_binding(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=build_agent_request(episode),
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    binding = _target_checkpoint_binding(config_path)
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, binding, record)
    changed = {**binding, field: value}
    with pytest.raises(ValueError, match="binding|schema"):
        _load_target_checkpoint(checkpoint, changed)


@pytest.mark.parametrize(
    "field",
    ["episode_sha256", "request_sha256", "prompt_sha256"],
)
def test_target_checkpoint_rejects_preamendment_schema_and_changed_record_binding(
    tmp_path: Path,
    field: str,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    config = load_yaml(config_path)
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    request = build_agent_request(episode)
    record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=request,
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )
    binding = _target_checkpoint_binding(config_path)
    checkpoint = tmp_path / "checkpoint.json"
    _write_target_checkpoint(checkpoint, binding, record)

    expected_record_binding = {
        "episode_sha256": record.episode_sha256,
        "request_sha256": record.request_sha256,
        "prompt_sha256": record.prompt_sha256,
    }
    expected_record_binding[field] = "D" * 64
    with pytest.raises(ValueError, match="prompt binding mismatch"):
        _load_target_checkpoint(
            checkpoint,
            binding,
            expected_record_binding=expected_record_binding,
        )

    payload = json.loads(checkpoint.read_text(encoding="utf-8"))
    payload["schema_version"] = "2.0.0"
    checkpoint.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="checkpoint schema mismatch"):
        _load_target_checkpoint(checkpoint, binding)


def test_terminal_recovery_preserves_initial_timing_and_tokens(
    tmp_path: Path,
) -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    config = load_yaml(Path("configs/models/target_agent.yaml"))

    record = _parse_with_repairs(
        episode,
        "canonical prompt",
        '{"events": []}}',
        config,
        batch_id="target-batch-000001",
        batch_size=4,
        initial_model_request_wall_seconds=2.5,
        initial_input_token_count=10,
        initial_output_token_count=3,
        diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
    )

    assert record.generation_attempts == 1
    assert record.format_status == "repaired_json"
    assert record.observed_model_request_wall_seconds == pytest.approx(2.5)
    assert record.input_token_count == 10
    assert record.output_token_count == 3
    assert record.batch_id == "target-batch-000001"
    assert record.batch_size == 4
    assert record.parse_error_class == "json_decode_error"
    assert record.diagnostic_artifact_sha256

    binding = _target_checkpoint_binding(Path("configs/models/target_agent.yaml"))
    checkpoint = tmp_path / f"{episode.episode_id}.json"
    _write_target_checkpoint(checkpoint, binding, record)
    assert _load_target_checkpoint(checkpoint, binding) == record
    diagnostic = tmp_path / "_nonpublic_diagnostics" / f"{episode.episode_id}.malformed.json"
    diagnostic.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="diagnostic binding mismatch"):
        _load_target_checkpoint(checkpoint, binding)


def test_unique_terminal_recovery_keeps_initial_trajectory_without_model_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    initial_response = '{"events": []}}'

    monkeypatch.setattr(
        target_runner,
        "_generate_batch",
        lambda *args, **kwargs: pytest.fail(
            "a unique strict-valid one-delimiter recovery must not call the model again"
        ),
    )

    record = _parse_with_repairs(
        episode,
        "canonical prompt",
        initial_response,
        config,
        batch_id="target-batch-000001",
        batch_size=1,
        initial_model_request_wall_seconds=2.5,
        initial_input_token_count=10,
        initial_output_token_count=3,
        diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
    )

    assert record.trajectory.events == []
    assert record.generation_attempts == 1
    assert record.format_status == "repaired_json"
    assert record.parse_error_class == "json_decode_error"
    assert record.malformed_response_sha256 == target_runner.text_sha256(initial_response)


def test_response_that_reached_the_token_ceiling_is_never_delimiter_repaired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    ceiling = int(config["max_new_tokens"])
    # Without the ceiling guard this response is repairable by removing one
    # terminal delimiter, which would silently accept a partial trajectory.
    truncated_response = '{"events": []}}'

    monkeypatch.setattr(
        target_runner,
        "_generate_batch",
        lambda *args, **kwargs: pytest.fail("a truncated response must not call the model again"),
    )

    record = _parse_with_repairs(
        episode,
        "canonical prompt",
        truncated_response,
        config,
        batch_id="target-batch-000001",
        batch_size=1,
        initial_model_request_wall_seconds=2.5,
        initial_input_token_count=10,
        initial_output_token_count=ceiling,
        diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
    )

    assert record.format_status == "safe_fallback"
    assert record.format_fallback is True
    assert record.parse_error_class == "response_truncation_error"
    assert record.output_token_count == ceiling

    summary = target_runner._summarize_target_format_failures(
        [record],
        max_new_tokens=ceiling,
        total_episode_count=4800,
    )
    assert summary["status"] == "PASS"
    assert summary["errors"] == []
    assert summary["unusable_observation_count"] == 1
    assert summary["unusable_observation_budget"] == 48
    assert summary["truncated_response_count"] == 1
    assert summary["response_token_ceiling_reached_count"] == 1
    assert summary["response_token_ceiling"] == ceiling


def test_response_below_the_token_ceiling_still_uses_bounded_delimiter_repair(
    tmp_path: Path,
) -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    config = load_yaml(Path("configs/models/target_agent.yaml"))

    record = _parse_with_repairs(
        episode,
        "canonical prompt",
        '{"events": []}}',
        config,
        batch_id="target-batch-000001",
        batch_size=1,
        initial_model_request_wall_seconds=2.5,
        initial_input_token_count=10,
        initial_output_token_count=int(config["max_new_tokens"]) - 1,
        diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
    )

    assert record.format_status == "repaired_json"
    assert record.parse_error_class == "json_decode_error"


def test_multiple_strict_valid_terminal_candidates_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    request = build_agent_request(episode)
    valid_record = parse_agent_response(
        '{"events": []}',
        episode=episode,
        request=request,
        model_id=str(config["model_id"]),
        model_revision=str(config["model_revision"]),
        tokenizer_revision=str(config["tokenizer_revision"]),
    )

    def fake_parse(response: str, **kwargs: object):
        if response == "ambiguous}":
            raise ValueError("initial parse failed")
        if response in {"ambiguous", "ambiguous}}"}:
            return valid_record
        raise ValueError("candidate parse failed")

    monkeypatch.setattr(target_runner, "parse_agent_response", fake_parse)
    record = _parse_with_repairs(
        episode,
        "canonical prompt",
        "ambiguous}",
        config,
        batch_id="target-batch-000001",
        batch_size=1,
        initial_model_request_wall_seconds=2.5,
        initial_input_token_count=10,
        initial_output_token_count=3,
        diagnostic_dir=tmp_path / "_nonpublic_diagnostics",
    )

    assert record.generation_attempts == 1
    assert record.format_status == "safe_fallback"
    assert record.format_fallback is True
    assert record.fallback_status == "used"


def test_batched_initial_timing_is_amortized_across_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    episodes = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[:2]
    assert len(episodes) == 2

    monkeypatch.setenv("VIPIBENCH_CONFIRMATORY_RUN_APPROVED", "YES")
    monkeypatch.setattr(
        target_runner,
        "validate_target_protocol",
        lambda _: {"status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        target_runner,
        "check_runtime_profile_path",
        lambda *args, **kwargs: {"status": "PASS", "hardware_observed": True, "errors": []},
    )
    monkeypatch.setattr(
        target_runner,
        "build_strict_capacity_receipt",
        lambda *args, **kwargs: {"receipt_type": "strict_runtime_capacity_80gb"},
    )
    monkeypatch.setattr(
        target_runner,
        "strict_capacity_receipt_sha256",
        lambda *args, **kwargs: "C" * 64,
    )
    telemetry_call: dict[str, object] = {}

    def fake_record_stage_interval(**kwargs: object) -> dict[str, object]:
        telemetry_call["stage"] = kwargs
        return {"interval_id": kwargs["interval_id"]}

    def fake_write_telemetry_ledger(*args: object, **kwargs: object) -> dict[str, object]:
        telemetry_call["output_path"] = args[0]
        telemetry_call["records"] = args[1]
        telemetry_call.update(kwargs)
        return {"ledger_sha256": "D" * 64}

    monkeypatch.setattr(target_runner, "record_stage_interval", fake_record_stage_interval)
    monkeypatch.setattr(target_runner, "write_telemetry_ledger", fake_write_telemetry_ledger)
    monkeypatch.setattr(
        target_runner,
        "_load_model",
        lambda _: (object(), object(), object(), {"status": "PASS"}),
    )
    monkeypatch.setattr(target_runner, "load_executable_episodes", lambda _: episodes)
    monkeypatch.setattr(
        target_runner,
        "_measure_capacity",
        lambda *args, **kwargs: {"status": "PASS", "selected": {"batch_size": 2}},
    )
    monkeypatch.setattr(
        target_runner,
        "_order_pending_by_input_length",
        lambda tokenizer, pending, config: (
            pending,
            {"strategy": "stable_token_length_descending", "pending_count": len(pending)},
        ),
    )

    def fake_generate(*args, **kwargs) -> GeneratedBatch:
        prompts = args[3]
        assert len(prompts) == 2
        return GeneratedBatch(
            responses=['{"events": []}}', '{"events": []}'],
            input_token_counts=[8, 8],
            output_token_counts=[2, 2],
        )

    monkeypatch.setattr(
        target_runner,
        "_validate_production_capacity",
        lambda *args, **kwargs: (
            args[-1],
            fake_generate(None, None, None, ["one", "two"]),
            5.0,
        ),
    )
    monkeypatch.setattr(
        target_runner,
        "_generate_batch",
        lambda *args, **kwargs: pytest.fail(
            "the first production batch must recover locally without another model request"
        ),
    )
    output_path = tmp_path / "trajectories.jsonl"
    result = run_target_agent(
        config_path=config_path,
        dataset_path=Path("data/splits/frozen/test.jsonl"),
        output_path=output_path,
        checkpoint_dir=tmp_path / "checkpoints",
    )

    assert result["schema_version"] == "2.5.0"
    assert result["status"] == "PASS"
    assert result["errors"] == []
    assert result["format_failure_summary"]["status"] == "PASS"
    assert result["format_repair_count"] == 1
    assert result["unresolved_parse_failure_count"] == 0
    records = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert {record["batch_id"] for record in records} == {"target-batch-000001"}
    assert {record["batch_size"] for record in records} == {2}
    assert [record["format_status"] for record in records] == ["repaired_json", "strict_json"]
    assert [record["observed_model_request_wall_seconds"] for record in records] == [2.5, 2.5]
    assert sum(record["observed_model_request_wall_seconds"] for record in records) == 5.0
    assert telemetry_call["output_path"] == output_path.with_suffix(".telemetry.json")
    assert telemetry_call["local_only"] is False
    assert telemetry_call["strict_capacity_receipt"] == {
        "receipt_type": "strict_runtime_capacity_80gb"
    }
    stage = telemetry_call["stage"]
    assert isinstance(stage, dict)
    assert stage["accelerator_stage"] is True
    assert stage["observed_device_receipt_sha256"] == "C" * 64


def test_target_run_aborts_before_next_batch_and_resume_uses_no_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = Path("configs/models/target_agent.yaml")
    dataset_path = Path("data/splits/frozen/test.jsonl")
    episodes = load_executable_episodes(dataset_path)[:3]
    assert len(episodes) == 3

    monkeypatch.setenv("VIPIBENCH_CONFIRMATORY_RUN_APPROVED", "YES")
    monkeypatch.setattr(
        target_runner,
        "validate_target_protocol",
        lambda _: {"status": "PASS", "errors": []},
    )
    monkeypatch.setattr(
        target_runner,
        "check_runtime_profile_path",
        lambda *args, **kwargs: {
            "status": "PASS",
            "hardware_observed": True,
            "errors": [],
        },
    )
    monkeypatch.setattr(
        target_runner,
        "build_strict_capacity_receipt",
        lambda *args, **kwargs: {"receipt_type": "strict_runtime_capacity_80gb"},
    )
    monkeypatch.setattr(
        target_runner,
        "strict_capacity_receipt_sha256",
        lambda *args, **kwargs: "C" * 64,
    )
    monkeypatch.setattr(target_runner, "load_executable_episodes", lambda _: episodes)
    model_loads: list[str] = []

    def fake_load_model(config: dict[str, object]):
        model_loads.append(str(config["model_id"]))
        return object(), object(), object(), {"status": "PASS"}

    monkeypatch.setattr(target_runner, "_load_model", fake_load_model)
    monkeypatch.setattr(
        target_runner,
        "_measure_capacity",
        lambda *args, **kwargs: {"status": "PASS", "selected": {"batch_size": 2}},
    )
    monkeypatch.setattr(
        target_runner,
        "_order_pending_by_input_length",
        lambda tokenizer, pending, config: (
            pending,
            {"strategy": "stable_token_length_descending", "pending_count": len(pending)},
        ),
    )
    monkeypatch.setattr(
        target_runner,
        "_validate_production_capacity",
        lambda *args, **kwargs: (
            args[-1],
            GeneratedBatch(
                responses=["not-json", '{"events": []}'],
                input_token_counts=[8, 8],
                output_token_counts=[2, 2],
            ),
            4.0,
        ),
    )
    monkeypatch.setattr(
        target_runner,
        "_generate_batch",
        lambda *args, **kwargs: pytest.fail(
            "an unresolved first-batch format failure must stop without another model request"
        ),
    )
    output_path = tmp_path / "trajectories.jsonl"
    checkpoint_dir = tmp_path / "checkpoints"

    result = run_target_agent(
        config_path=config_path,
        dataset_path=dataset_path,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
    )

    assert result["status"] == "FAIL"
    expected_errors = ["unobserved_response_budget_exceeded"]
    assert result["errors"] == expected_errors
    assert result["format_fallback_count"] == 1
    assert result["parse_failure_count"] == 1
    assert result["trajectory_artifact"] is None
    assert not output_path.exists()
    assert not output_path.with_suffix(".telemetry.json").exists()
    assert model_loads == ["Qwen/Qwen3-8B"]
    fail_fast = result["target_format_fail_fast"]
    assert fail_fast == {
        "policy": "abort_when_unobserved_response_budget_is_exceeded",
        "trigger_phase": "batch_boundary",
        "trigger_batch_id": "target-batch-000001",
        "recorded_episode_count": 2,
        "total_episode_count": 3,
        "unprocessed_episode_count": 1,
        "additional_model_batches_after_trigger": 0,
        "normal_trajectory_artifact_written": False,
        "raw_response_included": False,
        "truncated_response_policy": (
            "reject_truncated_response_without_delimiter_repair_v1"
        ),
        "response_token_ceiling": 4096,
        "truncated_response_count": 0,
        "unusable_observation_policy": "bounded_unusable_observation_two_sided_bounds_v1",
        "unusable_observation_count": 1,
        "unusable_observation_budget": 0,
    }
    run_receipt = output_path.with_suffix(".run.json")
    assert json.loads(run_receipt.read_text(encoding="utf-8")) == result
    assert len(list(checkpoint_dir.glob("*.json"))) >= 3

    monkeypatch.setattr(
        target_runner,
        "_load_model",
        lambda _: pytest.fail("resume loaded the model after a retained format failure"),
    )
    resumed = run_target_agent(
        config_path=config_path,
        dataset_path=dataset_path,
        output_path=output_path,
        checkpoint_dir=checkpoint_dir,
    )

    assert resumed["status"] == "FAIL"
    assert resumed["errors"] == expected_errors
    assert resumed["target_format_fail_fast"]["trigger_phase"] == "resume_pre_model_load"
    assert resumed["target_format_fail_fast"]["additional_model_batches_after_trigger"] == 0
    assert resumed["capacity_plan"] is None
    assert resumed["model_device_placement"] is None
    assert model_loads == ["Qwen/Qwen3-8B"]


def test_pending_inputs_are_stably_sorted_by_rendered_token_length() -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            prompt = messages[0]["content"]
            return list(range(len(prompt))) if tokenize else prompt

    pending = [
        (SimpleNamespace(episode_id="episode-long"), "123456"),
        (SimpleNamespace(episode_id="episode-short-b"), "12"),
        (SimpleNamespace(episode_id="episode-medium"), "1234"),
        (SimpleNamespace(episode_id="episode-short-a"), "12"),
    ]
    ordered, contract = target_runner._order_pending_by_input_length(
        FakeTokenizer(),
        pending,
        load_yaml(Path("configs/models/target_agent.yaml")),
    )

    assert [episode.episode_id for episode, _ in ordered] == [
        "episode-long",
        "episode-medium",
        "episode-short-a",
        "episode-short-b",
    ]
    assert contract == {
        "strategy": "stable_token_length_descending",
        "selection_inputs": ["rendered_input_token_count", "episode_id", "prompt_sha256"],
        "final_artifact_order": "restored_to_frozen_dataset_order",
        "pending_count": 4,
        "minimum_input_tokens": 2,
        "median_input_tokens": 3.0,
        "maximum_input_tokens": 6,
        "labels_or_outcomes_used": False,
    }


def test_terminal_recoveries_for_one_generation_round_use_no_model_retry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    episodes = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[:2]
    config = load_yaml(Path("configs/models/target_agent.yaml"))
    monkeypatch.setattr(
        target_runner,
        "_generate_batch",
        lambda *args, **kwargs: pytest.fail(
            "unique terminal-delimiter recoveries must not issue a model retry batch"
        ),
    )
    records = target_runner._parse_batch_with_repairs(
        [(episodes[0], "prompt one"), (episodes[1], "prompt two")],
        GeneratedBatch(
            responses=['{"events": []}}', '{"events": []}}'],
            input_token_counts=[10, 11],
            output_token_counts=[3, 4],
        ),
        config,
        batch_id="target-batch-000001",
        batch_size=2,
        initial_model_request_wall_seconds=[1.0, 1.0],
        diagnostic_dir=tmp_path / "diagnostics",
    )

    assert [record.generation_attempts for record in records] == [1, 1]
    assert [record.format_status for record in records] == ["repaired_json", "repaired_json"]
    assert [record.observed_model_request_wall_seconds for record in records] == [1.0, 1.0]
    assert [record.input_token_count for record in records] == [10, 11]
    assert [record.output_token_count for record in records] == [3, 4]


def test_capacity_scout_stops_larger_candidates_after_oom(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3

    class FakeOOM(RuntimeError):
        pass

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
        memory = FakeMemory()

        def synchronize(self):
            return None

    class FakeTokenizer:
        def apply_chat_template(self, messages, *, tokenize, **kwargs):
            prompt = messages[0]["content"]
            return [1] * len(prompt) if tokenize else prompt

    generated_batch_sizes: list[int] = []

    def fake_generate(*args, **kwargs) -> GeneratedBatch:
        batch_size = len(args[3])
        generated_batch_sizes.append(batch_size)
        if batch_size == 2:
            raise FakeOOM("CUDA out of memory")
        if batch_size > 2:
            pytest.fail("a larger batch ran after a monotonic OOM")
        return GeneratedBatch(["stable"], [1], [1])

    config = load_yaml(Path("configs/models/target_agent.yaml"))
    config["batch_candidates"] = [1, 2, 4]
    config["capacity_reference_batch_size"] = 1
    monkeypatch.setattr(target_runner, "_generate_batch", fake_generate)
    times = iter([0.0, 1.0, 1.0, 2.0, 2.0, 3.0])
    monkeypatch.setattr(target_runner.time, "perf_counter", lambda: next(times))

    result = target_runner._measure_capacity(
        SimpleNamespace(accelerator=FakeAccelerator(), OutOfMemoryError=FakeOOM),
        FakeTokenizer(),
        object(),
        ["development prompt"],
        config,
    )

    assert result["status"] == "PASS"
    assert result["selected"]["batch_size"] == 1
    assert result["candidate_output_equivalence"]["batch-2"]["errors"] == ["capacity_oom"]
    assert result["candidate_output_equivalence"]["batch-4"]["errors"] == [
        "skipped_after_smaller_batch_oom"
    ]
    assert generated_batch_sizes == [1, 1, 1, 1, 2]


def test_production_validation_falls_back_on_reference_hash_mismatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gib = 1024**3

    class FakeMemory:
        def empty_cache(self):
            return None

        def reset_peak_memory_stats(self):
            return None

        def max_memory_reserved(self):
            return 40 * gib

    class FakeAccelerator:
        memory = FakeMemory()

        def synchronize(self):
            return None

    def fake_generate(*args, **kwargs) -> GeneratedBatch:
        batch_size = len(args[3])
        if batch_size == 1:
            return GeneratedBatch(["reference"], [1], [1])
        return GeneratedBatch(["different", "second"], [1, 1], [1, 1])

    config = load_yaml(Path("configs/models/target_agent.yaml"))
    config["capacity_reference_batch_size"] = 1
    capacity = {
        "status": "PASS",
        "errors": [],
        "selected": {"candidate_id": "batch-2", "batch_size": 2},
        "ranked_candidate_ids": ["batch-2", "batch-1"],
        "measurements": [
            {
                "candidate_id": "batch-1",
                "batch_size": 1,
                "samples_per_second": 1.0,
                "peak_reserved_gib": 40.0,
                "total_memory_gib": 80.0,
                "completed": True,
                "utilization": 0.5,
            },
            {
                "candidate_id": "batch-2",
                "batch_size": 2,
                "samples_per_second": 2.0,
                "peak_reserved_gib": 40.0,
                "total_memory_gib": 80.0,
                "completed": True,
                "utilization": 0.5,
            },
        ],
    }
    monkeypatch.setattr(target_runner, "_generate_batch", fake_generate)
    times = iter([0.0, 1.0, 1.0, 2.0])
    monkeypatch.setattr(target_runner.time, "perf_counter", lambda: next(times))

    result, generated, wall_seconds = target_runner._validate_production_capacity(
        SimpleNamespace(accelerator=FakeAccelerator(), OutOfMemoryError=RuntimeError),
        object(),
        object(),
        ["one", "two"],
        config,
        capacity,
    )

    assert result["status"] == "PASS"
    assert result["selected"]["candidate_id"] == "batch-1"
    assert [attempt["status"] for attempt in result["production_validation"]["attempts"]] == [
        "FAIL",
        "PASS",
    ]
    assert generated.responses == ["reference"]
    assert wall_seconds == pytest.approx(1.0)


def test_generate_batch_resilient_splits_on_oom(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeOOM(RuntimeError):
        pass

    class FakeMemory:
        def empty_cache(self) -> None:
            return None

    class FakeAccelerator:
        memory = FakeMemory()

        def synchronize(self) -> None:
            return None

    attempted_batch_sizes: list[int] = []

    def fake_generate(*args, **kwargs) -> GeneratedBatch:
        batch_size = len(args[3])
        attempted_batch_sizes.append(batch_size)
        if batch_size >= 4:
            raise FakeOOM("CUDA out of memory")
        return GeneratedBatch(
            [f"response-{index}" for index in range(batch_size)],
            [10 + index for index in range(batch_size)],
            [3 + index for index in range(batch_size)],
        )

    monkeypatch.setattr(target_runner, "_generate_batch", fake_generate)
    monkeypatch.setattr(target_runner, "_release_generation_memory", lambda _torch: None)
    monkeypatch.setattr(target_runner.gc, "collect", lambda: None)

    generated, elapsed = target_runner._generate_batch_resilient(
        SimpleNamespace(accelerator=FakeAccelerator(), OutOfMemoryError=FakeOOM),
        object(),
        object(),
        ["one", "two", "three", "four"],
        load_yaml(Path("configs/models/target_agent.yaml")),
        max_new_tokens=4096,
    )

    assert elapsed >= 0.0
    assert generated.responses == [
        "response-0",
        "response-1",
        "response-0",
        "response-1",
    ]
    assert attempted_batch_sizes == [4, 2, 2]
