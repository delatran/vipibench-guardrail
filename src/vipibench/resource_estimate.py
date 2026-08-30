from __future__ import annotations

from pathlib import Path

from vipibench.dataio import sha256_file, write_json
from vipibench.modeling import load_yaml


def validate_resource_estimate(
    project_root: Path,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    config = load_yaml(config_path)
    experiment = load_yaml(root / "configs/experiments/exec_system.yaml")
    target = load_yaml(root / "configs/models/target_agent.yaml")
    generator = load_yaml(root / "configs/generation/adaptive_generator.yaml")
    profile = load_yaml(root / "configs/profiles/accelerator_80gb.yaml")
    errors: list[str] = []

    if config.get("schema_version") != "2.0.0":
        errors.append("schema_version_must_equal_2_0_0")
    if config.get("status") != "prelaunch_planning_envelope":
        errors.append("status_must_equal_prelaunch_planning_envelope")
    if config.get("hardware_contract_profile") != "configs/profiles/accelerator_80gb.yaml":
        errors.append("hardware_contract_profile_must_bind_accelerator_80gb")

    hardware = _mapping(config, "hardware_minimums", errors)
    for estimate_key, profile_key in (
        ("observed_device_memory_gib", "minimum_device_memory_gib"),
        ("system_ram_gib", "minimum_system_ram_gib"),
        ("free_disk_gib_per_distinct_storage_root", "minimum_disk_free_gib"),
        ("target_memory_utilization", "target_memory_utilization"),
    ):
        if hardware.get(estimate_key) != profile.get(profile_key):
            errors.append(f"hardware_profile_mismatch:{estimate_key}")

    storage = _mapping(config, "model_storage", errors)
    _validate_model_storage(
        storage,
        "target_agent",
        target,
        expected_shards=5,
        expected_bytes=16381516776,
        errors=errors,
    )
    _validate_model_storage(
        storage,
        "attack_generator",
        generator,
        expected_shards=3,
        expected_bytes=8044982000,
        errors=errors,
    )
    if storage.get("maximum_concurrent_large_weight_sets") != 1:
        errors.append("maximum_concurrent_large_weight_sets_must_equal_1")
    if storage.get("ephemeral_cache_rotation_required") is not True:
        errors.append("ephemeral_cache_rotation_must_be_required")
    if storage.get("cache_cleanup_requires_ownership_marker") is not True:
        errors.append("cache_cleanup_must_require_ownership_marker")

    static = _mapping(experiment, "static_evaluation", errors)
    attack = _mapping(experiment, "attack_search_evaluation", errors)
    static_episodes = _integer(static.get("episodes"), "static_episodes", errors)
    static_arms = len(static.get("arms", [])) if isinstance(static.get("arms"), list) else 0
    attack_episodes = _integer(attack.get("episodes"), "attack_episodes", errors)
    attack_strategies = (
        len(attack.get("strategies", [])) if isinstance(attack.get("strategies"), list) else 0
    )
    candidates_per_strategy = _integer(
        attack.get("queries_per_strategy"), "candidates_per_strategy", errors
    )
    defended_arms = (
        len(attack.get("defended_arms", []))
        if isinstance(attack.get("defended_arms"), list)
        else 0
    )
    generated_candidates = attack_episodes * attack_strategies * candidates_per_strategy
    static_trajectories = static_episodes * static_arms
    attack_trajectories = generated_candidates * defended_arms
    maximum_trajectories = static_trajectories + attack_trajectories
    target_calls = static_episodes + generated_candidates
    generator_tokens = generated_candidates * int(generator["max_new_tokens"])
    target_tokens = target_calls * int(target["max_new_tokens"])
    total_tokens = generator_tokens + target_tokens

    workload = _mapping(config, "workload", errors)
    expected_workload = {
        "encoder_runs": 9,
        "encoder_max_epochs_per_run": 5,
        "static_test_episodes": static_episodes,
        "static_system_arms": static_arms,
        "static_evaluation_trajectories": static_trajectories,
        "attack_base_episodes": attack_episodes,
        "attack_strategies": attack_strategies,
        "candidates_per_strategy": candidates_per_strategy,
        "generated_attack_candidates": generated_candidates,
        "defended_attack_arms": defended_arms,
        "attack_evaluation_trajectories": attack_trajectories,
        "maximum_confirmatory_trajectories": maximum_trajectories,
    }
    for key, value in expected_workload.items():
        if workload.get(key) != value:
            errors.append(f"workload_mismatch:{key}")

    generation = _mapping(config, "generation_upper_bound", errors)
    expected_generation = {
        "attack_generator_calls": generated_candidates,
        "attack_generator_max_new_tokens": int(generator["max_new_tokens"]),
        "attack_generator_tokens": generator_tokens,
        "target_agent_calls": target_calls,
        "target_agent_max_new_tokens": int(target["max_new_tokens"]),
        "target_agent_tokens": target_tokens,
        "total_generated_tokens": total_tokens,
    }
    for key, value in expected_generation.items():
        if generation.get(key) != value:
            errors.append(f"generation_upper_bound_mismatch:{key}")

    timing = _mapping(config, "timing_estimate", errors)
    scenarios = timing.get("aggregate_generation_throughput_scenarios")
    scenario_results: list[dict[str, float]] = []
    if not isinstance(scenarios, list) or not scenarios:
        errors.append("timing_scenarios_missing")
    else:
        for index, item in enumerate(scenarios):
            if not isinstance(item, dict):
                errors.append(f"timing_scenario_invalid:{index}")
                continue
            throughput = float(item.get("tokens_per_second", 0))
            if throughput <= 0:
                errors.append(f"timing_throughput_invalid:{index}")
                continue
            calculated = total_tokens / throughput / 3600
            recorded = float(item.get("generation_only_hours", -1))
            if abs(recorded - calculated) > 1e-6:
                errors.append(f"timing_scenario_mismatch:{index}")
            scenario_results.append(
                {
                    "tokens_per_second": throughput,
                    "generation_only_hours": round(calculated, 6),
                }
            )
    if timing.get("operational_planning_window_hours") != [40, 163]:
        errors.append("operational_planning_window_must_equal_40_163")
    if timing.get("status") != "unverified_until_live_capacity_and_warmup_measurement":
        errors.append("timing_status_must_remain_unverified")

    calculated = {
        "static_evaluation_trajectories": static_trajectories,
        "generated_attack_candidates": generated_candidates,
        "attack_evaluation_trajectories": attack_trajectories,
        "maximum_confirmatory_trajectories": maximum_trajectories,
        "target_agent_calls": target_calls,
        "attack_generator_token_upper_bound": generator_tokens,
        "target_agent_token_upper_bound": target_tokens,
        "total_generated_token_upper_bound": total_tokens,
        "timing_scenarios": scenario_results,
    }
    result: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "calculated": calculated,
        "live_measurement_required": True,
        "claim_boundary": (
            "PASS verifies pinned model-storage bytes, deterministic workload arithmetic, and "
            "the stated timing formula. It does not observe accelerator throughput, duration, "
            "cost, hardware allocation, or model quality."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _mapping(
    value: dict[str, object],
    key: str,
    errors: list[str],
) -> dict[str, object]:
    item = value.get(key)
    if not isinstance(item, dict):
        errors.append(f"mapping_missing:{key}")
        return {}
    return item


def _integer(value: object, name: str, errors: list[str]) -> int:
    if isinstance(value, bool):
        errors.append(f"integer_invalid:{name}")
        return 0
    try:
        parsed = int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        errors.append(f"integer_invalid:{name}")
        return 0
    if parsed < 0:
        errors.append(f"integer_negative:{name}")
        return 0
    return parsed


def _validate_model_storage(
    storage: dict[str, object],
    key: str,
    model_config: dict[str, object],
    *,
    expected_shards: int,
    expected_bytes: int,
    errors: list[str],
) -> None:
    entry = storage.get(key)
    if not isinstance(entry, dict):
        errors.append(f"model_storage_missing:{key}")
        return
    expected = {
        "model_id": model_config.get("model_id"),
        "revision": model_config.get("model_revision"),
        "safetensors_shard_count": expected_shards,
        "safetensors_bytes": expected_bytes,
    }
    for field, value in expected.items():
        if entry.get(field) != value:
            errors.append(f"model_storage_mismatch:{key}:{field}")
    recorded_gib = float(entry.get("safetensors_gib", 0))
    calculated_gib = expected_bytes / (1024**3)
    if abs(recorded_gib - calculated_gib) > 0.001:
        errors.append(f"model_storage_mismatch:{key}:safetensors_gib")
