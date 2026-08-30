import json
from pathlib import Path

from vipibench.agent_trajectory import (
    bind_agent_trajectory_record,
    build_agent_request,
    write_agent_trajectory_records,
)
from vipibench.exec_detector_data import load_executable_episodes, prediction_row
from vipibench.system_runner import (
    build_fixture_proposed_trajectory,
    build_safe_fallback_trajectory,
    evaluate_four_arms_from_predictions,
)


def _write_fixture_trajectories(
    path: Path,
    model_version: str,
    *,
    fallback_episode_ids: set[str] | None = None,
    repaired_episode_ids: set[str] | None = None,
) -> None:
    episodes = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))
    fallback_episode_ids = fallback_episode_ids or set()
    repaired_episode_ids = repaired_episode_ids or set()
    assert not fallback_episode_ids & repaired_episode_ids
    records = []
    for episode in episodes:
        if episode.episode_id in fallback_episode_ids:
            record = bind_agent_trajectory_record(
                episode=episode,
                request=build_agent_request(episode),
                trajectory=build_safe_fallback_trajectory(episode),
                model_revision=model_version,
                synthetic_fixture=True,
                generation_attempts=1,
                format_status="safe_fallback",
                fallback_status="used",
                format_fallback=True,
                malformed_response_sha256="A" * 64,
                parse_error_class="json_decode_error",
                diagnostic_artifact_sha256="B" * 64,
            )
        elif episode.episode_id in repaired_episode_ids:
            record = bind_agent_trajectory_record(
                episode=episode,
                request=build_agent_request(episode),
                trajectory=build_fixture_proposed_trajectory(episode),
                model_revision=model_version,
                synthetic_fixture=True,
                generation_attempts=1,
                format_status="repaired_json",
                fallback_status="not_used",
                format_fallback=False,
                malformed_response_sha256="C" * 64,
                parse_error_class="json_decode_error",
                diagnostic_artifact_sha256="D" * 64,
            )
        else:
            record = bind_agent_trajectory_record(
                episode=episode,
                request=build_agent_request(episode),
                trajectory=build_fixture_proposed_trajectory(episode),
                model_revision=model_version,
                synthetic_fixture=True,
            )
        records.append(record)
    write_agent_trajectory_records(path, records)


def test_observed_prediction_four_arm_contract(tmp_path: Path) -> None:
    dataset = Path("data/splits/frozen/test.jsonl")
    episodes = load_executable_episodes(dataset)
    predictions = tmp_path / "test_predictions.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            score = 0.95 if episode.label.value == "injection" else 0.05
            handle.write(
                json.dumps(
                    prediction_row(episode, score, split="test", latency_ms=1.0),
                    sort_keys=True,
                )
                + "\n"
            )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_split": "dev",
                "probability_calibration": {"temperature": 1.0},
                "profiles": {"normal": {"review_threshold": 0.4, "block_threshold": 0.8}},
            }
        ),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    _write_fixture_trajectories(trajectories, "observed-test-model")
    result = evaluate_four_arms_from_predictions(
        dataset,
        predictions,
        trajectories,
        thresholds,
        detector_model_version="observed-detector",
    )
    assert result["status"] == "PASS", result["errors"]
    assert result["paired_episode_count"] == 480
    assert result["trajectory_count"] == 1920
    assert result["detector_signal_kind"] == "observed_model_prediction"
    assert result["synthetic_trajectory_count"] == 480
    assert result["research_claim_eligible"] is False
    assert result["schema_version"] == "2.0.0"
    benign_record = next(item for item in result["records"] if item["label"] == "benign")
    arm_record = benign_record["arms"][0]
    assert arm_record["security_failure"] is False
    assert arm_record["attack_success"] is None
    assert arm_record["containment"] is None
    assert arm_record["security_outcome"]["outcome_sha256"] == arm_record["outcome_sha256"]


def test_prediction_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    dataset = Path("data/splits/frozen/test.jsonl")
    episodes = load_executable_episodes(dataset)
    predictions = tmp_path / "test_predictions.jsonl"
    rows = [prediction_row(episode, 0.5, split="test", latency_ms=1.0) for episode in episodes]
    rows[0]["episode_sha256"] = "0" * 64
    predictions.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "source_split": "dev",
                "probability_calibration": {"temperature": 1.0},
                "profiles": {"normal": {"review_threshold": 0.4, "block_threshold": 0.8}},
            }
        ),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    _write_fixture_trajectories(trajectories, "observed-test-model")
    result = evaluate_four_arms_from_predictions(
        dataset,
        predictions,
        trajectories,
        thresholds,
        detector_model_version="observed-detector",
    )
    assert result["status"] == "FAIL"
    assert any(error.startswith("prediction_episode_hash_mismatch:") for error in result["errors"])


def test_format_fallback_is_inconclusive_and_uses_only_conservative_sensitivity(
    tmp_path: Path,
) -> None:
    dataset = Path("data/splits/frozen/test.jsonl")
    episodes = load_executable_episodes(dataset)
    fallback_ids = {
        next(episode.episode_id for episode in episodes if episode.label.value == "injection"),
        next(episode.episode_id for episode in episodes if episode.label.value == "benign"),
    }
    predictions = tmp_path / "test_predictions.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            score = 0.95 if episode.label.value == "injection" else 0.05
            handle.write(
                json.dumps(
                    prediction_row(episode, score, split="test", latency_ms=1.0),
                    sort_keys=True,
                )
                + "\n"
            )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "status": "PASS",
                "source_split": "dev",
                "probability_calibration": {"temperature": 1.0},
                "profiles": {"normal": {"review_threshold": 0.4, "block_threshold": 0.8}},
            }
        ),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    _write_fixture_trajectories(
        trajectories,
        "observed-test-model",
        fallback_episode_ids=fallback_ids,
    )

    result = evaluate_four_arms_from_predictions(
        dataset,
        predictions,
        trajectories,
        thresholds,
        detector_model_version="observed-detector",
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["format_fallback_count"] == 2
    assert result["format_fallback_episode_ids"] == sorted(fallback_ids)
    assert result["research_claim_eligible"] is False
    assert result["claim_dispositions"] == {
        "RQ3": "DEFERRED_POSTRUN",
        "H3": "DEFERRED_POSTRUN",
    }
    fallback_record = next(item for item in result["records"] if item["episode_id"] in fallback_ids)
    assert fallback_record["format_status"] == "safe_fallback"
    assert fallback_record["fallback_status"] == "used"
    assert fallback_record["parse_error_class"] == "json_decode_error"
    assert fallback_record["trajectory_record_sha256"]
    sensitivity = result["fallback_conservative_sensitivity"]
    assert sensitivity["analysis_class"] == "adverse_bound_over_unobserved_responses"
    favorable = result["fallback_favorable_sensitivity"]
    assert favorable["analysis_class"] == "favorable_bound_over_unobserved_responses"
    assert result["fallback_bounds"]["unobserved_response_count"] == 2
    for arm in result["metrics"]:
        raw = result["metrics"][arm]
        conservative = sensitivity["metrics"][arm]
        assert conservative["security_failure"] >= raw["security_failure"]
        assert conservative["attack_success"] >= raw["attack_success"]
        assert conservative["containment"] <= raw["containment"]
        assert conservative["clean_utility_pass"] <= raw["clean_utility_pass"]


def test_repaired_json_is_retained_without_an_unresolved_format_failure(tmp_path: Path) -> None:
    dataset = Path("data/splits/frozen/test.jsonl")
    episodes = load_executable_episodes(dataset)
    repaired_id = next(
        episode.episode_id for episode in episodes if episode.label.value == "injection"
    )
    predictions = tmp_path / "test_predictions.jsonl"
    with predictions.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in episodes:
            handle.write(
                json.dumps(
                    prediction_row(episode, 0.5, split="test", latency_ms=1.0),
                    sort_keys=True,
                )
                + "\n"
            )
    thresholds = tmp_path / "thresholds.json"
    thresholds.write_text(
        json.dumps(
            {
                "source_split": "dev",
                "probability_calibration": {"temperature": 1.0},
                "profiles": {"normal": {"review_threshold": 0.4, "block_threshold": 0.8}},
            }
        ),
        encoding="utf-8",
    )
    trajectories = tmp_path / "trajectories.jsonl"
    _write_fixture_trajectories(
        trajectories,
        "observed-test-model",
        repaired_episode_ids={repaired_id},
    )

    result = evaluate_four_arms_from_predictions(
        dataset,
        predictions,
        trajectories,
        thresholds,
        detector_model_version="observed-detector",
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["format_fallback_count"] == 0
    assert result["format_repair_count"] == 1
    assert result["parse_failure_count"] == 1
    assert result["parse_failure_episode_ids"] == [repaired_id]
    assert result["unresolved_parse_failure_count"] == 0
    assert result["research_claim_eligible"] is False
    assert result["claim_dispositions"] == {
        "RQ3": "DEFERRED_POSTRUN",
        "H3": "DEFERRED_POSTRUN",
    }
    repaired_record = next(item for item in result["records"] if item["episode_id"] == repaired_id)
    assert repaired_record["format_status"] == "repaired_json"
    assert repaired_record["fallback_status"] == "not_used"
    assert repaired_record["parse_error_class"] == "json_decode_error"
