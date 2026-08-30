from pathlib import Path

import pytest

from vipibench.exec_detector_data import (
    detector_text,
    load_executable_episodes,
    prediction_row,
)


def test_frozen_test_detector_adapter_has_locked_counts_and_narrow_input() -> None:
    episodes = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))
    assert len(episodes) == 480
    assert sum(episode.label.value == "injection" for episode in episodes) == 240
    first = episodes[0]
    text_only = detector_text(first, "text_only")
    role_only = detector_text(first, "role_only")
    text_role = detector_text(first, "text_role")
    assert first.episode_id not in text_role
    assert first.content_sha256 not in text_role
    assert role_only == (
        "[ROLE=RETRIEVED][TRUST=UNTRUSTED] [SEGMENT] "
        "[ROLE=USER][TRUST=TRUSTED]"
    )
    assert first.trusted_user_goal in text_only
    assert "[ROLE=" not in text_only
    assert first.trusted_user_goal in text_role
    assert "[ROLE=USER][TRUST=TRUSTED]" in text_role
    assert "[ROLE=RETRIEVED][TRUST=UNTRUSTED]" in text_role


def test_prediction_row_preserves_grouping_and_hash_evidence() -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    row = prediction_row(
        episode,
        0.25,
        split="test",
        latency_ms=1.5,
        input_mode="text_role",
    )
    assert row["sample_id"] == episode.episode_id
    assert row["episode_sha256"] == episode.content_sha256
    assert row["template_family"] == episode.metadata.family_id
    assert row["input_mode"] == "text_role_provenance"
    assert isinstance(row["model_input_sha256"], str)
    assert len(row["model_input_sha256"]) == 64
    assert row["fixture_only"] is False


def test_detector_adapter_rejects_unknown_mode() -> None:
    episode = load_executable_episodes(Path("data/splits/frozen/test.jsonl"))[0]
    with pytest.raises(ValueError, match="unsupported executable detector input mode"):
        detector_text(episode, "unknown")
