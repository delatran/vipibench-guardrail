from __future__ import annotations

from pathlib import Path

import pytest

from vipibench.detector_view import detector_view_from_episode
from vipibench.episode import ExecutableEpisode
from vipibench.shortcut_audit import audit_exec_shortcuts

DATASET_PATH = Path("data/processed/vipibench_exec.jsonl")


def test_shortcut_audit_is_at_chance_for_all_declared_features() -> None:
    result = audit_exec_shortcuts(DATASET_PATH)
    assert result["status"] == "PASS"
    reports = {
        **result["role_label"]["features"],
        **result["template_generator_style"]["features"],
    }
    for report in reports.values():
        assert report["status"] == "PASS"
        assert report["categorical_majority_accuracy"] == 0.5
        assert report["mutual_information_nats"] == 0.0


def test_detector_view_excludes_label_and_lineage_fields() -> None:
    first_line = DATASET_PATH.read_text(encoding="utf-8").splitlines()[0]
    episode = ExecutableEpisode.model_validate_json(first_line)
    view = detector_view_from_episode(episode)
    visible = view.model_input("text_role_provenance")
    assert episode.episode_id not in visible
    for forbidden in (
        "attack_intent",
        "hard_negative",
        "matched_pair_id",
        "generator_id",
        "template_id",
        "family_id",
        "label",
    ):
        assert forbidden not in visible
    with pytest.raises(ValueError, match="unsupported detector input mode"):
        view.model_input("unsupported")  # type: ignore[arg-type]
