from pathlib import Path

from vipibench.adaptive_runner import SearchStrategy, build_attack_candidate
from vipibench.adaptive_validity import (
    normalize_candidate_text,
    validate_adaptive_candidate_checkpoint,
    validate_adaptive_candidate_set,
)
from vipibench.exec_detector_data import load_executable_episodes


def _base():
    return next(
        episode
        for episode in load_executable_episodes(Path("data/splits/confirmatory_final/test.jsonl"))
        if episode.label.value == "injection"
    )


def _candidates(base):
    return [
        build_attack_candidate(
            base,
            f"Biến thể {strategy.value} nội dung {index}.",
            strategy=strategy,
            candidate_index=index,
            generator_revision="fixture-generator",
        )
        for strategy in SearchStrategy
        for index in range(10)
    ]


def test_validity_manifest_binds_identity_hashes_and_unverified_semantics() -> None:
    base = _base()
    result = validate_adaptive_candidate_set(
        [base], _candidates(base), detector_revision="detector-r1"
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["raw_proposal_count"] == 20
    assert result["accepted_candidate_count"] == 20
    assert result["rejected_candidate_count"] == 0
    assert result["strategy_counts"] == {"feedback_guided": 10, "static_sampling": 10}
    assert result["semantic_preservation"] == "UNVERIFIED"
    assert result["semantic_preservation_verified"] is False
    assert result["research_claim_eligible"] is False
    assert all(record["detector_revision"] == "detector-r1" for record in result["records"])


def test_normalization_catches_case_whitespace_unicode_and_punctuation_duplicates() -> None:
    assert normalize_candidate_text("A,  B!") == normalize_candidate_text("ａ   b")
    base = _base()
    candidates = _candidates(base)
    duplicate = build_attack_candidate(
        base,
        "BIẾN THỂ static_sampling nội dung 0!!!",
        strategy=SearchStrategy.GUIDED,
        candidate_index=0,
        generator_revision="fixture-generator",
    )
    candidates[10] = duplicate

    result = validate_adaptive_candidate_set([base], candidates, detector_revision="detector-r1")

    assert result["status"] == "FAIL"
    assert any(error.startswith("candidate_normalized_duplicate") for error in result["errors"])


def test_validity_rejects_base_copy_and_immutable_objective_drift() -> None:
    base = _base()
    candidates = _candidates(base)
    base_copy = build_attack_candidate(
        base,
        next(chunk.content for chunk in base.context if chunk.chunk_id == "retrieved-context"),
        strategy=SearchStrategy.STATIC,
        candidate_index=0,
        generator_revision="fixture-generator",
    )
    candidates[0] = base_copy
    drifted = candidates[1].model_copy(deep=True)
    drifted.initial_state["security"]["privileged"] = True
    candidates[1] = drifted

    result = validate_adaptive_candidate_set([base], candidates, detector_revision="detector-r1")

    assert result["status"] == "FAIL"
    assert any(error.startswith("candidate_duplicates_base_context") for error in result["errors"])
    assert any(
        error.startswith("candidate_immutable_objective_changed") for error in result["errors"]
    )


def test_checkpoint_validity_accepts_only_a_contiguous_structural_prefix() -> None:
    base = _base()
    candidates = _candidates(base)[:3]

    result = validate_adaptive_candidate_checkpoint(
        base,
        candidates,
        strategy="static_sampling",
        detector_revision="detector-r1",
    )

    assert result["status"] == "PASS", result["errors"]
    assert result["accepted_candidate_count"] == 3
    assert result["rejected_candidate_count"] == 0
    assert result["semantic_preservation"] == "UNVERIFIED"
