import json
from pathlib import Path

from vipibench.ablation_analysis import analyze_encoder_ablations
from vipibench.provenance_contrast import DIAGNOSTIC_CONDITIONS
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS


def _row(
    pair_index: int,
    label: str,
    score: float,
    *,
    split: str = "test",
    condition: str = "canonical",
    input_hash: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": f"{split}-pair-{pair_index:03d}-{label}",
        "episode_sha256": f"{pair_index * 2 + int(label == 'injection'):064X}",
        "model_input_sha256": input_hash or f"{pair_index + 1:064X}",
        "matched_pair_id": f"{split}-pair-{pair_index:03d}",
        "label": label,
        "score": score,
        "split": split,
        "benchmark_track": "provenance_contrast",
        "diagnostic_condition": condition,
        "source_family": f"family-{pair_index % 16:02d}",
    }


def test_paired_ablation_analysis_recovers_provenance_gain(tmp_path: Path) -> None:
    (tmp_path / "model_selection.json").write_text(
        json.dumps({"status": "PASS", "test_accessed": False}),
        encoding="utf-8",
    )
    for mode in LOCKED_INPUT_MODES:
        for seed in LOCKED_SEEDS:
            run_dir = tmp_path / f"mdeberta-{mode}-s{seed}"
            run_dir.mkdir(parents=True)
            rows = []
            scheduled_pairs = [
                *[(pair_index, "canonical") for pair_index in range(200)],
                *[
                    (200 + condition_index * 40 + local_index, condition)
                    for condition_index, condition in enumerate(DIAGNOSTIC_CONDITIONS)
                    for local_index in range(40)
                ],
            ]
            for pair_index, condition in scheduled_pairs:
                if mode == "text_role":
                    benign_score, injection_score = 0.05, 0.95
                    benign_hash = f"{pair_index * 2 + 1:064X}"
                    injection_hash = f"{pair_index * 2 + 2:064X}"
                else:
                    benign_score = injection_score = 0.5
                    benign_hash = injection_hash = f"{pair_index + 1:064X}"
                rows.extend(
                    [
                        _row(
                            pair_index,
                            "benign",
                            benign_score,
                            condition=condition,
                            input_hash=benign_hash,
                        ),
                        _row(
                            pair_index,
                            "injection",
                            injection_score,
                            condition=condition,
                            input_hash=injection_hash,
                        ),
                    ]
                )
            (run_dir / "test_predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in rows) + "\n",
                encoding="utf-8",
            )
            dev_rows = [
                _row(index, label, 0.1 if label == "benign" else 0.9, split="dev")
                for index in range(32)
                for label in ("benign", "injection")
            ]
            (run_dir / "dev_predictions.jsonl").write_text(
                "\n".join(json.dumps(row) for row in dev_rows) + "\n",
                encoding="utf-8",
            )
    result = analyze_encoder_ablations(
        tmp_path,
        tmp_path / "analysis.json",
        bootstrap_iterations=100,
    )
    assert result["status"] == "PASS", result["errors"]
    effect = result["primary_effects"]["content_provenance_minus_text_only"]
    assert effect["primary_metric"] == "dev_calibrated_signed_probability_margin_difference"
    assert effect["estimate"] > 0.9
    assert effect["pairwise_ordering_difference"]["estimate"] == 0.5
    assert effect["pair_count"] == 200
    assert effect["family_count"] == 16
    assert effect["weighting"] == "equal_source_family"
    assert effect["bootstrap"]["valid_iterations"] == 100
    assert result["control_identity_max_abs_margin"] == {
        "text_only": 0.0,
        "role_only": 0.0,
    }
    assert result["h2_identity_gate"]["status"] == "PASS"
    assert result["h2_identity_gate"]["observed_relevant_pair_count"] == 400
    assert result["h2_identity_gate"]["model_input_byte_identity"] is True
    assert effect["small_family_sensitivity"]["decision_agrees_with_primary"] is True


def test_h2_fails_closed_when_control_model_input_hash_differs(tmp_path: Path) -> None:
    test_paired_ablation_analysis_recovers_provenance_gain(tmp_path)
    path = tmp_path / "mdeberta-text_only-s17" / "test_predictions.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    rows[0]["model_input_sha256"] = "F" * 64
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = analyze_encoder_ablations(
        tmp_path,
        tmp_path / "analysis-invalid.json",
        bootstrap_iterations=20,
    )

    assert result["status"] == "FAIL"
    assert result["h2_identity_gate"]["status"] == "FAIL"
    assert result["h2_identity_gate"]["input_identity_violation_count"] == 1
    assert (
        result["primary_effects"]["content_provenance_minus_text_only"]["h1_decision"]
        == "INVALID_H2_IDENTITY_GATE"
    )
