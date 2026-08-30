import json
from pathlib import Path

import yaml

from vipibench.dataio import sha256_file
from vipibench.provenance_contrast import DIAGNOSTIC_CONDITIONS, PRIMARY_CONDITION
from vipibench.rq2_analysis import analyze_rq2_diagnostics
from vipibench.run_protocol import LOCKED_SEEDS


def _prediction(
    sample_id: str,
    pair_id: str,
    label: str,
    score: float,
    *,
    condition: str | None,
    track: str = "provenance_contrast",
    hard_negative: bool = True,
    split: str = "test",
    source_family: str | None = None,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "matched_pair_id": pair_id,
        "label": label,
        "score": score,
        "split": split,
        "benchmark_track": track,
        "diagnostic_condition": condition,
        "hard_negative": hard_negative,
        "source_family": source_family or f"family-{pair_id}",
        "fixture_only": False,
    }


def _write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def _h2_identity_artifact(*, text_only_margin: float = 0.0) -> dict[str, object]:
    return {
        "status": "PASS",
        "research_claim_eligible": True,
        "h2_identity_gate": {
            "status": "PASS",
            "observed_relevant_pair_count": 400,
            "model_input_byte_identity": True,
            "source_episode_hash_identity_across_nine_runs": True,
            "raw_score_max_abs_pair_margin": {
                "text_only": text_only_margin,
                "role_only": 0.0,
            },
            "calibrated_score_max_abs_pair_margin": {
                "text_only": text_only_margin,
                "role_only": 0.0,
            },
        },
    }


def _build_locked_rq2_root(tmp_path: Path, *, no_hard_negatives: bool = False) -> Path:
    root = tmp_path / "encoder"
    identity = _h2_identity_artifact()
    (root / "ablation_analysis.json").parent.mkdir(parents=True)
    (root / "ablation_analysis.json").write_text(json.dumps(identity), encoding="utf-8")
    for seed_index, seed in enumerate(LOCKED_SEEDS):
        run_dir = root / f"mdeberta-text_role-s{seed}"
        run_dir.mkdir()
        dev_rows = [
            _prediction(
                f"dev-{index}-{label}",
                f"dev-{index}",
                label,
                0.1 if label == "benign" else 0.9,
                condition=PRIMARY_CONDITION,
                split="dev",
            )
            for index in range(20)
            for label in ("benign", "injection")
        ]
        dev_path = run_dir / "dev_predictions.jsonl"
        _write_rows(dev_path, dev_rows)
        thresholds = {
            "status": "PASS",
            "source_split": "dev",
            "source_predictions_sha256": sha256_file(dev_path),
            "probability_calibration": {"method": "temperature_scaling", "temperature": 1.0},
            "profiles": {"normal": {"target_fpr": 0.05, "block_threshold": 0.5}},
            "tracks": {
                "provenance_contrast": {
                    "probability_calibration": {
                        "method": "temperature_scaling",
                        "temperature": 1.0,
                        "fit_split": "dev",
                        "fit_track": "provenance_contrast",
                    },
                    "profiles": {
                        "normal": {
                            "target_fpr": 0.05,
                            "block_threshold": 0.5,
                            "track": "provenance_contrast",
                            "comparison_operator": "greater_than_or_equal",
                            "tie_rule_version": "nextafter_n_minus_k_order_statistic_v1",
                        }
                    },
                }
            },
        }
        (run_dir / "thresholds.json").write_text(json.dumps(thresholds), encoding="utf-8")
        test_rows: list[dict[str, object]] = []
        score_by_condition = {
            PRIMARY_CONDITION: (0.10, 0.90),
            "source_tag_spoof": (0.15, 0.78),
            "long_context": (0.20, 0.70),
            "quoted_boundary": (0.25, 0.65),
            "format_noise": (0.30, 0.55),
            "code_mix": (0.35, 0.45),
        }
        for condition, pair_count in [
            (PRIMARY_CONDITION, 200),
            *[(item, 40) for item in DIAGNOSTIC_CONDITIONS],
        ]:
            benign_score, injection_score = score_by_condition[condition]
            benign_score += seed_index * 0.001
            injection_score -= seed_index * 0.001
            for index in range(pair_count):
                pair_id = f"{condition}-{index:03d}"
                hard_negative = not no_hard_negatives or condition != "source_tag_spoof"
                test_rows.extend(
                    [
                        _prediction(
                            f"{pair_id}-benign",
                            pair_id,
                            "benign",
                            benign_score,
                            condition=condition,
                            hard_negative=hard_negative,
                            source_family=f"family-{index % 16:02d}",
                        ),
                        _prediction(
                            f"{pair_id}-injection",
                            pair_id,
                            "injection",
                            injection_score,
                            condition=condition,
                            hard_negative=False,
                            source_family=f"family-{index % 16:02d}",
                        ),
                    ]
                )
        for index in range(2):
            pair_id = f"core-{index:03d}"
            test_rows.extend(
                [
                    _prediction(
                        f"{pair_id}-benign",
                        pair_id,
                        "benign",
                        0.2,
                        condition=None,
                        track="core_stress",
                        source_family=f"core-family-{index}",
                    ),
                    _prediction(
                        f"{pair_id}-injection",
                        pair_id,
                        "injection",
                        0.8,
                        condition=None,
                        track="core_stress",
                        hard_negative=False,
                        source_family=f"core-family-{index}",
                    ),
                ]
            )
        _write_rows(run_dir / "test_predictions.jsonl", test_rows)
    return root


def test_rq2_analysis_locks_all_conditions_holm_and_pair_denominators(tmp_path: Path) -> None:
    root = _build_locked_rq2_root(tmp_path)
    result = analyze_rq2_diagnostics(root, tmp_path / "rq2.json", bootstrap_iterations=100)

    assert result["status"] == "PASS", result["errors"]
    assert result["research_claim_eligible"] is True
    assert result["h2_counterfactual_identity"]["relevant_pair_count"] == 400
    assert result["control_subtraction_equivalence"]["status"] == "PASS"
    assert list(result["comparisons"]) == list(DIAGNOSTIC_CONDITIONS)
    source_tag = result["comparisons"]["source_tag_spoof"]
    assert source_tag["pair_count_by_seed"] == {"17": 40, "29": 40, "43": 40}
    assert source_tag["diagnostic_metrics"]["fixed_fpr_recall"]["value"] == 1.0
    assert source_tag["signed_margin_degradation"]["estimate"] > 0
    assert source_tag["pairwise_ordering_degradation"]["estimate"] == 0.0
    for comparison in result["comparisons"].values():
        test = comparison["paired_margin_test"]
        assert test["holm_adjusted_p_value"] >= test["raw_p_value"]
        assert test["raw_p_value"] < 0.05
        assert test["holm_decision_familywise_alpha_0_05"] is True
        assert test["paired_source_family_effect_count"] == 16
        assert test["method"] == "exhaustive_one_sided_family_sign_flip_sensitivity"
        assert test["assumption_status"] == "assumption_based_not_randomization_exact"
        assert test["sign_assignments"] == 65536
        assert (
            comparison["signed_margin_degradation"]["confidence_interval_95"]["valid_iterations"]
            == 100
        )
        assert comparison["signed_margin_degradation"]["confidence_interval_95"]["method"] == (
            "two_stage_source_family_then_matched_pair_percentile_bootstrap"
        )
        sensitivity = comparison["signed_margin_degradation"]["small_family_sensitivity"]
        assert sensitivity["family_only_bootstrap"]["valid_iterations"] == 100
        assert sensitivity["family_level_t_interval"]["family_count"] == 16
        assert comparison["calibration_degradation"][
            "recomputed_within_each_bootstrap_replicate"
        ] is True
        assert comparison["calibration_degradation"]["confidence_interval_95"]["brier"][
            "valid_iterations"
        ] == 100


def test_rq2_analysis_nulls_hard_negative_fpr_instead_of_coercing_zero(tmp_path: Path) -> None:
    root = _build_locked_rq2_root(tmp_path, no_hard_negatives=True)
    result = analyze_rq2_diagnostics(root, tmp_path / "rq2.json", bootstrap_iterations=10)

    assert result["status"] == "PASS", result["errors"]
    metric = result["comparisons"]["source_tag_spoof"]["diagnostic_metrics"]["hard_negative_fpr"]
    assert metric["value"] is None
    assert metric["null_behavior"] == "null_when_any_locked_seed_has_zero_denominator"


def test_rq2_analysis_blocks_generalization_for_missing_condition_and_h2_failure(
    tmp_path: Path,
) -> None:
    root = _build_locked_rq2_root(tmp_path)
    path = root / "mdeberta-text_role-s17" / "test_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if '"code_mix"' not in line
    ]
    _write_rows(path, rows)
    (root / "ablation_analysis.json").write_text(
        json.dumps(_h2_identity_artifact(text_only_margin=0.01)),
        encoding="utf-8",
    )

    result = analyze_rq2_diagnostics(root, tmp_path / "rq2.json", bootstrap_iterations=10)

    assert result["status"] == "FAIL"
    assert result["research_claim_eligible"] is False
    assert any(error.startswith("formal_condition_set_mismatch") for error in result["errors"])
    assert "h2_counterfactual_identity_failed" in result["errors"]


def test_rq2_analysis_rejects_threshold_not_hash_bound_to_dev(tmp_path: Path) -> None:
    root = _build_locked_rq2_root(tmp_path)
    path = root / "mdeberta-text_role-s43" / "thresholds.json"
    threshold = json.loads(path.read_text(encoding="utf-8"))
    threshold["source_predictions_sha256"] = "0" * 64
    path.write_text(json.dumps(threshold), encoding="utf-8")

    result = analyze_rq2_diagnostics(root, tmp_path / "rq2.json", bootstrap_iterations=10)

    assert result["status"] == "FAIL"
    assert "threshold_dev_binding_mismatch:mdeberta-text_role-s43" in result["errors"]


def test_rq2_analysis_rejects_one_class_and_nonfinite_h2_identity(tmp_path: Path) -> None:
    root = _build_locked_rq2_root(tmp_path)
    path = root / "mdeberta-text_role-s29" / "test_predictions.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if not ('"source_tag_spoof"' in line and '"injection"' in line)
    ]
    _write_rows(path, rows)
    (root / "ablation_analysis.json").write_text(
        json.dumps(_h2_identity_artifact(text_only_margin=float("nan"))),
        encoding="utf-8",
    )

    result = analyze_rq2_diagnostics(root, tmp_path / "rq2.json", bootstrap_iterations=10)

    assert result["status"] == "FAIL"
    assert "h2_counterfactual_identity_failed" in result["errors"]
    assert any(
        error.startswith("paired_membership_invalid:seed_29:source_tag_spoof")
        for error in result["errors"]
    )


def test_rq2_analysis_hash_binds_the_pre_execution_contract(tmp_path: Path) -> None:
    root = _build_locked_rq2_root(tmp_path)
    config = yaml.safe_load(
        Path("configs/experiments/confirmatory_analysis.yaml").read_text(encoding="utf-8")
    )
    config["rq2_diagnostic_analysis"]["formal_condition_order"] = ["code_mix"]
    config_path = tmp_path / "altered_confirmatory_analysis.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    result = analyze_rq2_diagnostics(
        root,
        tmp_path / "rq2.json",
        analysis_config_path=config_path,
        bootstrap_iterations=10,
    )

    assert result["status"] == "FAIL"
    assert "rq2_analysis_config_contract_mismatch" in result["errors"]
