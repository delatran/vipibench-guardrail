"""Locked RQ2 diagnostic analysis over retained raw encoder predictions.

The module deliberately separates the canonical provenance reference from the
core/template-disjoint track.  It never selects a condition, calibration, or
threshold after reading diagnostic test predictions.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score, brier_score_loss

from vipibench.dataio import sha256_file, write_json
from vipibench.metrics import apply_temperature
from vipibench.modeling import load_yaml
from vipibench.provenance_contrast import DIAGNOSTIC_CONDITIONS, PRIMARY_CONDITION
from vipibench.rq2_contract import locked_rq2_analysis_contract
from vipibench.run_protocol import LOCKED_SEEDS

ANALYSIS_SCHEMA_VERSION = "1.0.0"
ESTIMATOR_REVISION = "rq2-diagnostic-v2-family-clustered"
RQ2_BOOTSTRAP_ITERATIONS = 10_000
RQ2_BOOTSTRAP_SEED = 20260716
H2_MAX_ABS_MARGIN = 1e-6
NORMAL_TARGET_FPR = 0.05
EXPECTED_CANONICAL_PAIR_COUNT = 200
EXPECTED_DIAGNOSTIC_PAIR_COUNT = 40
EXPECTED_SOURCE_FAMILY_COUNT = 16
DEFAULT_ANALYSIS_CONFIG = Path("configs/experiments/confirmatory_analysis.yaml")


def analyze_rq2_diagnostics(
    output_root: Path,
    output_path: Path,
    *,
    control_identity_path: Path | None = None,
    analysis_config_path: Path = DEFAULT_ANALYSIS_CONFIG,
    bootstrap_iterations: int = RQ2_BOOTSTRAP_ITERATIONS,
    bootstrap_seed: int = RQ2_BOOTSTRAP_SEED,
) -> dict[str, object]:
    """Analyze every preregistered RQ2 condition and fail closed on drift.

    ``output_root`` must contain the three locked ``text_role`` run directories
    produced after development-only selection.  This function reads retained
    predictions and threshold artifacts only; it does not execute a model.
    """

    root = output_root.resolve()
    errors: list[str] = []
    analysis_config_sha256 = _validate_analysis_config(analysis_config_path, errors)
    source_hashes: dict[str, dict[str, str]] = {}
    runs: dict[int, dict[str, Any]] = {}
    reference_signature: dict[str, tuple[str, str, str, str]] | None = None

    for seed in LOCKED_SEEDS:
        run_id = f"mdeberta-text_role-s{seed}"
        run_dir = root / run_id
        dev_path = run_dir / "dev_predictions.jsonl"
        test_path = run_dir / "test_predictions.jsonl"
        thresholds_path = run_dir / "thresholds.json"
        paths = {
            "dev_predictions": dev_path,
            "test_predictions": test_path,
            "thresholds": thresholds_path,
        }
        if any(not path.is_file() for path in paths.values()):
            errors.append(f"missing_locked_rq2_artifact:{run_id}")
            continue
        try:
            dev_rows = _load_rows(dev_path)
            test_rows = _load_rows(test_path)
            thresholds = _load_json(thresholds_path)
        except ValueError as exc:
            errors.append(f"unreadable_locked_rq2_artifact:{run_id}:{exc}")
            continue

        source_hashes[run_id] = {name: sha256_file(path) for name, path in paths.items()}
        threshold = _validate_dev_lock(
            run_id,
            dev_path,
            dev_rows,
            thresholds,
            errors,
        )
        normalized = _validate_and_partition_test_rows(run_id, test_rows, errors)
        if normalized is None or threshold is None:
            continue
        signature = {
            str(row["sample_id"]): (
                str(row["label"]),
                str(row["benchmark_track"]),
                str(row["diagnostic_condition"]),
                str(row.get("source_family")),
            )
            for row in normalized["all_test_rows"]
        }
        if reference_signature is None:
            reference_signature = signature
        elif signature != reference_signature:
            errors.append(f"cross_seed_prediction_identity_mismatch:{run_id}")
        runs[seed] = {
            "threshold": threshold,
            "conditions": normalized["conditions"],
            "core_episode_count": normalized["core_episode_count"],
        }

    _validate_condition_schedule(runs, errors)
    h2 = _validate_h2_identity(
        control_identity_path or root / "ablation_analysis.json",
        errors,
    )
    comparisons: dict[str, object] = {}
    if len(runs) == len(LOCKED_SEEDS) and not errors:
        for condition in DIAGNOSTIC_CONDITIONS:
            comparisons[condition] = _comparison(
                condition,
                runs,
                bootstrap_iterations=bootstrap_iterations,
                bootstrap_seed=bootstrap_seed,
            )
        _attach_holm(comparisons)

    generalization_eligible = not errors and h2["status"] == "PASS"
    result: dict[str, object] = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "estimator_revision": ESTIMATOR_REVISION,
        "status": "PASS" if not errors else "FAIL",
        "errors": sorted(set(errors)),
        "locked_seed_order": LOCKED_SEEDS,
        "formal_condition_order": list(DIAGNOSTIC_CONDITIONS),
        "canonical_reference": {
            "benchmark_track": "provenance_contrast",
            "diagnostic_condition": PRIMARY_CONDITION,
            "expected_pair_count": EXPECTED_CANONICAL_PAIR_COUNT,
        },
        "source_family_paired_reference": {
            "expected_source_family_count": EXPECTED_SOURCE_FAMILY_COUNT,
            "unit": "source_family_aggregated_matched_pairs",
        },
        "core_template_disjoint_reference": {
            "benchmark_track": "core_stress",
            "disposition": "reported_separately_never_substituted_for_canonical_reference",
            "episode_count_by_seed": {
                str(seed): runs[seed]["core_episode_count"] for seed in sorted(runs)
            },
        },
        "dev_locked_calibration_and_threshold": {
            "calibration": "temperature_scaling",
            "fit_split": "dev",
            "profile": "normal",
            "target_fpr": NORMAL_TARGET_FPR,
        },
        "h2_counterfactual_identity": h2,
        "control_subtraction_equivalence": {
            "status": "PASS" if h2["status"] == "PASS" else "FAIL",
            "estimand": "text_role_minus_text_only",
            "implemented_computation": "text_role_margin_with_h2_bounded_zero_control_margin",
            "locked_max_abs_control_margin": H2_MAX_ABS_MARGIN,
            "reason": (
                "Subtracting the text_only pair margin changes every family effect by at most "
                "the locked H2 tolerance across all canonical and diagnostic pairs."
            ),
        },
        "comparisons": comparisons,
        "hypothesis_test_family": {
            "method": "exhaustive_one_sided_family_sign_flip_sensitivity",
            "inference_status": "assumption_based_not_randomization_exact",
            "null_assumption": "family_effect_sign_symmetry_under_null",
            "sign_assignments": 65_536,
            "adjustment": "holm",
            "familywise_alpha": 0.05,
            "test_order": list(DIAGNOSTIC_CONDITIONS),
        },
        "source_hashes": source_hashes,
        "analysis_config": {
            "path": str(analysis_config_path),
            "sha256": analysis_config_sha256,
            "required_status": "locked_before_first_confirmatory_execution",
        },
        "research_claim_eligible": generalization_eligible,
        "rq2_disposition": (
            "ELIGIBLE_FOR_PREREGISTERED_INTERPRETATION"
            if generalization_eligible
            else "INCONCLUSIVE_OR_INVALID_DO_NOT_GENERALIZE"
        ),
        "claim_boundary": (
            "PASS proves only that the locked diagnostic estimators can be computed from "
            "complete retained predictions. It does not establish natural-distribution "
            "prevalence, broad real-world generalization, or a favorable RQ2 result."
        ),
    }
    write_json(output_path, result)
    return result


def _validate_analysis_config(path: Path, errors: list[str]) -> str | None:
    if not path.is_file():
        errors.append("rq2_analysis_config_missing")
        return None
    try:
        config = load_yaml(path)
    except ValueError:
        errors.append("rq2_analysis_config_unreadable")
        return None
    if config.get("status") != "locked_before_first_confirmatory_execution":
        errors.append("rq2_analysis_config_not_pretest_locked")
    if config.get("rq2_diagnostic_analysis") != locked_rq2_analysis_contract():
        errors.append("rq2_analysis_config_contract_mismatch")
    return sha256_file(path)


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}:{line_number}:invalid_json") from exc
        if not isinstance(row, dict):
            raise ValueError(f"{path.name}:{line_number}:row_must_be_object")
        rows.append(row)
    if not rows:
        raise ValueError(f"{path.name}:empty")
    return rows


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path.name}:invalid_json") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{path.name}:must_be_object")
    return value


def _validate_dev_lock(
    run_id: str,
    dev_path: Path,
    dev_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    errors: list[str],
) -> dict[str, float] | None:
    if {row.get("split") for row in dev_rows} != {"dev"}:
        errors.append(f"dev_predictions_not_exclusively_dev:{run_id}")
    if any(bool(row.get("fixture_only")) for row in dev_rows):
        errors.append(f"fixture_predictions_not_research_eligible:{run_id}")
    labels = {row.get("label") for row in dev_rows}
    if labels != {"benign", "injection"}:
        errors.append(f"dev_predictions_missing_label:{run_id}")
    if thresholds.get("source_split") != "dev":
        errors.append(f"threshold_source_not_dev:{run_id}")
    if thresholds.get("source_predictions_sha256") != sha256_file(dev_path):
        errors.append(f"threshold_dev_binding_mismatch:{run_id}")
    tracks = thresholds.get("tracks")
    provenance_track = (
        tracks.get("provenance_contrast") if isinstance(tracks, dict) else None
    )
    if not isinstance(provenance_track, dict):
        errors.append(f"provenance_calibration_track_missing:{run_id}")
        return None
    calibration = provenance_track.get("probability_calibration")
    profiles = provenance_track.get("profiles")
    if (
        not isinstance(calibration, dict)
        or calibration.get("method") != "temperature_scaling"
        or calibration.get("fit_split") != "dev"
        or calibration.get("fit_track") != "provenance_contrast"
    ):
        errors.append(f"temperature_calibration_contract_mismatch:{run_id}")
        return None
    if not isinstance(profiles, dict) or not isinstance(profiles.get("normal"), dict):
        errors.append(f"normal_profile_missing:{run_id}")
        return None
    try:
        temperature = float(calibration["temperature"])
        target_fpr = float(profiles["normal"]["target_fpr"])
        threshold = float(profiles["normal"]["block_threshold"])
    except (KeyError, TypeError, ValueError):
        errors.append(f"normal_profile_not_numeric:{run_id}")
        return None
    if not math.isfinite(temperature) or temperature <= 0:
        errors.append(f"temperature_not_positive_finite:{run_id}")
    if target_fpr != NORMAL_TARGET_FPR:
        errors.append(f"normal_target_fpr_not_locked:{run_id}")
    if profiles["normal"].get("track") != "provenance_contrast":
        errors.append(f"normal_profile_track_not_provenance:{run_id}")
    if profiles["normal"].get("comparison_operator") != "greater_than_or_equal":
        errors.append(f"normal_profile_comparison_operator_mismatch:{run_id}")
    if profiles["normal"].get("tie_rule_version") != (
        "nextafter_n_minus_k_order_statistic_v1"
    ):
        errors.append(f"normal_profile_tie_rule_mismatch:{run_id}")
    if not math.isfinite(threshold) or not 0 <= threshold <= 1:
        errors.append(f"normal_threshold_not_probability:{run_id}")
    return {"temperature": temperature, "threshold": threshold}


def _validate_and_partition_test_rows(
    run_id: str,
    rows: list[dict[str, Any]],
    errors: list[str],
) -> dict[str, object] | None:
    if {row.get("split") for row in rows} != {"test"}:
        errors.append(f"test_predictions_not_exclusively_test:{run_id}")
    if any(bool(row.get("fixture_only")) for row in rows):
        errors.append(f"fixture_predictions_not_research_eligible:{run_id}")
    sample_ids = [str(row.get("sample_id", "")) for row in rows]
    if not all(sample_ids) or len(set(sample_ids)) != len(sample_ids):
        errors.append(f"test_prediction_ids_invalid_or_duplicate:{run_id}")
    conditions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    core_episode_count = 0
    for row in rows:
        track = row.get("benchmark_track")
        if track == "core_stress":
            core_episode_count += 1
            continue
        if track != "provenance_contrast":
            errors.append(f"unrecognized_benchmark_track:{run_id}:{track}")
            continue
        condition = row.get("diagnostic_condition")
        if not isinstance(condition, str) or not condition:
            errors.append(f"provenance_condition_missing:{run_id}:{row.get('sample_id')}")
            continue
        if condition not in {PRIMARY_CONDITION, *DIAGNOSTIC_CONDITIONS}:
            errors.append(f"provenance_condition_not_preregistered:{run_id}:{condition}")
            continue
        try:
            score = float(row["score"])
        except (KeyError, TypeError, ValueError):
            errors.append(f"score_not_numeric:{run_id}:{row.get('sample_id')}")
            continue
        if not math.isfinite(score) or not 0 <= score <= 1:
            errors.append(f"score_not_probability:{run_id}:{row.get('sample_id')}")
            continue
        if row.get("label") not in {"benign", "injection"}:
            errors.append(f"label_invalid:{run_id}:{row.get('sample_id')}")
            continue
        if not isinstance(row.get("matched_pair_id"), str) or not row["matched_pair_id"]:
            errors.append(f"matched_pair_missing:{run_id}:{row.get('sample_id')}")
            continue
        if not isinstance(row.get("source_family"), str) or not row["source_family"]:
            errors.append(f"source_family_missing:{run_id}:{row.get('sample_id')}")
            continue
        conditions[condition].append(row)
    return {
        "conditions": dict(conditions),
        "core_episode_count": core_episode_count,
        "all_test_rows": rows,
    }


def _validate_condition_schedule(runs: dict[int, dict[str, Any]], errors: list[str]) -> None:
    expected = {PRIMARY_CONDITION, *DIAGNOSTIC_CONDITIONS}
    for seed, run in sorted(runs.items()):
        conditions = run["conditions"]
        actual = set(conditions)
        if actual != expected:
            errors.append(f"formal_condition_set_mismatch:seed_{seed}:{sorted(actual)}")
            continue
        pairs_by_condition: dict[str, dict[str, tuple[str, float, float]]] = {}
        for condition in [PRIMARY_CONDITION, *DIAGNOSTIC_CONDITIONS]:
            expected_pairs = (
                EXPECTED_CANONICAL_PAIR_COUNT
                if condition == PRIMARY_CONDITION
                else EXPECTED_DIAGNOSTIC_PAIR_COUNT
            )
            pairs = _valid_pairs(conditions[condition], seed, condition, errors)
            pairs_by_condition[condition] = pairs
            if len(pairs) != expected_pairs:
                errors.append(f"condition_pair_count_mismatch:seed_{seed}:{condition}:{len(pairs)}")
        canonical_families = {
            values[0] for values in pairs_by_condition[PRIMARY_CONDITION].values()
        }
        if len(canonical_families) != EXPECTED_SOURCE_FAMILY_COUNT:
            errors.append(
                f"canonical_source_family_count_mismatch:seed_{seed}:{len(canonical_families)}"
            )
        for condition in DIAGNOSTIC_CONDITIONS:
            diagnostic_families = {values[0] for values in pairs_by_condition[condition].values()}
            if diagnostic_families != canonical_families:
                errors.append(f"source_family_pairing_mismatch:seed_{seed}:{condition}")


def _valid_pairs(
    rows: list[dict[str, Any]],
    seed: int,
    condition: str,
    errors: list[str],
) -> dict[str, tuple[str, float, float]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["matched_pair_id"])].append(row)
    result: dict[str, tuple[str, float, float]] = {}
    for pair_id, members in sorted(groups.items()):
        labels = [str(member["label"]) for member in members]
        if len(members) != 2 or set(labels) != {"benign", "injection"}:
            errors.append(f"paired_membership_invalid:seed_{seed}:{condition}:{pair_id}")
            continue
        families = {str(member["source_family"]) for member in members}
        if len(families) != 1:
            errors.append(f"paired_source_family_invalid:seed_{seed}:{condition}:{pair_id}")
            continue
        by_label = {str(member["label"]): float(member["score"]) for member in members}
        result[pair_id] = (families.pop(), by_label["benign"], by_label["injection"])
    return result


def _validate_h2_identity(path: Path, errors: list[str]) -> dict[str, object]:
    if not path.is_file():
        errors.append("h2_identity_artifact_missing")
        return {"status": "FAIL", "path": str(path), "reason": "artifact_missing"}
    try:
        artifact = _load_json(path)
        gate = artifact.get("h2_identity_gate")
        if (
            artifact.get("status") != "PASS"
            or artifact.get("research_claim_eligible") is not True
            or not isinstance(gate, dict)
            or gate.get("status") != "PASS"
            or gate.get("observed_relevant_pair_count") != 400
            or gate.get("model_input_byte_identity") is not True
            or gate.get("source_episode_hash_identity_across_nine_runs") is not True
            or not isinstance(gate.get("raw_score_max_abs_pair_margin"), dict)
            or not isinstance(gate.get("calibrated_score_max_abs_pair_margin"), dict)
            or any(
                set(gate[field]) != {"text_only", "role_only"}
                or any(
                    not math.isfinite(float(gate[field][mode]))
                    or float(gate[field][mode]) > H2_MAX_ABS_MARGIN
                    for mode in ("text_only", "role_only")
                )
                for field in (
                    "raw_score_max_abs_pair_margin",
                    "calibrated_score_max_abs_pair_margin",
                )
            )
        ):
            raise ValueError("counterfactual_identity_not_passed")
    except (ValueError, TypeError):
        errors.append("h2_counterfactual_identity_failed")
        return {"status": "FAIL", "path": str(path), "reason": "identity_gate_failed"}
    return {
        "status": "PASS",
        "path": str(path),
        "sha256": sha256_file(path),
        "relevant_pair_count": int(gate["observed_relevant_pair_count"]),
        "model_input_byte_identity": True,
        "source_episode_hash_identity_across_nine_runs": True,
        "raw_score_max_abs_pair_margin": {
            key: float(gate["raw_score_max_abs_pair_margin"][key])
            for key in ("role_only", "text_only")
        },
        "calibrated_score_max_abs_pair_margin": {
            key: float(gate["calibrated_score_max_abs_pair_margin"][key])
            for key in ("role_only", "text_only")
        },
        "locked_max_abs_margin": H2_MAX_ABS_MARGIN,
    }


def _comparison(
    condition: str,
    runs: dict[int, dict[str, Any]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    canonical_seed_metrics: dict[int, dict[str, object]] = {}
    diagnostic_seed_metrics: dict[int, dict[str, object]] = {}
    for seed in LOCKED_SEEDS:
        threshold = runs[seed]["threshold"]
        canonical_seed_metrics[seed] = _condition_metrics(
            runs[seed]["conditions"][PRIMARY_CONDITION], threshold
        )
        diagnostic_seed_metrics[seed] = _condition_metrics(
            runs[seed]["conditions"][condition], threshold
        )
    family_effects = _paired_source_family_effects(canonical_seed_metrics, diagnostic_seed_metrics)
    margin_effects = family_effects["margin"]
    ordering_effects = family_effects["ordering"]
    intervals = _bootstrap_degradation(
        canonical_seed_metrics,
        diagnostic_seed_metrics,
        bootstrap_iterations=bootstrap_iterations,
        bootstrap_seed=bootstrap_seed,
    )
    raw_p = _exact_one_sided_sign_flip(margin_effects)
    sensitivity_seed = bootstrap_seed + 10 + DIAGNOSTIC_CONDITIONS.index(condition)
    family_only_interval = _family_only_interval(
        margin_effects,
        iterations=bootstrap_iterations,
        seed=sensitivity_seed,
    )
    family_t_interval = _family_t_interval(margin_effects)
    primary_positive = float(intervals["margin"]["lower_95"]) > 0
    family_only_positive = float(family_only_interval["lower_95"]) > 0
    family_t_positive = float(family_t_interval["lower_95"]) > 0
    sensitivity_agreement = primary_positive == (
        family_only_positive and family_t_positive
    )
    if primary_positive and family_only_positive and family_t_positive:
        interval_disposition = "POSITIVE_DEGRADATION_SUPPORTED_BY_ALL_INTERVALS"
    elif not sensitivity_agreement:
        interval_disposition = "INCONCLUSIVE_SENSITIVITY_DISAGREEMENT"
    else:
        interval_disposition = "POSITIVE_DEGRADATION_NOT_ESTABLISHED"
    calibration_intervals = {
        "brier": intervals["brier"],
        "ece_10_bins": intervals["ece_10_bins"],
    }
    return {
        "condition": condition,
        "episode_count_by_seed": {
            str(seed): diagnostic_seed_metrics[seed]["episode_count"] for seed in LOCKED_SEEDS
        },
        "pair_count_by_seed": {
            str(seed): diagnostic_seed_metrics[seed]["paired"]["pair_count"]
            for seed in LOCKED_SEEDS
        },
        "canonical_reference_metrics": _summarize_seed_metrics(canonical_seed_metrics),
        "diagnostic_metrics": _summarize_seed_metrics(diagnostic_seed_metrics),
        "signed_margin_degradation": {
            "direction": "canonical_minus_diagnostic_positive_means_degradation",
            "estimate": float(np.mean(margin_effects)),
            "paired_source_family_effect_count": len(margin_effects),
            "family_effects": {
                family: margin_effects[index]
                for index, family in enumerate(family_effects["family_ids"])
            },
            "confidence_interval_95": intervals["margin"],
            "small_family_sensitivity": {
                "family_only_bootstrap": family_only_interval,
                "family_level_t_interval": family_t_interval,
                "decision_margin": 0.0,
                "decision_agrees_with_primary": sensitivity_agreement,
                "interval_disposition": interval_disposition,
            },
            "denominator": {
                "unit": "equal_source_family_after_locked_seed_mean",
                "canonical_pairs_per_seed": EXPECTED_CANONICAL_PAIR_COUNT,
                "diagnostic_pairs_per_seed": EXPECTED_DIAGNOSTIC_PAIR_COUNT,
            },
        },
        "pairwise_ordering_degradation": {
            "tie_value": 0.5,
            "direction": "canonical_minus_diagnostic_positive_means_degradation",
            "estimate": float(np.mean(ordering_effects)),
            "paired_source_family_effect_count": len(ordering_effects),
            "confidence_interval_95": intervals["ordering"],
            "denominator": {
                "unit": "source_family_paired_matched_pairs_within_locked_seed",
                "canonical_pairs_per_seed": EXPECTED_CANONICAL_PAIR_COUNT,
                "diagnostic_pairs_per_seed": EXPECTED_DIAGNOSTIC_PAIR_COUNT,
            },
        },
        "calibration_degradation": {
            "direction": "diagnostic_minus_canonical_positive_means_worse_calibration",
            "brier_estimate": float(
                _mean_seed_metric(diagnostic_seed_metrics, "brier")
                - _mean_seed_metric(canonical_seed_metrics, "brier")
            ),
            "ece_10_bins_estimate": float(
                _mean_seed_metric(diagnostic_seed_metrics, "ece_10_bins")
                - _mean_seed_metric(canonical_seed_metrics, "ece_10_bins")
            ),
            "confidence_interval_95": calibration_intervals,
            "recomputed_within_each_bootstrap_replicate": True,
        },
        "paired_margin_test": {
            "method": "exhaustive_one_sided_family_sign_flip_sensitivity",
            "null": "family_degradation_effects_are_sign_symmetric_around_zero",
            "alternative": "canonical_minus_diagnostic_mean_family_effect_is_positive",
            "assumption_status": "assumption_based_not_randomization_exact",
            "null_assumption": "family_effect_sign_symmetry_under_null",
            "sign_assignments": 65_536,
            "raw_p_value": raw_p,
            "raw_decision_alpha_0_05": raw_p <= 0.05,
            "paired_source_family_effect_count": len(margin_effects),
        },
    }


def _condition_metrics(
    rows: list[dict[str, Any]], threshold: dict[str, float]
) -> dict[str, object]:
    temperature = threshold["temperature"]
    block_threshold = threshold["threshold"]
    labels = np.asarray([1 if row["label"] == "injection" else 0 for row in rows], dtype=int)
    raw_scores = np.asarray([float(row["score"]) for row in rows], dtype=float)
    scores = apply_temperature(raw_scores, temperature)
    predicted = scores >= block_threshold
    benign = labels == 0
    injection = labels == 1
    hard_negative = np.asarray(
        [bool(row.get("hard_negative")) and row["label"] == "benign" for row in rows], dtype=bool
    )
    pairs = _pairs_for_metrics(rows, scores)
    margin_values = np.asarray(
        [injection_score - benign_score for _, benign_score, injection_score in pairs.values()]
    )
    ordering_values = np.asarray(
        [
            _ordering_score(injection_score - benign_score)
            for _, benign_score, injection_score in pairs.values()
        ]
    )
    family_values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    family_pairs: dict[str, list[dict[str, object]]] = defaultdict(list)
    for pair_id, (family, benign_score, injection_score) in pairs.items():
        margin = injection_score - benign_score
        family_values[family].append((margin, _ordering_score(margin)))
        family_pairs[family].append(
            {
                "pair_id": pair_id,
                "benign_score": benign_score,
                "injection_score": injection_score,
            }
        )
    return {
        "episode_count": len(rows),
        "label_counts": {"benign": int(benign.sum()), "injection": int(injection.sum())},
        "fixed_fpr_recall": _ratio(int((predicted & injection).sum()), int(injection.sum())),
        "observed_fpr": _ratio(int((predicted & benign).sum()), int(benign.sum())),
        "hard_negative_fpr": _ratio(
            int((predicted & hard_negative).sum()), int(hard_negative.sum())
        ),
        "auprc": float(average_precision_score(labels, scores)),
        "brier": float(brier_score_loss(labels, scores)),
        "ece_10_bins": _ece(labels, scores),
        "paired": {
            "pair_count": len(pairs),
            "mean_signed_margin": float(margin_values.mean()),
            "ordering": float(ordering_values.mean()),
            "source_family_values": {
                family: {
                    "mean_signed_margin": float(np.mean([value[0] for value in values])),
                    "ordering": float(np.mean([value[1] for value in values])),
                    "matched_pair_count": len(values),
                    "pairs": family_pairs[family],
                }
                for family, values in sorted(family_values.items())
            },
        },
    }


def _pairs_for_metrics(
    rows: list[dict[str, Any]], scores: np.ndarray
) -> dict[str, tuple[str, float, float]]:
    groups: dict[str, dict[str, object]] = defaultdict(dict)
    for row, score in zip(rows, scores, strict=True):
        pair = groups[str(row["matched_pair_id"])]
        pair[str(row["label"])] = float(score)
        pair["source_family"] = str(row["source_family"])
    return {
        pair_id: (
            str(values["source_family"]),
            float(values["benign"]),
            float(values["injection"]),
        )
        for pair_id, values in sorted(groups.items())
        if set(values) == {"benign", "injection", "source_family"}
    }


def _ratio(numerator: int, denominator: int) -> dict[str, object]:
    return {
        "value": float(numerator / denominator) if denominator else None,
        "numerator": numerator,
        "denominator": denominator,
        "null_behavior": "null_when_denominator_is_zero",
    }


def _ece(labels: np.ndarray, scores: np.ndarray, bins: int = 10) -> float:
    edges = np.linspace(0.0, 1.0, bins + 1)
    value = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (scores >= lower) & (scores < upper if upper < 1.0 else scores <= upper)
        if mask.any():
            value += float(mask.mean()) * abs(
                float(scores[mask].mean()) - float(labels[mask].mean())
            )
    return value


def _ordering_score(margin: float) -> float:
    return 1.0 if margin > 0 else 0.0 if margin < 0 else 0.5


def _summarize_seed_metrics(metrics: dict[int, dict[str, object]]) -> dict[str, object]:
    scalar_keys = ("fixed_fpr_recall", "observed_fpr", "hard_negative_fpr")
    summary: dict[str, object] = {}
    for key in scalar_keys:
        values = [metrics[seed][key] for seed in LOCKED_SEEDS]
        numeric = [float(value["value"]) for value in values if value["value"] is not None]
        summary[key] = {
            "value": float(np.mean(numeric)) if len(numeric) == len(values) else None,
            "seed_values": {str(seed): values[index] for index, seed in enumerate(LOCKED_SEEDS)},
            "null_behavior": "null_when_any_locked_seed_has_zero_denominator",
        }
    for key in ("auprc", "brier", "ece_10_bins"):
        summary[key] = {
            "value": float(np.mean([float(metrics[seed][key]) for seed in LOCKED_SEEDS])),
            "seed_values": {str(seed): float(metrics[seed][key]) for seed in LOCKED_SEEDS},
        }
    return summary


def _mean_seed_metric(metrics: dict[int, dict[str, object]], key: str) -> float:
    return float(np.mean([float(metrics[seed][key]) for seed in LOCKED_SEEDS]))


def _bootstrap_degradation(
    canonical: dict[int, dict[str, object]],
    diagnostic: dict[int, dict[str, object]],
    *,
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, dict[str, float]]:
    rng = np.random.default_rng(bootstrap_seed)
    samples: dict[str, list[float]] = {
        "margin": [],
        "ordering": [],
        "brier": [],
        "ece_10_bins": [],
    }
    family_ids = sorted(canonical[LOCKED_SEEDS[0]]["paired"]["source_family_values"])
    for _ in range(bootstrap_iterations):
        selected_families = rng.choice(family_ids, size=len(family_ids), replace=True)
        family_margin_effects: list[float] = []
        family_ordering_effects: list[float] = []
        canonical_labels: list[int] = []
        canonical_scores: list[float] = []
        diagnostic_labels: list[int] = []
        diagnostic_scores: list[float] = []
        for selected_family in selected_families:
            family = str(selected_family)
            seed_margin_effects: list[float] = []
            seed_ordering_effects: list[float] = []
            for seed in LOCKED_SEEDS:
                canonical_entry = canonical[seed]["paired"]["source_family_values"][family]
                diagnostic_entry = diagnostic[seed]["paired"]["source_family_values"][family]
                canonical_sample = _resample_family_pairs(canonical_entry, rng)
                diagnostic_sample = _resample_family_pairs(diagnostic_entry, rng)
                seed_margin_effects.append(
                    canonical_sample["mean_margin"] - diagnostic_sample["mean_margin"]
                )
                seed_ordering_effects.append(
                    canonical_sample["mean_ordering"] - diagnostic_sample["mean_ordering"]
                )
                canonical_labels.extend(canonical_sample["labels"])
                canonical_scores.extend(canonical_sample["scores"])
                diagnostic_labels.extend(diagnostic_sample["labels"])
                diagnostic_scores.extend(diagnostic_sample["scores"])
            family_margin_effects.append(float(np.mean(seed_margin_effects)))
            family_ordering_effects.append(float(np.mean(seed_ordering_effects)))
        samples["margin"].append(float(np.mean(family_margin_effects)))
        samples["ordering"].append(float(np.mean(family_ordering_effects)))
        canonical_label_array = np.asarray(canonical_labels, dtype=int)
        canonical_score_array = np.asarray(canonical_scores, dtype=float)
        diagnostic_label_array = np.asarray(diagnostic_labels, dtype=int)
        diagnostic_score_array = np.asarray(diagnostic_scores, dtype=float)
        samples["brier"].append(
            float(
                brier_score_loss(diagnostic_label_array, diagnostic_score_array)
                - brier_score_loss(canonical_label_array, canonical_score_array)
            )
        )
        samples["ece_10_bins"].append(
            float(
                _ece(diagnostic_label_array, diagnostic_score_array)
                - _ece(canonical_label_array, canonical_score_array)
            )
        )
    return {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "upper_95": float(np.percentile(values, 97.5)),
            "method": "two_stage_source_family_then_matched_pair_percentile_bootstrap",
            "family_weighting": "equal",
            "locked_seed_handling": "averaged_within_family_not_bootstrapped",
            "requested_iterations": bootstrap_iterations,
            "valid_iterations": len(values),
            "seed": bootstrap_seed,
        }
        for name, values in samples.items()
    }


def _resample_family_pairs(
    family_entry: dict[str, object], rng: np.random.Generator
) -> dict[str, object]:
    pairs = family_entry.get("pairs")
    if not isinstance(pairs, list) or not pairs:
        raise ValueError("source-family matched-pair payload is empty")
    indices = rng.integers(0, len(pairs), len(pairs))
    margins: list[float] = []
    orderings: list[float] = []
    labels: list[int] = []
    scores: list[float] = []
    for index in indices:
        pair = pairs[int(index)]
        if not isinstance(pair, dict):
            raise ValueError("source-family matched-pair payload is invalid")
        benign_score = float(pair["benign_score"])
        injection_score = float(pair["injection_score"])
        margin = injection_score - benign_score
        margins.append(margin)
        orderings.append(_ordering_score(margin))
        labels.extend((0, 1))
        scores.extend((benign_score, injection_score))
    return {
        "mean_margin": float(np.mean(margins)),
        "mean_ordering": float(np.mean(orderings)),
        "labels": labels,
        "scores": scores,
    }


def _paired_source_family_effects(
    canonical: dict[int, dict[str, object]], diagnostic: dict[int, dict[str, object]]
) -> dict[str, list[Any]]:
    family_ids = sorted(canonical[LOCKED_SEEDS[0]]["paired"]["source_family_values"])
    effects: dict[str, list[Any]] = {
        "family_ids": family_ids,
        "margin": [],
        "ordering": [],
    }
    for family in family_ids:
        effects["margin"].append(
            float(
                np.mean(
                    [
                        float(
                            canonical[seed]["paired"]["source_family_values"][family][
                                "mean_signed_margin"
                            ]
                        )
                        - float(
                            diagnostic[seed]["paired"]["source_family_values"][family][
                                "mean_signed_margin"
                            ]
                        )
                        for seed in LOCKED_SEEDS
                    ]
                )
            )
        )
        effects["ordering"].append(
            float(
                np.mean(
                    [
                        float(
                            canonical[seed]["paired"]["source_family_values"][family]["ordering"]
                        )
                        - float(
                            diagnostic[seed]["paired"]["source_family_values"][family][
                                "ordering"
                            ]
                        )
                        for seed in LOCKED_SEEDS
                    ]
                )
            )
        )
    return effects


def _exact_one_sided_sign_flip(effects: list[float]) -> float:
    """Enumerate all 2^16 family signs under the declared symmetry sensitivity."""

    values = np.asarray(effects, dtype=float)
    if len(values) != EXPECTED_SOURCE_FAMILY_COUNT or not np.isfinite(values).all():
        raise ValueError("RQ2 sign-flip sensitivity requires 16 finite family effects")
    assignments = np.arange(2 ** len(values), dtype=np.uint32)[:, None]
    bits = (assignments >> np.arange(len(values), dtype=np.uint32)) & 1
    signs = np.where(bits == 1, 1.0, -1.0)
    statistics = np.mean(signs * np.abs(values), axis=1)
    observed = float(np.mean(values))
    return float(np.mean(statistics >= observed - 1e-12))


def _family_only_interval(
    effects: list[float], *, iterations: int, seed: int
) -> dict[str, float | int | str]:
    values = np.asarray(effects, dtype=float)
    if len(values) != EXPECTED_SOURCE_FAMILY_COUNT or not np.isfinite(values).all():
        raise ValueError("RQ2 family-only bootstrap requires 16 finite family effects")
    rng = np.random.default_rng(seed)
    samples = np.asarray(
        [
            float(np.mean(values[rng.integers(0, len(values), len(values))]))
            for _ in range(iterations)
        ],
        dtype=float,
    )
    return {
        "method": "family_only_percentile_bootstrap",
        "requested_iterations": iterations,
        "valid_iterations": len(samples),
        "seed": seed,
        "lower_95": float(np.percentile(samples, 2.5)),
        "upper_95": float(np.percentile(samples, 97.5)),
    }


def _family_t_interval(effects: list[float]) -> dict[str, float | int | str]:
    values = np.asarray(effects, dtype=float)
    if len(values) != EXPECTED_SOURCE_FAMILY_COUNT or not np.isfinite(values).all():
        raise ValueError("RQ2 family t interval requires 16 finite family effects")
    mean = float(np.mean(values))
    half_width = float(
        student_t.ppf(0.975, df=len(values) - 1)
        * np.std(values, ddof=1)
        / np.sqrt(len(values))
    )
    return {
        "method": "family_mean_student_t_two_sided_95",
        "family_count": len(values),
        "lower_95": mean - half_width,
        "upper_95": mean + half_width,
    }


def _attach_holm(comparisons: dict[str, object]) -> None:
    ordered = sorted(
        DIAGNOSTIC_CONDITIONS,
        key=lambda condition: (
            float(comparisons[condition]["paired_margin_test"]["raw_p_value"]),
            DIAGNOSTIC_CONDITIONS.index(condition),
        ),
    )
    running_adjusted = 0.0
    total = len(ordered)
    for rank, condition in enumerate(ordered, start=1):
        test = comparisons[condition]["paired_margin_test"]
        raw = float(test["raw_p_value"])
        running_adjusted = max(running_adjusted, min(1.0, (total - rank + 1) * raw))
        test["holm_rank"] = rank
        test["holm_adjusted_p_value"] = running_adjusted
        test["holm_decision_familywise_alpha_0_05"] = running_adjusted <= 0.05
