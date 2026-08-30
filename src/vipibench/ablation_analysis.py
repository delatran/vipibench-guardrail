from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from scipy.stats import t as student_t
from sklearn.metrics import average_precision_score

from vipibench.dataio import sha256_file, write_json
from vipibench.metrics import apply_temperature, fit_temperature
from vipibench.run_protocol import LOCKED_INPUT_MODES, LOCKED_SEEDS

EXPECTED_PROVENANCE_PAIR_COUNT = 400
H2_MAX_ABS_MARGIN = 1e-6


def analyze_encoder_ablations(
    output_root: Path,
    output_path: Path,
    *,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20260716,
) -> dict[str, object]:
    selection_path = output_root / "model_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("model selection artifact is required before test analysis")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("test_accessed") is not False:
        raise ValueError("model selection artifact must be development-only")

    runs: dict[tuple[str, int], list[dict[str, Any]]] = {}
    prediction_hashes: dict[str, str] = {}
    dev_prediction_hashes: dict[str, str] = {}
    track_specific_temperatures: dict[str, float] = {}
    errors: list[str] = []
    h2_errors: list[str] = []
    reference_ids: set[str] | None = None
    reference_episode_hashes: dict[str, str] | None = None
    for mode in LOCKED_INPUT_MODES:
        for seed in LOCKED_SEEDS:
            run_id = f"mdeberta-{mode}-s{seed}"
            run_dir = output_root / run_id
            path = run_dir / "test_predictions.jsonl"
            dev_path = run_dir / "dev_predictions.jsonl"
            rows = _load_rows(path)
            dev_rows = _load_rows(dev_path)
            ids = {str(row["sample_id"]) for row in rows}
            if len(ids) != len(rows):
                errors.append(f"duplicate_prediction_ids:{run_id}")
            if reference_ids is None:
                reference_ids = ids
            elif ids != reference_ids:
                errors.append(f"prediction_id_set_mismatch:{run_id}")
            episode_hashes = {
                str(row["sample_id"]): str(row.get("episode_sha256") or "") for row in rows
            }
            if any(not _valid_sha256(value) for value in episode_hashes.values()):
                h2_errors.append(f"source_episode_hash_missing_or_invalid:{run_id}")
            if reference_episode_hashes is None:
                reference_episode_hashes = episode_hashes
            elif episode_hashes != reference_episode_hashes:
                h2_errors.append(f"cross_run_source_episode_hash_mismatch:{run_id}")
            if {str(row.get("split")) for row in rows} != {"test"}:
                errors.append(f"non_test_prediction_rows:{run_id}")
            if {str(row.get("split")) for row in dev_rows} != {"dev"}:
                errors.append(f"non_dev_calibration_rows:{run_id}")
            temperature = _track_specific_temperature(dev_rows, run_id, errors)
            runs[(mode, seed)] = _calibrated_rows(rows, temperature)
            prediction_hashes[run_id] = sha256_file(path)
            dev_prediction_hashes[run_id] = sha256_file(dev_path)
            track_specific_temperatures[run_id] = temperature

    primary_ids = _eligible_pair_ids(
        runs[("text_role", LOCKED_SEEDS[0])],
        benchmark_track="provenance_contrast",
        diagnostic_condition="canonical",
    )
    if len(primary_ids) != 200:
        errors.append(f"primary_pair_count_not_200:{len(primary_ids)}")
    primary_families = _pair_families(
        runs[("text_role", LOCKED_SEEDS[0])], set(primary_ids)
    )
    if len(set(primary_families.values())) != 16:
        errors.append(
            f"primary_source_family_count_not_16:{len(set(primary_families.values()))}"
        )
    relevant_pair_ids = _all_provenance_pair_ids(runs[("text_role", LOCKED_SEEDS[0])])
    if len(relevant_pair_ids) != EXPECTED_PROVENANCE_PAIR_COUNT:
        h2_errors.append(f"h2_relevant_pair_count_not_400:{len(relevant_pair_ids)}")
    control_raw_max_abs_margin: dict[str, float] = {}
    control_calibrated_max_abs_margin: dict[str, float] = {}
    input_identity_violations: list[str] = []
    for mode in ("text_only", "role_only"):
        raw_margins: list[float] = []
        calibrated_margins: list[float] = []
        for seed in LOCKED_SEEDS:
            grouped = _rows_by_pair(runs[(mode, seed)], set(relevant_pair_ids))
            for pair_id in relevant_pair_ids:
                members = grouped[pair_id]
                input_hashes = {
                    str(member.get("model_input_sha256") or "") for member in members
                }
                if len(input_hashes) != 1 or not _valid_sha256(next(iter(input_hashes))):
                    input_identity_violations.append(f"{mode}:s{seed}:{pair_id}")
                raw_margins.append(abs(_pair_margin(members, score_field="raw_score")))
                calibrated_margins.append(abs(_pair_margin(members)))
        control_raw_max_abs_margin[mode] = max(raw_margins, default=float("inf"))
        control_calibrated_max_abs_margin[mode] = max(
            calibrated_margins, default=float("inf")
        )
        if control_raw_max_abs_margin[mode] > H2_MAX_ABS_MARGIN:
            h2_errors.append(
                f"counterfactual_raw_identity_violated:{mode}:{control_raw_max_abs_margin[mode]}"
            )
        if control_calibrated_max_abs_margin[mode] > H2_MAX_ABS_MARGIN:
            h2_errors.append(
                "counterfactual_calibrated_identity_violated:"
                f"{mode}:{control_calibrated_max_abs_margin[mode]}"
            )
    if input_identity_violations:
        h2_errors.append(f"counterfactual_input_byte_identity_violated:{len(input_identity_violations)}")
    errors.extend(h2_errors)
    h2_gate = {
        "status": "PASS" if not h2_errors else "FAIL",
        "locked_max_abs_margin": H2_MAX_ABS_MARGIN,
        "expected_relevant_pair_count": EXPECTED_PROVENANCE_PAIR_COUNT,
        "observed_relevant_pair_count": len(relevant_pair_ids),
        "evaluated_mode_seed_pair_count": (
            2 * len(LOCKED_SEEDS) * len(relevant_pair_ids)
        ),
        "source_episode_hash_identity_across_nine_runs": not any(
            error.startswith("source_episode_hash_")
            or error.startswith("cross_run_source_episode_hash_")
            for error in h2_errors
        ),
        "model_input_byte_identity": not input_identity_violations,
        "input_identity_violation_count": len(input_identity_violations),
        "raw_score_max_abs_pair_margin": control_raw_max_abs_margin,
        "calibrated_score_max_abs_pair_margin": control_calibrated_max_abs_margin,
        "errors": h2_errors,
    }
    primary = {
        comparison: _paired_mode_effect(
            runs,
            left_mode="text_role",
            right_mode=right_mode,
            pair_ids=primary_ids,
            bootstrap_iterations=bootstrap_iterations,
            bootstrap_seed=bootstrap_seed,
        )
        for comparison, right_mode in {
            "content_provenance_minus_text_only": "text_only",
            "content_provenance_minus_role_only": "role_only",
        }.items()
    }
    for effect in primary.values():
        effect["h2_identity_gate_status"] = h2_gate["status"]
        if h2_gate["status"] != "PASS":
            effect["h1_decision"] = "INVALID_H2_IDENTITY_GATE"

    diagnostic_conditions = sorted(
        {
            str(row["diagnostic_condition"])
            for row in runs[("text_role", LOCKED_SEEDS[0])]
            if row.get("benchmark_track") == "provenance_contrast"
            and row.get("diagnostic_condition") not in {None, "canonical"}
        }
    )
    diagnostics = {
        condition: _condition_summary(runs, condition)
        for condition in diagnostic_conditions
    }
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "selection_sha256": sha256_file(selection_path),
        "prediction_hashes": prediction_hashes,
        "dev_prediction_hashes": dev_prediction_hashes,
        "track_specific_temperature_by_run": track_specific_temperatures,
        "primary_estimand": (
            "equal mean across 16 source families after averaging three preregistered seeds "
            "within each canonical matched pair, using provenance-track dev-temperature-scaled "
            "probabilities"
        ),
        "primary_effects": primary,
        "h2_identity_gate": h2_gate,
        "control_identity_max_abs_margin": control_calibrated_max_abs_margin,
        "diagnostic_conditions": diagnostics,
        "bootstrap_iterations": bootstrap_iterations,
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_method": "two_stage_source_family_then_matched_pair_percentile_bootstrap",
        "seed_population_inference": False,
        "research_claim_eligible": not errors,
        "claim_boundary": (
            "PASS estimates preregistered detector ablations after development-only selection. "
            "It does not establish natural-distribution prevalence or causal system impact."
        ),
    }
    write_json(output_path, result)
    return result


def _track_specific_temperature(
    rows: list[dict[str, Any]], run_id: str, errors: list[str]
) -> float:
    calibration_rows = [
        row for row in rows if row.get("benchmark_track") == "provenance_contrast"
    ]
    labels = np.asarray(
        [1 if row.get("label") == "injection" else 0 for row in calibration_rows], dtype=int
    )
    scores = np.asarray([float(row.get("score", float("nan"))) for row in calibration_rows])
    if (
        not calibration_rows
        or set(labels.tolist()) != {0, 1}
        or not np.isfinite(scores).all()
        or np.any((scores < 0) | (scores > 1))
    ):
        errors.append(f"track_specific_dev_calibration_invalid:{run_id}")
        return 1.0
    return fit_temperature(labels, scores)


def _calibrated_rows(rows: list[dict[str, Any]], temperature: float) -> list[dict[str, Any]]:
    scores = apply_temperature(
        np.asarray([float(row["score"]) for row in rows], dtype=float), temperature
    )
    return [
        {**row, "raw_score": float(row["score"]), "score": float(score)}
        for row, score in zip(rows, scores, strict=True)
    ]


def _load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _eligible_pair_ids(
    rows: list[dict[str, Any]],
    *,
    benchmark_track: str,
    diagnostic_condition: str,
) -> list[str]:
    pairs: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if (
            row.get("benchmark_track") == benchmark_track
            and row.get("diagnostic_condition") == diagnostic_condition
            and row.get("matched_pair_id")
        ):
            pairs[str(row["matched_pair_id"])].append(row)
    return sorted(
        pair_id
        for pair_id, members in pairs.items()
        if len(members) == 2 and {member["label"] for member in members} == {"benign", "injection"}
    )


def _paired_mode_effect(
    runs: dict[tuple[str, int], list[dict[str, Any]]],
    *,
    left_mode: str,
    right_mode: str,
    pair_ids: list[str],
    bootstrap_iterations: int,
    bootstrap_seed: int,
) -> dict[str, object]:
    by_run = {
        key: _rows_by_pair(rows, set(pair_ids))
        for key, rows in runs.items()
        if key[0] in {left_mode, right_mode}
    }
    pair_families = _pair_families(runs[(left_mode, LOCKED_SEEDS[0])], set(pair_ids))
    for mode in (left_mode, right_mode):
        for seed in LOCKED_SEEDS:
            if _pair_families(runs[(mode, seed)], set(pair_ids)) != pair_families:
                raise ValueError("paired ablation source-family binding mismatch")
    seed_effects = {
        str(seed): float(
            np.mean(
                list(
                    _family_means(
                        {
                            pair_id: _pair_margin(by_run[(left_mode, seed)][pair_id])
                            - _pair_margin(by_run[(right_mode, seed)][pair_id])
                            for pair_id in pair_ids
                        },
                        pair_families,
                    ).values()
                )
            )
        )
        for seed in LOCKED_SEEDS
    }
    pair_margin_effects = {
        pair_id: float(
            np.mean(
                [
                    _pair_margin(by_run[(left_mode, seed)][pair_id])
                    - _pair_margin(by_run[(right_mode, seed)][pair_id])
                    for seed in LOCKED_SEEDS
                ]
            )
        )
        for pair_id in pair_ids
    }
    family_margin_effects = _family_means(pair_margin_effects, pair_families)
    estimate = float(np.mean(list(family_margin_effects.values())))
    ordering_seed_effects = {
        str(seed): float(
            np.mean(
                list(
                    _family_means(
                        {
                            pair_id: _ordering_score(
                                _pair_margin(by_run[(left_mode, seed)][pair_id])
                            )
                            - _ordering_score(
                                _pair_margin(by_run[(right_mode, seed)][pair_id])
                            )
                            for pair_id in pair_ids
                        },
                        pair_families,
                    ).values()
                )
            )
        )
        for seed in LOCKED_SEEDS
    }
    auprc_seed_effects = {
        str(seed): _auprc_for_pairs(by_run[(left_mode, seed)], pair_ids)
        - _auprc_for_pairs(by_run[(right_mode, seed)], pair_ids)
        for seed in LOCKED_SEEDS
    }
    pair_ordering_effects = {
        pair_id: float(
            np.mean(
                [
                    _ordering_score(_pair_margin(by_run[(left_mode, seed)][pair_id]))
                    - _ordering_score(_pair_margin(by_run[(right_mode, seed)][pair_id]))
                    for seed in LOCKED_SEEDS
                ]
            )
        )
        for pair_id in pair_ids
    }
    margin_samples = _two_stage_samples(
        pair_margin_effects,
        pair_families,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed,
    )
    ordering_samples = _two_stage_samples(
        pair_ordering_effects,
        pair_families,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 1,
    )
    lower_95 = float(np.percentile(margin_samples, 2.5))
    family_only_samples = _family_only_samples(
        family_margin_effects,
        iterations=bootstrap_iterations,
        seed=bootstrap_seed + 2,
    )
    family_only_lower = float(np.percentile(family_only_samples, 2.5))
    family_only_upper = float(np.percentile(family_only_samples, 97.5))
    t_interval = _family_t_interval(list(family_margin_effects.values()))
    primary_support = lower_95 > 0
    sensitivity_support = family_only_lower > 0 and float(t_interval["lower"]) > 0
    if primary_support and sensitivity_support:
        h1_decision = "SUPPORTED"
    elif primary_support != sensitivity_support:
        h1_decision = "INCONCLUSIVE_SENSITIVITY_DISAGREEMENT"
    else:
        h1_decision = "NOT_SUPPORTED_OR_INCONCLUSIVE"
    return {
        "left_mode": left_mode,
        "right_mode": right_mode,
        "primary_metric": "dev_calibrated_signed_probability_margin_difference",
        "estimate": estimate,
        "lower_95": lower_95,
        "upper_95": float(np.percentile(margin_samples, 97.5)),
        "h1_decision": h1_decision,
        "seed_effects": seed_effects,
        "family_effects": family_margin_effects,
        "family_count": len(family_margin_effects),
        "weighting": "equal_source_family",
        "seed_handling": "averaged_within_pair_not_bootstrapped",
        "bootstrap": {
            "method": "two_stage_source_family_then_matched_pair_percentile_bootstrap",
            "requested_iterations": bootstrap_iterations,
            "valid_iterations": len(margin_samples),
            "seed": bootstrap_seed,
        },
        "small_family_sensitivity": {
            "locked_decision_margin": 0.0,
            "family_only_bootstrap": {
                "method": "family_only_percentile_bootstrap",
                "requested_iterations": bootstrap_iterations,
                "valid_iterations": len(family_only_samples),
                "seed": bootstrap_seed + 2,
                "lower_95": family_only_lower,
                "upper_95": family_only_upper,
                "supports_positive_effect": family_only_lower > 0,
            },
            "family_level_t_interval": {
                **t_interval,
                "supports_positive_effect": float(t_interval["lower"]) > 0,
            },
            "decision_agrees_with_primary": primary_support == sensitivity_support,
        },
        "pairwise_ordering_difference": {
            "estimate": float(np.mean(list(ordering_seed_effects.values()))),
            "lower_95": float(np.percentile(ordering_samples, 2.5)),
            "upper_95": float(np.percentile(ordering_samples, 97.5)),
            "seed_effects": ordering_seed_effects,
            "tie_value": 0.5,
        },
        "secondary_auprc_difference": {
            "estimate": float(np.mean(list(auprc_seed_effects.values()))),
            "seed_effects": auprc_seed_effects,
        },
        "pair_count": len(pair_ids),
    }


def _pair_families(
    rows: list[dict[str, Any]], pair_ids: set[str]
) -> dict[str, str]:
    grouped = _rows_by_pair(rows, pair_ids)
    result: dict[str, str] = {}
    for pair_id, members in grouped.items():
        families = {str(member.get("source_family") or "") for member in members}
        if len(families) != 1 or not next(iter(families)):
            raise ValueError("paired ablation source_family is missing or inconsistent")
        result[pair_id] = families.pop()
    return result


def _family_means(
    pair_values: dict[str, float], pair_families: dict[str, str]
) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for pair_id, value in pair_values.items():
        grouped[pair_families[pair_id]].append(value)
    return {family: float(np.mean(values)) for family, values in sorted(grouped.items())}


def _two_stage_samples(
    pair_values: dict[str, float],
    pair_families: dict[str, str],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    by_family: dict[str, list[str]] = defaultdict(list)
    for pair_id, family in pair_families.items():
        by_family[family].append(pair_id)
    family_ids = sorted(by_family)
    random = np.random.default_rng(seed)
    samples: list[float] = []
    for _ in range(iterations):
        selected_families = random.choice(family_ids, size=len(family_ids), replace=True)
        family_values: list[float] = []
        for family in selected_families:
            pair_ids = sorted(by_family[str(family)])
            selected_pairs = random.choice(pair_ids, size=len(pair_ids), replace=True)
            family_values.append(
                float(np.mean([pair_values[str(pair)] for pair in selected_pairs]))
            )
        samples.append(float(np.mean(family_values)))
    return samples


def _family_only_samples(
    family_values: dict[str, float],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    values = np.asarray([family_values[key] for key in sorted(family_values)], dtype=float)
    if len(values) == 0:
        return []
    random = np.random.default_rng(seed)
    return [
        float(np.mean(values[random.integers(0, len(values), len(values))]))
        for _ in range(iterations)
    ]


def _family_t_interval(values: list[float]) -> dict[str, float | int | str | None]:
    data = np.asarray(values, dtype=float)
    if len(data) < 2 or not np.isfinite(data).all():
        return {
            "method": "family_mean_student_t_two_sided_95",
            "family_count": len(data),
            "lower": None,
            "upper": None,
        }
    mean = float(np.mean(data))
    half_width = float(
        student_t.ppf(0.975, df=len(data) - 1)
        * np.std(data, ddof=1)
        / np.sqrt(len(data))
    )
    return {
        "method": "family_mean_student_t_two_sided_95",
        "family_count": len(data),
        "lower": mean - half_width,
        "upper": mean + half_width,
    }


def _rows_by_pair(
    rows: list[dict[str, Any]],
    pair_ids: set[str],
) -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        pair_id = row.get("matched_pair_id")
        if pair_id in pair_ids:
            result[str(pair_id)].append(row)
    if set(result) != pair_ids:
        raise ValueError("paired ablation rows are incomplete")
    return result


def _auprc_for_pairs(
    rows_by_pair: dict[str, list[dict[str, Any]]],
    pair_ids: list[str],
) -> float:
    rows = [row for pair_id in pair_ids for row in rows_by_pair[pair_id]]
    labels = [1 if row["label"] == "injection" else 0 for row in rows]
    scores = [float(row["score"]) for row in rows]
    return float(average_precision_score(labels, scores))


def _pair_margin(
    members: list[dict[str, Any]], *, score_field: str = "score"
) -> float:
    if len(members) != 2 or {str(member["label"]) for member in members} != {
        "benign",
        "injection",
    }:
        raise ValueError("paired ablation membership is invalid")
    by_label = {str(member["label"]): float(member[score_field]) for member in members}
    return by_label["injection"] - by_label["benign"]


def _all_provenance_pair_ids(rows: list[dict[str, Any]]) -> list[str]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row.get("benchmark_track") == "provenance_contrast" and row.get("matched_pair_id"):
            grouped[str(row["matched_pair_id"])].append(row)
    return sorted(
        pair_id
        for pair_id, members in grouped.items()
        if len(members) == 2
        and {str(member.get("label")) for member in members} == {"benign", "injection"}
    )


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _ordering_score(margin: float) -> float:
    if margin > 0:
        return 1.0
    if margin < 0:
        return 0.0
    return 0.5


def _condition_summary(
    runs: dict[tuple[str, int], list[dict[str, Any]]],
    condition: str,
) -> dict[str, object]:
    seed_auprc: dict[str, float] = {}
    counts: set[int] = set()
    for seed in LOCKED_SEEDS:
        rows = [
            row
            for row in runs[("text_role", seed)]
            if row.get("benchmark_track") == "provenance_contrast"
            and row.get("diagnostic_condition") == condition
        ]
        counts.add(len(rows))
        labels = [1 if row["label"] == "injection" else 0 for row in rows]
        scores = [float(row["score"]) for row in rows]
        seed_auprc[str(seed)] = float(average_precision_score(labels, scores))
    if len(counts) != 1:
        raise ValueError(f"diagnostic condition count mismatch: {condition}")
    return {
        "episode_count": counts.pop(),
        "seed_auprc": seed_auprc,
        "mean_auprc": float(np.mean(list(seed_auprc.values()))),
    }
