import json
from pathlib import Path

import numpy as np
import pytest

from vipibench.metrics import (
    _grouped_bootstrap,
    _matched_pair_bootstrap_samples,
    _matched_pair_bootstrap_samples_reference,
    _mcc_metric,
    apply_temperature,
    calibrate_thresholds,
    threshold_at_fpr,
)


def test_threshold_at_fpr_respects_negative_budget() -> None:
    labels = np.array([0, 0, 0, 1, 1])
    scores = np.array([0.01, 0.10, 0.20, 0.70, 0.90])
    threshold = threshold_at_fpr(labels, scores, 0.05)
    assert threshold > 0.20


def test_temperature_scaling_preserves_order() -> None:
    scores = np.array([0.1, 0.5, 0.9])
    scaled = apply_temperature(scores, 2.0)
    assert np.all(np.diff(scaled) > 0)


def test_threshold_at_fpr_is_conservative_when_benign_scores_tie() -> None:
    labels = np.array([0, 0, 1])
    scores = np.array([0.20, 0.20, 0.90])
    threshold = threshold_at_fpr(labels, scores, 0.50)
    assert threshold > 0.20
    assert int(((scores >= threshold) & (labels == 0)).sum()) == 0


def test_mcc_zero_denominator_is_null_with_explicit_flag() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    predicted = np.asarray([0, 0, 0, 0], dtype=int)

    result = _mcc_metric(labels, predicted)

    assert result["value"] is None
    assert result["denominator_zero"] is True
    assert result["denominator"] == 0.0
    assert result["confusion_counts"] == {
        "true_positive": 0,
        "true_negative": 2,
        "false_positive": 0,
        "false_negative": 2,
    }


def test_mcc_non_degenerate_value_is_preserved() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=int)
    predicted = np.asarray([0, 1, 1, 1], dtype=int)

    result = _mcc_metric(labels, predicted)

    assert result["denominator_zero"] is False
    assert result["value"] == pytest.approx(1 / np.sqrt(3))


def test_calibration_artifact_is_track_specific_and_records_tie_safe_budget(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "sample_id": f"{track}-{label}-{index}",
            "label": label,
            "score": 0.1 + 0.05 * index if label == "benign" else 0.8 + 0.05 * index,
            "split": "dev",
            "benchmark_track": track,
        }
        for track in ("core_stress", "provenance_contrast")
        for index in range(2)
        for label in ("benign", "injection")
    ]
    predictions = tmp_path / "dev.jsonl"
    output = tmp_path / "thresholds.json"
    predictions.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = calibrate_thresholds(predictions, output)

    assert result["primary_track"] == "core_stress"
    assert set(result["tracks"]) == {"core_stress", "provenance_contrast"}
    normal = result["tracks"]["provenance_contrast"]["profiles"]["normal"]
    assert normal["track"] == "provenance_contrast"
    assert normal["n_benign"] == 2
    assert normal["allowed_false_positives"] == 0
    assert normal["observed_false_positives"] == 0
    assert normal["comparison_operator"] == "greater_than_or_equal"
    assert normal["tie_rule_version"] == "nextafter_n_minus_k_order_statistic_v1"


def test_grouped_bootstrap_reports_equal_family_small_cluster_sensitivities() -> None:
    rows = [
        {
            "sample_id": f"family-{family:02d}-{label}",
            "matched_pair_id": f"pair-{family:02d}",
            "source_family": f"family-{family:02d}",
            "label": label,
            "score": 0.1 if label == "benign" else 0.9,
        }
        for family in range(16)
        for label in ("benign", "injection")
    ]

    result = _grouped_bootstrap(rows, 0.5, iterations=20, seed=7)

    assert result["family_count"] == 16
    assert result["family_weighting"] == "equal"
    assert result["equal_family_estimates"]["auprc"] == 1.0
    sensitivity = result["small_family_sensitivity"]
    assert sensitivity["family_only_bootstrap"]["intervals"]["auprc"][
        "valid_iterations"
    ] == 20
    assert sensitivity["family_level_t_intervals"]["intervals"]["auprc"][
        "family_count"
    ] == 16


def test_grouped_bootstrap_is_worker_count_invariant() -> None:
    rows = [
        {
            "sample_id": f"family-{family}-pair-{pair}-{label}",
            "matched_pair_id": f"family-{family}-pair-{pair}",
            "source_family": f"family-{family}",
            "label": label,
            "score": 0.1 + pair * 0.01 if label == "benign" else 0.9 - pair * 0.01,
        }
        for family in range(3)
        for pair in range(4)
        for label in ("benign", "injection")
    ]

    serial = _grouped_bootstrap(rows, 0.5, iterations=23, seed=19, workers=1)
    parallel = _grouped_bootstrap(rows, 0.5, iterations=23, seed=19, workers=2)

    assert serial["worker_count"] == 1
    assert parallel["worker_count"] == 2
    assert serial["intervals"] == parallel["intervals"]
    assert serial["equal_family_estimates"] == parallel["equal_family_estimates"]
    assert serial["small_family_sensitivity"] == parallel["small_family_sensitivity"]


def test_grouped_bootstrap_rejects_nonpositive_worker_count() -> None:
    rows = [
        {
            "sample_id": label,
            "source_family": "family",
            "label": label,
            "score": 0.1 if label == "benign" else 0.9,
        }
        for label in ("benign", "injection")
    ]

    with pytest.raises(ValueError, match="workers must be positive"):
        _grouped_bootstrap(rows, 0.5, iterations=2, seed=1, workers=0)


def test_chunked_matched_pair_bootstrap_matches_reference_rng_stream() -> None:
    ordered = np.asarray([0.0, 1.0, 1.0, 0.0, 1.0])
    decisions = np.asarray([1.0, 1.0, 0.0, 0.0, 1.0])
    margins = np.asarray([-0.2, 0.8, 0.1, -0.4, 0.5])

    reference = _matched_pair_bootstrap_samples_reference(
        ordered,
        decisions,
        margins,
        iterations=37,
        seed=11,
    )
    optimized = _matched_pair_bootstrap_samples(
        ordered,
        decisions,
        margins,
        iterations=37,
        seed=11,
        chunk_size=7,
    )

    assert optimized.keys() == reference.keys()
    for name in reference:
        np.testing.assert_array_equal(optimized[name], reference[name])


def test_chunked_matched_pair_bootstrap_rejects_invalid_shape_or_budget() -> None:
    values = np.asarray([0.0, 1.0])

    with pytest.raises(ValueError, match="iterations must be positive"):
        _matched_pair_bootstrap_samples(values, values, values, iterations=0, seed=1)
    with pytest.raises(ValueError, match="common positive length"):
        _matched_pair_bootstrap_samples(
            values,
            values[:1],
            values,
            iterations=2,
            seed=1,
        )
