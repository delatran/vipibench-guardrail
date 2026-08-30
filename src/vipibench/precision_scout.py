from __future__ import annotations

import json
import math
import os
import statistics
from pathlib import Path

import yaml

from vipibench.dataio import sha256_file, write_json
from vipibench.modeling import load_yaml
from vipibench.transformer_runner import _run_encoder_development_matrix

SCOUT_SEEDS = [17, 29, 43]
SCOUT_INPUT_MODES = ["text_role"]
_COMPARISON_KEYS = (
    "runtime_profile",
    "backbone",
    "model_revision",
    "tokenizer_revision",
    "max_length",
    "learning_rate",
    "epochs",
    "effective_train_batch_size",
    "batch_candidates",
    "gradient_checkpointing_options",
    "gradient_checkpointing_use_reentrant",
    "target_memory_utilization",
    "optimizer",
    "max_grad_norm",
    "numerics_policy",
    "numerics_canary_optimizer_steps",
    "capacity_probe_input_mode",
    "capacity_warmup_optimizer_steps",
    "capacity_measurement_optimizer_steps",
    "dataloader_worker_candidates",
    "dataloader_worker_warmup_batches",
    "dataloader_worker_measurement_batches",
    "dataloader_worker_repeats",
    "early_stopping_metric",
    "early_stopping_patience",
    "early_stopping_threshold",
    "threshold_source",
    "primary_fpr",
    "secondary_fpr",
    "system_input_mode",
    "contrast_dataset",
    "final_holdout_feedback_allowed",
)
_QUALITY_GATE = {
    "primary_metric": "dev_auprc",
    "aggregation": "mean_across_fixed_seeds",
    "mean_noninferiority_margin": 0.005,
    "per_seed_max_degradation": 0.01,
    "minimum_capacity_throughput_ratio": 1.10,
    "maximum_median_training_wall_time_ratio": 0.90,
    "same_live_device_observation_required": True,
    "all_finite_canaries_required": True,
}
_PROMOTION_POLICY = {
    "automatic_promotion_allowed": False,
    "final_holdout_may_influence_promotion": False,
    "required_next_step": "new_versioned_confirmatory_protocol_after_development_review",
}


def _bound_control_path(
    root: Path,
    scout: dict[str, object],
    errors: list[str],
) -> Path | None:
    binding = scout.get("control_config")
    if not isinstance(binding, dict):
        errors.append("control_config_binding_missing")
        return None
    relative = binding.get("path")
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        errors.append("control_config_path_invalid")
        return None
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        errors.append("control_config_path_outside_project")
        return None
    if not candidate.is_file():
        errors.append("control_config_missing")
        return None
    if binding.get("sha256") != sha256_file(candidate):
        errors.append("control_config_hash_mismatch")
        return None
    return candidate


def validate_precision_scout_protocol(
    project_root: Path,
    config_path: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    root = project_root.resolve()
    scout = load_yaml(config_path)
    errors: list[str] = []
    expected_scalars: dict[str, object] = {
        "schema_version": "1.0.0",
        "status": "exploratory_development_only",
        "execution_scope": "train_and_development_only",
        "decision_owner": "precision_scout_protocol",
        "mixed_precision": "bf16",
        "parameter_storage_dtype": "fp32",
        "autocast_dtype": "bfloat16",
        "optimizer_state_dtype": "fp32",
        "allowed_partitions": ["train", "dev"],
        "test_evaluation_allowed": False,
        "final_holdout_feedback_allowed": False,
        "precision_difference_only": True,
    }
    for key, expected in expected_scalars.items():
        if scout.get(key) != expected:
            errors.append(f"scout_contract_mismatch:{key}")
    if scout.get("input_modes") != SCOUT_INPUT_MODES:
        errors.append("scout_input_modes_must_equal_text_role")
    if scout.get("seeds") != SCOUT_SEEDS:
        errors.append("scout_seeds_must_equal_17_29_43")
    if scout.get("quality_gate") != _QUALITY_GATE:
        errors.append("scout_quality_gate_mismatch")
    if scout.get("promotion_policy") != _PROMOTION_POLICY:
        errors.append("scout_promotion_policy_mismatch")

    control_path = _bound_control_path(root, scout, errors)
    control: dict[str, object] = {}
    if control_path is not None:
        control = load_yaml(control_path)
        for key in _COMPARISON_KEYS:
            if scout.get(key) != control.get(key):
                errors.append(f"precision_scout_nonprecision_drift:{key}")
        if control.get("mixed_precision") != "fp32":
            errors.append("precision_scout_control_must_be_fp32")
        if control.get("input_modes") != ["role_only", "text_only", "text_role"]:
            errors.append("precision_scout_control_input_matrix_mismatch")

    run_matrix = [
        {
            "input_mode": "text_role",
            "seed": seed,
            "run_id": f"precision-bf16-text_role-s{seed}",
            "control_run_id": f"precision-fp32-text_role-s{seed}",
        }
        for seed in SCOUT_SEEDS
    ]
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "control_config_path": str(control_path) if control_path is not None else None,
        "control_config_sha256": (
            sha256_file(control_path) if control_path is not None else None
        ),
        "run_count_per_arm": len(run_matrix),
        "run_matrix": run_matrix,
        "test_access_allowed": False,
        "automatic_promotion_allowed": False,
        "claim_boundary": scout.get("claim_boundary"),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def _derived_fp32_control_config(
    control: dict[str, object],
    *,
    control_sha256: str,
) -> dict[str, object]:
    derived = dict(control)
    derived.update(
        {
            "run_name": "mdeberta_precision_scout_fp32_control",
            "status": "exploratory_development_only",
            "input_modes": ["text_role"],
            "scout_control_source_sha256": control_sha256,
            "test_evaluation_allowed": False,
            "final_holdout_feedback_allowed": False,
        }
    )
    return derived


def _protocol_arm(
    protocol: dict[str, object],
    *,
    arm: str,
) -> dict[str, object]:
    run_matrix = protocol["run_matrix"]
    assert isinstance(run_matrix, list)
    key = "run_id" if arm == "bf16" else "control_run_id"
    return {
        "status": "PASS",
        "run_count": len(run_matrix),
        "run_matrix": [
            {
                "input_mode": str(entry["input_mode"]),
                "seed": int(entry["seed"]),
                "run_id": str(entry[key]),
            }
            for entry in run_matrix
            if isinstance(entry, dict)
        ],
    }


def _load_json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _selected_throughput(summary: dict[str, object]) -> float:
    capacity = summary.get("capacity_plan")
    if not isinstance(capacity, dict) or capacity.get("status") != "PASS":
        raise ValueError("precision scout capacity plan is not PASS")
    selected = capacity.get("selected")
    if not isinstance(selected, dict):
        raise ValueError("precision scout capacity plan has no selected candidate")
    value = float(selected.get("samples_per_second", 0.0))
    if not math.isfinite(value) or value <= 0:
        raise ValueError("precision scout capacity throughput is not finite and positive")
    return value


def _same_observed_device(
    control_summary: dict[str, object],
    candidate_summary: dict[str, object],
) -> tuple[bool, dict[str, object]]:
    observations: list[dict[str, object]] = []
    for summary in (control_summary, candidate_summary):
        runtime = summary.get("runtime_check")
        if not isinstance(runtime, dict):
            return False, {"reason": "runtime_check_missing"}
        probe = runtime.get("probe")
        if (
            runtime.get("status") != "PASS"
            or runtime.get("hardware_observed") is not True
            or not isinstance(probe, dict)
        ):
            return False, {"reason": "runtime_check_not_observed_pass"}
        observations.append(probe)
    keys = ("device_name", "device_index", "compute_capability", "device_memory_gib")
    first = {key: observations[0].get(key) for key in keys}
    second = {key: observations[1].get(key) for key in keys}
    return first == second, {"fp32_control": first, "bf16_candidate": second}


def evaluate_precision_scout(
    project_root: Path,
    config_path: Path,
    output_root: Path,
    output_path: Path | None = None,
) -> dict[str, object]:
    protocol = validate_precision_scout_protocol(project_root, config_path)
    errors = list(protocol["errors"])
    control_root = output_root / "fp32-control"
    candidate_root = output_root / "bf16-candidate"
    try:
        control_summary = _load_json_object(control_root / "arm_summary.json")
        candidate_summary = _load_json_object(candidate_root / "arm_summary.json")
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        errors.append(f"arm_summary_invalid:{type(exc).__name__}")
        control_summary = {}
        candidate_summary = {}

    forbidden_names = {
        "test_predictions.jsonl",
        "test_prediction_manifest.json",
        "test_manifest.json",
        "evaluation.json",
    }
    forbidden_artifacts = sorted(
        path.relative_to(output_root).as_posix()
        for path in output_root.rglob("*")
        if path.is_file() and path.name in forbidden_names
    )
    if forbidden_artifacts:
        errors.append("scout_test_or_evaluation_artifact_present")

    comparisons: list[dict[str, object]] = []
    control_scores: list[float] = []
    candidate_scores: list[float] = []
    control_wall: list[float] = []
    candidate_wall: list[float] = []
    run_matrix = protocol.get("run_matrix")
    if isinstance(run_matrix, list):
        for entry in run_matrix:
            if not isinstance(entry, dict):
                errors.append("scout_run_matrix_invalid")
                continue
            seed = int(entry["seed"])
            control_id = str(entry["control_run_id"])
            candidate_id = str(entry["run_id"])
            try:
                control_metric = _load_json_object(
                    control_root / control_id / "development_metrics.json"
                )
                candidate_metric = _load_json_object(
                    candidate_root / candidate_id / "development_metrics.json"
                )
                if (
                    control_metric.get("status") != "PASS"
                    or candidate_metric.get("status") != "PASS"
                    or control_metric.get("test_accessed") is not False
                    or candidate_metric.get("test_accessed") is not False
                ):
                    raise ValueError("development metric is not a test-free PASS")
                control_score = float(control_metric["dev_auprc"])
                candidate_score = float(candidate_metric["dev_auprc"])
                control_seconds = float(control_metric["training_wall_seconds"])
                candidate_seconds = float(candidate_metric["training_wall_seconds"])
                values = (
                    control_score,
                    candidate_score,
                    control_seconds,
                    candidate_seconds,
                )
                if not all(math.isfinite(value) for value in values):
                    raise ValueError("non-finite development comparison")
                if control_seconds <= 0 or candidate_seconds <= 0:
                    raise ValueError("non-positive training wall time")
            except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as exc:
                errors.append(f"development_comparison_invalid:s{seed}:{type(exc).__name__}")
                continue
            control_scores.append(control_score)
            candidate_scores.append(candidate_score)
            control_wall.append(control_seconds)
            candidate_wall.append(candidate_seconds)
            comparisons.append(
                {
                    "seed": seed,
                    "fp32_dev_auprc": control_score,
                    "bf16_dev_auprc": candidate_score,
                    "bf16_minus_fp32_dev_auprc": candidate_score - control_score,
                    "fp32_training_wall_seconds": control_seconds,
                    "bf16_training_wall_seconds": candidate_seconds,
                }
            )

    throughput_ratio: float | None = None
    wall_time_ratio: float | None = None
    same_device = False
    device_evidence: dict[str, object] = {}
    if not errors:
        try:
            control_throughput = _selected_throughput(control_summary)
            candidate_throughput = _selected_throughput(candidate_summary)
            throughput_ratio = candidate_throughput / control_throughput
            wall_time_ratio = statistics.median(candidate_wall) / statistics.median(control_wall)
            same_device, device_evidence = _same_observed_device(
                control_summary,
                candidate_summary,
            )
        except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
            errors.append(f"performance_comparison_invalid:{type(exc).__name__}")

    gate_results: dict[str, bool] = {}
    if not errors and len(comparisons) == len(SCOUT_SEEDS):
        mean_control = statistics.mean(control_scores)
        mean_candidate = statistics.mean(candidate_scores)
        gate_results = {
            "mean_dev_auprc_noninferior": (
                mean_candidate
                >= mean_control - float(_QUALITY_GATE["mean_noninferiority_margin"])
            ),
            "every_seed_within_degradation_limit": all(
                control - candidate
                <= float(_QUALITY_GATE["per_seed_max_degradation"])
                for control, candidate in zip(control_scores, candidate_scores, strict=True)
            ),
            "capacity_throughput_improved": (
                throughput_ratio is not None
                and throughput_ratio
                >= float(_QUALITY_GATE["minimum_capacity_throughput_ratio"])
            ),
            "median_training_wall_time_improved": (
                wall_time_ratio is not None
                and wall_time_ratio
                <= float(_QUALITY_GATE["maximum_median_training_wall_time_ratio"])
            ),
            "same_observed_a100_device": same_device,
            "no_test_or_evaluation_artifacts": not forbidden_artifacts,
        }
    quality_gate_passed = bool(gate_results) and all(gate_results.values())
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "protocol_sha256": protocol.get("config_sha256"),
        "control_config_sha256": protocol.get("control_config_sha256"),
        "comparison_count": len(comparisons),
        "comparisons": comparisons,
        "mean_fp32_dev_auprc": statistics.mean(control_scores) if control_scores else None,
        "mean_bf16_dev_auprc": statistics.mean(candidate_scores) if candidate_scores else None,
        "capacity_throughput_ratio_bf16_over_fp32": throughput_ratio,
        "median_training_wall_time_ratio_bf16_over_fp32": wall_time_ratio,
        "same_device_evidence": device_evidence,
        "gate_results": gate_results,
        "quality_gate_passed": quality_gate_passed,
        "future_protocol_review_eligible": quality_gate_passed,
        "automatic_promotion_authorized": False,
        "final_holdout_accessed": False,
        "forbidden_artifacts": forbidden_artifacts,
        "disposition": (
            "ELIGIBLE_FOR_SEPARATELY_VERSIONED_PROTOCOL_REVIEW"
            if quality_gate_passed
            else "KEEP_FP32_CONFIRMATORY_PROTOCOL"
        ),
        "research_claim_eligible": False,
        "claim_boundary": protocol.get("claim_boundary"),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result


def run_precision_scout(
    project_root: Path,
    config_path: Path,
    split_dir: Path,
    output_root: Path,
) -> dict[str, object]:
    if os.environ.get("VIPIBENCH_PRECISION_SCOUT_APPROVED") != "YES":
        raise PermissionError(
            "VIPIBENCH_PRECISION_SCOUT_APPROVED=YES is required for live A100 scout execution"
        )
    protocol = validate_precision_scout_protocol(project_root, config_path)
    if protocol["status"] != "PASS":
        raise ValueError(protocol["errors"])
    control_path = Path(str(protocol["control_config_path"]))
    control = load_yaml(control_path)
    output_root.mkdir(parents=True, exist_ok=True)
    generated_protocol_dir = output_root / "protocols"
    generated_protocol_dir.mkdir(parents=True, exist_ok=True)
    derived_control_path = generated_protocol_dir / "fp32_control.yaml"
    derived_control = _derived_fp32_control_config(
        control,
        control_sha256=str(protocol["control_config_sha256"]),
    )
    derived_control_path.write_text(
        yaml.safe_dump(derived_control, sort_keys=False),
        encoding="utf-8",
        newline="\n",
    )

    control_root = output_root / "fp32-control"
    candidate_root = output_root / "bf16-candidate"
    control_result = _run_encoder_development_matrix(
        derived_control_path,
        split_dir,
        control_root,
        protocol=_protocol_arm(protocol, arm="fp32"),
        execution_scope="exploratory_precision_scout_fp32_control",
        partition_names=("train", "dev"),
    )
    write_json(control_root / "arm_summary.json", control_result)
    candidate_result = _run_encoder_development_matrix(
        config_path,
        split_dir,
        candidate_root,
        protocol=_protocol_arm(protocol, arm="bf16"),
        execution_scope="exploratory_precision_scout_bf16_candidate",
        partition_names=("train", "dev"),
    )
    write_json(candidate_root / "arm_summary.json", candidate_result)
    comparison_path = output_root / "precision_scout_evaluation.json"
    evaluation = evaluate_precision_scout(
        project_root,
        config_path,
        output_root,
        comparison_path,
    )
    return {
        "status": evaluation["status"],
        "phase": "exploratory_development_precision_scout",
        "fp32_control": control_result,
        "bf16_candidate": candidate_result,
        "evaluation": evaluation,
        "evaluation_path": str(comparison_path),
        "automatic_promotion_authorized": False,
        "research_claim_eligible": False,
    }
