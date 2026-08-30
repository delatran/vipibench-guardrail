from __future__ import annotations

import re
from pathlib import Path

from vipibench.modeling import load_yaml

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LOCKED_SEEDS = [17, 29, 43]
LOCKED_INPUT_MODES = ["role_only", "text_only", "text_role"]


def validate_encoder_protocol(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    errors: list[str] = []
    for key in ("model_revision", "tokenizer_revision"):
        if not SHA_RE.fullmatch(str(config.get(key, ""))):
            errors.append(f"{key}_must_be_immutable_40_hex_commit")
    if config.get("seeds") != LOCKED_SEEDS:
        errors.append("main_encoder_seeds_must_equal_17_29_43")
    if config.get("input_modes") != LOCKED_INPUT_MODES:
        errors.append("input_modes_must_equal_role_only_text_only_text_role")
    if config.get("threshold_source") != "dev_only":
        errors.append("threshold_source_must_be_dev_only")
    if float(config.get("primary_fpr", -1)) != 0.05:
        errors.append("primary_fpr_must_equal_0_05")
    if float(config.get("secondary_fpr", -1)) != 0.01:
        errors.append("secondary_fpr_must_equal_0_01")
    if int(config.get("max_length", 0)) != 512:
        errors.append("max_length_must_equal_512")
    if config.get("batch_candidates") != [16, 32, 64, 128]:
        errors.append("batch_candidates_must_equal_16_32_64_128")
    if config.get("gradient_checkpointing_options") != [False, True]:
        errors.append("checkpointing_ablation_must_equal_false_true")
    if config.get("gradient_checkpointing_use_reentrant") is not False:
        errors.append("gradient_checkpointing_must_use_non_reentrant")
    if int(config.get("effective_train_batch_size", 0)) != 64:
        errors.append("effective_train_batch_size_must_equal_64")
    if config.get("mixed_precision") != "fp32":
        errors.append("mixed_precision_must_equal_fp32")
    if config.get("optimizer") != "adamw_torch":
        errors.append("optimizer_must_equal_adamw_torch")
    if float(config.get("max_grad_norm", 0)) != 1.0:
        errors.append("max_grad_norm_must_equal_1")
    if config.get("numerics_policy") != "fail_fast_finite":
        errors.append("numerics_policy_must_equal_fail_fast_finite")
    if int(config.get("numerics_canary_optimizer_steps", 0)) != 2:
        errors.append("numerics_canary_optimizer_steps_must_equal_2")
    if config.get("capacity_probe_input_mode") != "text_role":
        errors.append("capacity_probe_input_mode_must_equal_text_role")
    if int(config.get("capacity_warmup_optimizer_steps", 0)) != 2:
        errors.append("capacity_warmup_optimizer_steps_must_equal_2")
    if int(config.get("capacity_measurement_optimizer_steps", 0)) != 5:
        errors.append("capacity_measurement_optimizer_steps_must_equal_5")
    if config.get("dataloader_worker_candidates") != [2, 4, 8]:
        errors.append("dataloader_worker_candidates_must_equal_2_4_8")
    if int(config.get("dataloader_worker_warmup_batches", 0)) != 2:
        errors.append("dataloader_worker_warmup_batches_must_equal_2")
    if int(config.get("dataloader_worker_measurement_batches", 0)) != 8:
        errors.append("dataloader_worker_measurement_batches_must_equal_8")
    if int(config.get("dataloader_worker_repeats", 0)) != 2:
        errors.append("dataloader_worker_repeats_must_equal_2")
    if config.get("dataloader_persistent_workers") is not True:
        errors.append("dataloader_persistent_workers_must_be_enabled")
    if int(config.get("dataloader_prefetch_factor", 0)) != 2:
        errors.append("dataloader_prefetch_factor_must_equal_2")
    if config.get("final_holdout_feedback_allowed") is not False:
        errors.append("final_holdout_feedback_must_remain_disabled")
    if not 0.80 <= float(config.get("target_memory_utilization", 0)) <= 0.90:
        errors.append("target_memory_utilization_outside_locked_range")
    if config.get("system_input_mode") != "text_role":
        errors.append("system_input_mode_must_be_preregistered_text_role")
    if config.get("contrast_dataset") != "data/processed/provenance_contrast.jsonl":
        errors.append("contrast_dataset_path_mismatch")
    if config.get("early_stopping_metric") != "dev_auprc":
        errors.append("early_stopping_metric_must_be_dev_auprc")
    if int(config.get("early_stopping_patience", 0)) < 1:
        errors.append("early_stopping_patience_must_be_positive")
    if not 0 < float(config.get("early_stopping_threshold", 0)) <= 0.01:
        errors.append("early_stopping_threshold_outside_locked_range")
    matrix = [
        {"input_mode": mode, "seed": seed, "run_id": f"mdeberta-{mode}-s{seed}"}
        for mode in LOCKED_INPUT_MODES
        for seed in LOCKED_SEEDS
    ]
    return {
        "status": "PASS" if not errors else "FAIL",
        "config_path": str(config_path),
        "errors": errors,
        "run_count": len(matrix),
        "run_matrix": matrix,
    }


def validate_public_detector_protocol(config_path: Path) -> dict[str, object]:
    config = load_yaml(config_path)
    errors: list[str] = []
    for key in ("model_revision", "tokenizer_revision"):
        if not SHA_RE.fullmatch(str(config.get(key, ""))):
            errors.append(f"{key}_must_be_immutable_40_hex_commit")
    if config.get("threshold_source") != "dev_only":
        errors.append("threshold_source_must_be_dev_only")
    if config.get("license_decision") not in {"allowed", "internal_evaluation_only"}:
        errors.append("license_decision_not_approved_for_evaluation")
    if int(config.get("injection_label_id", -1)) != 1:
        errors.append("public_detector_injection_label_id_must_equal_pinned_config")
    if config.get("runtime_profile") != "configs/profiles/accelerator_80gb.yaml":
        errors.append("public_detector_runtime_profile_mismatch")
    if config.get("contrast_dataset") != "data/processed/provenance_contrast.jsonl":
        errors.append("public_detector_contrast_dataset_mismatch")
    if config.get("batch_candidates") != [32, 64, 128, 256]:
        errors.append("public_detector_batch_candidates_must_equal_32_64_128_256")
    if not 0.80 <= float(config.get("target_memory_utilization", 0)) <= 0.90:
        errors.append("public_detector_target_memory_utilization_outside_locked_range")
    if int(config.get("capacity_warmup_batches", 0)) != 1:
        errors.append("public_detector_capacity_warmup_batches_must_equal_1")
    if int(config.get("capacity_measurement_batches", 0)) != 2:
        errors.append("public_detector_capacity_measurement_batches_must_equal_2")
    if int(config.get("capacity_repeats", 0)) != 2:
        errors.append("public_detector_capacity_repeats_must_equal_2")
    return {
        "status": "PASS" if not errors else "FAIL",
        "config_path": str(config_path),
        "errors": errors,
    }
