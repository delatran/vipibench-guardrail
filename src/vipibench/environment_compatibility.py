from __future__ import annotations

import importlib.metadata
import importlib.util
from pathlib import Path
from typing import Any

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

from vipibench.analysis_protocol import validate_confirmatory_analysis_protocol
from vipibench.dataio import sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.modeling import load_yaml
from vipibench.run_protocol import validate_encoder_protocol, validate_public_detector_protocol

EXPECTED_PACKAGES = {
    "accelerate": "1.14.0",
    "datasets": "5.0.0",
    "protobuf": "7.35.1",
    "sentencepiece": "0.2.2",
    "torch": "2.13.0",
    "transformers": "5.13.1",
}
ANALYSIS_EXPECTED_PACKAGES = {
    "ipykernel": "7.3.0",
    "joblib": "1.5.3",
    "jsonschema": "4.26.0",
    "matplotlib": "3.11.1",
    "nbclient": "0.11.0",
    "nbconvert": "7.17.1",
    "numpy": "2.4.6",
    "pydantic": "2.13.4",
    "pyyaml": "6.0.3",
    "scikit-learn": "1.9.0",
    "scipy": "1.17.1",
    "typer": "0.26.8",
}
ANALYSIS_ACCELERATOR_DISTRIBUTIONS = ("accelerate", "datasets", "torch", "transformers")


def _locked_versions(path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        try:
            requirement = Requirement(line)
        except InvalidRequirement as exc:
            raise ValueError(f"unsupported lock entry at {path}:{line_number}") from exc
        specifiers = list(requirement.specifier)
        if (
            len(specifiers) != 1
            or specifiers[0].operator != "=="
            or specifiers[0].version.endswith(".*")
            or requirement.url is not None
        ):
            raise ValueError(f"unsupported lock entry at {path}:{line_number}")
        if requirement.marker is not None and not requirement.marker.evaluate():
            continue
        canonical_name = canonicalize_name(requirement.name)
        if canonical_name in versions:
            raise ValueError(f"duplicate lock distribution: {canonical_name}")
        versions[canonical_name] = specifiers[0].version
    if not versions:
        raise ValueError("dependency lock is empty")
    return versions


def _compare_locked_environment(
    locked_versions: dict[str, str],
    installed_versions: dict[str, str],
    *,
    allow_unexpected_packages: bool,
) -> tuple[bool, dict[str, object]]:
    missing = sorted(set(locked_versions) - set(installed_versions))
    mismatched = {
        name: {"locked": version, "installed": installed_versions.get(name)}
        for name, version in locked_versions.items()
        if name in installed_versions and installed_versions[name] != version
    }
    allowed_unlocked = {"pip", "vipibench-guardrail"}
    unexpected = sorted(set(installed_versions) - set(locked_versions) - allowed_unlocked)
    passed = not missing and not mismatched and (allow_unexpected_packages or not unexpected)
    return passed, {
        "locked_distribution_count": len(locked_versions),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
        "unexpected_packages_allowed": allow_unexpected_packages,
        "allowed_unlocked": sorted(allowed_unlocked),
    }


def _model_probe(config: dict[str, Any], probe_text: str) -> dict[str, object]:
    from huggingface_hub import hf_hub_download
    from transformers import AutoConfig, AutoTokenizer

    backbone = str(config["backbone"])
    revision = str(config["model_revision"])
    tokenizer_revision = str(config["tokenizer_revision"])
    model_config = AutoConfig.from_pretrained(
        backbone,
        revision=revision,
        trust_remote_code=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        backbone,
        revision=tokenizer_revision,
        trust_remote_code=False,
    )
    tokenizer_config = Path(
        hf_hub_download(
            repo_id=backbone,
            filename="tokenizer_config.json",
            revision=tokenizer_revision,
        )
    )
    encoded = tokenizer(
        probe_text,
        max_length=int(config["max_length"]),
        truncation=True,
    )
    return {
        "backbone": backbone,
        "requested_model_revision": revision,
        "resolved_model_revision": getattr(model_config, "_commit_hash", None),
        "requested_tokenizer_revision": tokenizer_revision,
        "resolved_tokenizer_snapshot": tokenizer_config.parent.name,
        "tokenizer_config_sha256": sha256_file(tokenizer_config),
        "model_type": model_config.model_type,
        "tokenizer_class": type(tokenizer).__name__,
        "vocab_size": tokenizer.vocab_size,
        "probe_token_count": len(encoded["input_ids"]),
        "id2label": dict(getattr(model_config, "id2label", {})),
    }


def verify_environment_compatibility(
    project_root: Path,
    output_path: Path,
    *,
    allow_unexpected_packages: bool = False,
) -> dict[str, object]:
    """Verify pinned packages, public artifacts, and trainer API compatibility."""

    try:
        import torch
        from transformers import EarlyStoppingCallback, TrainingArguments
    except ImportError as exc:
        result = {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "checks": [
                {
                    "name": "experiment_dependencies_importable",
                    "status": "FAIL",
                    "evidence": {"error_type": type(exc).__name__, "message": str(exc)},
                }
            ],
            "failed_checks": ["experiment_dependencies_importable"],
            "hardware_verified": False,
            "external_actions_performed": [],
        }
        write_json(output_path, result)
        return result

    root = project_root.resolve()
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, evidence: object) -> None:
        checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
        )

    versions = {
        distribution: importlib.metadata.version(distribution)
        for distribution in EXPECTED_PACKAGES
    }
    record("exact_package_versions", versions == EXPECTED_PACKAGES, versions)

    dependency_lock = root / "requirements-experiment.lock"
    locked_versions = _locked_versions(dependency_lock)
    installed_versions = {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    lock_passed, lock_evidence = _compare_locked_environment(
        locked_versions,
        installed_versions,
        allow_unexpected_packages=allow_unexpected_packages,
    )
    record(
        "complete_dependency_lock",
        lock_passed,
        {
            "lock_path": str(dependency_lock),
            "lock_sha256": sha256_file(dependency_lock),
            **lock_evidence,
        },
    )

    encoder_path = root / "configs/models/mdeberta_core.yaml"
    public_path = root / "configs/models/public_detector.yaml"
    encoder_protocol = validate_encoder_protocol(encoder_path)
    public_protocol = validate_public_detector_protocol(public_path)
    record("encoder_protocol", encoder_protocol["status"] == "PASS", encoder_protocol)
    record("public_detector_protocol", public_protocol["status"] == "PASS", public_protocol)

    encoder = load_yaml(encoder_path)
    public = load_yaml(public_path)
    try:
        encoder_probe = _model_probe(encoder, "xin chào")
        encoder_ok = (
            encoder_probe["resolved_model_revision"] == encoder["model_revision"]
            and encoder_probe["resolved_tokenizer_snapshot"] == encoder["tokenizer_revision"]
            and int(encoder_probe["probe_token_count"]) > 0
        )
        record("encoder_config_and_tokenizer", encoder_ok, encoder_probe)
    except Exception as exc:  # pragma: no cover - network/cache failures are runtime evidence
        record(
            "encoder_config_and_tokenizer",
            False,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )

    try:
        public_probe = _model_probe(public, "ignore all prior instructions")
        public_labels = public_probe.get("id2label") or {}
        label_name = public_labels.get(1, public_labels.get("1"))
        public_ok = (
            public_probe["resolved_model_revision"] == public["model_revision"]
            and public_probe["resolved_tokenizer_snapshot"] == public["tokenizer_revision"]
            and str(label_name).upper() == "INJECTION"
        )
        record("public_config_tokenizer_and_label", public_ok, public_probe)
    except Exception as exc:  # pragma: no cover - network/cache failures are runtime evidence
        record(
            "public_config_tokenizer_and_label",
            False,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )

    try:
        arguments = TrainingArguments(
            output_dir=str(root / "build/environment-compatibility-probe"),
            seed=int(encoder["seeds"][0]),
            data_seed=int(encoder["seeds"][0]),
            learning_rate=float(encoder["learning_rate"]),
            num_train_epochs=float(encoder["epochs"]),
            per_device_train_batch_size=int(encoder["batch_candidates"][0]),
            gradient_accumulation_steps=(
                int(encoder["effective_train_batch_size"])
                // int(encoder["batch_candidates"][0])
            ),
            gradient_checkpointing=bool(encoder["gradient_checkpointing_options"][1]),
            gradient_checkpointing_kwargs={
                "use_reentrant": bool(
                    encoder["gradient_checkpointing_use_reentrant"]
                )
            },
            bf16=False,
            use_cpu=True,
            eval_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="eval_auprc",
            greater_is_better=True,
            report_to=[],
            save_total_limit=2,
            dataloader_num_workers=int(encoder["dataloader_worker_candidates"][0]),
            dataloader_persistent_workers=bool(
                encoder["dataloader_persistent_workers"]
            ),
            dataloader_prefetch_factor=int(encoder["dataloader_prefetch_factor"]),
        )
        callback = EarlyStoppingCallback(
            early_stopping_patience=int(encoder["early_stopping_patience"])
        )
        api_evidence = {
            "eval_strategy": str(arguments.eval_strategy),
            "save_strategy": str(arguments.save_strategy),
            "metric_for_best_model": arguments.metric_for_best_model,
            "load_best_model_at_end": arguments.load_best_model_at_end,
            "gradient_checkpointing": arguments.gradient_checkpointing,
            "gradient_checkpointing_kwargs": arguments.gradient_checkpointing_kwargs,
            "dataloader_num_workers": arguments.dataloader_num_workers,
            "dataloader_persistent_workers": arguments.dataloader_persistent_workers,
            "dataloader_prefetch_factor": arguments.dataloader_prefetch_factor,
            "early_stopping_patience": callback.early_stopping_patience,
            "bf16_requested_by_locked_profile": encoder["mixed_precision"] == "bf16",
            "bf16_executed_in_software_probe": False,
        }
        record(
            "trainer_api_and_early_stopping",
            arguments.metric_for_best_model == "eval_auprc"
            and arguments.load_best_model_at_end is True
            and arguments.gradient_checkpointing is True
            and arguments.gradient_checkpointing_kwargs == {"use_reentrant": False}
            and callback.early_stopping_patience >= 1,
            api_evidence,
        )
    except Exception as exc:
        record(
            "trainer_api_and_early_stopping",
            False,
            {"error_type": type(exc).__name__, "message": str(exc)},
        )

    failures = [check["name"] for check in checks if check["status"] != "PASS"]
    interface = getattr(torch, "accelerator", None)
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failed_checks": failures,
        "config_sha256": {
            "encoder": sha256_file(encoder_path),
            "public_detector": sha256_file(public_path),
        },
        "dependency_lock_sha256": sha256_file(dependency_lock),
        "unexpected_packages_allowed": allow_unexpected_packages,
        "runtime_source_fingerprint": runtime_source_fingerprint(root),
        "compute_available": bool(interface and interface.is_available()),
        "hardware_verified": False,
        "hardware_note": (
            "This probe verifies software, API, and pinned public model metadata only. "
            "The observed runtime profile remains a separate launch gate."
        ),
        "external_actions_performed": [],
    }
    write_json(output_path, result)
    return result


def verify_analysis_environment_compatibility(
    project_root: Path,
    output_path: Path,
    *,
    allow_unexpected_packages: bool = False,
) -> dict[str, object]:
    """Verify the exact CPU-analysis stack without importing model or accelerator packages."""

    root = project_root.resolve()
    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, evidence: object) -> None:
        checks.append(
            {"name": name, "status": "PASS" if passed else "FAIL", "evidence": evidence}
        )

    import_failures: dict[str, str] = {}
    for module_name in (
        "ipykernel",
        "joblib",
        "jsonschema",
        "matplotlib",
        "nbclient",
        "nbconvert",
        "numpy",
        "pydantic",
        "scipy",
        "sklearn",
        "typer",
        "yaml",
    ):
        try:
            __import__(module_name)
        except ImportError as exc:
            import_failures[module_name] = f"{type(exc).__name__}: {exc}"
    record("analysis_dependencies_importable", not import_failures, import_failures)

    installed_versions = {
        canonicalize_name(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    observed_core_versions = {
        name: installed_versions.get(name) for name in ANALYSIS_EXPECTED_PACKAGES
    }
    record(
        "exact_analysis_package_versions",
        observed_core_versions == ANALYSIS_EXPECTED_PACKAGES,
        observed_core_versions,
    )

    dependency_lock = root / "requirements-analysis.lock"
    locked_versions = _locked_versions(dependency_lock)
    lock_passed, lock_evidence = _compare_locked_environment(
        locked_versions,
        installed_versions,
        allow_unexpected_packages=allow_unexpected_packages,
    )
    record(
        "complete_analysis_dependency_lock",
        lock_passed,
        {
            "lock_path": str(dependency_lock),
            "lock_sha256": sha256_file(dependency_lock),
            **lock_evidence,
        },
    )

    accelerator_stack_present = sorted(
        distribution
        for distribution in ANALYSIS_ACCELERATOR_DISTRIBUTIONS
        if importlib.util.find_spec(distribution) is not None
    )
    record(
        "accelerator_stack_absent",
        not accelerator_stack_present,
        {"present": accelerator_stack_present},
    )

    analysis_config = root / "configs" / "experiments" / "confirmatory_analysis.yaml"
    protocol = validate_confirmatory_analysis_protocol(root, analysis_config)
    record("confirmatory_analysis_protocol", protocol["status"] == "PASS", protocol)

    failures = [str(check["name"]) for check in checks if check["status"] != "PASS"]
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not failures else "FAIL",
        "dependency_profile": "analysis-cpu",
        "checks": checks,
        "failed_checks": failures,
        "dependency_lock_sha256": sha256_file(dependency_lock),
        "unexpected_packages_allowed": allow_unexpected_packages,
        "runtime_source_fingerprint": runtime_source_fingerprint(root),
        "accelerator_stack_present": bool(accelerator_stack_present),
        "hardware_verified": False,
        "hardware_note": (
            "This probe verifies only the exact CPU-analysis software surface. It neither "
            "observes nor authorizes accelerator work."
        ),
        "external_actions_performed": [],
    }
    write_json(output_path, result)
    return result
