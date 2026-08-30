"""Prepare and verify the pinned Colab environment without importing it in the live kernel."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import re
import subprocess
import sys
from pathlib import Path

UNUSED_OPTIONAL_DISTRIBUTIONS = ("peft", "torchvision", "torchaudio")
DEPENDENCY_PROFILES = {
    "accelerator": "requirements-experiment.lock",
    "analysis-cpu": "requirements-analysis.lock",
}
MAX_DIAGNOSTIC_LINES = 80
IMPORT_PROBE = r"""
import importlib.util
import inspect
import json
from pathlib import Path

import ipykernel
import matplotlib
import nbconvert
import numpy
import scipy
import sklearn
import torch
import transformers
from transformers import EarlyStoppingCallback, TrainerCallback, TrainingArguments

unused_optional_present = [
    name
    for name in ("peft", "torchvision", "torchaudio")
    if importlib.util.find_spec(name) is not None
]
training_argument_parameters = inspect.signature(TrainingArguments).parameters
numerics_guard_api = {
    "gradient_checkpointing_kwargs": "gradient_checkpointing_kwargs"
    in training_argument_parameters,
    "dataloader_persistent_workers": "dataloader_persistent_workers"
    in training_argument_parameters,
    "dataloader_prefetch_factor": "dataloader_prefetch_factor"
    in training_argument_parameters,
    "on_pre_optimizer_step": callable(
        getattr(TrainerCallback, "on_pre_optimizer_step", None)
    ),
    "on_optimizer_step": callable(getattr(TrainerCallback, "on_optimizer_step", None)),
}
payload = {
    "status": (
        "PASS"
        if not unused_optional_present and all(numerics_guard_api.values())
        else "FAIL"
    ),
    "unused_optional_present": unused_optional_present,
    "numerics_guard_api": numerics_guard_api,
    "versions": {
        "numpy": numpy.__version__,
        "ipykernel": ipykernel.__version__,
        "matplotlib": matplotlib.__version__,
        "nbconvert": nbconvert.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    },
    "module_paths": {
        "numpy": str(Path(numpy.__file__).resolve()),
        "ipykernel": str(Path(ipykernel.__file__).resolve()),
        "matplotlib": str(Path(matplotlib.__file__).resolve()),
        "nbconvert": str(Path(nbconvert.__file__).resolve()),
        "scipy": str(Path(scipy.__file__).resolve()),
        "scikit-learn": str(Path(sklearn.__file__).resolve()),
        "torch": str(Path(torch.__file__).resolve()),
        "transformers": str(Path(transformers.__file__).resolve()),
    },
    "trainer_api": {
        "EarlyStoppingCallback": EarlyStoppingCallback.__name__,
        "TrainerCallback": TrainerCallback.__name__,
        "TrainingArguments": TrainingArguments.__name__,
    },
}
print(json.dumps(payload, sort_keys=True))
"""
ANALYSIS_IMPORT_PROBE = r"""
import importlib.metadata
import importlib.util
import json
from pathlib import Path

import ipykernel
import joblib
import jsonschema
import matplotlib
import nbclient
import nbconvert
import numpy
import pydantic
import scipy
import sklearn
import typer
import yaml

accelerator_distributions = ("accelerate", "datasets", "torch", "transformers")
accelerator_stack_present = [
    name for name in accelerator_distributions if importlib.util.find_spec(name) is not None
]
payload = {
    "status": "PASS" if not accelerator_stack_present else "FAIL",
    "accelerator_stack_present": accelerator_stack_present,
    "versions": {
        "ipykernel": ipykernel.__version__,
        "joblib": joblib.__version__,
        "jsonschema": importlib.metadata.version("jsonschema"),
        "matplotlib": matplotlib.__version__,
        "nbclient": nbclient.__version__,
        "nbconvert": nbconvert.__version__,
        "numpy": numpy.__version__,
        "pydantic": pydantic.__version__,
        "PyYAML": importlib.metadata.version("PyYAML"),
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "typer": typer.__version__,
    },
    "module_paths": {
        "numpy": str(Path(numpy.__file__).resolve()),
        "matplotlib": str(Path(matplotlib.__file__).resolve()),
        "scipy": str(Path(scipy.__file__).resolve()),
        "scikit-learn": str(Path(sklearn.__file__).resolve()),
        "nbconvert": str(Path(nbconvert.__file__).resolve()),
    },
}
print(json.dumps(payload, sort_keys=True))
"""


class PreparationStageError(RuntimeError):
    """A preparation failure bound to the stage that produced it."""

    def __init__(
        self,
        stage: str,
        message: str,
        *,
        command: list[str] | None = None,
        returncode: int | None = None,
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.command = command
        self.returncode = returncode


def _canonicalize_distribution(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _diagnostic_tail(*streams: str) -> str:
    lines = [
        line
        for stream in streams
        for line in stream.splitlines()
        if line.strip()
    ]
    return "\n".join(lines[-MAX_DIAGNOSTIC_LINES:])


def run_checked_command(command: list[str], *, stage: str) -> subprocess.CompletedProcess[str]:
    print(
        f"VIPIBENCH_COLAB_STAGE {stage} START "
        + json.dumps(command, ensure_ascii=True),
        flush=True,
    )
    completed = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.stdout:
        print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
    if completed.stderr:
        print(
            completed.stderr,
            end="" if completed.stderr.endswith("\n") else "\n",
            file=sys.stderr,
            flush=True,
        )
    if completed.returncode != 0:
        tail = _diagnostic_tail(completed.stdout, completed.stderr)
        detail = f"\nLast command output:\n{tail}" if tail else ""
        raise PreparationStageError(
            stage,
            f"Stage {stage!r} failed with exit code {completed.returncode}.{detail}",
            command=command,
            returncode=completed.returncode,
        )
    print(f"VIPIBENCH_COLAB_STAGE {stage} PASS", flush=True)
    return completed


def locked_versions(lock_path: Path) -> dict[str, str]:
    versions: dict[str, str] = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if ";" in line:
            line, marker = (part.strip() for part in line.split(";", maxsplit=1))
            if marker != 'platform_system == "Linux"':
                raise ValueError(f"Unsupported experiment lock marker: {marker}")
            if sys.platform != "linux":
                continue
        if line.count("==") != 1:
            raise ValueError(f"Experiment lock entry must be an exact pin: {line}")
        name, version = line.split("==", maxsplit=1)
        versions[_canonicalize_distribution(name)] = version
    if not versions:
        raise ValueError(f"Experiment lock is empty: {lock_path}")
    return versions


def observed_lock_drift(lock_path: Path) -> dict[str, object]:
    expected = locked_versions(lock_path)
    installed = {
        _canonicalize_distribution(distribution.metadata["Name"]): distribution.version
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    }
    missing = sorted(set(expected) - set(installed))
    mismatched = {
        name: {"expected": version, "observed": installed.get(name)}
        for name, version in expected.items()
        if name in installed and installed[name] != version
    }
    allowed_unlocked = {"pip"}
    unexpected = sorted(set(installed) - set(expected) - allowed_unlocked)
    return {
        "status": (
            "PASS" if not missing and not mismatched and not unexpected else "FAIL"
        ),
        "locked_distribution_count": len(expected),
        "missing": missing,
        "mismatched": mismatched,
        "unexpected": unexpected,
        "allowed_unlocked": sorted(allowed_unlocked),
    }


def build_lock_install_command(python: str, lock_path: Path) -> list[str]:
    return [
        python,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--progress-bar",
        "off",
        "--requirement",
        str(lock_path),
    ]


def run_fresh_import_probe(python: str) -> dict[str, object]:
    completed = run_checked_command(
        [python, "-c", IMPORT_PROBE],
        stage="trainer_import_probe",
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise PreparationStageError(
            "trainer_import_probe",
            "Fresh runtime import probe produced no output",
        )
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise PreparationStageError(
            "trainer_import_probe",
            "Fresh runtime import probe did not end with a JSON result: "
            + output_lines[-1],
        ) from exc
    if result.get("status") != "PASS":
        raise PreparationStageError(
            "trainer_import_probe",
            f"Fresh runtime import probe did not pass: {result}",
        )
    return result


def run_fresh_analysis_import_probe(python: str) -> dict[str, object]:
    completed = run_checked_command(
        [python, "-c", ANALYSIS_IMPORT_PROBE],
        stage="analysis_import_probe",
    )
    output_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not output_lines:
        raise PreparationStageError(
            "analysis_import_probe",
            "Fresh analysis runtime import probe produced no output",
        )
    try:
        result = json.loads(output_lines[-1])
    except json.JSONDecodeError as exc:
        raise PreparationStageError(
            "analysis_import_probe",
            "Fresh analysis runtime import probe did not end with a JSON result: "
            + output_lines[-1],
        ) from exc
    if result.get("status") != "PASS":
        raise PreparationStageError(
            "analysis_import_probe",
            f"Fresh analysis runtime import probe did not pass: {result}",
        )
    return result


def run_notebook_runner_probe(python: str) -> str:
    completed = run_checked_command(
        [python, "-m", "nbconvert", "--version"],
        stage="notebook_runner_probe",
    )
    version = completed.stdout.strip()
    if not version:
        raise PreparationStageError(
            "notebook_runner_probe",
            "The isolated Jupyter nbconvert runner produced no version output",
        )
    return version


def prepare_colab_environment(
    project_root: Path,
    *,
    probe_only: bool,
    dependency_profile: str = "accelerator",
) -> dict[str, object]:
    root = project_root.resolve()
    if dependency_profile not in DEPENDENCY_PROFILES:
        raise PreparationStageError(
            "runtime_validation",
            f"Unsupported dependency profile: {dependency_profile}",
        )
    lock_path = root / DEPENDENCY_PROFILES[dependency_profile]
    if not lock_path.is_file():
        raise PreparationStageError(
            "runtime_validation",
            f"Missing exact experiment lock: {lock_path}",
        )
    if not ((3, 11) <= sys.version_info[:2] < (3, 13)):
        raise PreparationStageError(
            "runtime_validation",
            "ViPIBench requires Python 3.11 or 3.12; "
            f"observed {sys.version_info.major}.{sys.version_info.minor}",
        )
    if sys.prefix == sys.base_prefix:
        raise PreparationStageError(
            "runtime_validation",
            "Refusing to mutate the live Colab interpreter. Run this preparer with the "
            "isolated /content/vipibench_runtime Python created by RUN_EXPERIMENT.ipynb.",
        )

    if not probe_only:
        run_checked_command(
            build_lock_install_command(sys.executable, lock_path),
            stage="dependency_install",
        )

    drift = observed_lock_drift(lock_path)
    if drift["status"] != "PASS":
        raise PreparationStageError(
            "lock_verification",
            "Pinned Colab environment is incomplete or drifted: "
            + json.dumps(drift, sort_keys=True),
        )
    imports = (
        run_fresh_import_probe(sys.executable)
        if dependency_profile == "accelerator"
        else run_fresh_analysis_import_probe(sys.executable)
    )
    notebook_runner_version = run_notebook_runner_probe(sys.executable)
    return {
        "schema_version": "1.0.0",
        "status": "PASS",
        "mode": "probe_only" if probe_only else "prepare",
        "dependency_profile": dependency_profile,
        "python": sys.executable,
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
        "lock": str(lock_path),
        "lock_verification": drift,
        "import_probe": imports,
        "notebook_runner": {
            "status": "PASS",
            "nbconvert_version": notebook_runner_version,
        },
        "claim_boundary": (
            "PASS proves that a fresh child interpreter sees the selected exact pinned "
            "dependency set, imports its declared execution surface, and starts the isolated "
            "Jupyter nbconvert runner. It is not live accelerator or model-quality evidence."
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare and verify the exact ViPIBench Colab dependency environment."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument(
        "--dependency-profile",
        choices=sorted(DEPENDENCY_PROFILES),
        default="accelerator",
    )
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Do not install; only verify exact pins and imports in a fresh interpreter.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = prepare_colab_environment(
            args.project_root,
            probe_only=args.probe_only,
            dependency_profile=args.dependency_profile,
        )
    except Exception as exc:
        failure = {
            "schema_version": "1.0.0",
            "status": "FAIL",
            "stage": getattr(exc, "stage", "internal"),
            "error_type": type(exc).__name__,
            "message": str(exc),
            "python": sys.executable,
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
            "project_root": str(args.project_root.resolve()),
        }
        command = getattr(exc, "command", None)
        returncode = getattr(exc, "returncode", None)
        if command is not None:
            failure["command"] = command
        if returncode is not None:
            failure["returncode"] = returncode
        print(
            "VIPIBENCH_COLAB_PREPARE_FAILURE "
            + json.dumps(failure, ensure_ascii=True, sort_keys=True),
            file=sys.stderr,
            flush=True,
        )
        return 1
    versions = result["import_probe"]["versions"]
    if result["dependency_profile"] == "accelerator":
        summary = (
            "OK fresh-process trainer import"
            f" | numpy {versions['numpy']}"
            f" | scipy {versions['scipy']}"
            f" | torch {versions['torch']}"
            f" | transformers {versions['transformers']}"
            f" | nbconvert {versions['nbconvert']}"
        )
    else:
        summary = (
            "OK fresh-process CPU analysis import"
            f" | numpy {versions['numpy']}"
            f" | scipy {versions['scipy']}"
            f" | scikit-learn {versions['scikit-learn']}"
            f" | nbconvert {versions['nbconvert']}"
        )
    print(summary)
    print(json.dumps(result, ensure_ascii=True, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
