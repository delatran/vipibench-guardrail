from __future__ import annotations

import tomllib
from pathlib import Path

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from vipibench.environment_compatibility import (
    _compare_locked_environment,
    _locked_versions,
)


def test_experiment_lock_contains_every_direct_dependency_at_the_exact_pin() -> None:
    project = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))
    locked = _locked_versions(Path("requirements-experiment.lock"))
    direct = list(project["project"]["dependencies"])
    for group in project["project"]["optional-dependencies"].values():
        direct.extend(group)
    assert len(locked) >= 70
    for value in direct:
        requirement = Requirement(value)
        name = canonicalize_name(requirement.name)
        assert name in locked
        assert str(requirement.specifier) == f"=={locked[name]}"


def test_analysis_lock_is_an_exact_cpu_only_subset_of_experiment_lock() -> None:
    experiment = _locked_versions(Path("requirements-experiment.lock"))
    analysis = _locked_versions(Path("requirements-analysis.lock"))

    assert len(analysis) == 76
    assert analysis["matplotlib"] == "3.11.1"
    assert analysis["narwhals"] == "2.24.0"
    assert analysis.items() <= experiment.items()
    for accelerator_package in (
        "accelerate",
        "datasets",
        "huggingface-hub",
        "torch",
        "transformers",
        "triton",
    ):
        assert accelerator_package not in analysis
    assert not any(name.startswith("nvidia-") for name in analysis)


def test_lock_parser_applies_environment_markers(tmp_path: Path) -> None:
    lock = tmp_path / "requirements.lock"
    lock.write_text(
        'numpy==2.4.6\n'
        'cuda-toolkit==13.0.3.0; platform_system == "Never"\n',
        encoding="utf-8",
    )

    assert _locked_versions(lock) == {"numpy": "2.4.6"}


def test_permissive_mode_allows_unrelated_packages_but_never_version_drift() -> None:
    locked = {"torch": "2.13.0", "transformers": "5.13.1"}
    installed = {**locked, "ipywidgets": "8.1.7"}
    strict_passed, strict_evidence = _compare_locked_environment(
        locked,
        installed,
        allow_unexpected_packages=False,
    )
    permissive_passed, permissive_evidence = _compare_locked_environment(
        locked,
        installed,
        allow_unexpected_packages=True,
    )
    drifted, drift_evidence = _compare_locked_environment(
        locked,
        {**installed, "torch": "2.12.0"},
        allow_unexpected_packages=True,
    )
    assert strict_passed is False
    assert strict_evidence["unexpected"] == ["ipywidgets"]
    assert permissive_passed is True
    assert permissive_evidence["unexpected_packages_allowed"] is True
    assert drifted is False
    assert drift_evidence["mismatched"]["torch"] == {
        "locked": "2.13.0",
        "installed": "2.12.0",
    }


def test_bootstrap_and_wheel_gates_cannot_bypass_the_experiment_lock() -> None:
    bootstrap = Path("scripts/workspace_bootstrap.py").read_text(encoding="utf-8")
    clean = Path("scripts/verify_clean_environment.ps1").read_text(encoding="utf-8")
    wheel = Path("scripts/verify_current_wheel.ps1").read_text(encoding="utf-8")

    assert 'ROOT / "requirements-experiment.lock"' in bootstrap
    assert 'str(wheel) + "[experiment]"' not in bootstrap
    assert '"--no-deps",\n        wheel,' in bootstrap
    for script in (clean, wheel):
        assert 'pythonpath=' in script
        assert '$env:PYTHONPATH = $null' in script
