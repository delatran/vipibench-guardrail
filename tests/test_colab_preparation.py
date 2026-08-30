import importlib.util
import inspect
import json
import subprocess
import sys
from pathlib import Path

import pytest

from vipibench.environment_compatibility import verify_analysis_environment_compatibility

SCRIPT_PATH = Path("scripts/prepare_colab.py")
SPEC = importlib.util.spec_from_file_location("vipibench_prepare_colab", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
PREPARE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PREPARE)


def test_lock_parser_requires_exact_pins(tmp_path: Path) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("NumPy==2.4.6\nscikit_learn==1.9.0\n", encoding="utf-8")

    assert PREPARE.locked_versions(lock_path) == {
        "numpy": "2.4.6",
        "scikit-learn": "1.9.0",
    }

    lock_path.write_text(
        'numpy==2.4.6\ncuda-toolkit==13.0.3.0; platform_system == "Linux"\n',
        encoding="utf-8",
    )
    expected = {"numpy": "2.4.6"}
    if sys.platform == "linux":
        expected["cuda-toolkit"] = "13.0.3.0"
    assert PREPARE.locked_versions(lock_path) == expected

    lock_path.write_text("numpy>=2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="exact pin"):
        PREPARE.locked_versions(lock_path)


def test_lock_drift_rejects_unlisted_transitive_dependency(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "requirements.lock"
    lock_path.write_text("numpy==2.4.6\n", encoding="utf-8")

    class Distribution:
        def __init__(self, name: str, version: str) -> None:
            self.metadata = {"Name": name}
            self.version = version

    monkeypatch.setattr(
        PREPARE.importlib.metadata,
        "distributions",
        lambda: [
            Distribution("numpy", "2.4.6"),
            Distribution("pip", "25.0"),
            Distribution("unlisted-transitive", "1.0"),
        ],
    )

    result = PREPARE.observed_lock_drift(lock_path)

    assert result["status"] == "FAIL"
    assert result["unexpected"] == ["unlisted-transitive"]
    assert result["allowed_unlocked"] == ["pip"]


def test_dependency_mutations_use_the_selected_child_interpreter() -> None:
    lock_command = PREPARE.build_lock_install_command(
        "/usr/local/bin/python",
        Path("/content/project/requirements-experiment.lock"),
    )

    assert lock_command[:4] == ["/usr/local/bin/python", "-m", "pip", "install"]
    assert lock_command[-2:] == [
        "--requirement",
        str(Path("/content/project/requirements-experiment.lock")),
    ]


def test_import_probe_runs_in_a_fresh_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        assert kwargs == {"check": False, "capture_output": True, "text": True}
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps({"status": "PASS", "versions": {}}) + "\n",
            stderr="",
        )

    monkeypatch.setattr(PREPARE.subprocess, "run", fake_run)

    assert PREPARE.run_fresh_import_probe("/usr/local/bin/python")["status"] == "PASS"
    assert observed == [["/usr/local/bin/python", "-c", PREPARE.IMPORT_PROBE]]


def test_import_probe_fails_closed_when_unused_optional_packages_remain(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = {"status": "FAIL", "unused_optional_present": ["peft"]}
    completed = subprocess.CompletedProcess(
        ["python", "-c", PREPARE.IMPORT_PROBE],
        0,
        stdout=json.dumps(payload) + "\n",
        stderr="",
    )
    monkeypatch.setattr(PREPARE.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(RuntimeError, match="did not pass"):
        PREPARE.run_fresh_import_probe("python")


def test_import_probe_requires_the_numerics_guard_api_surface() -> None:
    assert "import matplotlib" in PREPARE.IMPORT_PROBE
    assert "TrainerCallback" in PREPARE.IMPORT_PROBE
    assert "gradient_checkpointing_kwargs" in PREPARE.IMPORT_PROBE
    assert "dataloader_persistent_workers" in PREPARE.IMPORT_PROBE
    assert "dataloader_prefetch_factor" in PREPARE.IMPORT_PROBE
    assert "on_pre_optimizer_step" in PREPARE.IMPORT_PROBE
    assert "on_optimizer_step" in PREPARE.IMPORT_PROBE
    assert "numerics_guard_api" in PREPARE.IMPORT_PROBE


def test_analysis_profile_uses_cpu_lock_and_excludes_accelerator_imports() -> None:
    assert "import matplotlib" in PREPARE.ANALYSIS_IMPORT_PROBE
    assert PREPARE.DEPENDENCY_PROFILES["analysis-cpu"] == "requirements-analysis.lock"
    assert 'accelerator_distributions = ("accelerate", "datasets", "torch", "transformers")' in (
        PREPARE.ANALYSIS_IMPORT_PROBE
    )
    assert "import torch\n" not in PREPARE.ANALYSIS_IMPORT_PROBE
    assert "import transformers\n" not in PREPARE.ANALYSIS_IMPORT_PROBE


def test_analysis_import_probe_runs_in_a_fresh_child_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []
    payload = {
        "status": "PASS",
        "accelerator_stack_present": [],
        "versions": {},
    }

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        return subprocess.CompletedProcess(
            command,
            0,
            stdout=json.dumps(payload) + "\n",
            stderr="",
        )

    monkeypatch.setattr(PREPARE.subprocess, "run", fake_run)

    assert PREPARE.run_fresh_analysis_import_probe("/usr/local/bin/python") == payload
    assert observed == [["/usr/local/bin/python", "-c", PREPARE.ANALYSIS_IMPORT_PROBE]]


def test_analysis_environment_validator_passes_project_root_to_protocol() -> None:
    source = inspect.getsource(verify_analysis_environment_compatibility)

    assert "validate_confirmatory_analysis_protocol(root, analysis_config)" in source


def test_notebook_runner_probe_uses_the_isolated_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        observed.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="7.17.1\n", stderr="")

    monkeypatch.setattr(PREPARE.subprocess, "run", fake_run)

    assert PREPARE.run_notebook_runner_probe("/content/runtime/bin/python") == "7.17.1"
    assert observed == [
        [
            "/content/runtime/bin/python",
            "-m",
            "nbconvert",
            "--version",
        ]
    ]


def test_failed_child_process_preserves_stage_and_command_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        ["python", "-m", "pip"],
        17,
        stdout="resolver detail\n",
        stderr="disk detail\n",
    )
    monkeypatch.setattr(PREPARE.subprocess, "run", lambda *args, **kwargs: completed)

    with pytest.raises(PREPARE.PreparationStageError) as raised:
        PREPARE.run_checked_command(
            ["python", "-m", "pip"],
            stage="dependency_install",
        )

    assert raised.value.stage == "dependency_install"
    assert raised.value.returncode == 17
    assert "resolver detail" in str(raised.value)
    assert "disk detail" in str(raised.value)


def test_preparer_refuses_to_mutate_non_isolated_interpreter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "requirements-experiment.lock").write_text(
        "numpy==2.4.6\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(PREPARE.sys, "prefix", "/usr")
    monkeypatch.setattr(PREPARE.sys, "base_prefix", "/usr")

    with pytest.raises(PREPARE.PreparationStageError, match="live Colab interpreter") as raised:
        PREPARE.prepare_colab_environment(tmp_path, probe_only=False)

    assert raised.value.stage == "runtime_validation"
