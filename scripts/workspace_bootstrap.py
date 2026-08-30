"""Verify a transferred release bundle and prepare an isolated experiment environment."""

import base64
import hashlib
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(
    os.environ.get("VIPIBENCH_WORKSPACE_ROOT", "/workspace/vipibench-guardrail")
)
ARTIFACT_ROOT = pathlib.Path(
    os.environ.get("VIPIBENCH_ARTIFACT_ROOT", "/workspace/vipibench-artifacts")
)
VENV = pathlib.Path(os.environ.get("VIPIBENCH_ENVIRONMENT_ROOT", "/workspace/vipibench-env"))
ROOT.mkdir(parents=True, exist_ok=True)
ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

files_b64 = __FILES_B64__  # noqa: F821 - replaced in the release-bundle renderer
for relative_path, payload in files_b64.items():
    target = ROOT / relative_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(base64.b64decode(payload))

wheel = ROOT / "build/current-wheel/vipibench_guardrail-0.1.0-py3-none-any.whl"
authorization = ROOT / "data/release_decision.yaml"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


assert sha256(wheel) == "__WHEEL_SHA__"
assert sha256(authorization) == "__AUTH_SHA__"
print(
    "TRANSFER_VERIFIED",
    {
        "wheel_sha256": sha256(wheel),
        "authorization_sha256": sha256(authorization),
    },
    flush=True,
)

subprocess.check_call(
    [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--progress-bar",
        "off",
        "virtualenv==20.35.4",
    ]
)
subprocess.check_call([sys.executable, "-m", "virtualenv", "--clear", VENV])

python = VENV / "bin/python"
environment = os.environ.copy()
environment["VIPIBENCH_ARTIFACT_ROOT"] = str(ARTIFACT_ROOT)


def run(arguments: list[object]) -> None:
    print("RUN", " ".join(map(str, arguments)), flush=True)
    subprocess.check_call([str(argument) for argument in arguments], env=environment)


run(
    [
        python,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--progress-bar",
        "off",
        "-r",
        ROOT / "requirements-experiment.lock",
    ]
)
run(
    [
        python,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--progress-bar",
        "off",
        "--no-deps",
        wheel,
    ]
)
run([python, "-m", "pip", "check"])
run([python, "-m", "vipibench.cli", "validate-run-protocol"])
run([python, "-m", "vipibench.cli", "validate-target-protocol"])
run([python, "-m", "vipibench.cli", "validate-attack-search"])
run([python, "-m", "vipibench.cli", "validate-resource-estimate", "--project-root", ROOT])
run([python, "-m", "vipibench.cli", "check-notebook", ROOT / "notebooks/confirmatory_run.ipynb"])
print("WORKSPACE_BOOTSTRAP_PASS", flush=True)
