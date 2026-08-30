"""Verify and refresh a release bundle without starting confirmatory execution."""

import base64
import hashlib
import os
import pathlib
import subprocess

ROOT = pathlib.Path(
    os.environ.get("VIPIBENCH_WORKSPACE_ROOT", "/workspace/vipibench-guardrail")
)
PYTHON = pathlib.Path(
    os.environ.get("VIPIBENCH_ENVIRONMENT_PYTHON", "/workspace/vipibench-env/bin/python")
)

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
    "REFRESH_TRANSFER_VERIFIED",
    {
        "wheel_sha256": sha256(wheel),
        "authorization_sha256": sha256(authorization),
    },
    flush=True,
)

environment = os.environ.copy()


def run(arguments: list[object]) -> None:
    print("RUN", " ".join(map(str, arguments)), flush=True)
    subprocess.check_call([str(argument) for argument in arguments], env=environment)


run(
    [
        PYTHON,
        "-m",
        "pip",
        "install",
        "--no-input",
        "--no-deps",
        "--force-reinstall",
        wheel,
    ]
)
run([PYTHON, "-m", "pip", "check"])
run([PYTHON, "-m", "vipibench.cli", "validate-run-protocol"])
run([PYTHON, "-m", "vipibench.cli", "validate-target-protocol"])
run([PYTHON, "-m", "vipibench.cli", "validate-attack-search"])
run([PYTHON, "-m", "vipibench.cli", "validate-resource-estimate", "--project-root", ROOT])
run([PYTHON, "-m", "vipibench.cli", "check-notebook", ROOT / "notebooks/confirmatory_run.ipynb"])
print("WORKSPACE_REFRESH_PASS", flush=True)
