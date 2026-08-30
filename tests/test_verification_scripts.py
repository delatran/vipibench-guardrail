import re
from pathlib import Path


def _default_venv_name(path: Path) -> str:
    source = path.read_text(encoding="utf-8-sig")
    match = re.search(r'\[string\]\$VenvName\s*=\s*"([^"]+)"', source)
    assert match is not None
    return match.group(1)


def test_clean_environment_and_current_wheel_share_default_venv() -> None:
    clean_environment = _default_venv_name(Path("scripts/verify_clean_environment.ps1"))
    current_wheel = _default_venv_name(Path("scripts/verify_current_wheel.ps1"))

    assert clean_environment == "clean-base-verify-venv"
    assert current_wheel == clean_environment
