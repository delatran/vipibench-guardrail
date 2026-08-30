from __future__ import annotations

import re
from pathlib import Path

import yaml

INACTIVE_ACCELERATOR_TERMS = (
    "nvidia l4",
    "h100",
    "l40",
    "rtx 4090",
    "tesla t4",
)
ACTIVE_DEVICE_NAMES = [
    "NVIDIA A100-SXM4-80GB",
    "NVIDIA A100-PCIE-80GB",
    "NVIDIA A100 80GB",
]


def _text_files(root: str) -> list[Path]:
    suffixes = {".py", ".toml", ".yaml", ".yml", ".json", ".ipynb"}
    return [
        path
        for path in Path(root).rglob("*")
        if path.is_file() and path.suffix.casefold() in suffixes
    ]


def _scrub_hashes(text: str) -> str:
    return re.sub(r"[0-9a-f]{8,}", " ", text.casefold())


def test_runtime_contract_contains_no_inactive_accelerator_sku() -> None:
    violations: list[str] = []
    for root in ("src", "scripts", "notebooks", "configs"):
        for path in _text_files(root):
            text = _scrub_hashes(path.read_text(encoding="utf-8"))
            for term in INACTIVE_ACCELERATOR_TERMS:
                if term in text:
                    violations.append(f"{path}:{term}")
    assert not violations, violations


def test_active_profile_names_exact_registered_devices() -> None:
    profile = yaml.safe_load(
        Path("configs/profiles/accelerator_80gb.yaml").read_text(encoding="utf-8")
    )
    assert profile["required_device_names"] == ACTIVE_DEVICE_NAMES


def test_inactive_skus_are_confined_to_runtime_negative_controls() -> None:
    allowed_names = {"test_runtime_capacity.py", Path(__file__).name}
    violations: list[str] = []
    for path in _text_files("tests"):
        if path.name in allowed_names:
            continue
        text = _scrub_hashes(path.read_text(encoding="utf-8"))
        for term in INACTIVE_ACCELERATOR_TERMS:
            if term in text:
                violations.append(f"{path}:{term}")
    assert not violations, violations


def test_current_facing_documents_use_the_active_a100_limits() -> None:
    documents = {
        "README.md": Path("README.md").read_text(encoding="utf-8"),
        "docs/implementation_contract.md": Path(
            "docs/implementation_contract.md"
        ).read_text(encoding="utf-8"),
    }
    combined = "\n".join(documents.values())
    for required in (
        "a100_80gb-response-truncation-guard-v8-2026-08-12",
        "compute capability 8.0",
        "70-82 GiB",
    ):
        assert required in combined
    for forbidden in (
        "ACCELERATED_24GB",
        "at most 26 GiB",
        "CUDA 8.9",
        "21–26 GiB",
        "## 24 GB accelerator contract",
        "ready to launch an authorized 24 GB accelerator run",
    ):
        assert forbidden not in combined
