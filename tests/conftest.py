from pathlib import Path

import pytest

PRIVATE_INTEGRATION_SENTINELS = (
    Path("artifact_manifest.json"),
    Path("outputs/executable_benchmark_compile.json"),
    Path("outputs/confirmatory_analysis_validation.json"),
    Path("outputs/training_authorization_verification.json"),
)


def _private_integration_available(root: Path) -> bool:
    return all((root / path).is_file() for path in PRIVATE_INTEGRATION_SENTINELS)


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if _private_integration_available(Path(str(config.rootpath))):
        return

    skip_private_integration = pytest.mark.skip(
        reason=(
            "requires the private integration bundle and generated evidence; "
            "not included in a source-only checkout"
        )
    )
    for item in items:
        if "private_integration" in item.keywords:
            item.add_marker(skip_private_integration)
