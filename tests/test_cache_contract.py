from pathlib import Path

import pytest

from vipibench.cache_contract import (
    CACHE_MARKER_NAME,
    CACHE_MARKER_TEXT,
    reset_ephemeral_model_cache,
)


def test_cache_reset_requires_exact_ownership_marker(tmp_path: Path) -> None:
    cache = tmp_path / "vipibench-model-cache"
    cache.mkdir()
    (cache / "foreign.bin").write_text("preserve", encoding="utf-8")

    with pytest.raises(PermissionError, match="ownership marker"):
        reset_ephemeral_model_cache(cache)

    assert (cache / "foreign.bin").read_text(encoding="utf-8") == "preserve"


def test_cache_reset_is_bounded_to_owned_root(tmp_path: Path) -> None:
    sibling = tmp_path / "do-not-delete.txt"
    sibling.write_text("preserve", encoding="utf-8")
    cache = tmp_path / "vipibench-model-cache"
    (cache / "hub" / "models--qwen").mkdir(parents=True)
    (cache / "hub" / "models--qwen" / "weight.bin").write_text("fixture", encoding="utf-8")
    marker = cache / CACHE_MARKER_NAME
    marker.write_text(CACHE_MARKER_TEXT, encoding="utf-8")

    result = reset_ephemeral_model_cache(cache)

    assert result["status"] == "PASS"
    assert result["removed_entries"] == ["hub"]
    assert marker.read_text(encoding="utf-8") == CACHE_MARKER_TEXT
    assert sibling.read_text(encoding="utf-8") == "preserve"
