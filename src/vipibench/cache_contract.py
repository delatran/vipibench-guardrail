from __future__ import annotations

import shutil
from pathlib import Path

CACHE_MARKER_NAME = ".vipibench-owned-cache"
CACHE_MARKER_TEXT = "vipibench-owned-ephemeral-model-cache"


def reset_ephemeral_model_cache(cache_root: Path) -> dict[str, object]:
    """Purge only a marker-owned cache root while preserving the root and its marker."""

    root = cache_root.resolve()
    if root == Path(root.anchor) or len(root.parts) < 2:
        raise ValueError(f"unsafe model cache root: {root}")
    marker = root / CACHE_MARKER_NAME
    if not marker.is_file() or marker.read_text(encoding="utf-8") != CACHE_MARKER_TEXT:
        raise PermissionError("model-cache cleanup requires the exact ownership marker")
    removed: list[str] = []
    for child in root.iterdir():
        if child == marker:
            continue
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)
        else:
            raise ValueError(f"unsupported cache entry: {child}")
        removed.append(child.name)
    return {
        "status": "PASS",
        "cache_root": str(root),
        "removed_entries": sorted(removed),
        "ownership_marker_preserved": marker.is_file(),
    }
