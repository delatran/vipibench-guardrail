from __future__ import annotations

import hashlib
from pathlib import Path

from vipibench.dataio import canonical_json, sha256_file


def directory_fingerprint(path: Path) -> str:
    root = path.resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"artifact directory missing: {root}")
    artifacts = [item for item in sorted(root.rglob("*")) if item.is_file()]
    if not artifacts:
        raise FileNotFoundError(f"artifact directory is empty: {root}")
    bindings = [
        {"path": item.relative_to(root).as_posix(), "sha256": sha256_file(item)}
        for item in artifacts
    ]
    return hashlib.sha256(canonical_json(bindings).encode()).hexdigest().upper()
