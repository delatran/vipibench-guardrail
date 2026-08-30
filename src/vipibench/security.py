from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path

from vipibench.dataio import canonical_json, sha256_file, write_json

PATTERNS = {
    "openai_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "huggingface_token": re.compile(r"\bhf_[A-Za-z0-9]{20,}\b"),
    "aws_access_key": re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    "private_key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "email_address": re.compile(
        r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        r"(?![\w-]|\.[A-Za-z0-9])"
    ),
    "vietnam_phone": re.compile(r"(?<![A-Za-z0-9])(?:\+?84|0)(?:3|5|7|8|9)\d{8}(?![A-Za-z0-9])"),
}
TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".json",
    ".jsonl",
    ".yaml",
    ".yml",
    ".toml",
    ".txt",
    ".ipynb",
    ".csv",
}
EXCLUDED_PARTS = {
    ".git",
    ".venv",
    "build",
    "outputs",
    "__pycache__",
    ".pytest_cache",
    ".ruff_cache",
}
EXCLUDED_DIRECTORY_SUFFIXES = {".egg-info"}
EXCLUDED_RELATIVE_PATHS = {
    "artifact_manifest.json",
    "data/release_decision.yaml",
}


def scan_secrets(root: Path, output_path: Path | None = None) -> dict[str, object]:
    findings: list[dict[str, object]] = []
    scanned_files: list[dict[str, str]] = []
    for current, directories, filenames in os.walk(root):
        directories[:] = sorted(
            directory
            for directory in directories
            if directory not in EXCLUDED_PARTS
            and not any(
                directory.casefold().endswith(suffix)
                for suffix in EXCLUDED_DIRECTORY_SUFFIXES
            )
        )
        for filename in sorted(filenames):
            path = Path(current) / filename
            relative_path = path.relative_to(root).as_posix()
            if path.suffix.casefold() not in TEXT_SUFFIXES:
                continue
            if relative_path in EXCLUDED_RELATIVE_PATHS:
                continue
            scanned_files.append({"path": relative_path, "sha256": sha256_file(path)})
            text = path.read_text(encoding="utf-8", errors="replace")
            for name, pattern in PATTERNS.items():
                for match in pattern.finditer(text):
                    findings.append(
                        {
                            "path": relative_path,
                            "line": text.count("\n", 0, match.start()) + 1,
                            "pattern": name,
                        }
                    )
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not findings else "FAIL",
        "scanned_file_count": len(scanned_files),
        "scanned_file_set_sha256": hashlib.sha256(
            canonical_json(scanned_files).encode("utf-8")
        ).hexdigest().upper(),
        "scan_policy": {
            "text_suffixes": sorted(TEXT_SUFFIXES),
            "excluded_directory_names": sorted(EXCLUDED_PARTS),
            "excluded_directory_suffixes": sorted(EXCLUDED_DIRECTORY_SUFFIXES),
            "excluded_relative_paths": sorted(EXCLUDED_RELATIVE_PATHS),
        },
        "findings": findings,
        "note": (
            "Secret and direct-identifier pattern scan reduces risk but is not proof that no "
            "sensitive data exists."
        ),
    }
    if output_path:
        write_json(output_path, result)
    return result
