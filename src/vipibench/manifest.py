from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from vipibench.dataio import sha256_file, write_json

MANIFEST_ROOT_FILES = [
    Path("pyproject.toml"),
    Path("requirements.lock"),
    Path("requirements-experiment.lock"),
    Path("requirements-analysis.lock"),
    Path("README.md"),
    Path("LICENSE.md"),
    Path("PROJECT_STATE.md"),
    Path("DECISIONS.md"),
    Path("EVIDENCE.md"),
    Path(".env.example"),
    Path(".gitattributes"),
    Path(".gitignore"),
    Path("outputs/readiness_report.schema.json"),
    Path("outputs/prelaunch_readiness.schema.json"),
    # The post-run finalizer loads this schema at runtime; it is executable
    # payload, not merely an audit-side convenience file.
    Path("outputs/postrun_audit.schema.json"),
]

MANIFEST_SCHEMA_VERSION = "1.0.0"
MANIFEST_PROJECT = "vipibench-guardrail"
_MANIFEST_FIELDS = frozenset({"schema_version", "project", "generated_at", "status", "artifacts"})
_ARTIFACT_FIELDS = frozenset({"path", "sha256", "size_bytes"})


def default_manifest_paths(project_root: Path) -> list[Path]:
    paths = list(MANIFEST_ROOT_FILES)
    for directory in (
        "configs",
        "src/vipibench",
        "tests",
        "docs",
        "notebooks",
        "reports/thesis",
        "scripts",
    ):
        paths.extend(
            path.relative_to(project_root)
            for path in (project_root / directory).rglob("*")
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
        )
    paths.extend(
        [
            Path("data/provenance_ledger.yaml"),
            Path("data/release_decision.yaml"),
        ]
    )
    return sorted(set(paths), key=lambda path: path.as_posix())


def runtime_source_fingerprint(project_root: Path) -> str:
    paths = [
        Path("pyproject.toml"),
        Path("requirements.lock"),
        Path("requirements-experiment.lock"),
        Path("requirements-analysis.lock"),
    ]
    for directory in ("configs", "src/vipibench", "tests", "notebooks", "scripts"):
        paths.extend(
            path.relative_to(project_root)
            for path in (project_root / directory).rglob("*")
            if path.is_file() and not path.is_symlink() and "__pycache__" not in path.parts
        )
    paths.extend(
        [
            Path("outputs/readiness_report.schema.json"),
            Path("outputs/prelaunch_readiness.schema.json"),
        ]
    )
    payload = "\n".join(
        f"{path.as_posix()}:{sha256_file(project_root / path)}"
        for path in sorted(set(paths), key=lambda item: item.as_posix())
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def readiness_manifest_paths(project_root: Path) -> list[Path]:
    paths = default_manifest_paths(project_root)
    paths.extend(
        Path(path)
        for path in (
            "data/processed/vipibench_exec.jsonl",
            "data/processed/vipibench_exec_templates.jsonl",
            "data/processed/provenance_contrast.jsonl",
            "data/splits/frozen/train.jsonl",
            "data/splits/frozen/dev.jsonl",
            "data/splits/frozen/test.jsonl",
            "data/splits/frozen/split_manifest.json",
            "data/splits/frozen/split_audit.json",
            "data/splits/frozen/holdout_manifest.json",
            "data/splits/confirmatory_final/train.jsonl",
            "data/splits/confirmatory_final/dev.jsonl",
            "data/splits/confirmatory_final/test.jsonl",
            "data/splits/confirmatory_final/templates.jsonl",
            "data/splits/confirmatory_final/manifest.json",
            "outputs/exec_oracle_verification.json",
            "outputs/exec_composition_audit.json",
            "outputs/executable_benchmark_compile.json",
            "outputs/executable_benchmark_validation.json",
            "outputs/role_label_leakage.json",
            "outputs/template_generator_leakage.json",
            "outputs/policy_gate_verification.json",
            "outputs/four_arm_fixture_verification.json",
            "outputs/experiment_protocol_validation.json",
            "outputs/confirmatory_analysis_validation.json",
            "outputs/provenance_contrast_manifest.json",
            "outputs/provenance_contrast_audit.json",
            "outputs/provenance_verification.json",
            "outputs/training_authorization_verification.json",
            "outputs/secret_scan.json",
            "outputs/proposal_coverage_audit.json",
            "outputs/resource_estimate_validation.json",
            "outputs/full/tfidf/baseline_manifest.json",
        )
    )
    return sorted(set(paths), key=lambda path: path.as_posix())


def build_manifest(project_root: Path, paths: list[Path], output_path: Path) -> dict[str, object]:
    root = project_root.resolve()
    artifacts: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for candidate in sorted(paths, key=lambda path: path.as_posix()):
        path = _safe_manifest_member(root, candidate)
        if path is None:
            raise ValueError(
                f"manifest path must be a regular file within project root: {candidate}"
            )
        relative_path = path.relative_to(root).as_posix()
        if relative_path in seen_paths:
            raise ValueError(f"duplicate manifest path: {relative_path}")
        seen_paths.add(relative_path)
        artifacts.append(
            {
                "path": relative_path,
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "project": MANIFEST_PROJECT,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "PASS",
        "artifacts": artifacts,
    }
    write_json(output_path, manifest)
    return manifest


def verify_manifest(
    project_root: Path,
    manifest_path: Path,
    *,
    expected_paths: list[Path] | None = None,
) -> dict[str, object]:
    """Verify a manifest's schema, containment, completeness, and byte bindings.

    ``expected_paths`` is mandatory for readiness consumers.  Keeping it
    explicit preserves the generic helper for small unit-test manifests while
    ensuring the CLI/readiness/preflight paths reject an empty or partial
    project manifest.
    """

    root = project_root.resolve()
    errors: list[str] = []
    try:
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return {
            "status": "FAIL",
            "manifest": str(manifest_path),
            "artifact_count": 0,
            "errors": [f"manifest_invalid_json:{type(exc).__name__}"],
        }
    if not isinstance(value, dict):
        return {
            "status": "FAIL",
            "manifest": str(manifest_path),
            "artifact_count": 0,
            "errors": ["manifest_not_object"],
        }
    manifest = value
    if set(manifest) != _MANIFEST_FIELDS:
        errors.append("manifest_fields_invalid")
    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        errors.append("manifest_schema_version_invalid")
    if manifest.get("project") != MANIFEST_PROJECT:
        errors.append("manifest_project_invalid")
    if not _valid_generated_at(manifest.get("generated_at")):
        errors.append("manifest_generated_at_invalid")
    if manifest.get("status") != "PASS":
        errors.append("manifest_status_not_pass")
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        errors.append("manifest_artifacts_missing_or_empty")
        artifacts = []

    observed_paths: list[str] = []
    for index, artifact in enumerate(artifacts):
        if not isinstance(artifact, dict) or set(artifact) != _ARTIFACT_FIELDS:
            errors.append(f"artifact_fields_invalid:{index}")
            continue
        raw_path = artifact.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            errors.append(f"artifact_path_invalid:{index}")
            continue
        path = _safe_manifest_member(root, Path(raw_path))
        if path is None or path.relative_to(root).as_posix() != raw_path:
            errors.append(f"artifact_path_unsafe:{raw_path}")
            continue
        if raw_path in observed_paths:
            errors.append(f"artifact_path_duplicate:{raw_path}")
            continue
        observed_paths.append(raw_path)
        expected_hash = artifact.get("sha256")
        if not _is_sha256(expected_hash):
            errors.append(f"artifact_hash_invalid:{raw_path}")
        elif sha256_file(path) != expected_hash:
            errors.append(f"hash_mismatch:{raw_path}")
        expected_size = artifact.get("size_bytes")
        if (
            not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 0
        ):
            errors.append(f"artifact_size_invalid:{raw_path}")
        elif path.stat().st_size != expected_size:
            errors.append(f"size_mismatch:{raw_path}")

    if expected_paths is not None:
        expected = _expected_manifest_paths(root, expected_paths)
        if observed_paths != expected:
            errors.append("manifest_artifact_paths_mismatch")
    return {
        "status": "PASS" if not errors else "FAIL",
        "manifest": str(manifest_path),
        "artifact_count": len(artifacts),
        "errors": sorted(set(errors)),
    }


def _expected_manifest_paths(project_root: Path, paths: list[Path]) -> list[str]:
    expected: list[str] = []
    for candidate in sorted(paths, key=lambda path: path.as_posix()):
        path = _safe_manifest_member(project_root, candidate)
        if path is None:
            raise ValueError(f"expected manifest path unsafe or missing: {candidate}")
        relative = path.relative_to(project_root).as_posix()
        if relative in expected:
            raise ValueError(f"duplicate expected manifest path: {relative}")
        expected.append(relative)
    return expected


def _safe_manifest_member(project_root: Path, candidate: Path) -> Path | None:
    root = project_root.resolve()
    if not root.is_dir() or candidate.is_symlink():
        return None
    if candidate.is_absolute():
        path = candidate
    else:
        if ".." in candidate.parts:
            return None
        path = root / candidate
    if path.is_symlink() or not path.is_file():
        return None
    resolved = path.resolve()
    try:
        resolved.relative_to(root)
    except ValueError:
        return None
    return resolved


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789ABCDEF" for character in value)
    )


def _valid_generated_at(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return True
