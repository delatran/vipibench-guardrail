"""Fail-closed materialization of report-ready figures from accepted run evidence."""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from collections.abc import Mapping
from pathlib import Path

from vipibench.dataio import sha256_file, write_json
from vipibench.manifest import runtime_source_fingerprint
from vipibench.postrun_audit import (
    POSTRUN_AUDIT_FILENAME,
    REQUIRED_ARTIFACTS,
    validate_postrun_audit_for_finalization,
)

SCHEMA_VERSION = "1.0.0"
PACKAGE_DIRECTORY = "bao_cao_hinh_anh"
ASSET_DIRECTORY = "hinh_bao_cao"
ARCHIVE_FILENAME = "goi_hinh_bao_cao.zip"
MANIFEST_FILENAME = "report_assets_manifest.json"
CATALOG_FILENAME = "DANH_MUC_HINH.json"
CAPTION_FILENAME = "CHU_THICH_HINH.md"
VISIBLE_CODE_PATTERN = r"(?i)\b(?:rq|rg|h)\s*[-_]?\d+\b"
VISIBLE_VERSION_PATTERN = r"(?i)\bv\d+(?:\.\d+)*\b"
REQUIRED_REPORT_SOURCES = {
    "encoder_ablation": ("encoder_ablation", REQUIRED_ARTIFACTS["encoder_ablation"]),
    "diagnostic_analysis": ("rq2_analysis", REQUIRED_ARTIFACTS["rq2_analysis"]),
    "static_analysis": ("static_analysis", REQUIRED_ARTIFACTS["static_analysis"]),
    "joint_analysis": ("h3_analysis", REQUIRED_ARTIFACTS["h3_analysis"]),
    "adaptive_analysis": ("adaptive_analysis", REQUIRED_ARTIFACTS["adaptive_analysis"]),
    "runtime_telemetry": ("runtime_telemetry", REQUIRED_ARTIFACTS["runtime_telemetry"]),
}


def materialize_report_assets(
    *, project_root: Path, output_root: Path
) -> dict[str, object]:
    """Create one atomic figure package only from independently accepted evidence."""

    root = project_root.resolve()
    run_root = output_root.resolve()
    errors: list[str] = []
    if not root.is_dir() or project_root.is_symlink():
        errors.append("project_root_missing_or_unsafe")
    if not run_root.is_dir() or output_root.is_symlink():
        errors.append("output_root_missing_or_unsafe")
    if errors:
        return _failure(errors)

    audit_path = run_root / POSTRUN_AUDIT_FILENAME
    try:
        audit, audit_errors = validate_postrun_audit_for_finalization(
            project_root=root,
            output_root=run_root,
            postrun_audit_path=audit_path,
        )
    except Exception as exc:  # boundary converts verifier diagnostics into stage evidence
        return _failure([f"postrun_audit_revalidation_error:{type(exc).__name__}:{exc}"])
    errors.extend(f"postrun_audit:{error}" for error in audit_errors)

    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = _load_object(run_manifest_path, "run_manifest", errors)
    _validate_terminal_acceptance(run_manifest, audit, audit_path, errors)
    artifacts, source_sha256 = _load_report_sources(run_root, audit, errors)
    if errors:
        return _failure(errors)

    input_sha256 = {
        "run_manifest.json": sha256_file(run_manifest_path),
        POSTRUN_AUDIT_FILENAME: sha256_file(audit_path),
        **source_sha256,
        "src/vipibench/report_assets.py": sha256_file(Path(__file__).resolve()),
        "src/vipibench/report_figures.py": sha256_file(
            Path(__file__).with_name("report_figures.py").resolve()
        ),
    }
    package_root = run_root / PACKAGE_DIRECTORY
    try:
        reused = _reuse_existing_package(package_root, input_sha256)
    except (OSError, ValueError) as exc:
        return _failure([f"existing_report_package_verification_error:{type(exc).__name__}:{exc}"])
    if reused is not None:
        if reused.get("status") == "PASS":
            return {
                **reused,
                "cache_reused": True,
                "package_root": str(package_root),
                "manifest_path": str(package_root / MANIFEST_FILENAME),
                "archive_path": str(package_root / ARCHIVE_FILENAME),
            }
        return reused

    staging = Path(tempfile.mkdtemp(prefix=f".{PACKAGE_DIRECTORY}.tmp-", dir=run_root))
    published = False
    try:
        from vipibench.report_figures import render_report_figures

        asset_root = staging / ASSET_DIRECTORY
        catalog = render_report_figures(artifacts, asset_root)
        write_json(
            asset_root / CATALOG_FILENAME,
            {
                "status": "HOAN_THANH",
                "figure_count": len(catalog),
                "figures": catalog,
                "reader_note": (
                    "Mỗi hình đi kèm bảng CSV chứa đúng các giá trị đã dùng để dựng hình. "
                    "Chú thích cần được đối chiếu với văn bản báo cáo trước khi nộp."
                ),
            },
        )
        (asset_root / CAPTION_FILENAME).write_text(
            _caption_guide(catalog), encoding="utf-8", newline="\n"
        )
        _validate_presentation_outputs(asset_root, catalog)
        archive_path = staging / ARCHIVE_FILENAME
        _write_deterministic_archive(asset_root, archive_path)
        entries = _package_entries(staging, exclude={MANIFEST_FILENAME})
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "status": "PASS",
            "errors": [],
            "runtime_source_fingerprint": runtime_source_fingerprint(root),
            "input_sha256": dict(sorted(input_sha256.items())),
            "figure_count": len(catalog),
            "formats_per_figure": ["png", "svg", "pdf", "csv"],
            "package_entries": entries,
            "archive_sha256": sha256_file(archive_path),
            "presentation_contract": {
                "language": "formal_academic_vietnamese",
                "internal_research_codes_visible": False,
                "notebook_version_label_visible": False,
                "caption_catalog": f"{ASSET_DIRECTORY}/{CAPTION_FILENAME}",
                "data_catalog": f"{ASSET_DIRECTORY}/{CATALOG_FILENAME}",
            },
            "claim_boundary": (
                "The package renders already accepted estimates and intervals without "
                "recomputing estimands or promoting statistical conclusions. A passing package "
                "is presentation evidence only, not a new scientific result."
            ),
            "external_actions_performed": [],
        }
        write_json(staging / MANIFEST_FILENAME, manifest)
        if package_root.exists() or package_root.is_symlink():
            return _failure(["report_package_target_appeared_during_materialization"])
        staging.replace(package_root)
        published = True
        return {
            **manifest,
            "cache_reused": False,
            "package_root": str(package_root),
            "manifest_path": str(package_root / MANIFEST_FILENAME),
            "archive_path": str(package_root / ARCHIVE_FILENAME),
        }
    except Exception as exc:  # rendering failures must fail the finalize stage cleanly
        return _failure([f"report_asset_materialization_error:{type(exc).__name__}:{exc}"])
    finally:
        if not published and staging.exists() and staging.parent == run_root:
            shutil.rmtree(staging)


def _validate_terminal_acceptance(
    run_manifest: Mapping[str, object],
    audit: Mapping[str, object],
    audit_path: Path,
    errors: list[str],
) -> None:
    if (
        run_manifest.get("status") != "PASS"
        or run_manifest.get("errors") != []
        or run_manifest.get("RUN_COMPLETE") != "PASS"
        or run_manifest.get("RESEARCH_EVIDENCE_ELIGIBLE") is not True
        or run_manifest.get("final_claim_audit_created") is not True
    ):
        errors.append("terminal_run_manifest_not_eligible")
    if audit_path.is_file() and run_manifest.get("postrun_audit_sha256") != sha256_file(audit_path):
        errors.append("terminal_run_manifest_audit_hash_mismatch")
    dispositions = audit.get("dispositions")
    if (
        audit.get("status") != "PASS"
        or audit.get("errors") != []
        or audit.get("fixture_only") is not False
        or audit.get("research_evidence_eligible") is not True
        or not isinstance(dispositions, Mapping)
        or dispositions.get("RUN_COMPLETE") != "PASS"
        or dispositions.get("RESEARCH_EVIDENCE_ELIGIBLE") is not True
    ):
        errors.append("postrun_audit_not_eligible_for_report_assets")


def _load_report_sources(
    run_root: Path,
    audit: Mapping[str, object],
    errors: list[str],
) -> tuple[dict[str, Mapping[str, object]], dict[str, str]]:
    audit_hashes = audit.get("artifact_sha256")
    if not isinstance(audit_hashes, Mapping):
        errors.append("postrun_audit_artifact_hashes_missing")
        audit_hashes = {}
    artifacts: dict[str, Mapping[str, object]] = {}
    source_sha256: dict[str, str] = {}
    for report_name, (audit_name, relative_path) in REQUIRED_REPORT_SOURCES.items():
        path = run_root / relative_path
        artifact = _load_object(path, report_name, errors)
        if not artifact:
            continue
        digest = sha256_file(path)
        if audit_hashes.get(audit_name) != digest:
            errors.append(f"report_source_hash_mismatch:{report_name}")
        if report_name == "runtime_telemetry":
            if (
                artifact.get("validation_status") != "PASS"
                or artifact.get("hardware_observed") is not True
                or artifact.get("local_only") is not False
                or artifact.get("execution_status") != "completed"
            ):
                errors.append("runtime_telemetry_not_live_complete")
        elif (
            artifact.get("status") != "PASS"
            or artifact.get("errors") != []
            or artifact.get("research_claim_eligible") is not True
        ):
            errors.append(f"report_source_not_eligible:{report_name}")
        artifacts[report_name] = artifact
        source_sha256[relative_path] = digest
    return artifacts, source_sha256


def _reuse_existing_package(
    package_root: Path, input_sha256: Mapping[str, str]
) -> dict[str, object] | None:
    if not package_root.exists() and not package_root.is_symlink():
        return None
    if not package_root.is_dir() or package_root.is_symlink():
        return _failure(["existing_report_package_is_unsafe"])
    errors: list[str] = []
    manifest = _load_object(package_root / MANIFEST_FILENAME, "report_assets_manifest", errors)
    if errors:
        return _failure(["existing_report_package_conflict", *errors])
    if (
        manifest.get("schema_version") != SCHEMA_VERSION
        or manifest.get("status") != "PASS"
        or manifest.get("errors") != []
        or manifest.get("input_sha256") != dict(sorted(input_sha256.items()))
    ):
        return _failure(["existing_report_package_conflict"])
    expected_entries = manifest.get("package_entries")
    if not isinstance(expected_entries, list) or not expected_entries:
        return _failure(["existing_report_package_manifest_entries_invalid"])
    observed = _package_entries(package_root, exclude={MANIFEST_FILENAME})
    if observed != expected_entries:
        return _failure(["existing_report_package_file_hash_mismatch"])
    archive_path = package_root / ARCHIVE_FILENAME
    if not archive_path.is_file() or archive_path.is_symlink():
        return _failure(["existing_report_package_archive_missing_or_unsafe"])
    if manifest.get("archive_sha256") != sha256_file(archive_path):
        return _failure(["existing_report_package_archive_hash_mismatch"])
    return dict(manifest)


def _validate_presentation_outputs(
    asset_root: Path, catalog: list[dict[str, object]]
) -> None:
    forbidden_code = re.compile(VISIBLE_CODE_PATTERN)
    forbidden_version = re.compile(VISIBLE_VERSION_PATTERN)
    visible_text = json.dumps(catalog, ensure_ascii=False)
    for path in sorted(asset_root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"presentation output cannot be a symlink: {path.name}")
        if path.suffix.lower() in {".svg", ".csv", ".md", ".json"}:
            visible_text += "\n" + path.read_text(encoding="utf-8-sig")
    if forbidden_code.search(visible_text):
        raise ValueError("presentation output contains an internal research code")
    if forbidden_version.search(visible_text):
        raise ValueError("presentation output contains a version-style label")
    extensions = [path.suffix.lower() for path in asset_root.iterdir() if path.is_file()]
    for suffix in (".png", ".svg", ".pdf", ".csv"):
        if extensions.count(suffix) != len(catalog):
            raise ValueError(f"presentation output count mismatch for {suffix}")


def _caption_guide(catalog: list[dict[str, object]]) -> str:
    lines = [
        "# DANH MỤC VÀ CHÚ THÍCH HÌNH ĐỀ XUẤT",
        "",
        (
            "Các chú thích dưới đây được tạo từ kết quả đã hậu kiểm. Trước khi nộp, cần đặt "
            "hình vào đúng chương, đối chiếu số thứ tự với mục lục và biên tập lại theo mẫu "
            "chính thức."
        ),
        "",
    ]
    for item in catalog:
        lines.extend(
            [
                f"## Hình {item['number']}. {item['title']}",
                "",
                f"**Chú thích đề xuất:** {item['caption']}",
                "",
                f"**Vị trí đề xuất:** {item['recommended_section']}",
                "",
                f"**Nguồn dữ liệu:** {item['data_source']}",
                "",
                f"**Đơn vị:** {item['unit']}",
                "",
                f"**Cỡ mẫu:** {item['sample_size']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Ranh giới diễn giải",
            "",
            (
                "Hình trực quan hóa các ước lượng đã có; hình không thay thế bảng số liệu, kiểm "
                "định thống kê, phân tích độ nhạy hoặc thảo luận về giới hạn của thiết kế."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def _write_deterministic_archive(asset_root: Path, archive_path: Path) -> None:
    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(asset_root.rglob("*")):
            if not path.is_file() or path.is_symlink():
                continue
            relative = path.relative_to(asset_root).as_posix()
            info = zipfile.ZipInfo(relative, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            info.create_system = 3
            archive.writestr(info, path.read_bytes(), compresslevel=9)
    if not archive_path.is_file() or archive_path.stat().st_size <= 0:
        raise RuntimeError("report archive was not created")


def _package_entries(package_root: Path, *, exclude: set[str]) -> list[dict[str, object]]:
    entries: list[dict[str, object]] = []
    for path in sorted(package_root.rglob("*")):
        if not path.is_file() or path.name in exclude:
            continue
        if path.is_symlink():
            raise ValueError(f"report package cannot contain a symlink: {path.name}")
        entries.append(
            {
                "path": path.relative_to(package_root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not entries:
        raise ValueError("report package is empty")
    return entries


def _load_object(path: Path, label: str, errors: list[str]) -> dict[str, object]:
    if not path.is_file() or path.is_symlink():
        errors.append(f"{label}_missing_or_unsafe")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"{label}_invalid_json:{type(exc).__name__}")
        return {}
    if not isinstance(value, dict):
        errors.append(f"{label}_not_object")
        return {}
    return value


def _failure(errors: list[str]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "FAIL",
        "errors": sorted(set(errors)),
        "figure_count": 0,
        "claim_boundary": (
            "No report figure is eligible when terminal evidence, source hashes, rendering, or "
            "presentation-safety validation fails."
        ),
        "external_actions_performed": [],
    }
