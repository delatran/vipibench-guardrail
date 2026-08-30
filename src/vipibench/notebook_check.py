from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

from vipibench.dataio import sha256_file, write_json

REQUIRED_TAGS = {
    "configuration",
    "install",
    "readiness",
    "smoke",
    "runtime",
    "data-build",
    "development",
    "model-selection",
    "target-trajectories",
    "attack-search",
    "checkpoint-resume",
}
NOTEBOOK_REQUIRED_TAGS = {
    "confirmatory_run.ipynb": {
        "cache-contract",
        "analysis",
        "environment",
        "evidence",
        "hash-verification",
        "postrun-audit",
    }
}
REQUIRED_TEXT = {
    "VIPIBENCH_PROJECT_ROOT",
    "VIPIBENCH_DATASET_PATH",
    "VIPIBENCH_SPLIT_DIR",
    "VIPIBENCH_OUTPUT_ROOT",
    "VIPIBENCH_MODE",
    "VIPIBENCH_CONFIRMATORY_RUN_APPROVED",
    "compile-provenance-contrast",
    "audit-provenance-contrast",
    "run-public-detector-benchmark",
    "run-tfidf-baseline",
    "analyze-encoder-ablations",
    "run-target-agent",
    "evaluate-four-arms",
    "generate-attack-candidates",
    "evaluate-attack-search",
}

NOTEBOOK_REQUIRED_TEXT = {
    "experiment_workflow.ipynb": {
        "evaluate_launch_readiness",
        "check-runtime",
        "accelerator_80gb.yaml",
        "run-encoder-matrix",
    },
    "confirmatory_run.ipynb": {
        "requirements-experiment.lock",
        "run-encoder-accelerator-matrix",
        "run-encoder-test-analysis",
        "project_source_import_mismatch",
        "verify-environment-compatibility",
        "preflight",
        "--verify-hash",
        "--runtime-environment-compatibility",
        "--active-isolated-runtime",
        "check-accelerator",
        "accelerator_80gb.yaml",
        'DEPENDENCY_PROFILE != "accelerator"',
        "Every public stage requires the accelerator dependency profile",
        "VIPIBENCH_DURABLE_OUTPUT_CONFIRMED",
        "VIPIBENCH_UPLOAD_AUTHORIZED",
        "VIPIBENCH_PAID_COMPUTE_AUTHORIZED",
        "VIPIBENCH_LAUNCH_AUTHORIZATION_PATH",
        "VIPIBENCH_LAUNCH_AUTHORIZATION_SHA256",
        "VIPIBENCH_RESUME_EXISTING_RUN",
        "VIPIBENCH_STAGE",
        "VIPIBENCH_STAGE_PLAN_PATH",
        "VIPIBENCH_STAGE_PLAN_SHA256",
        "VIPIBENCH_EPHEMERAL_MODEL_CACHE_ROOT",
        "VIPIBENCH_RUNTIME_STORAGE_PLAN_PATH",
        "VIPIBENCH_RUNTIME_STORAGE_PLAN_SHA256",
        "VIPIBENCH_RUNTIME_STORAGE_PLAN_FILE_SHA256",
        "accelerator_capacity_check.json",
        "reset_ephemeral_model_cache",
        "launch_record.json",
        "run_manifest.json",
        "NVIDIA A100-SXM4-80GB",
        "build-strict-capacity-receipt",
        "consolidate-runtime-telemetry",
        "analyze-static-system",
        "analyze-rq2-diagnostics",
        "analyze-h3",
        "analyze-adaptive-search",
        "prepare-postrun-supporting-evidence",
        "write-postrun-run-context",
        "build-postrun-raw-manifests",
        "audit-postrun",
        "finalize-confirmatory-run",
        "materialize-report-assets",
        "stage-selection-skip",
        "complete_public_stage",
    },
}


def check_notebook(path: Path, output_path: Path | None = None) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if notebook.get("nbformat") != 4:
        errors.append("notebook_format_must_be_4")
    cells = notebook.get("cells", [])
    tags = {tag for cell in cells for tag in cell.get("metadata", {}).get("tags", [])}
    required_tags = REQUIRED_TAGS | NOTEBOOK_REQUIRED_TAGS.get(path.name, set())
    missing_tags = sorted(required_tags - tags)
    if missing_tags:
        errors.append(f"missing_tags:{','.join(missing_tags)}")
    text = "\n".join("".join(cell.get("source", [])) for cell in cells)
    required_text = REQUIRED_TEXT | NOTEBOOK_REQUIRED_TEXT.get(path.name, set())
    for required in sorted(required_text):
        if required not in text:
            errors.append(f"missing_contract_text:{required}")
    if "drive.mount" in text or "files.upload(" in text:
        errors.append("automatic_upload_or_external_mount_forbidden")
    if path.name == "confirmatory_run.ipynb":
        install_text = "\n".join(
            "".join(cell.get("source", []))
            for cell in cells
            if "install" in cell.get("metadata", {}).get("tags", [])
        )
        if '"-e"' in install_text or "'-e'" in install_text:
            errors.append("editable_install_forbidden_for_launch_notebook")
        ordered_tags = [
            "configuration",
            "install",
            "environment",
            "hash-verification",
            "runtime",
            "data-build",
            "development",
            "model-selection",
            "target-trajectories",
            "attack-search",
            "evidence",
            "analysis",
            "postrun-audit",
        ]
        positions = {
            tag: min(
                index
                for index, cell in enumerate(cells)
                if tag in cell.get("metadata", {}).get("tags", [])
            )
            for tag in ordered_tags
            if tag in tags
        }
        if len(positions) == len(ordered_tags) and list(positions.values()) != sorted(
            positions.values()
        ):
            errors.append("launch_stage_order_invalid")
    for index, cell in enumerate(cells):
        if cell.get("cell_type") == "code":
            source = "".join(cell.get("source", []))
            try:
                ast.parse(source, filename=f"{path.name}:cell_{index}")
            except SyntaxError as exc:
                errors.append(f"invalid_python:cell_{index}:line_{exc.lineno}")
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(f"notebook_must_ship_clean:cell_{index}")
    result = {
        "schema_version": "1.0.0",
        "status": "PASS" if not errors else "FAIL",
        "path": str(path),
        "sha256": sha256_file(path),
        "cell_count": len(cells),
        "tags": sorted(tags),
        "errors": errors,
    }
    if output_path:
        write_json(output_path, result)
    return result


def execute_smoke_cells(path: Path, project_root: Path) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    namespace: dict[str, object] = {"__name__": "__vipibench_notebook_smoke__"}
    executed = 0
    old_cwd = Path.cwd()
    import os

    previous_mode = os.environ.get("VIPIBENCH_MODE")
    previous_root = os.environ.get("VIPIBENCH_PROJECT_ROOT")
    previous_runtime_python = os.environ.get("VIPIBENCH_RUNTIME_PYTHON")
    previous_python_utf8 = os.environ.get("PYTHONUTF8")
    try:
        os.chdir(project_root)
        os.environ["VIPIBENCH_MODE"] = "smoke"
        os.environ["VIPIBENCH_PROJECT_ROOT"] = str(project_root)
        os.environ["VIPIBENCH_RUNTIME_PYTHON"] = str(Path(sys.executable).absolute())
        os.environ["PYTHONUTF8"] = "1"
        for cell in notebook["cells"]:
            tags = cell.get("metadata", {}).get("tags", [])
            if "smoke-executable" in tags:
                exec("".join(cell["source"]), namespace)
                executed += 1
    finally:
        os.chdir(old_cwd)
        if previous_mode is None:
            os.environ.pop("VIPIBENCH_MODE", None)
        else:
            os.environ["VIPIBENCH_MODE"] = previous_mode
        if previous_root is None:
            os.environ.pop("VIPIBENCH_PROJECT_ROOT", None)
        else:
            os.environ["VIPIBENCH_PROJECT_ROOT"] = previous_root
        if previous_runtime_python is None:
            os.environ.pop("VIPIBENCH_RUNTIME_PYTHON", None)
        else:
            os.environ["VIPIBENCH_RUNTIME_PYTHON"] = previous_runtime_python
        if previous_python_utf8 is None:
            os.environ.pop("PYTHONUTF8", None)
        else:
            os.environ["PYTHONUTF8"] = previous_python_utf8
    return {"status": "PASS", "executed_smoke_cells": executed, "mode": "smoke"}
