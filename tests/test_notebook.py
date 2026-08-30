import ast
import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from vipibench.dataio import write_json
from vipibench.notebook_check import check_notebook, execute_smoke_cells

OUTER_NOTEBOOK_PATH = Path(__file__).parents[2] / "RUN_EXPERIMENT.ipynb"
requires_outer_notebook = pytest.mark.skipif(
    not OUTER_NOTEBOOK_PATH.is_file(),
    reason="private outer operator notebook is not present in this source checkout",
)


def _load_stream_helper():
    notebook = json.loads(
        OUTER_NOTEBOOK_PATH.read_text(encoding="utf-8")
    )
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code" and "def _run_visible" in "".join(cell["source"])
    )
    tree = ast.parse(source)
    constants = {
        "WORK_ROOT",
        "_HEARTBEAT_SECONDS",
        "_PROCESS_POLL_SECONDS",
        "_ANSI_ESCAPE",
        "_INTERNAL_CLAIM_CODE",
        "_STAGE_LABELS",
        "_CHECKPOINT_DIRS",
    }
    functions = {
        "_normalized_stream_line",
        "_reader_stage_label",
        "_reader_friendly_line",
        "_is_dynamic_progress",
        "_checkpoint_summary",
        "_restore_phase_hint",
        "_emit_coalesced_progress",
        "_emit_stream_heartbeat",
        "_run_visible",
    }
    selected = []
    for node in tree.body:
        if (
            isinstance(node, (ast.Import, ast.ImportFrom))
            or (
                isinstance(node, ast.Assign)
                and any(
                    isinstance(target, ast.Name) and target.id in constants
                    for target in node.targets
                )
            )
            or (isinstance(node, ast.FunctionDef) and node.name in functions)
        ):
            selected.append(node)
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=selected, type_ignores=[]), "<launcher-cell>", "exec"), namespace)
    return namespace["_run_visible"]


@requires_outer_notebook
def test_operator_notebook_binds_all_controller_commands_to_its_fresh_work_root() -> None:
    notebook = json.loads(
        OUTER_NOTEBOOK_PATH.read_text(encoding="utf-8")
    )
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert (
        "WORK_ROOT = Path('/content/vipibench_run_response_truncation_guard_2026_08_12')"
        in source
    )
    assert "Path('/content/vipibench_run/run-output')" not in source
    assert source.count("'--work-root', str(WORK_ROOT)") == 4
    assert source.count("'--bundle', BUNDLE") == 5
    assert "--verify-upload-inventory" in source
    assert "--stage-upload-payload" in source
    assert "--prune-upload-pollutants" in source
    assert "USE_LOCAL_BUNDLE_MIRROR" in source
    assert "_print_resume_guide" in source
    assert "_verify_launch_bindings" in source
    assert source.count("'--verify-upload-inventory'") >= 2
    assert source.count("'--prune-upload-pollutants'") >= 2
    assert "shutil.copytree(DRIVE_BUNDLE_PATH" not in source


def _load_checkpointed_stage_helper(namespace: dict[str, object]):
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    source = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if (
            cell.get("cell_type") == "code"
            and "def run_checkpointed_stage" in "".join(cell["source"])
        )
    )
    tree = ast.parse(source)
    function_names = (
        "stage_reader_label",
        "reader_safe_failure_line",
        "stage_failure_receipts",
        "print_stage_failure_tail",
        "run_checkpointed_stage",
    )
    stage_functions = [
        next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == name
        )
        for name in function_names
    ]
    exec(
        compile(
            ast.Module(body=stage_functions, type_ignores=[]),
            "<checkpointed-stage>",
            "exec",
        ),
        namespace,
    )
    return namespace["run_checkpointed_stage"]


@pytest.mark.parametrize(
    "path",
    [
        Path("notebooks/experiment_workflow.ipynb"),
        Path("notebooks/confirmatory_run.ipynb"),
    ],
)
def test_notebook_structure_and_smoke_cells(path: Path) -> None:
    result = check_notebook(path)
    assert result["status"] == "PASS"
    executed = execute_smoke_cells(path, Path.cwd())
    assert executed == {"status": "PASS", "executed_smoke_cells": 2, "mode": "smoke"}


def test_launch_notebook_rejects_editable_install(tmp_path: Path) -> None:
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    install_cell = next(
        cell for cell in notebook["cells"] if "install" in cell.get("metadata", {}).get("tags", [])
    )
    install_cell["source"] = ['subprocess.check_call(["pip", "install", "-e", "."])\n']
    path = tmp_path / "confirmatory_run.ipynb"
    path.write_text(json.dumps(notebook), encoding="utf-8")

    result = check_notebook(path)

    assert result["status"] == "FAIL"
    assert "editable_install_forbidden_for_launch_notebook" in result["errors"]


def test_launch_notebook_does_not_replace_dependency_wheels_in_live_kernel() -> None:
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    install_cell = next(
        cell for cell in notebook["cells"] if "install" in cell.get("metadata", {}).get("tags", [])
    )
    source = "".join(install_cell["source"])

    assert "RUNTIME_PROBE_COMMAND" in source
    assert 'LOCK_PATH = PROJECT_ROOT / "requirements-experiment.lock"' in source
    assert "requirements-analysis.lock" not in source
    assert 'scripts" / "prepare_colab.py' in source
    assert '"--probe-only"' in source
    assert "PROJECT_SOURCE_ROOT" in source
    assert "project_source_import_mismatch" in source
    assert "import vipibench" in source
    assert "PROJECT_INSTALL_COMMAND" not in source
    assert "LOCK_INSTALL_COMMAND" not in source
    assert '"--requirement"' not in source
    assert "--allow-unexpected-packages" not in source


def test_confirmatory_notebook_checkpoints_all_analyses_before_terminal_status() -> None:
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    evidence_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "postrun-audit" in cell.get("metadata", {}).get("tags", [])
    )

    assert "--strict-capacity-receipt" in source
    assert "consolidate-runtime-telemetry" in source
    assert 'candidate_dataset.with_suffix(".validity.json")' in source
    for command in (
        "analyze-static-system",
        "analyze-rq2-diagnostics",
        "analyze-h3",
        "analyze-adaptive-search",
        "build-postrun-raw-manifests",
        "audit-postrun",
        "finalize-confirmatory-run",
        "materialize-report-assets",
    ):
        assert command in evidence_cell
    assert "write_json(\n        RUN_MANIFEST_PATH" not in evidence_cell
    assert 'final_run_manifest["RESEARCH_EVIDENCE_ELIGIBLE"]' in evidence_cell


def test_confirmatory_notebook_uses_stable_stage_bindings_and_fresh_session_a100() -> None:
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    source = "\n".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )
    runtime_cell = next(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if "def run_checkpointed_stage" in "".join(cell.get("source", []))
    )
    checkpoint_function = runtime_cell.split(
        "def run_checkpointed_stage", maxsplit=1
    )[1].split("def complete_public_stage", maxsplit=1)[0]

    assert "checkpoint_metadata(command, RUN_BINDING)" in checkpoint_function
    assert '"preflight_sha256": sha256_file(PREFLIGHT_PATH)' not in checkpoint_function
    assert 'metadata["accelerator_sha256"]' not in checkpoint_function
    assert "subprocess.check_call(ACCELERATOR_COMMAND, cwd=PROJECT_ROOT)" in runtime_cell
    assert 'SESSION_EVIDENCE_ROOT / "runtime_sessions"' in source
    assert 'DEPENDENCY_PROFILE != "accelerator"' in source
    assert 'DEPENDENCY_PROFILE == "accelerator"' in runtime_cell
    assert "write_cpu_analysis_transition_receipt" not in runtime_cell
    assert "load_bound_accelerator_preflight" not in source
    assert "check-analysis-runtime" not in source
    assert "analysis-cpu" not in source
    assert "minimum_free_disk_gib = 80.0" in runtime_cell
    assert "strict_capacity_receipt_sha256" in runtime_cell
    assert 'SESSION_RUNTIME_ROOT / "accelerator_capacity_check.json"' in runtime_cell
    assert "verify_runtime_storage_plan" in runtime_cell
    assert "VIPIBENCH_RUNTIME_STORAGE_PLAN_PATH" in runtime_cell
    assert "VIPIBENCH_RUNTIME_STORAGE_PLAN_FILE_SHA256" in runtime_cell
    assert '"session_temp": Path(str(runtime_storage_plan["session_temp_root"]))' in runtime_cell
    assert '"ephemeral": Path(str(runtime_storage_plan["ephemeral_root"]))' in runtime_cell
    assert "if not STRICT_CAPACITY_RECEIPT_PATH.is_file():" in runtime_cell
    assert 'complete_public_stage("preflight",direct_outputs=[' in "".join(
        runtime_cell.split()
    )
    compact_source = "".join(source.split())
    for public_stage in (
        "data",
        "baselines",
        "encoder",
        "core",
        "attack-generate",
        "attack-evaluate",
        "analysis",
        "finalize",
    ):
        assert f'complete_public_stage("{public_stage}"' in compact_source


def test_confirmatory_notebook_persists_checkpointed_stage_failure_diagnostics(
    tmp_path: Path,
) -> None:
    class IncompleteLedger:
        def verified_complete(self, stage_id, metadata):
            return False

        def complete(self, stage_id, outputs, metadata):
            raise AssertionError("a failed stage must not be marked complete")

    output_root = tmp_path / "run-output"
    run_binding = {
        "runtime_source_fingerprint": "A" * 64,
        "launch_hashes": {"artifact_manifest": "B" * 64},
        "launch_authorization_sha256": "C" * 64,
        "stage_plan_sha256": "D" * 64,
        "protocol_amendment": "a100-80gb-hardware-only-2026-08-08",
        "durable_lineage": "staged-test-lineage",
    }
    namespace: dict[str, object] = {
        "subprocess": subprocess,
        "json": json,
        "PROJECT_ROOT": tmp_path,
        "OUTPUT_ROOT": output_root,
        "write_json": write_json,
        "ORCHESTRATION_LEDGER": IncompleteLedger(),
        "STAGE_GROUP_LEDGER": object(),
        "MEMBER_TO_PUBLIC_STAGE": {"synthetic-stage": "synthetic-public-stage"},
        "SELECTED_STAGE": "synthetic-public-stage",
        "STAGE_PLAN": {},
        "RUN_BINDING": run_binding,
        "stage_enabled": lambda selected, stage: selected == stage,
        "require_stage_prerequisite": lambda **kwargs: None,
        "checkpoint_metadata": lambda command, binding: {
            "command": [str(value) for value in command],
            **binding,
        },
    }
    run_stage = _load_checkpointed_stage_helper(namespace)

    with pytest.raises(RuntimeError, match="Không hoàn tất bước xử lý thí nghiệm"):
        run_stage(
            "synthetic-stage",
            [
                sys.executable,
                "-c",
                "import sys; print('synthetic root cause', file=sys.stderr); raise SystemExit(7)",
            ],
            [],
        )

    receipt = json.loads(
        (output_root / "stage_failures" / "synthetic-stage.json").read_text(encoding="utf-8")
    )
    assert receipt["status"] == "FAIL"
    assert receipt["returncode"] == 7
    assert "synthetic root cause" in receipt["output_tail"]


def test_confirmatory_notebook_exposes_failed_dataloader_worker_measurements(
    tmp_path: Path,
) -> None:
    class IncompleteLedger:
        def verified_complete(self, stage_id, metadata):
            return False

        def complete(self, stage_id, outputs, metadata):
            raise AssertionError("a failed stage must not be marked complete")

    output_root = tmp_path / "run-output"
    worker_plan_path = output_root / "mdeberta" / "dataloader_worker_plan.json"
    write_json(
        worker_plan_path,
        {
            "status": "FAIL",
            "measurements": [
                {
                    "num_workers": 2,
                    "status": "FAIL",
                    "error_type": "ValueError",
                    "error_message": "dataloader scout batch has no labels",
                }
            ],
        },
    )
    namespace: dict[str, object] = {
        "subprocess": subprocess,
        "json": json,
        "PROJECT_ROOT": tmp_path,
        "OUTPUT_ROOT": output_root,
        "write_json": write_json,
        "ORCHESTRATION_LEDGER": IncompleteLedger(),
        "STAGE_GROUP_LEDGER": object(),
        "MEMBER_TO_PUBLIC_STAGE": {"encoder-matrix": "encoder"},
        "SELECTED_STAGE": "encoder",
        "STAGE_PLAN": {},
        "RUN_BINDING": {
            "runtime_source_fingerprint": "A" * 64,
            "launch_hashes": {"artifact_manifest": "B" * 64},
            "launch_authorization_sha256": "C" * 64,
            "stage_plan_sha256": "D" * 64,
            "protocol_amendment": "a100-80gb-hardware-only-2026-08-08",
            "durable_lineage": "staged-test-lineage",
        },
        "stage_enabled": lambda selected, stage: selected == stage,
        "require_stage_prerequisite": lambda **kwargs: None,
        "checkpoint_metadata": lambda command, binding: {
            "command": [str(value) for value in command],
            **binding,
        },
    }
    run_stage = _load_checkpointed_stage_helper(namespace)

    with pytest.raises(RuntimeError, match="Không hoàn tất bước xử lý thí nghiệm"):
        run_stage(
            "encoder-matrix",
            [sys.executable, "-c", "raise SystemExit(7)"],
            [],
        )

    receipt = json.loads(
        (output_root / "stage_failures" / "encoder-matrix.json").read_text(encoding="utf-8")
    )
    related = receipt["related_receipts"]
    assert len(related) == 1
    assert related[0]["path"] == str(worker_plan_path)
    assert related[0]["status"] == "FAIL"
    assert related[0]["measurements"][0]["error_message"] == (
        "dataloader scout batch has no labels"
    )


@pytest.mark.parametrize(
    ("stage_id", "public_stage", "receipt_name"),
    [
        ("core-target-trajectories", "core", "core_target_trajectories.run.json"),
        (
            "attack-target-trajectories",
            "attack-evaluate",
            "attack_target_trajectories.run.json",
        ),
    ],
)
def test_confirmatory_notebook_presents_target_failure_without_internal_claim_codes(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    stage_id: str,
    public_stage: str,
    receipt_name: str,
) -> None:
    class IncompleteLedger:
        def verified_complete(self, stage_id, metadata):
            return False

        def complete(self, stage_id, outputs, metadata):
            raise AssertionError("a failed stage must not be marked complete")

    output_root = tmp_path / "run-output"
    write_json(
        output_root / receipt_name,
        {
            "status": "FAIL",
            "errors": ["target_run_failures_observed"],
            "format_failure_summary": {
                "status": "FAIL",
                "format_fallback_count": 8,
                "parse_failure_count": 8,
                "parse_error_class_counts": {"response_shape_error": 8},
                "raw_response_included": False,
            },
            "target_format_fail_fast": {
                "recorded_episode_count": 8,
                "total_episode_count": 480,
                "unprocessed_episode_count": 472,
                "additional_model_batches_after_trigger": 0,
            },
            "claim_dispositions": {
                "RQ3": "INCONCLUSIVE_FORMAT_FALLBACK",
                "H3": "INCONCLUSIVE_FORMAT_FALLBACK",
                "H4": "INCONCLUSIVE_FORMAT_FALLBACK",
            },
        },
    )
    namespace: dict[str, object] = {
        "subprocess": subprocess,
        "json": json,
        "PROJECT_ROOT": tmp_path,
        "OUTPUT_ROOT": output_root,
        "write_json": write_json,
        "ORCHESTRATION_LEDGER": IncompleteLedger(),
        "STAGE_GROUP_LEDGER": object(),
        "MEMBER_TO_PUBLIC_STAGE": {stage_id: public_stage},
        "SELECTED_STAGE": public_stage,
        "STAGE_PLAN": {},
        "RUN_BINDING": {
            "runtime_source_fingerprint": "A" * 64,
            "launch_hashes": {"artifact_manifest": "B" * 64},
            "launch_authorization_sha256": "C" * 64,
            "stage_plan_sha256": "D" * 64,
            "protocol_amendment": "fixture-amendment",
            "durable_lineage": "fixture-lineage",
        },
        "stage_enabled": lambda selected, stage: selected == stage,
        "require_stage_prerequisite": lambda **kwargs: None,
        "checkpoint_metadata": lambda command, binding: {
            "command": [str(value) for value in command],
            **binding,
        },
    }
    run_stage = _load_checkpointed_stage_helper(namespace)

    with pytest.raises(RuntimeError, match="sinh quỹ đạo phản hồi"):
        run_stage(
            stage_id,
            [
                sys.executable,
                "-c",
                (
                    "print({'claim_dispositions': {'RQ3': 'INCONCLUSIVE_FORMAT_FALLBACK', "
                    "'H3': 'INCONCLUSIVE_FORMAT_FALLBACK', "
                    "'H4': 'INCONCLUSIVE_FORMAT_FALLBACK'}}); raise SystemExit(1)"
                ),
            ],
            [],
        )

    visible = capsys.readouterr().out
    assert "Thí nghiệm đã dừng vì phản hồi của mô hình không đúng cấu trúc JSON" in visible
    assert "Số phản hồi sai cấu trúc: 8; số mẫu chưa xử lý: 472." in visible
    assert "Lần chạy này chưa đủ điều kiện để rút ra kết luận nghiên cứu." in visible
    assert "RQ3" not in visible
    assert "H3" not in visible
    assert "H4" not in visible
    assert "claim_dispositions" not in visible
    technical_receipt = json.loads(
        (output_root / "stage_failures" / f"{stage_id}.json").read_text(
            encoding="utf-8"
        )
    )
    assert "RQ3" in technical_receipt["output_tail"]


def test_confirmatory_notebook_names_response_truncation_as_the_stopping_cause(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class IncompleteLedger:
        def verified_complete(self, stage_id, metadata):
            return False

        def complete(self, stage_id, outputs, metadata):
            raise AssertionError("a failed stage must not be marked complete")

    stage_id = "attack-target-trajectories"
    public_stage = "attack-evaluate"
    output_root = tmp_path / "run-output"
    write_json(
        output_root / "attack_target_trajectories.run.json",
        {
            "status": "FAIL",
            "errors": [
                "final_holdout_format_fallback_observed",
                "target_response_truncation_observed",
                "target_run_failures_observed",
            ],
            "format_failure_summary": {
                "status": "FAIL",
                "format_fallback_count": 3,
                "parse_failure_count": 30,
                "parse_error_class_counts": {"response_truncation_error": 3},
                "truncated_response_count": 3,
                "response_token_ceiling": 4096,
                "response_token_ceiling_reached_count": 3,
                "raw_response_included": False,
            },
            "target_format_fail_fast": {
                "recorded_episode_count": 2624,
                "total_episode_count": 4800,
                "unprocessed_episode_count": 2176,
                "additional_model_batches_after_trigger": 0,
            },
        },
    )
    namespace: dict[str, object] = {
        "subprocess": subprocess,
        "json": json,
        "PROJECT_ROOT": tmp_path,
        "OUTPUT_ROOT": output_root,
        "write_json": write_json,
        "ORCHESTRATION_LEDGER": IncompleteLedger(),
        "STAGE_GROUP_LEDGER": object(),
        "MEMBER_TO_PUBLIC_STAGE": {stage_id: public_stage},
        "SELECTED_STAGE": public_stage,
        "STAGE_PLAN": {},
        "RUN_BINDING": {
            "runtime_source_fingerprint": "A" * 64,
            "launch_hashes": {"artifact_manifest": "B" * 64},
            "launch_authorization_sha256": "C" * 64,
            "stage_plan_sha256": "D" * 64,
            "protocol_amendment": "fixture-amendment",
            "durable_lineage": "fixture-lineage",
        },
        "stage_enabled": lambda selected, stage: selected == stage,
        "require_stage_prerequisite": lambda **kwargs: None,
        "checkpoint_metadata": lambda command, binding: {
            "command": [str(value) for value in command],
            **binding,
        },
    }
    run_stage = _load_checkpointed_stage_helper(namespace)

    with pytest.raises(RuntimeError, match="sinh quỹ đạo phản hồi"):
        run_stage(stage_id, [sys.executable, "-c", "raise SystemExit(1)"], [])

    visible = capsys.readouterr().out
    assert "bị cắt ngang khi đạt giới hạn" in visible
    assert "Số phản hồi bị cắt ngang: 3; giới hạn token đầu ra hiện tại: 4096." in visible
    assert "Tổng số phản hồi không dùng được: 30; số mẫu chưa xử lý: 2176." in visible
    assert "Số phản hồi sai cấu trúc" not in visible
    assert "không đúng cấu trúc JSON" not in visible


@requires_outer_notebook
def test_launch_notebook_reports_exited_parent_even_when_descendant_keeps_stdout() -> None:
    run_visible = _load_stream_helper()
    parent_code = "\n".join(
        [
            "import subprocess, sys",
            (
                "subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(2)'], "
                "stdout=sys.stdout, stderr=sys.stderr)"
            ),
            "print('parent exiting', flush=True)",
            "raise SystemExit(7)",
        ]
    )
    started = time.monotonic()

    with pytest.raises(RuntimeError, match="mã thoát 7"):
        run_visible(
            "inherited-pipe-regression",
            [sys.executable, "-c", parent_code],
            heartbeat_seconds=0.2,
            process_poll_seconds=0.05,
        )

    assert time.monotonic() - started < 1.0


@requires_outer_notebook
def test_launch_notebook_stream_uses_reader_friendly_failure_terms(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_visible = _load_stream_helper()
    machine_output = (
        "RQ3 H3 H4 INCONCLUSIVE_FORMAT_FALLBACK core-target-trajectories"
    )

    with pytest.raises(RuntimeError, match="Đánh giá phản hồi mô hình trên tập kiểm tra"):
        run_visible(
            "confirmatory_core",
            [sys.executable, "-c", f"print({machine_output!r}); raise SystemExit(1)"],
            heartbeat_seconds=0.2,
            process_poll_seconds=0.05,
        )

    visible = capsys.readouterr().out
    assert "RQ3" not in visible
    assert "H3" not in visible
    assert "H4" not in visible
    assert "INCONCLUSIVE_FORMAT_FALLBACK" not in visible
    assert "core-target-trajectories" not in visible
    assert "kết luận nghiên cứu" in visible
    assert "CHƯA THỂ KẾT LUẬN DO ĐẦU RA SAI CẤU TRÚC" in visible
    assert "sinh quỹ đạo phản hồi của mô hình" in visible


@requires_outer_notebook
def test_launch_notebook_stream_does_not_rewrite_lowercase_dependency_names(
    capsys: pytest.CaptureFixture[str],
) -> None:
    run_visible = _load_stream_helper()

    run_visible(
        "runtime_prepare",
        [sys.executable, "-c", "print('Collecting h11==0.16.0')"],
        heartbeat_seconds=0.2,
        process_poll_seconds=0.05,
    )

    visible = capsys.readouterr().out
    assert "Collecting h11==0.16.0" in visible
    assert "kết luận nghiên cứu==0.16.0" not in visible


def test_public_detector_stage_binds_its_capacity_plan() -> None:
    notebook = json.loads(Path("notebooks/confirmatory_run.ipynb").read_text(encoding="utf-8"))
    source = "".join(
        "".join(cell.get("source", []))
        for cell in notebook["cells"]
        if cell.get("cell_type") == "code"
    )

    assert 'public_root / "capacity_plan.json"' in source
