from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from vipibench import __version__
from vipibench.ablation_analysis import analyze_encoder_ablations
from vipibench.adaptive_analysis import analyze_adaptive_search_from_artifacts
from vipibench.adaptive_runner import (
    evaluate_attack_search,
    generate_attack_candidates,
    validate_attack_search_config,
)
from vipibench.analysis_protocol import validate_confirmatory_analysis_protocol
from vipibench.baseline_runner import run_tfidf_baseline
from vipibench.compiler import (
    compile_catalog_path,
    compile_confirmatory_holdout_path,
    verify_confirmatory_holdout_package,
)
from vipibench.coverage import audit_proposal_coverage
from vipibench.dataio import write_json
from vipibench.environment_compatibility import (
    verify_analysis_environment_compatibility,
    verify_environment_compatibility,
)
from vipibench.episode import export_episode_schema
from vipibench.exec_splits import (
    audit_exec_splits,
    freeze_exec_splits,
    seal_frozen_split_package,
)
from vipibench.exec_validation import validate_exec_benchmark
from vipibench.experiment_protocol import validate_exec_experiment_protocol
from vipibench.h3_analysis import analyze_h3_from_artifacts
from vipibench.manifest import build_manifest, readiness_manifest_paths, verify_manifest
from vipibench.metrics import calibrate_thresholds, evaluate_predictions
from vipibench.notebook_check import check_notebook
from vipibench.oracle_verification import verify_oracle_fixtures
from vipibench.policy_gate import verify_policy_gate
from vipibench.postrun_audit import audit_postrun, build_postrun_raw_manifests
from vipibench.postrun_preparation import (
    finalize_confirmatory_run,
    prepare_postrun_supporting_evidence,
    write_postrun_run_context,
)
from vipibench.precision_scout import (
    evaluate_precision_scout,
    run_precision_scout,
    validate_precision_scout_protocol,
)
from vipibench.preflight import evaluate_confirmatory_launch_readiness
from vipibench.provenance import verify_provenance, verify_training_authorization
from vipibench.provenance_contrast import (
    audit_provenance_contrast_path,
    compile_provenance_contrast,
)
from vipibench.readiness import evaluate_launch_readiness
from vipibench.report_assets import materialize_report_assets
from vipibench.resource_estimate import validate_resource_estimate
from vipibench.rq2_analysis import analyze_rq2_diagnostics
from vipibench.run_protocol import validate_encoder_protocol, validate_public_detector_protocol
from vipibench.runtime_capacity import check_runtime_profile_path
from vipibench.runtime_telemetry import (
    build_strict_capacity_receipt,
    consolidate_live_telemetry_ledgers,
)
from vipibench.runtime_transition import check_analysis_cpu_runtime_path
from vipibench.sample_size import validate_sample_size_protocol
from vipibench.security import scan_secrets
from vipibench.shortcut_audit import audit_exec_shortcuts
from vipibench.system_analysis import analyze_static_system
from vipibench.system_runner import evaluate_four_arms_from_predictions, verify_four_arm_fixture
from vipibench.target_runner import run_target_agent, validate_target_protocol
from vipibench.transformer_runner import (
    run_encoder_accelerator_matrix,
    run_encoder_matrix,
    run_encoder_test_analysis_matrix,
    run_encoder_test_prediction_matrix,
    run_public_detector,
    run_public_detector_benchmark,
)

app = typer.Typer(
    no_args_is_help=True,
    help="ViPIBench-Exec local engineering and readiness gates.",
)


def _emit(result: dict[str, object], *, fail_closed: bool = True) -> None:
    # ASCII-safe JSON keeps the CLI reliable under legacy Windows console encodings.
    typer.echo(json.dumps(result, ensure_ascii=True, indent=2))
    if fail_closed and result.get("status") != "PASS":
        raise typer.Exit(code=1)


@app.command("version")
def version() -> None:
    """Print the package version."""

    typer.echo(__version__)


@app.command("doctor")
def doctor(
    project_root: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    """Check the revised proposal binding and active coverage contract."""

    root = project_root.resolve()
    coverage = audit_proposal_coverage(root, root / "docs/proposal_coverage.yaml")
    _emit(
        {
            "schema_version": "2.0.0",
            "status": coverage["status"],
            "project_root": str(root),
            "proposal_coverage": coverage,
            "note": "A PASS is an engineering-contract check, not launch readiness.",
        }
    )


@app.command("audit-proposal-coverage")
def audit_proposal_coverage_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    manifest: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    root = project_root.resolve()
    manifest_path = manifest or (root / "docs/proposal_coverage.yaml")
    _emit(audit_proposal_coverage(root, manifest_path, output))


@app.command("check-runtime")
def check_runtime(
    profile: Annotated[Path, typer.Option()] = Path("configs/profiles/accelerator_80gb.yaml"),
    disk_path: Annotated[Path, typer.Option()] = Path("."),
) -> None:
    _emit(check_runtime_profile_path(profile, disk_path))


@app.command("check-accelerator")
def check_accelerator(
    profile: Annotated[Path, typer.Option()] = Path("configs/profiles/accelerator_80gb.yaml"),
    disk_path: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Fail closed unless the observed runtime satisfies the strict accelerator launch contract."""

    result = check_runtime_profile_path(profile, disk_path)
    if output is not None:
        write_json(output, result)
    _emit(result)


@app.command("check-analysis-runtime")
def check_analysis_runtime(
    profile: Annotated[Path, typer.Option()] = Path("configs/profiles/analysis_cpu.yaml"),
    disk_path: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Require an observed no-accelerator runtime for analysis or finalization."""

    result = check_analysis_cpu_runtime_path(profile, disk_path)
    if output is not None:
        write_json(output, result)
    _emit(result)


@app.command("verify-readiness")
def verify_readiness_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path | None, typer.Option()] = Path("outputs/readiness_report.json"),
) -> None:
    """Evaluate fail-closed local readiness for confirmatory execution."""

    _emit(evaluate_launch_readiness(project_root, output_path=output))


@app.command("preflight")
def preflight_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    verify_hash: Annotated[bool, typer.Option("--verify-hash")] = False,
    output: Annotated[Path, typer.Option()] = Path("outputs/prelaunch_readiness.json"),
    runtime_environment_compatibility: Annotated[Path | None, typer.Option()] = None,
    active_isolated_runtime: Annotated[bool, typer.Option()] = False,
) -> None:
    """Verify the current hash-bound package before any paid-compute or hosted launch action."""

    _emit(
        evaluate_confirmatory_launch_readiness(
            project_root,
            verify_hash=verify_hash,
            output_path=output,
            runtime_environment_compatibility_path=runtime_environment_compatibility,
            active_isolated_runtime=active_isolated_runtime,
        )
    )


@app.command("validate-resource-estimate")
def validate_resource_estimate_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path("configs/resources/resource_estimate.yaml"),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/resource_estimate_validation.json"
    ),
) -> None:
    """Validate pinned storage bytes, workload arithmetic, and timing scenarios."""

    _emit(validate_resource_estimate(project_root, config, output))


@app.command("validate-confirmatory-analysis")
def validate_confirmatory_analysis_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path(
        "configs/experiments/confirmatory_analysis.yaml"
    ),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/confirmatory_analysis_validation.json"
    ),
) -> None:
    """Verify the pretest final-holdout, MDE, power, and multiplicity lock."""

    _emit(validate_confirmatory_analysis_protocol(project_root, config, output))


@app.command("build-artifact-manifest")
def build_artifact_manifest_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path, typer.Option()] = Path("artifact_manifest.json"),
) -> None:
    """Build the hash manifest for source, frozen data, and deterministic evidence."""

    root = project_root.resolve()
    _emit(build_manifest(root, readiness_manifest_paths(root), root / output))


@app.command("verify-artifact-manifest")
def verify_artifact_manifest_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    manifest: Annotated[Path, typer.Option()] = Path("artifact_manifest.json"),
) -> None:
    """Verify every artifact bound by the readiness manifest."""

    root = project_root.resolve()
    _emit(
        verify_manifest(
            root,
            root / manifest,
            expected_paths=readiness_manifest_paths(root),
        )
    )


@app.command("verify-environment-compatibility")
def verify_environment_compatibility_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path, typer.Option()] = Path("outputs/environment_compatibility.json"),
    allow_unexpected_packages: Annotated[bool, typer.Option()] = False,
) -> None:
    """Verify pinned packages, public artifacts, and trainer API compatibility."""

    _emit(
        verify_environment_compatibility(
            project_root,
            output,
            allow_unexpected_packages=allow_unexpected_packages,
        )
    )


@app.command("verify-analysis-environment-compatibility")
def verify_analysis_environment_compatibility_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path, typer.Option()] = Path(
        "outputs/analysis_environment_compatibility.json"
    ),
    allow_unexpected_packages: Annotated[bool, typer.Option()] = False,
) -> None:
    """Verify the exact CPU-only analysis dependency profile."""

    _emit(
        verify_analysis_environment_compatibility(
            project_root,
            output,
            allow_unexpected_packages=allow_unexpected_packages,
        )
    )


@app.command("validate-run-protocol")
def validate_run_protocol(
    encoder: Annotated[Path, typer.Option()] = Path("configs/models/mdeberta_core.yaml"),
    public_detector: Annotated[Path, typer.Option()] = Path("configs/models/public_detector.yaml"),
) -> None:
    encoder_result = validate_encoder_protocol(encoder)
    public_result = validate_public_detector_protocol(public_detector)
    _emit(
        {
            "status": (
                "PASS" if encoder_result["status"] == public_result["status"] == "PASS" else "FAIL"
            ),
            "encoder": encoder_result,
            "public_detector": public_result,
        }
    )


@app.command("run-tfidf-baseline")
def run_tfidf_baseline_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path("configs/models/tfidf_core.yaml"),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/full/tfidf/baseline_manifest.json"
    ),
) -> None:
    """Train and evaluate the deterministic frozen TF-IDF baseline."""

    _emit(run_tfidf_baseline(config, project_root=project_root, output_path=output))


@app.command("run-public-detector")
def run_public_detector_command(
    dataset: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    split: Annotated[str, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/models/public_detector.yaml"),
) -> None:
    """Run the pinned public detector on one frozen split."""

    if split not in {"dev", "test"}:
        raise typer.BadParameter("split must be dev or test")
    _emit(run_public_detector(config, dataset, output, split=split))


@app.command("run-public-detector-benchmark")
def run_public_detector_benchmark_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/public_detector.yaml"),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest/public_detector"),
) -> None:
    """Evaluate the pinned public detector on the merged frozen benchmark tracks."""

    _emit(run_public_detector_benchmark(config, split_dir, output_root))


@app.command("validate-target-protocol")
def validate_target_protocol_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/target_agent.yaml"),
) -> None:
    _emit(validate_target_protocol(config))


@app.command("run-target-agent")
def run_target_agent_command(
    dataset: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    checkpoint_dir: Annotated[Path, typer.Option()],
    config: Annotated[Path, typer.Option()] = Path("configs/models/target_agent.yaml"),
    strict_capacity_receipt: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Generate hash-bound, oracle-blind target-agent trajectories."""

    _emit(
        run_target_agent(
            config_path=config,
            dataset_path=dataset,
            output_path=output,
            checkpoint_dir=checkpoint_dir,
            strict_capacity_receipt_path=strict_capacity_receipt,
        )
    )


@app.command("build-strict-capacity-receipt")
def build_strict_capacity_receipt_command(
    runtime_check: Annotated[Path, typer.Option()],
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path, typer.Option()] = Path("outputs/latest/strict_capacity_receipt.json"),
) -> None:
    """Bind one observed strict-accelerator runtime check for all measured run stages."""

    raw = json.loads(runtime_check.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise typer.BadParameter("runtime check must be a JSON object")
    receipt = build_strict_capacity_receipt(raw, project_root=project_root)
    write_json(output, receipt)
    _emit({"status": "PASS", "errors": [], "output": str(output), "receipt": receipt})


@app.command("consolidate-runtime-telemetry")
def consolidate_runtime_telemetry_command(
    telemetry: Annotated[list[Path], typer.Option()],
    strict_capacity_receipt: Annotated[Path, typer.Option()],
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path, typer.Option()] = Path("outputs/latest/runtime_telemetry.json"),
) -> None:
    """Fail closed unless retained target-stage ledgers share one strict accelerator receipt."""

    ledger = consolidate_live_telemetry_ledgers(
        telemetry,
        strict_capacity_receipt_path=strict_capacity_receipt,
        output_path=output,
        project_root=project_root,
    )
    _emit(
        {
            "status": ledger["validation_status"],
            "errors": [],
            "output": str(output),
            "runtime_telemetry": ledger,
        }
    )


@app.command("calibrate-thresholds")
def calibrate_thresholds_command(
    predictions: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Calibrate probabilities and fixed-FPR thresholds from development predictions only."""

    _emit(calibrate_thresholds(predictions, output))


@app.command("evaluate-predictions")
def evaluate_predictions_command(
    predictions: Annotated[Path, typer.Option()],
    thresholds: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
) -> None:
    """Evaluate frozen test predictions against development-only thresholds."""

    _emit(evaluate_predictions(predictions, thresholds, output))


@app.command("run-encoder-matrix")
def run_encoder_matrix_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/mdeberta_core.yaml"),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/full/mdeberta"),
) -> None:
    """Run the hash-resumable three-mode by three-seed mDeBERTa matrix."""

    _emit(run_encoder_matrix(config, split_dir, output_root))


@app.command("validate-precision-scout")
def validate_precision_scout_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path(
        "configs/models/mdeberta_bf16_scout.yaml"
    ),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Validate the development-only FP32-versus-BF16 A100 scout protocol."""

    _emit(validate_precision_scout_protocol(project_root, config, output))


@app.command("run-precision-scout")
def run_precision_scout_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path(
        "configs/models/mdeberta_bf16_scout.yaml"
    ),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/confirmatory_final"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/precision_scout"),
) -> None:
    """Run an explicitly authorized, development-only paired precision scout on A100."""

    _emit(run_precision_scout(project_root, config, split_dir, output_root))


@app.command("evaluate-precision-scout")
def evaluate_precision_scout_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path(
        "configs/models/mdeberta_bf16_scout.yaml"
    ),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/precision_scout"),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/precision_scout/precision_scout_evaluation.json"
    ),
) -> None:
    """Evaluate retained scout evidence without granting automatic BF16 promotion."""

    _emit(evaluate_precision_scout(project_root, config, output_root, output))


@app.command("run-encoder-accelerator-matrix")
def run_encoder_accelerator_matrix_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/mdeberta_core.yaml"),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/full/mdeberta"),
) -> None:
    """Run encoder development, selection, and test prediction without CPU analysis."""

    _emit(run_encoder_accelerator_matrix(config, split_dir, output_root))


@app.command("run-encoder-test-predictions")
def run_encoder_test_predictions_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/mdeberta_core.yaml"),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/full/mdeberta"),
) -> None:
    """Run only the accelerator-dependent encoder test predictions."""

    _emit(run_encoder_test_prediction_matrix(config, split_dir, output_root))


@app.command("run-encoder-test-analysis")
def run_encoder_test_analysis_command(
    config: Annotated[Path, typer.Option()] = Path("configs/models/mdeberta_core.yaml"),
    split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/full/mdeberta"),
) -> None:
    """Analyze hash-bound encoder predictions without requiring an accelerator."""

    _emit(run_encoder_test_analysis_matrix(config, split_dir, output_root))


@app.command("analyze-encoder-ablations")
def analyze_encoder_ablations_command(
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest/mdeberta"),
    output: Annotated[Path, typer.Option()] = Path("outputs/latest/encoder_ablation_analysis.json"),
) -> None:
    """Estimate paired detector-mode effects on the frozen provenance contrast set."""

    _emit(analyze_encoder_ablations(output_root, output))


@app.command("validate-attack-search")
def validate_attack_search_command(
    config: Annotated[Path, typer.Option()] = Path("configs/generation/adaptive_generator.yaml"),
) -> None:
    _emit(validate_attack_search_config(config))


@app.command("generate-attack-candidates")
def generate_attack_candidates_command(
    detector_model_dir: Annotated[Path, typer.Option()],
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_dataset: Annotated[Path, typer.Option()] = Path(
        "outputs/latest/attack_candidates.jsonl"
    ),
    output_scores: Annotated[Path, typer.Option()] = Path(
        "outputs/latest/attack_candidate_scores.jsonl"
    ),
    checkpoint_dir: Annotated[Path, typer.Option()] = Path(
        "outputs/latest/attack_search_checkpoints"
    ),
    config: Annotated[Path, typer.Option()] = Path("configs/generation/adaptive_generator.yaml"),
) -> None:
    """Generate equal-budget static and detector-feedback attack candidates."""

    _emit(
        generate_attack_candidates(
            project_root=project_root,
            detector_model_dir=detector_model_dir,
            config_path=config,
            output_dataset=output_dataset,
            output_scores=output_scores,
            checkpoint_dir=checkpoint_dir,
        )
    )


@app.command("evaluate-attack-search")
def evaluate_attack_search_command(
    candidate_dataset: Annotated[Path, typer.Option()],
    candidate_scores: Annotated[Path, typer.Option()],
    target_trajectories: Annotated[Path, typer.Option()],
    thresholds: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path("outputs/latest/attack_search_evaluation.json"),
) -> None:
    """Evaluate equal-query attack-search strategies on observed target trajectories."""

    _emit(
        evaluate_attack_search(
            candidate_dataset=candidate_dataset,
            candidate_scores=candidate_scores,
            target_trajectories=target_trajectories,
            thresholds_path=thresholds,
            output_path=output,
        )
    )


@app.command("analyze-adaptive-search")
def analyze_adaptive_search_command(
    candidate_dataset: Annotated[Path, typer.Option()],
    candidate_validity: Annotated[Path, typer.Option()],
    candidate_manifest: Annotated[Path, typer.Option()],
    adaptive_report: Annotated[Path, typer.Option()],
    generator_config: Annotated[Path, typer.Option()] = Path(
        "configs/generation/adaptive_generator.yaml"
    ),
    telemetry: Annotated[Path | None, typer.Option()] = None,
    strict_capacity_receipt: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = Path("outputs/latest/adaptive_analysis.json"),
) -> None:
    """Analyze H4 from retained adaptive raw artifacts without broadening its threat model."""

    receipt = (
        json.loads(strict_capacity_receipt.read_text(encoding="utf-8"))
        if strict_capacity_receipt is not None
        else None
    )
    _emit(
        analyze_adaptive_search_from_artifacts(
            candidate_dataset_path=candidate_dataset,
            candidate_validity_path=candidate_validity,
            candidate_manifest_path=candidate_manifest,
            adaptive_report_path=adaptive_report,
            generator_config_path=generator_config,
            telemetry_path=telemetry,
            strict_capacity_receipt=receipt,
            output_path=output,
        )
    )


@app.command("audit-postrun")
def audit_postrun_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Audit retained post-run evidence and refuse any fixture-only thesis claim."""

    _emit(
        audit_postrun(
            project_root=project_root,
            output_root=output_root,
            output_path=output,
        )
    )


@app.command("prepare-postrun-supporting-evidence")
def prepare_postrun_supporting_evidence_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
    launch_authorization_source: Annotated[Path, typer.Option()] = Path(
        "launch_authorization.json"
    ),
    strict_capacity_receipt: Annotated[Path, typer.Option()] = Path(
        "outputs/latest/strict_capacity_receipt.json"
    ),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Build deterministic authorization, matrix, fallback, and failure evidence."""

    _emit(
        prepare_postrun_supporting_evidence(
            project_root=project_root,
            output_root=output_root,
            launch_authorization_source=launch_authorization_source,
            strict_capacity_receipt_path=strict_capacity_receipt,
            output_path=output,
        )
    )


@app.command("write-postrun-run-context")
def write_postrun_run_context_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Hash-bind required post-run inputs before raw-manifest construction."""

    _emit(
        write_postrun_run_context(
            project_root=project_root,
            output_root=output_root,
            output_path=output,
        )
    )


@app.command("build-postrun-raw-manifests")
def build_postrun_raw_manifests_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Build allowlisted raw-evidence manifests before the final audit."""

    _emit(
        build_postrun_raw_manifests(
            project_root=project_root,
            output_root=output_root,
            output_path=output,
        )
    )


@app.command("finalize-confirmatory-run")
def finalize_confirmatory_run_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
    postrun_audit: Annotated[Path, typer.Option()] = Path("outputs/latest/postrun_audit.json"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    """Write terminal status only from the independent post-run audit."""

    _emit(
        finalize_confirmatory_run(
            project_root=project_root,
            output_root=output_root,
            postrun_audit_path=postrun_audit,
            output_path=output,
        )
    )


@app.command("materialize-report-assets")
def materialize_report_assets_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    output_root: Annotated[Path, typer.Option()] = Path("outputs/latest"),
) -> None:
    """Render report-ready figures only from an independently accepted live run."""

    _emit(materialize_report_assets(project_root=project_root, output_root=output_root))


@app.command("analyze-static-system")
def analyze_static_system_command(
    four_arm_report: Annotated[Path, typer.Option()],
    trajectories: Annotated[Path, typer.Option()],
    telemetry: Annotated[Path | None, typer.Option()] = None,
    strict_capacity_receipt: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = Path("outputs/latest/static_analysis.json"),
) -> None:
    """Analyze the complete static four-arm system from retained raw records."""

    receipt = (
        json.loads(strict_capacity_receipt.read_text(encoding="utf-8"))
        if strict_capacity_receipt is not None
        else None
    )
    _emit(
        analyze_static_system(
            four_arm_report_path=four_arm_report,
            trajectories_path=trajectories,
            telemetry_path=telemetry,
            strict_capacity_receipt=receipt,
            output_path=output,
        )
    )


@app.command("validate-sample-size-protocol")
def validate_sample_size_protocol_command(
    config: Annotated[Path, typer.Option()] = Path("configs/experiments/sample_size_scaling.yaml"),
) -> None:
    _emit(validate_sample_size_protocol(config))


@app.command("check-notebook")
def check_notebook_command(
    notebook: Annotated[Path, typer.Argument()] = Path("notebooks/confirmatory_run.ipynb"),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    _emit(check_notebook(notebook, output))


@app.command("scan-secrets")
def scan_secrets_command(
    root: Annotated[Path, typer.Option()] = Path("."),
    output: Annotated[Path | None, typer.Option()] = None,
) -> None:
    _emit(scan_secrets(root.resolve(), output))


@app.command("export-exec-schema")
def export_exec_schema(
    output: Annotated[Path, typer.Option()] = Path("docs/schema/vipibench_episode.schema.json"),
) -> None:
    """Export the executable episode JSON Schema."""

    _emit(export_episode_schema(output))


@app.command("verify-exec-oracle")
def verify_exec_oracle(
    output: Annotated[Path | None, typer.Option()] = Path("outputs/exec_oracle_verification.json"),
) -> None:
    """Run deterministic golden fixtures and emit machine-readable evidence."""

    _emit(verify_oracle_fixtures(output))


@app.command("compile-exec-benchmark")
def compile_exec_benchmark(
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/vipibench_exec.jsonl"),
    template_output: Annotated[Path, typer.Option()] = Path(
        "data/processed/vipibench_exec_templates.jsonl"
    ),
    manifest: Annotated[Path, typer.Option()] = Path("outputs/executable_benchmark_compile.json"),
) -> None:
    """Compile the locked 80-family executable benchmark without network or model calls."""

    _emit(compile_catalog_path(config, output, template_output, manifest))


@app.command("compile-confirmatory-holdout")
def compile_confirmatory_holdout_command(
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    frozen_split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    output_dir: Annotated[Path, typer.Option()] = Path("data/splits/confirmatory_final"),
) -> None:
    """Create the sealed final surface-realization holdout without model execution."""

    _emit(compile_confirmatory_holdout_path(config, frozen_split_dir, output_dir))


@app.command("verify-confirmatory-holdout")
def verify_confirmatory_holdout_command(
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    frozen_split_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    holdout_dir: Annotated[Path, typer.Option()] = Path("data/splits/confirmatory_final"),
) -> None:
    """Verify the sealed final-holdout package and all byte bindings."""

    _emit(verify_confirmatory_holdout_package(config, frozen_split_dir, holdout_dir))


@app.command("compile-provenance-contrast")
def compile_provenance_contrast_command(
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/provenance_contrast.yaml"),
    output: Annotated[Path, typer.Option()] = Path("data/processed/provenance_contrast.jsonl"),
    manifest: Annotated[Path, typer.Option()] = Path("outputs/provenance_contrast_manifest.json"),
) -> None:
    """Compile the deterministic source-provenance counterfactual benchmark."""

    _emit(compile_provenance_contrast(config, output, manifest))


@app.command("audit-provenance-contrast")
def audit_provenance_contrast_command(
    dataset: Annotated[Path, typer.Argument()] = Path("data/processed/provenance_contrast.jsonl"),
    output: Annotated[Path | None, typer.Option()] = Path("outputs/provenance_contrast_audit.json"),
) -> None:
    """Audit pair identity, source binding, counts, splits, and diagnostic slices."""

    _emit(audit_provenance_contrast_path(dataset, output))


@app.command("validate-exec-benchmark")
def validate_exec_benchmark_command(
    dataset: Annotated[Path, typer.Argument()] = Path("data/processed/vipibench_exec.jsonl"),
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    templates: Annotated[Path, typer.Option()] = Path(
        "data/processed/vipibench_exec_templates.jsonl"
    ),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/executable_benchmark_validation.json"
    ),
    composition_output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/exec_composition_audit.json"
    ),
) -> None:
    """Validate the persisted 2,400-episode benchmark independently of compilation."""

    _emit(
        validate_exec_benchmark(
            dataset,
            config,
            templates,
            output_path=output,
            composition_output_path=composition_output,
        )
    )


@app.command("freeze-exec-splits")
def freeze_exec_splits_command(
    dataset: Annotated[Path, typer.Argument()] = Path("data/processed/vipibench_exec.jsonl"),
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    output_dir: Annotated[Path, typer.Option()] = Path("data/splits/frozen"),
    near_duplicate_threshold: Annotated[float, typer.Option()] = 0.90,
) -> None:
    """Freeze family-grouped train/dev/test files and fail closed on leakage."""

    _emit(
        freeze_exec_splits(
            dataset,
            config,
            output_dir,
            near_duplicate_threshold=near_duplicate_threshold,
        )
    )


@app.command("audit-exec-splits")
def audit_exec_splits_command(
    split_dir: Annotated[Path, typer.Argument()] = Path("data/splits/frozen"),
    dataset: Annotated[Path, typer.Option()] = Path("data/processed/vipibench_exec.jsonl"),
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
    output: Annotated[Path | None, typer.Option()] = None,
    holdout_output: Annotated[Path | None, typer.Option()] = None,
    near_duplicate_threshold: Annotated[float, typer.Option()] = 0.90,
) -> None:
    """Re-audit frozen split isolation, duplicates, reconciliation, and holdout folds."""

    _emit(
        audit_exec_splits(
            split_dir,
            dataset,
            config,
            near_duplicate_threshold=near_duplicate_threshold,
            output_path=output,
            holdout_output_path=holdout_output,
        )
    )


@app.command("audit-exec-shortcuts")
def audit_exec_shortcuts_command(
    dataset: Annotated[Path, typer.Argument()] = Path("data/processed/vipibench_exec.jsonl"),
    role_output: Annotated[Path | None, typer.Option()] = Path("outputs/role_label_leakage.json"),
    template_output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/template_generator_leakage.json"
    ),
) -> None:
    """Audit role, template, generator, domain, language, and explicit marker shortcuts."""

    _emit(
        audit_exec_shortcuts(
            dataset,
            role_output_path=role_output,
            template_output_path=template_output,
        )
    )


@app.command("seal-exec-splits")
def seal_exec_splits_command(
    split_dir: Annotated[Path, typer.Argument()] = Path("data/splits/frozen"),
    dataset: Annotated[Path, typer.Option()] = Path("data/processed/vipibench_exec.jsonl"),
    config: Annotated[Path, typer.Option()] = Path("configs/benchmark/exec_catalog.yaml"),
) -> None:
    """Refresh frozen audit bindings only when the complete package validates."""

    _emit(seal_frozen_split_package(split_dir, dataset, config))


@app.command("verify-policy-gate")
def verify_policy_gate_command(
    output: Annotated[Path | None, typer.Option()] = Path("outputs/policy_gate_verification.json"),
) -> None:
    """Verify authorization, capability, provenance, and detector threshold fixtures."""

    _emit(verify_policy_gate(output))


@app.command("validate-exec-experiment-protocol")
def validate_exec_experiment_protocol_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    config: Annotated[Path, typer.Option()] = Path("configs/experiments/exec_system.yaml"),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/experiment_protocol_validation.json"
    ),
) -> None:
    """Validate the frozen static and adaptive execution budgets and source bindings."""

    _emit(validate_exec_experiment_protocol(project_root, config, output))


@app.command("verify-provenance")
def verify_provenance_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    ledger: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = Path("outputs/provenance_verification.json"),
) -> None:
    """Verify current benchmark lineage, license scope, and immutable artifact bindings."""

    _emit(verify_provenance(project_root, ledger_path=ledger, output_path=output))


@app.command("verify-training-authorization")
def verify_training_authorization_command(
    project_root: Annotated[Path, typer.Option()] = Path("."),
    decision: Annotated[Path | None, typer.Option()] = None,
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/training_authorization_verification.json"
    ),
) -> None:
    """Verify bounded internal preparation permission without enabling external mutations."""

    _emit(
        verify_training_authorization(
            project_root,
            decision_path=decision,
            output_path=output,
        )
    )


@app.command("verify-four-arm-fixture")
def verify_four_arm_fixture_command(
    test_dataset: Annotated[Path, typer.Argument()] = Path("data/splits/frozen/test.jsonl"),
    output: Annotated[Path | None, typer.Option()] = Path(
        "outputs/four_arm_fixture_verification.json"
    ),
) -> None:
    """Run all 480 frozen test episodes through the four paired system arms."""

    _emit(verify_four_arm_fixture(test_dataset, output_path=output))


@app.command("evaluate-four-arms")
def evaluate_four_arms_command(
    predictions: Annotated[Path, typer.Option()],
    trajectories: Annotated[Path, typer.Option()],
    thresholds: Annotated[Path, typer.Option()],
    detector_model_version: Annotated[str, typer.Option()],
    test_dataset: Annotated[Path, typer.Option()] = Path("data/splits/frozen/test.jsonl"),
    output: Annotated[Path, typer.Option()] = Path(
        "outputs/latest/static_four_arm_evaluation.json"
    ),
) -> None:
    """Evaluate four paired system arms from hash-bound observed detector predictions."""

    _emit(
        evaluate_four_arms_from_predictions(
            test_dataset,
            predictions,
            trajectories,
            thresholds,
            detector_model_version=detector_model_version,
            output_path=output,
        )
    )


@app.command("analyze-rq2-diagnostics")
def analyze_rq2_diagnostics_command(
    output_root: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path("outputs/encoder/rq2_diagnostic_analysis.json"),
    control_identity: Annotated[Path | None, typer.Option()] = None,
    analysis_config: Annotated[Path, typer.Option()] = Path(
        "configs/experiments/confirmatory_analysis.yaml"
    ),
) -> None:
    """Analyze the five locked RQ2 diagnostics from retained raw predictions."""

    _emit(
        analyze_rq2_diagnostics(
            output_root,
            output,
            control_identity_path=control_identity,
            analysis_config_path=analysis_config,
        )
    )


@app.command("analyze-h3")
def analyze_h3_command(
    four_arm_report: Annotated[Path, typer.Option()],
    static_analysis: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()] = Path("outputs/latest/h3_analysis.json"),
    analysis_config: Annotated[Path, typer.Option()] = Path(
        "configs/experiments/confirmatory_analysis.yaml"
    ),
) -> None:
    """Apply the locked H3 paired simultaneous decision to raw static artifacts."""

    _emit(
        analyze_h3_from_artifacts(
            four_arm_report_path=four_arm_report,
            static_analysis_path=static_analysis,
            output_path=output,
            analysis_config_path=analysis_config,
        )
    )


if __name__ == "__main__":
    app()
