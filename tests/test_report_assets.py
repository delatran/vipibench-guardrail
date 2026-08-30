from __future__ import annotations

import json
import re
from pathlib import Path

from vipibench.dataio import sha256_file
from vipibench.report_assets import (
    ARCHIVE_FILENAME,
    MANIFEST_FILENAME,
    PACKAGE_DIRECTORY,
    materialize_report_assets,
)
from vipibench.report_figures import (
    CONDITION_LABELS,
    VISIBLE_CODE_PATTERN,
    VISIBLE_VERSION_PATTERN,
    render_report_figures,
)


def _analysis_artifacts() -> dict[str, dict[str, object]]:
    effects = {
        "content_provenance_minus_text_only": {
            "estimate": 0.20,
            "lower_95": 0.12,
            "upper_95": 0.28,
            "pair_count": 200,
            "family_count": 16,
        },
        "content_provenance_minus_role_only": {
            "estimate": 0.15,
            "lower_95": 0.07,
            "upper_95": 0.23,
            "pair_count": 200,
            "family_count": 16,
        },
    }
    comparisons: dict[str, object] = {}
    for index, condition in enumerate(CONDITION_LABELS):
        estimate = 0.04 + 0.01 * index
        comparisons[condition] = {
            "pair_count_by_seed": {"42": 100, "314": 100, "2718": 100},
            "signed_margin_degradation": {
                "estimate": estimate,
                "confidence_interval_95": {
                    "lower_95": estimate - 0.02,
                    "upper_95": estimate + 0.02,
                },
            },
            "calibration_degradation": {
                "brier_estimate": 0.01 + index * 0.002,
                "ece_10_bins_estimate": 0.015 + index * 0.002,
                "confidence_interval_95": {
                    "brier": {
                        "lower_95": 0.002 + index * 0.002,
                        "upper_95": 0.018 + index * 0.002,
                    },
                    "ece_10_bins": {
                        "lower_95": 0.005 + index * 0.002,
                        "upper_95": 0.025 + index * 0.002,
                    },
                },
            },
        }
    arm_rates = {
        "none": (0.80, 0.10, 0.98, 0.01),
        "detector_only": (0.35, 0.62, 0.90, 0.08),
        "policy_only": (0.15, 0.85, 0.88, 0.10),
        "hybrid": (0.08, 0.92, 0.86, 0.12),
    }
    metric_names = (
        "attack_success_rate",
        "containment_rate",
        "clean_utility_rate",
        "false_block_rate",
    )
    metrics: dict[str, object] = {}
    confidence_arms: dict[str, object] = {}
    for arm, rates in arm_rates.items():
        metrics[arm] = {
            name: {"value": value, "denominator": 240}
            for name, value in zip(metric_names, rates, strict=True)
        }
        confidence_arms[arm] = {
            name: {
                "lower_95": max(0.0, value - 0.05),
                "upper_95": min(1.0, value + 0.05),
            }
            for name, value in zip(metric_names, rates, strict=True)
        }
    taxonomy = {
        strategy: {
            arm: {
                "violation_code_counts": {
                    "ATTACK_OBJECTIVE_ACHIEVED": 2 if strategy == "feedback_guided" else 1
                }
            }
            for arm in ("detector_only", "policy_only", "hybrid")
        }
        for strategy in ("static_sampling", "feedback_guided")
    }
    return {
        "encoder_ablation": {
            "status": "PASS",
            "errors": [],
            "research_claim_eligible": True,
            "primary_effects": effects,
        },
        "diagnostic_analysis": {
            "status": "PASS",
            "errors": [],
            "research_claim_eligible": True,
            "formal_condition_order": list(CONDITION_LABELS),
            "source_family_paired_reference": {"expected_source_family_count": 16},
            "comparisons": comparisons,
        },
        "static_analysis": {
            "status": "PASS",
            "errors": [],
            "research_claim_eligible": True,
            "metrics": metrics,
            "confidence_intervals": {
                "paired_episode_count": 480,
                "arms": confidence_arms,
            },
            "pareto_frontier": {
                "status": "PASS",
                "points": {
                    arm: {
                        "attack_success_rate": rates[0],
                        "clean_utility_rate": rates[2],
                    }
                    for arm, rates in arm_rates.items()
                },
                "frontier_arms": ["none", "detector_only", "policy_only", "hybrid"],
            },
        },
        "joint_analysis": {
            "status": "PASS",
            "errors": [],
            "research_claim_eligible": True,
            "sample_sizes": {"injection": 240, "benign": 240},
            "security": {
                "point_effect": 0.27,
                "marginal_one_sided_lower_bound_95": 0.18,
                "locked_margin": 0.05,
                "paired_episode_count": 240,
                "family_count": 16,
                "bound_passes_locked_margin": True,
            },
            "utility": {
                "point_effect": -0.01,
                "marginal_one_sided_lower_bound_95": -0.018,
                "locked_margin": -0.02,
                "paired_episode_count": 240,
                "family_count": 16,
                "bound_passes_locked_margin": True,
            },
        },
        "adaptive_analysis": {
            "status": "PASS",
            "errors": [],
            "research_claim_eligible": True,
            "paired_base_episode_count": 240,
            "family_count": 16,
            "paired_guided_minus_static": {
                arm: {
                    "effect": 0.10 + index * 0.02,
                    "confidence_interval_95": {
                        "lower": 0.02 + index * 0.02,
                        "upper": 0.18 + index * 0.02,
                    },
                    "hypothesis_decision": "SUPPORTED",
                }
                for index, arm in enumerate(("detector_only", "policy_only", "hybrid"))
            },
            "unique_failure_taxonomy": taxonomy,
        },
        "runtime_telemetry": {
            "validation_status": "PASS",
            "hardware_observed": True,
            "local_only": False,
            "execution_status": "completed",
            "unique_interval_count": 5,
            "compute_hours": 0.5,
            "records": [
                {
                    "stage_id": stage,
                    "elapsed_seconds": duration,
                    "accelerator_stage": accelerated,
                    "status": "completed",
                }
                for stage, duration, accelerated in (
                    ("preflight", 60.0, False),
                    ("encoder-matrix", 900.0, True),
                    ("core-target-trajectories", 300.0, True),
                    ("attack-candidate-generation", 420.0, True),
                    ("final-analysis", 120.0, False),
                )
            ],
        },
    }


def _write_terminal_fixture(root: Path) -> tuple[dict[str, object], dict[str, object]]:
    artifacts = _analysis_artifacts()
    path_by_name = {
        "encoder_ablation": "encoder_ablation_analysis.json",
        "diagnostic_analysis": "rq2_analysis.json",
        "static_analysis": "static_analysis.json",
        "joint_analysis": "h3_analysis.json",
        "adaptive_analysis": "adaptive_analysis.json",
        "runtime_telemetry": "runtime_telemetry.json",
    }
    audit_key = {
        "encoder_ablation": "encoder_ablation",
        "diagnostic_analysis": "rq2_analysis",
        "static_analysis": "static_analysis",
        "joint_analysis": "h3_analysis",
        "adaptive_analysis": "adaptive_analysis",
        "runtime_telemetry": "runtime_telemetry",
    }
    hashes: dict[str, str] = {}
    for report_name, filename in path_by_name.items():
        path = root / filename
        path.write_text(json.dumps(artifacts[report_name]), encoding="utf-8")
        hashes[audit_key[report_name]] = sha256_file(path)
    audit = {
        "status": "PASS",
        "errors": [],
        "fixture_only": False,
        "research_evidence_eligible": True,
        "artifact_sha256": hashes,
        "dispositions": {
            "RUN_COMPLETE": "PASS",
            "RESEARCH_EVIDENCE_ELIGIBLE": True,
        },
    }
    audit_path = root / "postrun_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    manifest = {
        "status": "PASS",
        "errors": [],
        "RUN_COMPLETE": "PASS",
        "RESEARCH_EVIDENCE_ELIGIBLE": True,
        "final_claim_audit_created": True,
        "postrun_audit_sha256": sha256_file(audit_path),
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return audit, manifest


def test_report_figures_render_all_formats_with_presentation_safe_text(tmp_path: Path) -> None:
    output = tmp_path / "figures"
    catalog = render_report_figures(_analysis_artifacts(), output)

    assert len(catalog) == 11
    assert len(list(output.glob("*.png"))) == 11
    assert len(list(output.glob("*.svg"))) == 11
    assert len(list(output.glob("*.pdf"))) == 11
    assert len(list(output.glob("*.csv"))) == 11
    assert all(path.read_bytes().startswith(b"\x89PNG") for path in output.glob("*.png"))
    assert all(path.read_bytes().startswith(b"%PDF") for path in output.glob("*.pdf"))
    visible = json.dumps(catalog, ensure_ascii=False)
    visible += "\n" + "\n".join(path.read_text(encoding="utf-8") for path in output.glob("*.svg"))
    assert re.search(VISIBLE_CODE_PATTERN, visible) is None
    assert re.search(VISIBLE_VERSION_PATTERN, visible) is None


def test_materializer_fails_closed_without_terminal_evidence(tmp_path: Path) -> None:
    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()

    result = materialize_report_assets(project_root=project, output_root=output)

    assert result["status"] == "FAIL"
    assert result["figure_count"] == 0
    assert not (output / PACKAGE_DIRECTORY).exists()


def test_materializer_publishes_atomically_reuses_and_detects_tampering(
    monkeypatch, tmp_path: Path
) -> None:
    from vipibench import report_assets

    project = tmp_path / "project"
    output = tmp_path / "output"
    project.mkdir()
    output.mkdir()
    audit, _ = _write_terminal_fixture(output)
    monkeypatch.setattr(
        report_assets,
        "validate_postrun_audit_for_finalization",
        lambda **_: (audit, []),
    )
    monkeypatch.setattr(report_assets, "runtime_source_fingerprint", lambda _: "A" * 64)

    first = materialize_report_assets(project_root=project, output_root=output)
    package = output / PACKAGE_DIRECTORY

    assert first["status"] == "PASS", first["errors"]
    assert first["cache_reused"] is False
    assert first["figure_count"] == 11
    assert (package / MANIFEST_FILENAME).is_file()
    assert (package / ARCHIVE_FILENAME).is_file()

    second = materialize_report_assets(project_root=project, output_root=output)
    assert second["status"] == "PASS"
    assert second["cache_reused"] is True

    svg = next((package / "hinh_bao_cao").glob("*.svg"))
    svg.write_text(svg.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    tampered = materialize_report_assets(project_root=project, output_root=output)
    assert tampered["status"] == "FAIL"
    assert "existing_report_package_file_hash_mismatch" in tampered["errors"]
