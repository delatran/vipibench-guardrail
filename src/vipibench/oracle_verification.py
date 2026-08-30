from __future__ import annotations

from pathlib import Path
from typing import Any

from vipibench.dataio import write_json
from vipibench.exec_fixture import build_oracle_fixture_matrix
from vipibench.oracle import SandboxOracle


def verify_oracle_fixtures(output_path: Path | None = None) -> dict[str, Any]:
    """Verify exact fixture agreement and repeatability for the sandbox oracle."""

    observations: list[dict[str, Any]] = []
    mismatches: list[str] = []
    for case in build_oracle_fixture_matrix():
        first = SandboxOracle().evaluate(case["episode"], case["trajectory"])
        second = SandboxOracle().evaluate(case["episode"], case["trajectory"])
        observed_violations = sorted(item.code for item in first.violations)
        expected_violations = sorted(case["expected_violations"])
        exact_repeat = first.model_dump(mode="json") == second.model_dump(mode="json")
        exact_expected = (
            first.status == case["expected_status"]
            and observed_violations == expected_violations
        )
        if not exact_expected:
            mismatches.append(f"{case['name']}:expected_contract_mismatch")
        if not exact_repeat:
            mismatches.append(f"{case['name']}:nondeterministic_repeat")
        observations.append(
            {
                "name": case["name"],
                "expected_status": case["expected_status"],
                "observed_status": first.status,
                "expected_violations": expected_violations,
                "observed_violations": observed_violations,
                "security_status": first.security_status,
                "utility_status": first.utility_status,
                "result_sha256": first.result_sha256,
                "exact_expected": exact_expected,
                "exact_repeat": exact_repeat,
            }
        )

    agreement_count = sum(item["exact_expected"] for item in observations)
    repeat_count = sum(item["exact_repeat"] for item in observations)
    total = len(observations)
    result: dict[str, Any] = {
        "schema_version": "1.0.0",
        "status": "PASS" if not mismatches and total > 0 else "FAIL",
        "oracle_is_deterministic_in_memory": True,
        "external_tool_calls": 0,
        "llm_judge_calls": 0,
        "fixture_count": total,
        "exact_contract_agreement": agreement_count / total if total else 0.0,
        "exact_repeat_agreement": repeat_count / total if total else 0.0,
        "mismatches": mismatches,
        "fixtures": observations,
        "claim_boundary": (
            "PASS covers the committed deterministic fixtures only; it does not prove "
            "benchmark-wide oracle correctness until the 2,400-episode validation passes."
        ),
    }
    if output_path is not None:
        write_json(output_path, result)
    return result
