from __future__ import annotations

from pathlib import Path


def validate_new_run_output_root(
    output_root: Path,
    session_evidence_root: Path,
    *,
    resume_existing: bool,
) -> None:
    """Reject stale run output while permitting the required prior smoke evidence."""

    if resume_existing or not output_root.exists():
        return
    children = list(output_root.iterdir())
    if not children:
        return
    resolved_output = output_root.resolve()
    resolved_evidence = session_evidence_root.resolve()
    smoke_evidence_only = (
        len(children) == 1
        and children[0].resolve() == resolved_evidence
        and resolved_evidence.parent == resolved_output
        and resolved_evidence.is_dir()
    )
    if not smoke_evidence_only:
        raise FileExistsError(
            "Output directory contains data other than the required smoke session-evidence "
            "directory; set VIPIBENCH_RESUME_EXISTING_RUN=YES only to resume a bound run"
        )
