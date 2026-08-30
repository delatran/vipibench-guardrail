# ADR 0001: Python, PyTorch, and Hugging Face Stack

- Status: accepted
- Date: 2026-07-15

## Decision

Use Python 3.11, scikit-learn for the sparse baseline, PyTorch/Transformers for encoder training,
Pydantic/JSON Schema for data contracts, and FastAPI for the local guardrail boundary.

## Rationale

This implements the stack and model family locked by the proposal, keeps the fixture path CPU
compatible, and permits identical model/split/seed manifests locally and in Colab.

## Consequences

GPU packages have a separate optional dependency group. Full-run images and Hugging Face assets
must be pinned by immutable revision. Windows editable installation is not the clean-environment
gate because its non-ASCII parent path is not representable by the active legacy code page;
wheel installation is the verified path.
