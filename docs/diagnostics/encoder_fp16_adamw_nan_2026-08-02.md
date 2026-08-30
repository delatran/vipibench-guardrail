# Encoder FP16 Checkpoint / AdamW Numerical Failure Diagnosis

## Scope and exit condition

This packet diagnoses the `encoder-matrix` capacity failure observed in the hosted confirmatory
run. The repair is complete locally only when the pinned checkpoint is converted to the locked FP32
trainable-parameter dtype before every optimizer is built, the regression test passes, and all
hash-bound package gates are regenerated. A complete hosted encoder run is intentionally out of
scope for local verification.

## Observed evidence

- The retained Colab log has SHA-256
  `36E5C20DD0329E774906819FA379671740A400B34313329E2B46112A1EAAA03F`.
- Its `capacity_plan.json` summary reports `no_numerically_stable_capacity_candidate`.
- All six admissible candidates (`16`, `32`, and `64`, checkpointing on and off) fail on optimizer
  step 1. Each failure identifies all 192,768,000 elements of
  `deberta.embeddings.word_embeddings.weight` as non-finite.
- A read-only inspection of the pinned checkpoint revision
  `a0484667b22365f84929a935b5e50a51f71f159d` reports the same parameter shape
  `[251000, 768]` with dtype `torch.float16`.
- A two-element local AdamW reproduction with an FP16 parameter, one zero gradient, learning rate
  `2e-5`, and weight decay `0.0` changes the zero-gradient parameter to NaN on the first step. The
  same path remains finite after the parameter is converted to FP32.

## Root cause

The protocol declared `mixed_precision: fp32`, but the three trainable model-load paths did not pass
an explicit dtype to `from_pretrained`. Transformers therefore preserved the pinned checkpoint's
FP16 storage dtype. AdamW's default epsilon (`1e-8`) is below the representable normal/subnormal
range used by this FP16 optimizer state. For the many embedding entries with zero gradient, the
first update evaluates with a zero denominator and produces `0/0`, poisoning the full dense
embedding parameter. Batch size and gradient checkpointing cannot correct this systematic dtype
mismatch, which explains the identical failure across all candidates.

## Repair

`_load_trainable_encoder_model` now:

1. requires the locked FP32 protocol;
2. loads the checkpoint with `dtype=torch.float32`;
3. verifies every floating-point trainable parameter is FP32; and
4. is shared by capacity measurement, the numerical canary, and all nine development runs.

The new durable lineage is `encoder-fp32-parameters-2026-08-02-v1`, so the failed capacity plan and
its stale runner binding cannot be restored into the repaired execution.

## Claim ledger

| Claim | Label | Evidence | Gap / falsifier |
|---|---|---|---|
| The hosted failure is numerical rather than capacity exhaustion. | FACT | Six receipt attempts are `FloatingPointError`; none is OOM. | A different fresh receipt with OOM would describe a separate failure. |
| The checkpoint/optimizer dtype mismatch causes the step-1 NaN pattern. | FACT | Exact checkpoint dtype, tensor shape/count match, and minimal AdamW reproduction. | A reproduction remaining non-finite after verified FP32 loading would falsify sufficiency. |
| The source repair covers every trainable encoder load. | FACT | Shared helper call sites plus regression/source tests. | Any direct trainable backbone load outside the helper invalidates the claim. |
| The repaired bundle will complete the L4 encoder matrix. | UNVERIFIED | No post-repair hosted execution exists yet. | Fresh capacity and stage receipts are required. |

## Verification record

- Pre-fix regression baseline: failed during collection because the FP32 loader did not exist.
- Focused post-fix test: `20 passed` in `tests/test_transformer_numerics.py`.
- Pinned stack load: Torch 2.13.0 and Transformers 5.13.1 load the exact checkpoint with only
  `torch.float32` trainable parameters after the repair.
- Full source-tree suite after binding refresh: `364 passed, 1 skipped`; the skip is the existing
  local missing-ipykernel probe.
- Ruff, encoder/public protocol validation, current-source notebook validation, training
  authorization, project artifact-manifest verification, and outer bundle verification pass.
- Residual verifier: a fresh exact-L4 capacity receipt must select a candidate and advance the
  encoder matrix; local tests cannot establish that hosted outcome.
