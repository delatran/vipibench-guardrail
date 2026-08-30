# A100 target-format fail-fast v1 amendment

## Status and trigger

The active amendment is `a100_80gb-target-format-fail-fast-v1-2026-08-11`, with durable lineage
`fp32-a100_80gb-target-format-fail-fast-v1-2026-08-11`. It supersedes the execution lineage
`fp32-a100_80gb-target-throughput-v2-2026-08-10` after that lineage completed target generation but
retained strict-format failures that were surfaced only by the near-final post-run preparation
gate.

The failed lineage remains immutable evidence. This amendment cannot restore or combine its
authorization, stage markers, checkpoints, content store, model outputs, fallback ledger, or
analysis artifacts.

## Preserved scientific protocol

The pinned models and tokenizer revisions, BF16 target and generator precision, FP32 confirmatory
encoder, greedy decoding, disabled thinking, prompts, 4,096 input-token limit, 512 output-token
limit, two-attempt ceiling, response parser, datasets, splits, query budgets, seeds, thresholds,
estimands, analysis, and post-run eligibility gates are unchanged. No malformed response is
accepted, normalized, regenerated into eligibility, or removed from historical evidence.

The target-engine v2 capacity ladder, batch-8 response-equivalence reference, production-token
validation, decreasing-token-length scheduling, same-round repair batching, memory reserve, and
CUDA-only placement contract are also unchanged.

## Fail-fast transport contract

Every target run derives a raw-content-free format-failure summary from the records observed so far.
The summary contains only episode identifiers already present in the bound run, aggregate counts,
stable parse-error classes, and an explicit `raw_response_included: false` marker.

The target-run result is `PASS` only when both `parse_failure_episode_ids` and
`format_fallback_episode_ids` are empty. Any first-pass parse failure makes the result `FAIL`, keeps
the dependent claims inconclusive, writes the checkpoint-bound diagnostic evidence and a target-run
failure receipt, intentionally withholds the incomplete normal trajectory artifact, and causes the
CLI to exit non-zero.

The production-capacity generation is reused as the first scheduled target batch, so the first
format gate adds no model call. After every batch, all records and diagnostics from that batch are
persisted before the gate is evaluated. A repaired response is still a retained first-pass parse
failure. A repaired response or exhausted fallback therefore stops the run before the next target
batch. Resuming the same lineage detects the retained disposition before model loading or capacity
measurement and exits non-zero without accelerator/model work. Later evaluation and post-run stages
cannot consume or present the incomplete target run as successful.

This changes failure timing and compute containment, not successful-run model outputs. Repeatedly
rerunning a failed final-holdout lineage until it passes is forbidden; any future output-generation
change requires a separate predeclared study and a fresh lineage.

## Claim boundary and open gate

Local tests can prove status propagation, safe aggregate reporting, source binding, and lineage
isolation. They cannot prove that Qwen3-8B will produce zero strict-format failures, identify the
recorded failure subtype, validate the prior run, or establish any research result.

The exact retained parse diagnostic remains the distinguishing evidence for JSON syntax, response
shape, trajectory validation, or output-budget exhaustion. Prompt, parser, token-limit, or
constrained-decoding changes remain out of scope until that evidence is inspected under a separate
change contract.
