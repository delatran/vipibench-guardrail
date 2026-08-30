# A100 Target Response Envelope v2

Date: 2026-08-11

Protocol amendment: `a100_80gb-target-response-envelope-v2-2026-08-11`

Durable lineage: `fp32-a100_80gb-target-response-envelope-v2-2026-08-11`

## Trigger and evidence boundary

The previous A100 lineage stopped in the first production target batch. Its retained aggregate
receipt recorded eight first-pass strict parse failures, eight exhausted format fallbacks, and 472
unprocessed requests. A read-only inspection of one retained nonpublic diagnostic established that
both attempts were valid JSON with an extra `output_contract` wrapper. The first event also combined
assistant-message content with tool-call-only fields. Raw model text was inspected only to classify
the failure and is not reproduced in this protocol record.

Source inspection identified a contract mismatch. The prompt demonstrated the wrapped, mixed-field
shape, while the strict parser required an exact top-level `events` object and mutually exclusive
event-type fields. This amendment corrects the instruction surface; it does not reinterpret the
failed responses as valid.

## Amended output contract

The target prompt now requires one JSON object with exactly one top-level key, `events`.

Each event must match exactly one of these variants:

- assistant message: `event_type`, `event_id`, and `content`;
- tool call: `event_type`, `event_id`, `tool`, `arguments`, `authorization_refs`, and
  `supporting_context_ids`.

No outer wrapper, Markdown fence, prose prefix, or cross-variant field is allowed. The bounded
format-repair turn repeats this contract and explicitly identifies the observed wrapper as invalid.
The strict parser remains authoritative. It does not unwrap, coerce, or salvage a nonconforming
response.

## Preserved protocol

The amendment preserves the pinned target and generator model/tokenizer revisions, decoding
settings, maximum generation length, frozen datasets and splits, encoder seeds, FP32 confirmatory
encoder arithmetic, capacity ladder and parity checks, attack budget, detector thresholds,
estimands, multiplicity correction, bootstrap count, and bootstrap seed. The existing fail-fast
rule still stops after the affected batch, preserves raw diagnostics in machine receipts, and
withholds the incomplete normal trajectory artifact.

## Lineage and claim boundary

Target-facing prompt bytes changed after final-holdout model behavior was observed. Therefore the
failed `fp32-a100_80gb-target-format-fail-fast-v1-2026-08-11` lineage is immutable, ineligible, and
cannot be resumed or combined with this amendment. All source-bound manifests, authorization, and
local compatibility receipts must be regenerated for the new lineage.

Local tests can establish only prompt/parser consistency, fail-fast behavior, reader-facing output,
and lineage isolation. They cannot establish that Qwen will satisfy the revised contract, that the
target stage will complete, or that any registered research claim is supported. Those outcomes
remain `UNVERIFIED` until a separately authorized A100 run reaches terminal post-run acceptance.
