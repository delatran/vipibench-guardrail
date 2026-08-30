# A100 Deterministic Terminal-Recovery Amendment V5

Date: 2026-08-12

Protocol amendment: `a100_80gb-deterministic-terminal-recovery-v5-2026-08-12`

Durable lineage: `fp32-a100_80gb-deterministic-terminal-recovery-v5-2026-08-12`

Registered policy: `unique_single_terminal_delimiter_recovery_v1`

## Triggering evidence

The immutable v4 A100 run stopped after recording 2,080 of 4,800 attack-target trajectories. It
retained eleven initial JSON decode incidents, ten accepted repairs, one safe fallback, and 2,720
unprocessed records. The unresolved episode was
`customer_support-policy_forgery-59-feedback-guided-01`.

The checkpoint-bound nonpublic diagnostic showed the following structural facts without requiring
labels, oracle outcomes, detector scores, utilities, or attack-success measurements:

- the first response was 729 UTF-8 bytes and had exactly one extra terminal `}`;
- removing that one terminal delimiter produced the sole strict-valid candidate;
- the second response was 812 UTF-8 bytes and passed the strict parser;
- the second response added `supporting_context_ids` to two tool-call events, changed the trajectory
  hash, and was therefore correctly rejected by v4;
- the strict A100 receipt, CUDA-only placement receipt, capacity plan, and model revision all passed.

The v4 failure is therefore not attributed to accelerator capacity, model placement, event-ID
length, or an ambiguous local delimiter edit. It is caused by using a second generative response as
the confirmation and selected value for a syntax correction that was already unique and
deterministic from the original response.

## Registered recovery predicate

For each original model response, the target runner applies this exact sequence:

1. Parse the unchanged response with the strict trajectory parser.
2. If strict parsing fails, retain the original parse-error class and response SHA-256.
3. Enumerate only these one-character candidates: remove one terminal `}` or `]` when present,
   append one terminal `}`, and append one terminal `]`.
4. Parse every candidate with the unchanged episode-, request-, event-, authorization-, and
   trajectory-validation contract.
5. Accept a recovered trajectory only when exactly one candidate passes the strict parser.
6. Bind the accepted record to the original malformed-response hash, nonpublic diagnostic hash,
   one generation attempt, and `repaired_json` status.
7. If zero or more than one candidate passes, create the existing safe fallback, withhold the normal
   trajectory artifact, and stop before the next target batch.

The runner does not make a second model request. It never selects text from a retry, unwraps a
response, edits an interior character, adds an event field, truncates an identifier, or coerces a
schema value.

## Version and isolation boundary

- agent trajectory record and manifest schema: `2.2.0`;
- target configuration schema: `2.2.0`;
- target checkpoint and checkpoint-binding schema: `5.0.0`;
- maximum model generation attempts per target response: `1`;
- model-repair batching: `not_applicable`;
- durable namespace: new and disjoint from v4 and every earlier lineage.

Source hashes, the target configuration hash, model and tokenizer revisions, timing schema, event-ID
maximum, response format, and oracle source remain checkpoint-bound. No v4 checkpoint, fallback
ledger, authorization, snapshot blob index, stage marker, output, or completion claim is eligible for
v5 restore.

## Preserved scientific and runtime boundary

The model and tokenizer revisions, BF16 target arithmetic, deterministic decoding, input and output
token limits, capacity ladder, input ordering, frozen datasets, encoder protocol, detector
thresholds, attack budgets, estimands, multiplicity rules, bootstrap count, and bootstrap seed are
unchanged. Local recovery time is excluded from model-request wall time, just as parsing and
checkpoint I/O were already excluded.

This amendment was informed by structural final-holdout format behavior. That fact must remain
disclosed. A fresh v5 run is operationally isolated but does not by itself resolve whether an
advisor or institution will treat the amended use of the same frozen holdout as confirmatory;
external acceptance remains `UNVERIFIED`.

## Claim boundary

Local tests can prove the exact one-character predicate, absence of a model retry, fail-closed
negative cases, source binding, and lineage isolation. They cannot prove A100 completion, future
format incidence, throughput, model quality, attack resistance, statistical hypotheses, or thesis
claim eligibility. Those require a separately authorized live run, complete post-run audit, and the
applicable external review.
