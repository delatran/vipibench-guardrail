# A100 Event-ID Envelope v4

Date: 2026-08-11

Protocol amendment: `a100_80gb-event-id-envelope-v4-2026-08-11`

Durable lineage: `fp32-a100_80gb-event-id-envelope-v4-2026-08-11`

Registered maximum event identifier length: 96 ASCII characters

## Trigger and blinded diagnosis

The bounded-format-repair-v3 A100 run stopped in the attack target-trajectory stage after 192 of
4,800 records. Its raw-content-free run receipt recorded 11 initial parse incidents, ten accepted
one-terminal-delimiter repairs, one exhausted repair, and 4,608 unprocessed records. The normal
trajectory artifact was intentionally withheld.

Read-only structural inspection of the checkpoint-bound nonpublic diagnostic established that both
responses for the unresolved episode were strict JSON objects with the sole top-level `events`
field and the correct event-specific field sets. The first response used an 80-character assistant
message event identifier; the registered repair used a 72-character tool-call event identifier.
The then-current parser admitted at most 64 characters, while neither the initial prompt nor the
repair prompt disclosed that maximum. No outcome label, metric, oracle decision, target utility, or
attack-success result was used to select this correction. Raw response content is not reproduced in
this protocol record.

The v3 run remains failed, incomplete, immutable, and ineligible. This amendment is prospective and
uses a fresh durable lineage.

## Registered response contract

Every trajectory event identifier must:

1. contain between 2 and 96 ASCII characters, inclusive;
2. match `^[a-z][a-z0-9_-]{1,95}$`;
3. be unique within its trajectory.

The initial prompt and the one permitted repair prompt state the same bound and pattern. The strict
parser accepts a conforming identifier exactly as generated. It does not truncate, hash, rename,
canonicalize, or otherwise rewrite model output. A 97-character identifier and every other schema
violation remain fail-closed.

The 96-character ceiling aligns event identifiers with the existing trajectory and episode
identifier envelope. It covers both observed failing identifiers without introducing an unbounded
string surface.

## Version and resume isolation

- trajectory schema: `1.1.0`;
- agent trajectory record schema: `2.1.0`;
- target configuration schema: `2.1.0`;
- target checkpoint binding schema: `4.0.0`.

The checkpoint binding includes the event identifier maximum, trajectory schema version, and
SHA-256 digest of the oracle source in addition to the existing target-runner, prompt, model,
tokenizer, decoding, and repair-policy bindings. Consequently, a v3 checkpoint or fallback ledger
cannot be admitted into v4 even if copied into the new namespace.

## Preserved protocol

The Qwen model and tokenizer revisions, BF16 target arithmetic, deterministic decoding, 512-token
output ceiling, batch-capacity protocol, input ordering, one-terminal-delimiter repair predicate,
two-attempt maximum, frozen data, encoder seeds and FP32 arithmetic, attack budget, thresholds,
estimands, multiplicity rules, bootstrap count, and bootstrap seed remain unchanged.

The ten accepted v3 delimiter repairs are not imported as results. Their diagnostic pattern is only
evidence that the registered v3 repair implementation operated as designed before the independent
event-identifier mismatch stopped the stage.

## Rejected alternatives

A prompt-only correction would retain the 64-character parser cap but would leave acceptance
probabilistic; one repair could still exceed the undisclosed historical limit. Deterministic
truncation or renaming would alter raw model output and could collapse distinct identifiers.
Constrained decoding would change the decoding and runtime surface without a current same-device
parity study. The bounded raw-envelope expansion is narrower: it accepts the two observed forms
without changing their semantic fields or generated identifier text.

## Evidence and claim boundary

Local regression can prove parser/prompt/config agreement, acceptance of the observed 72- and
80-character shapes without rewriting, rejection at 97 characters, checkpoint isolation, and
downstream gate consistency. It cannot prove that a fresh Qwen run will avoid other format errors,
complete on A100, improve throughput, satisfy post-run eligibility, or support a research claim.
Those hosted and scientific outcomes remain `UNVERIFIED` until a separately authorized v4 run
reaches terminal audit acceptance.
