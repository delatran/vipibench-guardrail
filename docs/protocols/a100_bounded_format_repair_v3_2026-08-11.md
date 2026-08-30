# A100 Bounded Format Repair v3

Date: 2026-08-11

Protocol amendment: `a100_80gb-bounded-format-repair-v3-2026-08-11`

Durable lineage: `fp32-a100_80gb-bounded-format-repair-v3-2026-08-11`

Registered policy: `one_terminal_delimiter_retry_strict_equivalence_v1`

## Trigger and observed boundary

The response-envelope-v2 A100 run recorded 112 of 480 target responses and then stopped with 368
requests unprocessed. The aggregate receipt recorded one first-pass JSON decode error and no
exhausted format fallback. Read-only inspection of the checkpoint-bound nonpublic diagnostic showed
that removing one extra terminal `}` made the first response strict-valid. The already-recorded
second response was strict-valid, used two total generation attempts, did not use a fallback, and
produced exactly the same trajectory hash as the uniquely recovered first response. Raw model text
is not reproduced in this protocol record.

The prior rule classified every first-pass parse incident as terminal even when the single repair
succeeded. Therefore the observed v2 run remains a failed, incomplete, immutable run. This amendment
changes the eligibility predicate prospectively under a new lineage; it does not reclassify, resume,
or combine the old observations.

## Acceptance predicate

The unchanged strict trajectory parser is applied to the first response. If it passes, the response
is recorded as `strict_json` with one generation attempt.

If the first response fails, exactly one model repair attempt is allowed. The second response is
accepted as `repaired_json` only when all of the following hold:

1. the second response passes the unchanged strict trajectory parser;
2. adding one terminal `}` or `]`, or removing one existing terminal `}` or `]`, from the original
   response yields exactly one strict-valid candidate;
3. the strict-valid candidate and the strict second response have exactly the same trajectory hash;
4. no safe fallback has been recorded for that episode.

The accepted trajectory is the strict second response. The original parse-error class, malformed
response hash, diagnostic-artifact hash, two-attempt count, and `repaired_json` status remain in the
record. Initial incidents and accepted repairs therefore remain inspectable rather than disappearing
from provenance.

## Fail-closed cases

The episode becomes `safe_fallback`, and the target stage stops before another target batch, when
any of these conditions holds:

- the second response is not strict-valid;
- zero or more than one terminal-delimiter candidate from the first response is strict-valid;
- the second response changes the recovered trajectory hash;
- the observed defect requires a wrapper edit, interior edit, coercion, or any change larger than
  one terminal closer;
- the registered repair budget is exhausted.

The normal incomplete trajectory artifact is withheld. Raw diagnostics remain nonpublic and
checkpoint-bound. A same-lineage resume reuses the recorded disposition and may not regenerate a
failed final-holdout episode.

## Preserved protocol

This amendment does not change the target prompt, parser schema, Qwen model or tokenizer revisions,
deterministic decoding, token limits, target capacity ladder, output ordering, frozen datasets or
splits, encoder seeds, FP32 confirmatory encoder arithmetic, attack budget, thresholds, estimands,
multiplicity correction, bootstrap count, or bootstrap seed.

Arbitrary strict second-response acceptance and automatic wrapper removal are rejected because they
could hide semantic changes after final-holdout behavior was observed. Constrained decoding remains
a separate development study because it changes the decoding and runtime surface and lacks a current
same-device parity result.

## Evidence and claim boundary

Local regression tests can establish the exact positive predicate, semantic-change rejection,
exhausted-repair failure, downstream incidence accounting, fail-fast behavior, and lineage
isolation. They cannot establish Qwen behavior on a fresh A100 session, completion, throughput,
research eligibility, or advisor approval. All hosted and scientific outcomes remain `UNVERIFIED`
until a separately authorized fresh lineage reaches terminal post-run acceptance.
