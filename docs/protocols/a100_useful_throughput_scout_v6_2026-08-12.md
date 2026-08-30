# A100 Useful-Throughput Scout V6 Amendment

Protocol amendment: `a100_80gb-useful-throughput-scout-v6-2026-08-12`

Durable lineage: `fp32-a100_80gb-useful-throughput-scout-v6-2026-08-12`

## Evidence-triggered scope

The retained event-ID-envelope-v4 resource receipt is evidence for a failed historical lineage, not
an eligible research result. Its stage-level observations showed that the encoder, core target, and
attack-target workloads could be compute-saturated while occupying substantially less than the
available 80 GiB. Adaptive attack generation was the distinct avoidable bottleneck: median GPU
utilization was 39%, peak utilization reached 100% only transiently, and median/peak device memory
was approximately 12.4 GiB. The amendment therefore optimizes useful measured throughput rather than
VRAM occupancy.

No attack outcome, oracle decision, utility score, detector threshold, hypothesis result, or final
claim informed the optimization choice. Historical output remains immutable and cannot enter v6.

## Locked scientific invariants

The following remain unchanged:

- Qwen3-4B generator model and tokenizer revisions and BF16 inference arithmetic;
- prompt templates, root seed, derived per-batch seeds, temperature, top-p, disabled thinking, and
  maximum input/output token limits;
- static batch candidates `[1, 2, 4]`, guided batch candidates `[1, 2, 4, 8]`, proposal grouping, and
  selected-batch semantics;
- 240 frozen injection bases, ten static and ten guided proposals per base, detector-only feedback,
  all candidate validity rules, and the equal-query estimand;
- target-agent model, batch ladder, deterministic decoding, response envelope, terminal-delimiter
  recovery, datasets, thresholds, system arms, trajectory budgets, multiplicity rules, and analyses.

Execution-validation generations are out-of-budget capacity/canary calls. They are recorded in the
capacity plan and never enter candidate, detector-score, target-trajectory, or analysis artifacts.

## Execution candidates

After the existing capacity scout selects one batch size per attack-search strategy, the runner
measures these execution candidates at those exact batch sizes:

1. `dynamic_eager`: the pre-amendment generation call, retained as the mandatory baseline;
2. `static_compile`: an on-device fixed-size KV cache covering the locked 2,048-token input plus
   384-token output envelope, automatic `torch.compile` decode with the pinned Transformers
   `CompileConfig(backend="inductor", mode="reduce-overhead", fullgraph=false)`. Tokenized inputs
   remain on the model's CUDA device, as required by the pinned generation implementation.

Each strategy uses one synchronized warm-up and three synchronized measurements at the production
384-token ceiling. Candidate evidence records repeat proposal rates, decoded-text hashes, peak
reserved memory, total device memory, and an equal-budget time estimate for 2,400 proposals per
strategy.

## Selection and fail-closed production canary

An execution candidate is eligible only when:

- both strategy measurements complete;
- every repeat's decoded-text SHA-256 exactly matches the corresponding dynamic/eager baseline;
- all repeat rates are finite and positive;
- peak reserved memory divided by total observed device memory is at most `0.88`; and
- its estimated total time for both equal strategy budgets is lower than the eligible alternatives.

Ties prefer the baseline. A missing or invalid baseline aborts the scout. An optional-mode error is
stored by type and message hash, without placing raw environment error text in the public capacity
record.

If `static_compile` is selected, the first real generation batch is executed once through each mode
with the identical seed. Exact decoded-text hashes are required again. A mismatch or optional-mode
exception returns the already computed baseline output, releases the model's optional static cache
and compiled-call state, resets the compiler cache when available, empties unreferenced accelerator
cache, records the fallback, and latches subsequent production calls to `dynamic_eager`. A later
optional runtime error follows the same fallback path after resetting the seed for the baseline
call. The durable capacity plan is validated before every update and before reuse.

## Rejected designs

- Dummy tensors or allocator reservations are rejected because they increase occupancy without
  samples, tokens, or completed work.
- Larger target-agent batches are rejected because retained capacity evidence showed exact response
  hash divergence even when raw throughput was higher and memory remained available.
- Continuous batching and paged attention are rejected for this amendment because they change request
  scheduling, KV allocation, and stochastic RNG consumption without a current exact-parity result.
- Relaxing equivalence to approximate logits, semantic similarity, or aggregate metric tolerance is
  rejected because sampled candidate text can change the guided search history and estimand.

## Lineage and claim boundary

The adaptive generator configuration, capacity-plan schema, candidate-checkpoint schema, source
bindings, outer protocol, and durable namespace are new. V5 and all earlier state are forbidden from
restore, reclassification, or combination with v6.

Local tests and package gates can establish configuration locking, measurement derivation, exact-hash
selection, production fallback, durable revalidation, and lineage isolation. They cannot establish
that the optional path will be selected on an A100, will consume a particular amount of VRAM, will
improve utilization or wall-clock time, will complete the experiment, or will support a research
hypothesis. Those claims require a fresh authorized A100 execution, the v6 capacity plan, all
stage-bound telemetry, and terminal post-run acceptance.
