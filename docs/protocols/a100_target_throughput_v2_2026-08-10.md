# A100 target-throughput v2 amendment

> Superseded on 2026-08-11 by
> `a100_80gb-target-format-fail-fast-v1-2026-08-11` after the first complete target-generation
> attempt retained strict-format failures and exposed them only in the near-final analysis stage.

## Status and claim boundary

The workspace owner authorized local optimization after stopping the incomplete hosted run on
2026-08-10. This historical amendment was `a100_80gb-target-throughput-v2-2026-08-10`, and its
durable lineage was `fp32-a100_80gb-target-throughput-v2-2026-08-10`. It cannot restore or combine any
authorization, checkpoint, content store, model output, or stage marker from the superseded
`fp32-a100_80gb-throughput-autotuned-2026-08-10` lineage.

Local tests and manifests can prove only source-level implementation readiness. They do not prove a
speedup, maximum A100 utilization, complete hosted run, model quality, or research result. Those
claims remain `UNVERIFIED` until a fresh authorized A100 execution produces hash-bound capacity,
telemetry, trajectory, and terminal-acceptance artifacts.

## Preserved scientific and safety invariants

The pinned Qwen3-8B model and tokenizer revision, BF16 arithmetic, greedy decoding, disabled
thinking, prompts, maximum input and output tokens, response schema, format-attempt ceiling,
datasets, splits, query budgets, frozen analysis, and final artifact order remain unchanged. Model
offload, host fallback, revision fallback, labels, detector scores, attack outcomes, and final
analysis results remain forbidden capacity-selection inputs.

The confirmatory encoder remains FP32. Public-detector and attack-generator capacity contracts from
the superseded amendment remain unchanged. Only the target-agent execution engine is amended.

## Target-engine v2

The target batch ladder is `[8, 16, 24, 32, 48, 64]`. Each candidate receives one warm-up and three
synchronized 128-token measurements. The prompt batches are nested deterministic token-length
quantiles, so a larger candidate adds work rather than replacing a smaller candidate with an easier
shape. Allocation failure at one candidate skips every larger candidate. Selection still maximizes
median measured samples per second within the 0.88 peak-reserved-memory boundary.

Batch 8 is the live output-equivalence reference. Every wider candidate must reproduce the exact
raw response SHA-256 values for the reference prompts across all measured repeats. The provisional
winner is then tested at the production 512-token limit on the first scheduled batch. It is accepted
only if the reference hashes still match and peak reserved memory remains inside the boundary;
otherwise the next ranked candidate is tested. If no candidate passes, the stage fails closed.

Pending prompts are stably sorted by decreasing rendered input-token length with episode ID and
prompt hash as deterministic tie breakers. This reduces left-padding waste and makes the first
production validation batch the worst scheduled input-length batch; later batches cannot have a
longer input. Checkpoints remain episode-addressed, and the final trajectory artifact is restored to
frozen dataset order before hashing. The production validation generation becomes the first real
target batch rather than being discarded.

Malformed responses from the same initial batch are repaired together in one generation call per
format-attempt round. Each record retains its own prompt, previous response, parse error, token
counts, diagnostic hash, fallback disposition, and amortized synchronized wall time. Fallback facts
are still written before checkpoints and remain immutable on resume.

The stage writes `_capacity_plan.json` before long target generation. It binds the config, dataset,
checkpoint source, attempted batches, repeat rates, response hashes, input-scheduling contract,
production validation, and final selected candidate. This file is an operational receipt, not a
speedup claim.

## Rival design and adoption threshold

Paged KV-cache continuous batching through a separately pinned vLLM environment may outperform
static Hugging Face batches for variable output lengths. It is not added to the confirmatory
dependency lock because no same-A100 output-equivalence and end-to-end performance result exists.
Adoption requires a separate development-only engine study, exact model/tokenizer binding, raw
output and format-fallback comparison, dependency/CUDA compatibility proof, and at least 1.5x
end-to-end valid-trajectory throughput on the same A100 after startup overhead.

No quantization, constrained decoding, `torch.compile`, alternate attention backend, or precision
change is authorized by this amendment.

## Live closure

A performance claim requires a named baseline, at least three comparable live measurement windows
or complete runs, valid trajectories per hour, scout overhead, repair/fallback rates, stage-bound
GPU telemetry, output hashes, and terminal acceptance. A wider selected batch, higher VRAM use, or a
single recent counter window does not by itself establish a significant speedup.
