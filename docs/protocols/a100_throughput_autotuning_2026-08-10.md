# A100 bounded throughput-autotuning amendment

> Superseded on 2026-08-10 by `a100_80gb-target-throughput-v2-2026-08-10` after the first live
> A100 target-stage observation showed that the target ladder stopped at batch 16. This document
> remains immutable design provenance for the superseded lineage.

## Status and evidence boundary

This amendment is locked before the next live confirmatory execution. Its identifier is
`a100_80gb-throughput-autotuned-single-runtime-2026-08-10`; its durable lineage is
`fp32-a100_80gb-throughput-autotuned-2026-08-10`.

The amendment improves the execution path for the dominant GPU workloads while preserving the
frozen scientific arm. Local source checks, tests, and manifests can establish implementation
readiness only. They do not establish a speedup, maximum A100 utilization, completed execution,
model quality, or a research result. Those claims require a fresh authorized A100 run and its
hash-bound capacity, telemetry, output, and terminal-acceptance artifacts.

## Preserved scientific invariants

The confirmatory encoder remains FP32. Model and tokenizer revisions, datasets, splits, input modes,
seeds, effective train batch size, optimizer, maximum length, development-only early stopping,
threshold policy, target/generator decoding rules, attack query budgets, estimands, multiplicity
rules, bootstrap count, and bootstrap seed remain unchanged. Historical outputs remain bound to
their original lineage and are ineligible for restore into this one.

## Bounded live selection

The runtime selects capacity from explicit ladders rather than assuming that the largest batch is
fastest:

- encoder training: the existing micro-batch/checkpointing scout, numerical canary, and effective
  batch-size contract remain authoritative;
- public detector: batches `[32, 64, 128, 256]`, selected using development inputs only;
- target agent: batches `[1, 2, 4, 8, 16]` with deterministic capacity-probe generation;
- static attack generator: batches `[1, 2, 4]`; and
- guided attack generator: batches `[1, 2, 4, 8]`.

Every inference scout performs one warm-up batch, synchronizes the accelerator around measured
work, records repeated rates, aggregates with the median, and selects maximum measured throughput
within the 0.88 peak-reserved-memory boundary. Candidate allocation failures are recorded and
skipped; other runtime failures remain fatal. PyTorch documents why synchronization is required for
meaningful CUDA wall-clock timing in its
[CUDA semantics note](https://docs.pytorch.org/docs/main/notes/cuda.html).

The public detector is loaded once and reused for development and test. Its capacity decision is
bound to the development input hash and explicitly records that test was not accessed. Target and
generator scouts run only inside their already-authorized execution stages and may observe input
shape and device throughput, but selection cannot use labels, detector scores, attack success,
system outcomes, or final-holdout feedback.

## Input-pipeline execution

The train-only dataloader scout continues to select from `[2, 4, 8]` workers. The selected count is
now passed with persistent workers and a prefetch factor of two into both training and final
prediction. The isolated-environment probe rejects a Transformers build that does not expose those
arguments. Hugging Face documents that persistent workers avoid worker shutdown after each pass and
that prefetching keeps batches ready when workers are enabled in its
[Trainer recipes](https://huggingface.co/docs/transformers/main/trainer_recipes).

## Host-only cells and excluded optimizations

Data validation, JSON and manifest construction, statistical analysis, hashing, and final audit are
CPU or I/O work even though the single-runtime contract keeps the A100 allocated. Artificial GPU
allocations or unrelated kernels would not speed those cells and are forbidden as utilization
evidence.

`torch.compile`, continuous batching, a different attention backend, and automatic BF16 promotion
remain outside this amendment. Any one of them requires a separately versioned development-only
equivalence and performance study because it may change numerical or stochastic behavior.

## Live falsification and closure

The next authorized A100 run must produce, at minimum:

1. passing strict-device and CUDA-only placement receipts;
2. capacity plans with every attempted candidate, repeat rate, peak reserved memory, and selection;
3. stage-bound GPU/host telemetry for all nine public stages;
4. output hashes and the unchanged scientific-contract bindings;
5. wall-clock and throughput comparisons against an explicitly named baseline; and
6. terminal post-run and final-claim acceptance.

A wider candidate ladder or an excluded optimization may be faster. Such a result would falsify any
claim that this bounded ladder is globally optimal; it would not invalidate the integrity of the
recorded run within this amendment.
