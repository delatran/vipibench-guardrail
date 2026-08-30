# A100 resource-observed single-runtime and analysis-engine amendment

> Historical protocol: superseded before execution by
> `a100_80gb-throughput-autotuned-single-runtime-2026-08-10`. Retained for provenance only.

## Status and evidence boundary

This amendment is locked before the next live confirmatory execution. Its identifier is
`a100_80gb-resource-observed-single-runtime-2026-08-09`; its durable lineage is
`fp32-a100_80gb-resource-observed-2026-08-09`.

The amendment preserves the revised capacity measurement and deterministic statistical engine while
requiring one strict A100 80 GB runtime for every public stage. It adds verified ephemeral-storage
routing, stage-bound GPU/host/disk observations, and a train-only dataloader-worker scout. It does
not report an A100 speedup, full GPU saturation, model-quality change, completed hypothesis, or
research result. Those remain unverified until a fresh, explicitly authorized run produces
hash-bound live artifacts and reaches terminal post-run acceptance.

## Frozen scientific invariants

The confirmatory arm remains FP32. The following items are unchanged:

- mDeBERTa revision and tokenizer revision;
- input modes `role_only`, `text_only`, and `text_role`;
- seeds 17, 29, and 43;
- maximum sequence length 512;
- effective train batch size 64;
- optimizer, learning rate, epoch ceiling, development-only early stopping, and gradient clipping;
- sealed train, development, and final-holdout artifacts;
- threshold policy, estimands, multiplicity rules, bootstrap count, and bootstrap seed;
- Qwen target/generator revisions and static/adaptive query budgets.

Historical outputs remain bound to their original source and lineage. They are neither restored nor
combined with this amendment.

## Capacity measurement revision

The previous capacity probe measured one fixed-order step and could not distinguish warm-up effects
from steady-state throughput. The revised probe:

1. rejects a micro-batch that does not divide the fixed effective batch instead of labeling it as
   out-of-memory;
2. measures complete optimizer updates, including gradient accumulation;
3. performs two warm-up optimizer steps followed by five synchronized measured optimizer steps;
4. interleaves checkpoint-on and checkpoint-off candidates to reduce fixed-order bias;
5. selects by median samples per second inside the locked memory-reserve range; and
6. accepts the selected candidate only after the two-step finite-state canary passes for every
   required input mode and seed.

CUDA work is asynchronous, so synchronized boundaries are required for meaningful wall-clock
measurements. PyTorch documents this timing constraint in its
[CUDA semantics note](https://docs.pytorch.org/docs/main/notes/cuda.html). The capacity result is a
live device measurement, not a prediction from allocated VRAM.

## Single-runtime stage placement

The same strict A100 80 GB runtime remains allocated for the full public-stage chain:

- launch-package, dependency, hardware, and durable-output preflight;
- deterministic data rebuild and baselines;
- encoder development, development-only selection, and test prediction;
- target-agent trajectories;
- adaptive candidate generation;
- attack target trajectories and associated model inference;
- uncertainty, ablation, RQ2, H3, and adaptive statistical analysis; and
- raw-manifest construction, final audit, and terminal acceptance.

The encoder model and Trainer are released before calibration/analysis begins. The public `encoder`
stage ends after accelerator-dependent prediction. Bootstrap uncertainty, ablation analysis, RQ2,
H3, adaptive analysis, raw-manifest construction, and final audit run in the later `analysis` and
`finalize` stages without changing runtime type. Every stage process uses
`requirements-experiment.lock`, re-observes a registered A100 80 GB device, and validates the same
launch authorization, source fingerprint, stage plan, and durable prerequisite chain. A CPU, T4,
L4, smaller A100, or other accelerator resume fails closed.

NumPy bootstrap, JSON serialization, hashing, and file-system audit remain host CPU or I/O work
inside the A100 runtime. Keeping the accelerator allocated removes transition risk and environment
reconstruction; it does not make those operations CUDA kernels and may spend additional accelerator
compute time with low GPU utilization.

## Storage and utilization evidence amendment

The launcher discovers an ephemeral storage candidate before notebook execution. An explicit
`VIPIBENCH_LOCAL_SCRATCH_ROOT` is accepted only if it is a writable local non-root directory with at
least 80 GiB free and no overlap with the bundle, project, or durable output. Automatic selection
prefers a verified named local-scratch mount, then a separate local mount, then the existing local
work volume. The launcher claims only the marker-bound `vipibench-ephemeral` child. Hugging Face,
Torch, XDG, and temporary paths are routed below that child; durable checkpoints, receipts, raw
observations, and final evidence remain under the checksum-bound output root.

One outer observer samples each public-stage process at the locked policy interval. Raw JSONL
samples include GPU utilization, allocated memory, power, temperature, SM clock, host CPU/RAM, and
disk occupancy for the durable output and selected ephemeral roots. A per-session summary is bound
to the strict A100 capacity receipt and storage plan. The cumulative
`resource_measurement.json` is `PASS` only after all nine expected public stages have completed,
their raw samples and summaries reverify, and no stage is missing. Capacity availability, allocated
80 GiB VRAM, or a startup display showing zero allocated VRAM is not utilization evidence.

After the capacity candidate is locked, the encoder stage benchmarks dataloader worker candidates
`[2, 4, 8]` using only training records, the preregistered `text_role` probe representation, two
warm-up batches, eight measured batches, and two repeats. Selection maximizes median samples per
second with deterministic ties toward fewer workers. The resulting plan is bound to config, runner,
train set, capacity plan, and selected batch size. It cannot access test data, change scientific
hyperparameters, or authorize final-holdout feedback.

## Statistical engine revision

The estimands, resampling units, 10,000 iterations, and seed remain fixed. The executable host-side
engine remains explicitly protocol-bound as `cpu-analysis-v2` inside the single A100 runtime:

- grouped bootstrap replicates use one NumPy `SeedSequence([seed, replicate_index])` per replicate
  and are invariant to worker count;
- grouped bootstrap execution uses at most eight CPU worker processes;
- matched-pair bootstrap uses chunked NumPy sampling and is tested against the retained sequential
  reference for exact RNG-equivalent output.

The grouped bootstrap's per-replicate RNG partition differs from the old sequential random stream.
Therefore its Monte Carlo interval bytes may differ even though the estimand, seed, iteration count,
and resampling design remain fixed. The engine and its output contract are unchanged by the
single-runtime placement amendment. Worker-count invariance and reference-parity tests are required
before packaging.

## Development-only BF16 precision scout

`configs/models/mdeberta_bf16_scout.yaml` defines a separate, optional FP32-versus-BF16 AMP scout.
It is not part of the confirmatory stage plan. One command executes a three-seed FP32 control and a
three-seed BF16 candidate sequentially on the same observed A100 device, using train/development
partitions only. Both arms keep FP32 master parameters and optimizer state; only autocast precision
changes. PyTorch's [automatic mixed precision documentation](https://docs.pytorch.org/docs/stable/amp.html)
describes the autocast mechanism, while the pinned Transformers Trainer receives `bf16=True` only
for the candidate arm.

The scout requires all of the following predeclared gates:

- mean development AUPRC degradation no greater than 0.005;
- per-seed development AUPRC degradation no greater than 0.01;
- at least 1.10x median capacity throughput;
- median end-to-end training wall time no greater than 0.90x the FP32 control;
- identical observed A100 device identity for both arms;
- finite canaries and training states; and
- no test prediction, evaluation, or final-holdout feedback artifact.

Even if all gates pass, the evaluator emits only
`ELIGIBLE_FOR_SEPARATELY_VERSIONED_PROTOCOL_REVIEW`. Automatic promotion is always false. The
confirmatory FP32 arm remains active until a future, separately reviewed protocol explicitly changes
it.

## Rival design and falsification

The cost-oriented rival design disconnects A100 after `attack-evaluate` and reconstructs
`analysis` and `finalize` on a CPU-only runtime. It spends fewer accelerator hours, but adds a second
dependency environment, runtime transition, receipt-transfer surface, and operator sequence. The
single-runtime design is preferred for continuity and simpler fail-closed execution. That preference
would change if observed A100 idle cost is material and a separately reviewed CPU transition proves
reliable without changing artifacts.

The amendment is falsified or blocked by any of the following:

- a worker-count change alters grouped-bootstrap output;
- the dataloader worker plan is unbound, uses test feedback, or cannot be rebuilt from its locked
  train-only contract;
- any public stage lacks a verified raw resource observation or safe ephemeral-storage plan;
- the chunked matched-pair engine differs from its sequential oracle;
- any public stage runs without the accelerator dependency profile and a current registered A100
  80 GB observation;
- the runtime type changes before terminal acceptance;
- a capacity candidate is selected from unsynchronized or non-finite measurements;
- BF16 misses any quality, numerical, device, throughput, wall-time, or test-isolation gate; or
- a live A100 run fails to improve useful throughput despite higher memory occupancy.

Runtime continuity, synchronized samples per second, accelerator-stage wall time, end-to-end wall
time, and compute hours are the operational objectives. High host RAM, VRAM allocation, or an A100
remaining attached during host-side work is not treated as GPU utilization or scientific success.
