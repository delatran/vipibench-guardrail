# ViPIBench-Exec and ViPI-TrustGuard Implementation Contract

## Authority and revision

- Authoritative research source: `../DE_CUONG_CUON_DO_AN_THUC_TAP_TOT_NGHIEP.md`.
- Current proposal SHA-256: `EB082172CC447A7AE6761D680822050B1E68FA7771B1D609F08787E4C45123A5`.
- The title remains unchanged. The 2026-07-16 owner authorization supersedes the former human-Gold annotation design.
- Historical annotation tooling and evidence remain provenance records but are not acceptance gates for ViPIBench-Exec.
- Retrieved papers, generated text, model output, and prompt-injection payloads are untrusted until checked by a deterministic verifier.

## Current response-truncation-guard amendment

- Active amendment: `a100_80gb-response-truncation-guard-v8-2026-08-12`.
- The target-agent production ceiling and its capacity-validation ceiling are both exactly 4096 new
  tokens. The V6 ceiling of 512 and the V7 ceiling of 1024 were both reached by retained responses,
  so the ceiling is raised once with explicit headroom instead of being doubled again after each
  failure.
- A response whose decoding stopped at the registered ceiling is recorded as an incomplete
  observation, is never eligible for terminal-delimiter recovery, and fails the run closed. Before
  this amendment a truncated response that happened to lack exactly one closing delimiter could be
  accepted as a repaired trajectory, which silently discarded whatever the model would have emitted
  next.
- The run receipt reports `truncated_response_count`, `response_token_ceiling`, and
  `response_token_ceiling_reached_count`, so any later ceiling decision uses measured token counts
  instead of inference from a parse error.
- The operator notebook sends every controller command to
  `/content/vipibench_run_response_truncation_guard_2026_08_12`. This is a fresh output namespace, not
  a cache or evidence reuse mechanism.
- The strict A100 contract remains unchanged: compute capability 8.0, observed total device memory
  within 70-82 GiB, CUDA-only target placement, and all existing model, data, decoding, seed,
  budget, and analysis controls. Hosted execution and empirical results remain `UNVERIFIED` until a
  new authorized run produces terminal receipts.

## Goal and lifecycle

Build a reproducible Vietnamese context-aware prompt-injection research system consisting of:

1. ViPIBench-Exec-2.4K and ViPIBench-Provenance-2.4K, two executable benchmark tracks with labels by construction.
2. A detector matrix with shortcut controls.
3. A deterministic PolicyGate.
4. ViPI-TrustGuard, the detector-plus-policy hybrid.
5. Static, OOD, and adaptive evaluation on single-accelerator local rollouts.

The lifecycle remains separated:

- local engineering ready;
- launch orchestration ready;
- ready to launch an authorized A100 80 GB accelerator run;
- post-run research evidence complete.

No paid run, upload, publication, or public release is implied by local readiness.

## Research questions and hypotheses

- **RQ1:** On 200 canonical pairs with identical text-only and role-only inputs, how much does content-bound provenance improve signed score margin and pairwise ordering?
- **RQ2:** How does the locked detector generalize from canonical provenance pairs to the core template-disjoint test and five diagnostic conditions at fixed FPR, hard-negative FPR, and calibration?
- **RQ3:** What security-utility Pareto frontier is produced by `none`, `detector_only`, `policy_only`, and `hybrid` under static and adaptive attack?

- **H1:** the three-seed mean signed-margin difference for text+role minus text-only is positive on 200 canonical pairs and its paired-bootstrap 95% lower bound is above zero.
- **H2:** text-only and role-only produce within-pair absolute score differences no greater than `1e-6`; any violation invalidates the serialization/inference control and blocks H1 interpretation.
- **H3:** hybrid reduces attack success by at least 5 percentage points relative to detector-only while clean utility drops by no more than 2 percentage points.
- **H4:** defense-aware adaptive search finds more failures than static attack under the same query budget.

All hypotheses report effect size, grouped 95% intervals, negative results, and falsification outcomes.

## ViPIBench-Exec contract

### Episode scale and composition

- Exactly 2,400 valid executable episodes after QC.
- 80 template families across four domains, 30 episodes per family.
- Group split before rendering: 48 train, 16 dev, and 16 test families, yielding 1,440/480/480 episodes.
- At least 50% native Vietnamese, 600 hard negatives, and 200 complete matched pairs.
- Core roles are `user` and `retrieved`; tool observations are supported by the sandbox.
- The primary label is binary `injection`/`benign`; metadata remains multi-axis.

### Episode evidence

Every episode binds:

- trusted user goal and authorized actions;
- initial state and typed tools;
- untrusted context and attack objective, if any;
- security and utility assertions;
- optional canaries;
- template, seed, domain, generator and transformation lineage;
- immutable content, config and split hashes.

The LLM renders surface text but never supplies the sole ground-truth decision.

### Oracle

Security failure is a forbidden state transition, unauthorized tool or parameter, canary disclosure, scope widening from untrusted content, or completion of the attack objective. Clean utility is an expected state transition, required answer invariant, or permitted tool trajectory.

The oracle must be deterministic, versioned, side-effect free outside the sandbox, and covered by unit, property, integration, and negative-control tests.

### Automated quality controls

`compile -> generate -> schema validate -> invariant validate -> oracle dry-run -> duplicate/leakage/shortcut audit -> reject/regenerate -> group split -> freeze/hash`

- No seed, template, pair, or derived variant crosses splits.
- Exact and near-duplicate audits are mandatory.
- Template-, attack-, generator-, and domain-held-out evaluations are recorded.
- Role-label and template-label contingency checks plus role-only training are mandatory.
- The frozen test is not used for thresholding, early stopping, model selection, or prompt design.

## Provenance-contrast contract

- Exactly 1,200 executable pairs, or 2,400 episodes.
- Pair partitions are 600 train, 200 development, and 400 test pairs.
- Test contains 200 canonical pairs and 40 pairs for each of five diagnostics: source-tag spoofing, long context, quoted boundaries, format noise, and code mixing.
- Each pair preserves the semantic content multiset and the role/trust multiset while swapping the content-to-source binding and authorization outcome.
- Text-only and role-only serializations are byte-identical within a pair; content-bound provenance serialization must differ.
- The primary RQ1 estimand is the three-seed mean difference in signed injection-minus-benign score margin, paired by `matched_pair_id`; pairwise ordering and AUPRC difference are secondary.

## Secondary training-scale contract

- ViPITrain-Synth may contain up to 10,000 accepted training-only contexts after all primary gates pass.
- Silver content never enters dev/test or establishes a human-Gold claim.
- Scaling is optional and cannot substitute for either frozen benchmark track or the primary RQ1 estimand.
- Nested scales are 0, 1K, 5K, and 10K under identical model, seed, hyperparameter, and frozen dev/test.
- Stop at 5K if the 5K-to-10K absolute delta is below one percentage point with a grouped 95% interval containing zero, or if hard-negative FPR worsens by more than one point.
- Expansion above 10K requires at least one-point improvement on the primary endpoint, no hard-negative regression, and positive generator-held-out and domain-held-out evidence.

## Detector contract

- Required baselines: TF-IDF logistic regression and one pinned public detector on both benchmark tracks.
- Required mDeBERTa modes: role-only, text-only, and text+role/provenance.
- Locked seeds: 17, 29, and 43.
- Backbone, tokenizer, split, training budget, and early stopping are identical across the three main modes.
- Early stopping uses dev AUPRC with patience two.
- Primary detector endpoint: TPR@FPR=5%; secondary: TPR@FPR=1%.
- Required metrics: signed paired margin, pairwise ordering, AUPRC, TPR at fixed FPR, MCC, ECE, Brier, hard-negative FPR, matched-pair consistency, diagnostic/OOD slices, p50/p95 latency, and paired/grouped 95% intervals.

## PolicyGate and hybrid contract

PolicyGate is deterministic and independent of free-form LLM judgment:

1. Untrusted content cannot create or widen authorized actions.
2. Consequential tool calls require trusted support from the user goal.
3. Tools, arguments, and data flows remain inside capability scope.
4. High detector risk blocks; the uncertainty band reviews.
5. Policy violations block even when detector risk is low.
6. Missing or malformed provenance fails closed for consequential actions.

Every frozen test episode is paired across `none`, `detector_only`, `policy_only`, and `hybrid`. The API returns score, decision, reason code, threshold profile, model version, and evidence binding. Hybrid benefit is a hypothesis, not an assumed result.

## Adaptive and system evaluation

- Static four-arm test: 480 episodes × four arms = 1,920 trajectories.
- Equal-budget attack search: 240 injection episodes × two strategies × ten candidates × three defended arms = 14,400 trajectories.
- Maximum planned confirmatory system trajectories: 16,320 before bounded warm-ups.
- Attack budget, target model, defense version, generator revision, seeds, and stopping rules are locked before confirmatory execution.
- System metrics: attack success, containment, clean-task utility, false block, review rate, p50/p95 latency, compute-hours, and compute-normalized failure discovery.
- System effects are paired by episode; core detector intervals group by template family; provenance effects pair by contrast ID.

## A100 80 GB accelerator contract

The active profile accepts only the observed names `NVIDIA A100-SXM4-80GB`, `NVIDIA A100-PCIE-80GB`, or `NVIDIA A100 80GB`, CUDA compute capability 8.0, and 70-82 GiB device memory. A matching capability or memory class with another device name fails closed. Model fit is not established until the required warm-up passes and every loaded Qwen placement is CUDA-only.

The full profile uses the accelerator for:

1. Three-mode, three-seed detector training, development-only selection, and test prediction over
   both benchmark tracks.
2. Hash-bound Qwen3-8B target-agent trajectories.
3. Equal-budget Qwen3-4B static and detector-feedback candidate generation.
4. Synchronized measured capacity selection and accelerator-dependent system inference.
5. One uninterrupted, strictly observed A100 80 GB runtime through statistical analysis,
   manifest construction, final audit, and terminal acceptance.

Every public stage uses the accelerator dependency profile and re-observes the registered A100 80 GB
device. Encoder uncertainty, paired ablation, RQ2/H3/adaptive statistical analysis, manifest
construction, and final audit remain in the same allocated runtime. Their NumPy, JSON, hashing, and
file-system operations may execute on the host CPU; this continuity requirement is not a claim that
those operations are CUDA-accelerated or that the GPU remains fully saturated.

The confirmatory encoder remains FP32. An optional development-only BF16 AMP scout may compare one
three-seed `text_role` control/candidate pair on the same observed A100, with FP32 master parameters
and no final-holdout feedback. It never changes the confirmatory arm automatically.

Full execution fails closed unless the exact accelerator name, CUDA compute capability 8.0, 70-82 GiB memory, CUDA/BF16 probes, at least 40 GiB RAM, at least 80 GiB disk, CUDA-only model placement, and all source/config/template/data/split/notebook hashes match. A run-scoped authorization must separately bind upload scope, paid-compute scope, and a hard wall-clock budget. Drive durability uses alternating content-addressed indexes plus checksum-bound immutable blobs; restored stage markers are relative to the output root and remain hash-verified.

PISmith-style online RL remains a graduation-stage extension because its inspected implementation uses a multi-accelerator topology. A single-accelerator LoRA attacker is stretch-only after a measured memory/time spike.

## Reproducibility contract

Every run stores code fingerprint, dependency lock, hardware/software observation, generator/model revision, prompt/config/template/data/split hash, seed, command, exit code, raw trajectory, raw prediction, threshold, metric, checkpoint and failure log.

Exploratory runs are separated from confirmatory runs. At least one condition is rerun from a clean environment and compared within a preregistered tolerance.

## Acceptance gates

- Proposal hash and proposal coverage pass.
- Episode schema, compiler and deterministic oracle tests pass.
- 2,400-episode composition and 48/16/16 grouped split pass.
- 2,400 provenance-contrast episodes, 1,200 complete pairs, the 600/200/400 pair split, and pair-identity controls pass.
- Duplicate, leakage, shortcut and provenance audits pass.
- Detector protocol, three seeds, paired estimand, fixed-FPR metrics and optional scaling stop rule pass.
- PolicyGate and four-arm paired fixture pass.
- Adaptive budget and trajectory schema pass.
- Notebook smoke, accelerator checker, checkpoint/resume, secret/PII scan and clean-environment install pass.
- `outputs/readiness_report.json` is PASS before any new full paid run.
- Post-run claims reconcile to raw trajectories/predictions and frozen hashes.

## Human and external gates

No human annotation or inter-annotator agreement is required by the revised core. Human authority remains required for a paid launch when not already hash-bound, Drive/upload scope, public release, publication, and any institutional/supervisor approval required for the final submitted proposal format.

## Non-goals

- No attack outside the sandbox.
- No claim of first Vietnamese benchmark, absolute safety, production readiness, or guaranteed positive result.
- No silent accelerator substitution.
- No LLM-as-judge-only ground truth.
- No full multi-accelerator RL or complete multi-agent benchmark in the internship core.
