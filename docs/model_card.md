# Detector Model Card

Status: `UNTRAINED_RESEARCH_CONFIGURATION`.

## Planned systems

- TF-IDF word/character logistic regression baseline.
- One pinned public prompt-injection detector baseline.
- `microsoft/mdeberta-v3-base` encoder in role-only, text-only, and text+role modes.
- Main encoder seeds: 17, 29, and 43 on identical frozen splits.
- Development-only seed selection for the preregistered text+role system arm; all nine runs remain required ablations.

## Decision policy

Temperature scaling and thresholds are fitted on dev only. Normal uses TPR at target FPR 5% as
the primary core operating point; 1% is secondary. The primary provenance estimand is the
three-seed mean signed injection-minus-benign score-margin difference on 200 canonical pairs, with
paired-bootstrap intervals. Evaluation also includes pairwise ordering, AUPRC, MCC, ECE, Brier
score, grouped confidence intervals, diagnostic slices, hard-negative FPR, matched-pair consistency,
and p50/p95 latency.

## Limitations

No trained encoder or research metric currently exists. A public detector's card is not evidence
of Vietnamese performance. Role-only performance is a leakage diagnostic, not a deployable
result. Text-only and role-only must remain within-pair identical controls on the provenance track.
API outcomes require application-specific utility and review-cost analysis.
