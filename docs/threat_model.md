# Threat Model

## Protected decision

The guardrail decides whether a text-bearing input should be allowed, reviewed, or blocked
before it reaches an LLM/RAG execution boundary. It does not execute tools, retrieve secrets,
or claim to make an application safe by itself.

## Trust boundaries

- `user`: direct, untrusted user input.
- `retrieved`: indirect, untrusted external context returned by retrieval.
- `tool`: stretch-only untrusted tool output; excluded from core acceptance evidence.
- system policy, model/config manifests, and frozen splits: trusted only after hash verification.
- web pages, model cards, datasets, and prompt payloads: evidence inputs, never instructions.

## In-scope failures

- Direct or indirect instructions that attempt to override trusted policy.
- False blocks caused by trigger words, quoted security discussion, or benign instructions.
- Shortcut learning from role tokens, source family, position, or duplicated variants.
- Threshold drift, test-set tuning, unpinned model/data revisions, and incomplete provenance.
- Unicode normalization or confusable handling that changes evidence semantics.
- Leakage of raw payloads, PII, credentials, canaries, or provenance identifiers into logs.

## Controls

- Binary label plus orthogonal role/delivery/language/lineage metadata.
- Group split before translation or augmentation; exact and near-duplicate audits.
- Role-only control and counterfactual matched pairs.
- Dev-only threshold selection and immutable test hashes.
- Safe response codes and aggregate logging only; raw payload logging is disabled.
- Fail-closed source/config/data/split/notebook hashes and accelerator contract.
- Human approval for paid compute, upload, public release, and unresolved licenses.

## Residual risk

Prompt injection has no foolproof detector. Distribution shift, adaptive attacks, tokenizer
truncation, retrieval context changes, and application-specific costs remain. The RAG case study
must therefore report utility and review load together with containment.
