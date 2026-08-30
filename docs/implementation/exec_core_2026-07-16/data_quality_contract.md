# Executable Episode Data Quality Contract

## Data product

- Product: ViPIBench-Exec episode and trajectory records.
- Owner: workspace owner; producer: deterministic compiler plus constrained renderer; consumers:
  oracle, detector dataset builder, system runner, evaluation, and thesis evidence generator.
- Criticality: launch-blocking research evidence.
- Privacy: no credentials, PII, real canaries, or external side effects.

## Hard-fail dimensions

- Schema validity: extra fields forbidden; IDs, enums, hashes, paths, and typed arguments valid.
- Completeness: trusted goal, initial state, tool contract, lineage, security assertions, and utility
  assertions present.
- Consistency: label equals attack-intent construction; hard negative implies benign; all action and
  context references resolve.
- Uniqueness: tool, context, action, predicate, event, and episode IDs are unique in their scope.
- Integrity: episode hash and every context content hash match canonical UTF-8 content.
- Capability safety: state operations are restricted to declared path prefixes; consequential
  calls require a matching trusted authorization.
- Reconciliation: oracle result binds episode and trajectory hashes and records final sandbox state.

## Quarantine policy

Any schema, hash, reference, capability, or oracle-determinism failure rejects the record. There is
no fail-open path into the benchmark. Generated surface text may be regenerated only from the same
locked family/split/seed contract and must receive a new content hash.

## Verifiers

- Passing and failing fixtures in `tests/test_episode_schema.py` and `tests/test_exec_oracle.py`.
- JSON Schema export and re-validation.
- CLI dry-run producing `outputs/exec_oracle_verification.json`.
- Repeated oracle evaluation with identical canonical result hashes.
