# ViPIBench-Exec Core Verification Protocol

## Scope and claim boundary

- [FACT] The revised proposal requires labels by construction plus a deterministic, versioned,
  side-effect-free security and utility oracle.
- This packet covers the typed episode/template contract, in-memory sandbox execution, trajectory
  schema, oracle result, negative controls, and machine-readable verifier.
- It does not claim that the 2,400-episode benchmark, PolicyGate, four-arm runner, adaptive search,
  or an A100 run exists.

## Hypotheses and controls

- Engineering hypothesis: every known forbidden transition, unauthorized consequential call,
  capability-scope violation, canary disclosure, and achieved attack objective is detected by the
  deterministic oracle.
- Utility hypothesis: a valid authorized trajectory satisfies declared expected outcomes without a
  security violation.
- Baseline/control: a benign authorized state update followed by the required safe answer.
- Negative controls: unknown tool, malformed argument, untrusted-only authorization, out-of-scope
  state path, protected-state mutation, canary disclosure, and explicit attack-objective success.
- Unit of analysis: one `(episode, trajectory, oracle_version)` tuple.
- Primary endpoint: 100% expected pass/fail agreement on the locked deterministic fixture matrix.
- Secondary endpoint: byte-identical canonical result hash across repeated evaluation.

## Reproducibility contract

- Oracle version is explicit and no clock, network, filesystem mutation, random source, or model
  judgment is used during evaluation.
- Fixture IDs, content hashes, tool schemas, authorized actions, and outcome predicates are fixed.
- Seeds are not applicable to oracle execution; future episode generation uses locked seed lineage.
- All state changes occur on a deep copy of the episode's JSON initial state.
- Raw verifier output: `outputs/exec_oracle_verification.json`.
- Schema artifact: `docs/schema/vipibench_episode.schema.json`.

## Contamination and falsification

- Prompt text and retrieved context are data, never executable instructions for the harness.
- A schema-valid trajectory that mutates protected state without a violation falsifies the core
  security claim.
- A valid authorized benign trajectory that fails utility or triggers a violation falsifies the
  no-regression control.
- A repeated evaluation with a different canonical result hash falsifies determinism.

## Acceptance gate

`pytest tests/test_episode_schema.py tests/test_exec_oracle.py` and the CLI verifier must pass.
This gate proves only the executable core fixture, not benchmark-scale coverage.

## Observed result (2026-07-16)

- Schema/oracle fixture tests: 12/12 PASS.
- CLI verifier: PASS on 6/6 locked fixtures.
- Exact expected-contract agreement: 1.0.
- Exact repeated-result agreement: 1.0.
- External tool calls: 0; LLM judge calls: 0.
- Evidence: `outputs/exec_oracle_verification.json`.
