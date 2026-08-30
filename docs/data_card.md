# ViPIBench Executable Tracks Data Card

## Scope

The project contains two internal research tracks for Vietnamese contextual prompt-injection
detection in LLM and RAG-style tool-use episodes. ViPIBench-Exec-2.4K contains 2,400 executable
episodes from 80 template families. ViPIBench-Provenance-2.4K contains 2,400 executable episodes
organized as 1,200 provenance-contrast pairs. Neither track samples production traffic, and neither
is approved for public release.

## Construction and labels

Both tracks are rendered by deterministic local compilers from locked configurations. No
model-generated text is included in either frozen track. Labels are assigned by construction and
checked against deterministic sandbox security and clean-utility oracles; a free-form LLM judge is
not ground truth.

The benchmark is balanced between 1,200 benign and 1,200 injection episodes. It includes 600 hard
negatives and 200 complete matched counterfactual pairs. Families are frozen into 48 train, 16
development, and 16 test families, giving 1,440/480/480 episodes.

The provenance track is frozen into 600 train, 200 development, and 400 test pairs. Test includes
200 canonical pairs and 200 diagnostic pairs split evenly across source-tag spoofing, long context,
quoted boundaries, format noise, and code mixing. Within each pair, the semantic text multiset and
role/trust multiset are identical. Only the binding between content and trusted/untrusted source is
swapped, which changes the correct authorization outcome. Text-only and role-only serializations
must therefore be byte-identical within each pair.

## Sources and license boundary

All current content is internally authored synthetic material under
`LicenseRef-Internal-Synthetic`, revision `exec-catalog-v1`. The catalog, compiler, templates,
datasets, split manifests, schemas, configurations, and oracle are hash-bound in
`data/provenance_ledger.yaml`.

Qwen3-8B is pinned only as the target system and Qwen3-4B only as the adaptive candidate generator;
neither produced the current core benchmark or the final surface-realization holdout. Any future
rendered training output must receive its own immutable manifest and pass provenance, duplication,
shortcut, and license review before training.

## Intended use

- Internal defensive research and reproducible evaluation.
- Detector comparison, policy-gate testing, and sandboxed four-arm system evaluation.
- Training preparation and later owner-authorized compute under the locked protocol.

## Prohibited or unapproved use

- Public release, publication, upload, redistribution, or production deployment without a new
  explicit decision.
- Treating synthetic fixture scores as empirical model results.
- Using the final holdout for threshold selection, training, or protocol adaptation.
- Executing attack payloads outside the deterministic sandbox.

## Known limitations

The tracks are synthetic and cover four domains and 20 mechanisms per domain. They may contain
lexical or semantic shortcuts not detected by the current categorical and explicit-marker audit.
They do not establish real-world prevalence, production safety, detector quality, or robustness to
all Vietnamese dialects, code-mixing patterns, generators, and adaptive attackers. Model and system
claims require the locked exact-accelerator run and post-run claim audit. The prior frozen-test TF-IDF
artifact is exploratory; the final 480-episode holdout is byte-distinct but does not claim semantic
independence from the synthetic renderer.

## Privacy and release

The compiler uses fictional internal content and no intended direct identifiers. Pattern scanning
is required before launch, but a passing scan is not proof that sensitive content is absent. Public
release remains denied by default. Internal preparation authorization is separate from paid-compute,
upload, publication, and release authorization.
