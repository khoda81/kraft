# Work queue

Updated: 2026-09-09. Work from the first unblocked item. Historical experiment records remain authoritative for completed campaigns.

## Now

- [x] P1 — Implement exact dynamic prediction groups with symbolic transition alternatives, full-vector equality, causal updates and independent oracle checks (D026, [E0h](experiments/E0-dynamic-prediction-groups.md)).
- [ ] P2 — Reduce symbolic maintenance overhead: E0h saved likelihood evaluations but was slower overall. Investigate state-label redundancy and finer count/weight factorization before introducing approximate predictive regions or claiming scalable inference.

- [x] P0 — Lock the causal prequential evaluation contract and Bayesian-mixture/MDL interpretation. Canonical definition: [prequential objective](prequential.md).
- [x] S0 — Implement and analyze sparse-DFA heuristic search, optimized dense scoring, multi-fidelity prefilter, audit, checkpoint output, and compressed artifact bundles.
- [x] A0 — Define prior-mass-preserving sparse-DFA regions through state count, topology, exception count, key subsets, destinations, and concrete leaves.
- [ ] A1 — A minimal sparse-DFA frontier is now runnable and maintains global evidence bounds; generalize/refactor only after the concrete experiment shows what the reusable frontier API actually needs.
- [x] A2 — Add a literal online sparse-DFA learner implementing `Model<u8>` and regression-test the current closed-form `SparseDfa::score()` evidence against `predict -> score -> observe` on fixed structures.
- [ ] A3 — Sparse key/destination inference is now data-directed and analytically marginalizes all unqueried transitions; likelihood bounds are linear in prefix length. Next evaluate convergence on longer prefixes, then marginalize `K` itself or add replay/chunking if it remains dominant.
- [ ] A4 — Add checkpoint/resume for the actual frontier rather than only winner/progress TSVs.

## Next

- [x] A5 — Implement unbounded state-count traversal for the telescoping sparse prior with no semantic `max_states` cutoff.
- [ ] A6 — Exact exception-count tail regions are implemented; next remove `max_exceptions` from the actual anytime search path and compare work-to-certificate under different prior concentration.
- [ ] A7 — Compare scheduler policies (`U`, `U/cost`, and controlled alternatives) on the same Bayesian target and causal prefix stream.
- [ ] A8 — Run tiny exact prequential mixture experiments and verify cumulative online loss equals negative log joint mixture evidence.
- [ ] L0 — Add a compact generated-transition language capable of expressing byte shift registers/n-grams with short descriptions; do not special-case benchmark order in the inference engine.
- [x] L0a — Implement generic finite-state feature partitions with a byte-history constructor and exact causal stop/split posterior. E0g passes tiny enumeration and 100k/1M comparisons; L0 remains open for broader constructors.
- [ ] L0b — Add useful non-context generators and matched stronger emission/smoothing controls; quantify incremental gains before full-corpus evaluation.
- [ ] L1 — Re-run n-gram comparisons under the richer transition-description prior.

## Exact-oracle / optimization backlog

- [ ] O0 — Continue ADD/WMC width-reduction work only when it directly helps resolve frontier regions or serves as a correctness oracle.
- [ ] O1 — Profile CPU batching after the anytime engine establishes realistic workloads.
- [ ] O2 — Benchmark a GPU backend only if CPU profiling shows enough independent candidate/region work to amortize it.

## Later research

- [ ] R0 — Study hierarchical/nested state-local predictors. Finite nesting remains finite-state in computational power but may provide much shorter descriptions and parameter sharing.
- [ ] R1 — Investigate richer program languages, stack machines, and eventually computably bounded Turing-complete families.
- [ ] R2 — Investigate learned/hierarchical priors over transition-program constructors without conflating learned proposal policy with the Bayesian target.

## Housekeeping

- [ ] H1 — Choose a license before encouraging external reuse or publishing the crate.
- [ ] H2 — Select a durable artifact backend before generating outputs too expensive to reproduce.

## Diagnostic follow-up (2026-09-08)

- [x] Expose precise anytime evidence intervals, per-category upper mass, forced-prefix lengths, refinement counts, and work-normalized progress; stop exhausted frontiers.
- [x] Run Rust validation gates and short-input CLI termination regression locally.
- [x] Repeat the 1,000-byte enwik8 experiment with `--diagnostics`; [E0f](experiments/E0-anytime-diagnostic.md) shows shallow trajectories and state-tail expansion. An exposed-tail scheduling ablation tightens slightly but takes more time.
- [ ] Develop symbolic large-state and trajectory likelihood bounds; account for the permanent `B(x)/65536` upper-evidence contribution before claiming arbitrarily tight convergence.
