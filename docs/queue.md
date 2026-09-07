# Work queue

Updated: 2026-09-07. Work from the first unblocked item. Historical experiment records remain authoritative for completed campaigns.

## Now

- [x] P0 — Lock the causal prequential evaluation contract and Bayesian-mixture/MDL interpretation. Canonical definition: [prequential objective](prequential.md).
- [x] S0 — Implement and analyze sparse-DFA heuristic search, optimized dense scoring, multi-fidelity prefilter, audit, checkpoint output, and compressed artifact bundles.
- [ ] A0 — Define the first anytime hypothesis-region types for the existing sparse-DFA prior. Completion: state-count, topology, exception-count, key-set, and destination regions form exact disjoint prior partitions with unit tests for mass conservation.
- [ ] A1 — Implement a generic anytime mixture frontier with `Partition`, `Tighten`, and `Resolve` refinements plus global lower/upper evidence accounting. Completion: toy finite spaces converge to exhaustive evidence independent of scheduler order.
- [ ] A2 — Add a literal online sparse-DFA learner implementing `Model<u8>` and regression-test the current closed-form `SparseDfa::score()` evidence against `predict -> score -> observe` on fixed structures.
- [ ] A3 — Connect concrete DFA leaves to chunked causal replay and rigorous suffix/evidence bounds. No rejected candidate may silently lose prior mass.
- [ ] A4 — Add checkpoint/resume for the actual frontier rather than only winner/progress TSVs.

## Next

- [ ] A5 — Implement unbounded state-count traversal for the telescoping sparse prior with no semantic `max_states` cutoff.
- [ ] A6 — Replace `max_exceptions` semantics with exact exception-count tail regions; compare work-to-certificate under different prior concentration.
- [ ] A7 — Compare scheduler policies (`U`, `U/cost`, and controlled alternatives) on the same Bayesian target and causal prefix stream.
- [ ] A8 — Run tiny exact prequential mixture experiments and verify cumulative online loss equals negative log joint mixture evidence.
- [ ] L0 — Add a compact generated-transition language capable of expressing byte shift registers/n-grams with short descriptions; do not special-case benchmark order in the inference engine.
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
