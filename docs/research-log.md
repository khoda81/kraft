# Research log

## 2026-09-06 — Symbolic exact DFA evidence

**Goal:** share computation across exact N = 2 DFA hypotheses rather than only sharing their stored histories.

**Work:** added `dfa-wmc`, a reduced ordered ADD evaluator over uniform Boolean transition-table variables. The symbolic recurrence carries the hidden state and state-one sufficient counts; the Dirichlet-1/2 joint log likelihood is rebuilt from count ADDs in a fixed factor order and averaged over the transition variables. Hash-consing shares identical algebraic subfunctions. Exact mark-and-rebuild garbage collection preserves all live roots and retries a node-limited observation once. Added independent complete-table enumeration tests, leaf-oracle evidence comparisons, GC continuation coverage, symbolic-node/payload/RSS diagnostics, and explicit node guards (D017). Discovery is now the default leaf quotient; predictive remains available for regressions.

**Results:** the ADD matched the persistent leaf oracle at every enwik8 prefix through byte 22, with maximum observed difference about `2.42e-12` nat. At byte 26 it used 7,529 live nodes, about 10 MB RSS, and 0.049 seconds, versus 9,289,728 leaves, 1.81 GB RSS, and 109.419 seconds for the leaf run. A ten-million-node workspace reached byte 44. A thirty-million-node workspace reached byte 46 with 5,612,035 live nodes, then exceeded the guard while constructing byte 47; see [E0c](experiments/E0-symbolic-dfa-wmc.md).

**Interpretation:** exact cross-leaf algebraic reuse is substantial and extends the tractable prefix by twenty bytes, but byte 64 remains unmet. The immediate symbolic bottleneck is temporary ADD apply width: byte 47 needs more than 24 million new intermediate nodes even though only 5.61 million nodes are live after byte 46. Variable ordering and within-observation factor scheduling/collection are the next controlled experiments.


## 2026-09-06 — Persistent exact partial-DFA state

**Goal:** quantify how much of the exact N = 2 posterior's memory growth came from cloning historical transition and emission state into every branch.

**Work:** replaced per-component edge vectors with immutable eight-byte transition nodes in a `u32`-indexed parent-linked arena. After a transition-only measurement showed cloned sparse emission statistics dominating payload, applied the same representation to emission observations and packed the common small-N logical-to-storage state mapping inline. Compact component fingerprints are only lookup accelerators; collision resolution compares semantic transition and emission contents through the arenas. Predictive canonicalization changes logical mappings without rebuilding shared histories. Added arena-node/RSS diagnostics and a vector-backed test oracle covering both quotient modes, sequential probabilities, evidence, component counts, retention, and coding ratios (D016).

**Results:** at the existing byte-22 / 1,032,192-component point, the payload estimate fell from 457.310 MB to 107.872 MB, a 76.41% reduction. The 20,840,448 logical transition records across leaves used 2,064,382 transition nodes; emission statistics used 1,205,503 update nodes. With a ten-million guard, both discovery and predictive quotients reached byte 26 / 9,289,728 components and stopped before byte 27's 10,838,016 prospective children. Discovery used 1,071.514 MB estimated payload and 1,809.519 MB sampled RSS. Predictive produced the same evidence-derived results and component count but was slower. See [E0b](experiments/E0-persistent-dfa-state.md).

**Interpretation:** duplicated physical histories were a large but not fundamental part of the explosion. Persistent storage permits roughly nine times the byte-22 leaf count in under 2 GB RSS, while exact retention still requires 6.21 million leaves at epsilon = 0.01 nat. Component count and leaf-wise CPU are now the limiting mechanisms. A weighted decision/arithmetic DAG is a possible separate next experiment; it was not implemented.


## 2026-09-06 — Predictive sufficient-state quotient

**Observation:** the first N=2 discovery-quotient enwik8 run reached 1,032,192 exact components after only 22 bytes and required 864,578 components for epsilon=0.01 nat (1,013,545 for epsilon=0.001). The user noted that paths differing only by state naming should not be distinct hypotheses, motivating a stronger exact quotient.

**Work:** added a predictive quotient that canonicalizes the complete future-relevant sufficient machine state under permutations of discovered state identities after every update. The current state is distinguished; historical state names, including which state was originally called A or B, are forgotten. Added a discovery control mode, generated-versus-merged child diagnostics, and exact tests requiring the two quotient modes to produce identical predictive probabilities and marginal evidence while predictive uses no more components. Brute-force permutation canonicalization is intentionally restricted to at most eight states for this oracle experiment.

**Next:** rerun the N=2 prefix with `--quotient predictive` and compare exact component growth against the recorded discovery baseline. If the reduction is large, pursue a scalable graph/sufficient-state canonicalizer before approximate pruning.


## 2026-09-06 — Exact lazy partial-DFA posterior oracle

**Goal:** measure whether a Bayesian posterior over byte-input DFA transition tables concentrates enough that a certified epsilon-KL truncation could make the otherwise exponential posterior practical.

**Work:** added an exact fixed-N oracle over all labeled transition tables with lazy edge instantiation, canonical aggregation of unused state labels with exact multiplicity, sparse per-state Dirichlet-1/2 byte emissions, exact component merging, retention diagnostics for the minimum top-mass set satisfying D_KL(Q || P) <= epsilon, and a dedicated `dfa-posterior` CLI. The first implementation intentionally does not prune; it measures whether pruning would be worthwhile before adding frontier/replay machinery.

**Validation:** CI run 34040037792 passed on pinned/stable Rust 1.98.1. Tests verify that N=1 exactly reproduces the byte KT unigram, unused-label aggregation preserves 1/N versus (N-k)/N branch masses, the retained-mass KL identity is correct, and predictions remain normalized after branching.

**Next:** run N=1 and N=2 on short enwik8 prefixes and inspect exact component growth versus retained component count at epsilon values such as 1e-2 and 1e-3.


## 2026-09-06 — Rust 1.98.1 becomes the project floor

**Request:** remove the Rust 1.85 compatibility target and move active development completely to Rust 1.98.1.

**Work:** raised `Cargo.toml`'s `rust-version` to 1.98.1, restored `rust-toolchain.toml` pinned to 1.98.1 with rustfmt/Clippy, and simplified CI to test the pinned toolchain plus current stable. Historical 1.85 results remain unchanged as historical records.

**Validation:** pending the new CI run on `feat/mixture-model`.


## 2026-09-06 — Bayesian learner mixture and coding ratio

**Request:** finish the generic learner mixture as exact Bayesian model averaging, where the model-weight ratio is multiplied by relative predictive likelihood, and replace bits-per-byte with a higher-is-better coding ratio.

**Work:** completed `Mixture<A, B>` with equal or explicit prior log odds, stable log-space predictive marginalization, posterior odds updates from `ln P_A - ln P_B`, and updates to both component learners. Added independent tests for equal-prior averaging, likelihood-ratio odds updates, the sequential Bayesian marginal-likelihood identity, and exact-zero support behavior. Replaced bits-per-byte with coding ratio `baseline_cost / model_cost`; the CLI now reports the uniform-relative ratio. Updated B1's recorded prefix metric accordingly.

**Validation:** GitHub CI run 34035684830 passed formatting, Clippy with warnings denied, tests, rustdoc, and documentation links on Rust 1.85.1 and stable 1.98.1. No new corpus experiment was run.

**Maintenance:** commit `3c6d09f` removed the development toolchain file, so this branch also repairs CI to use MSRV plus current stable (D014).


## 2026-09-06 — Toolchain update and first enwik8 prefix result

**User work:** upgraded the development toolchain to Rust 1.98.1 and pushed `1fc543fef87e`; supplied unigram results for the first million enwik8 bytes under Rust 1.85.0 and 1.98.1.

**Recorded:** [B1 prefix result](experiments/B1-enwik8-baseline.md), with equal printed coding metrics and explicit provenance/missing metadata. The two individual timings are not a controlled performance comparison. B1 remains partial.

**Maintenance:** CI now reads the development pin and minimum supported version from their respective TOML files, tests those plus current stable, and formats with the development pin (D011).

**Next:** complete B1 full-file outputs/input metadata; then continue finite-state model design.

## 2026-09-06 — Byte coding harness

**Request:** implement a simple generic predict/observe model interface and score the local text dataset byte by byte, minimizing total coding cost. The user supplied paths to enwik8/enwik9; no dataset is available in this execution environment.

**Work:** added generic distribution/model traits, a streaming byte evaluator with optional per-byte cost callback, uniform and Dirichlet-1/2 unigram baselines, and a local-file CLI with prefix limits and optional CSV costs. Updated methodology and queue to put text evaluation before the finite-model track (D010).

**Validation:** Rust 1.85.0 passed formatting, Clippy with warnings denied, four numerical tests, eight harness integration tests, one doctest, and rustdoc with warnings denied. Relative links in 16 Markdown files passed. Release CLI smoke checks scored 100,000 raw fixture bytes across buffer boundaries at exactly 800,000 uniform bits, verified CSV offsets and prefix limits, checked the analytic AAB unigram cost, and exercised empty prefixes, invalid arguments, missing inputs, and overwrite protection. These are synthetic infrastructure checks, not corpus results.

**Findings:** no corpus benchmark has run. No claims about learned program models follow from infrastructure tests.

**Next:** B1 — run and record the local enwik8 baselines.

## 2026-09-06 — Repository bootstrap

**Goal:** turn the empty `khoda81/kraft` repository into a usable Rust research project with enough context for a new collaborator to continue.

**Context recovered:** prior discussion chose KRAFT, Rust/CPU first with GPU planning, small finite-state/integer objects rather than a literal VM, description-length-plus-loss search, and interest in bounded approximation to exhaustive Bayesian inference. The precise multiply-shift formula was not available. It remains open rather than being reconstructed speculatively.

**Work:** created the Rust crate, a tested log-weight normalization primitive, CI definitions, contribution templates, dependency updates, and linked research records. Proposed an exact finite-class oracle before adaptive search and GPU work. Added explicit distinctions between posterior mass and scheduling utility, KL directions, and prior-preserving deduplication.

**Findings:** no research experiments have run. Numerical unit tests are infrastructure checks. The experiment hypotheses remain untested.

**Validation:** locally passed Rust 1.85.0 formatting, Clippy with warnings denied, four unit tests, one doctest, rustdoc with warnings denied, 15 Markdown files checked for relative file links, shell syntax, and Git whitespace checks. Published via the connected GitHub account after direct Git push lacked shell credentials. [CI run 34023389218](https://github.com/khoda81/kraft/actions/runs/34023389218) passed all three jobs (Rust 1.85.0, current stable, and documentation links) for implementation commit `9ab1d44118e0005ce16eccb1de67c76ba3e34e27`. This subsequent log edit records that result.

**Repository administration:** desired description/topics are encoded in `scripts/configure_repo.sh`. The connector exposes no settings mutation; a direct GitHub REST update returned HTTP 401 (requires authentication). Follow-up: the user ran the script successfully on 2026-09-06 and supplied the verification output showing the configured description and all seven topics. H2 is complete. No license or release tag was created.

**Next:** Q1 — settle the exact finite-state family and hand-compute its smallest example.

## 2026-09-04 — Design discussion (retrospective)

This entry is reconstructed from conversation context, not a contemporaneous experiment record.

The discussion explored Bayesian mixtures over programs, allocating compute according to description length and predictive evidence, and a computation-power ladder from finite-state objects toward richer machines. It identified that changing log base consistently does not add a free parameter when the score is literally Bayesian weight divided by compute. KRAFT and Rust/CPU-first were selected. No measurements or implementation are attributed to this discussion.

## 2026-09-07 — Prequential objective cleanup and anytime rewrite bootstrap

- Made causal Bayesian-mixture prequential coding the canonical KRAFT score in `docs/prequential.md`.
- Clarified that fixed structures selected using the complete evaluation corpus are hindsight/oracle diagnostics; candidate data cost plus negative log prior is a valid single-model upper bound on mixture cost, not the measured KRAFT code.
- Rewrote architecture/status/queue around prior-mass-preserving anytime inference: structural model choices stay inside the prior; compute only controls refinement and precision.
- Recorded the audited `N=8`, `next`, `K<=256` sparse-DFA run as E0e. The best candidate again hit the complexity ceiling and the `K=255 -> 256` step still improved the joint bound, motivating removal of semantic cutoffs.
- Added `SparseDfaLearner`, a literal online `predict -> score -> observe` implementation, and a regression test requiring the optimized integrated-evidence scorer to match it for prespecified DFAs.
- Renamed sparse search outputs to explicitly identify hindsight/oracle data costs and single-model mixture bounds.
- Added the first generic anytime primitives: prior regions, log evidence bounds, frontier nodes, and `Partition` / `Tighten` / `Resolve` refinements. No scheduler is part of these semantics.

## 2026-09-07 — Unbounded state-count region

- Added `PositiveNat`, backed by `num-bigint::BigUint`, so structural naturals are not capped by machine integer width. Zero is excluded by the public construction API.
- Added `StateCount` and `StateCountTail`; the root tail denotes all `N>=1` under `P(N)=1/[N(N+1)]`, and `split()` produces exact `N=n` plus the remaining `N>=n+1` tail.
- Tests verify telescoping mass conservation over repeated splits and explicitly cross the `u64` boundary.
- Simplified `anytime.rs` by removing internal defensive bound-validation machinery; arbitrary bound pairs are no longer publicly constructed.
- Recorded the rewrite style rule: compact code, invariants by construction, defensive validation at external boundaries.

## 2026-09-07 — Topology and exception-count prior regions

- Exact state-count mass now partitions into the three current sparse-DFA topology descriptions with `P(topology)=1/3`.
- Added a finite exception-count tail whose split preserves the normalized truncated prior `P(K|N) ∝ 1/((K+1)(K+2))` exactly.
- The tail stores `K+1` and remaining support mass/count structure, so reaching the mathematical endpoint returns `None` rather than requiring a `K<=max` guard.
- `N=1` naturally has only `K=0`; for `N>1` the support is `0..=256N`.
- Tests compare every exact `K` mass for `N=1,2,8` against the existing concrete prior and verify each tail split conserves mass.

## 2026-09-07 — First runnable sparse-DFA anytime certificate

- Added exact recursive uniform-subset refinement for exception keys using include/exclude probabilities `k/r` and `(r-k)/r`, avoiding enumeration of `choose(256N,K)` children at once.
- Added binary range refinement for each non-default destination; singleton ranges materialize trusted concrete sparse DFAs without re-validating invariants already guaranteed by the region construction.
- Added a minimal largest-upper-mass frontier over the full sparse prior. Unresolved regions use the rigorous likelihood upper bound 1; concrete leaves contribute exact prior times integrated prequential evidence.
- Added `sparse-dfa-anytime`, which reports an interval on exact mixture evidence and therefore on cumulative Bayesian prequential coding cost for a byte prefix.
- Tests verify evidence intervals tighten monotonically and that empty-sequence upper evidence remains exactly unit mass while the prior is repeatedly partitioned.
- This is an evidence-certificate experiment, not yet the finite-compute online codec; causal per-symbol approximate prediction and replay-aware tightening remain next.
