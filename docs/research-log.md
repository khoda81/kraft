# Research log

## 2026-09-06 — Exact finite-state oracle implementation

**Decision:** Q1 is fixed by D012: labeled deterministic binary transitions, state zero initially, integrated per-state Beta(1/2, 1/2) emissions, and MSB-first factorization inside the existing byte harness. The first exact prior is conditional on a fixed state count and uniform over all labeled transition tables.

**Work:** added scalar `BinaryKtFsm`, stable base-N transition-table ranks, exact labeled-table counts, and `ExactFsmMixture` with an explicit maximum-model allocation guard. Added analytic and normalization tests, including the one-state Beta-Bernoulli check and exact one-state mixture identity. The multiply-shift family is intentionally deferred so it can be evaluated against this oracle rather than define it.

**Validation:** pending GitHub CI on `feat/finite-state-oracle`; no experiment result is claimed yet. The available agent shell has no Rust toolchain and cannot resolve github.com, so repository CI is the validation path.

**Maintenance:** the user's commit `3c6d09f` removed `rust-toolchain.toml`; CI and documentation still referenced it. D013 records the resulting no-pin policy, with MSRV plus current stable CI.

**Next:** if CI passes, mark Q2 complete and add Q3 synthetic generators/runner integration.


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
