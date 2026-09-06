# Research log

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
