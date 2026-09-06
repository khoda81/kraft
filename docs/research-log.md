# Research log

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
