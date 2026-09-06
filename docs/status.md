# Current status

Updated: 2026-09-06 (UTC).

## Stage

Byte harness ready; the first exact finite-state model family is specified, implemented, and CI-validated on a feature branch. Full-corpus baseline comparison and finite-state experiments remain pending.

## Implemented

- Minimal dependency-free Rust library and pinned reference toolchain.
- Stable log-weight normalization with zero-mass/error handling and numerical unit tests.
- Generic `Model<T>` / `Distribution<T>` traits, streaming byte evaluator, optional per-byte cost sink, and local-file CLI.
- Uniform-byte and Dirichlet-1/2 adaptive unigram baselines. See [harness](harness.md).
- Q1 finite-state semantics: labeled binary transitions, state-zero start, MSB-first byte factorization, and integrated Beta(1/2, 1/2) emissions (D012).
- Scalar `BinaryKtFsm` and guarded exact uniform mixtures over a fixed labeled state count are implemented on `feat/finite-state-oracle`. CI run 34034335503 passed docs, formatting, Clippy, tests, and rustdoc on Rust 1.85.1 and stable 1.98.1.
- CI for formatting, Clippy, tests, rustdoc, and local Markdown file links; dependency-update configuration and contribution templates.
- Research context, proposed architecture, experiment protocols, and prioritized queue.
- GitHub description and all seven research topics configured; confirmed by user-provided CLI output.

The validation and remote setup outcome is recorded in the [bootstrap log](research-log.md). No scheduler, GPU backend, synthetic generator, automatic run-manifest system, cross-state-count model prior, or multiply-shift family is implemented yet. Dataset files remain local to the user.

## Immediate next step

**Q3: integrate synthetic binary generators and runner support for E0.** B1 full-file enwik8 metadata/results remain independently useful but require the user's local dataset.

## Open questions / blockers

- The multiply-shift mapping remains intentionally unspecified until the exact FSM oracle is validated; it will be compared against, not substituted for, that oracle.
- The prior across state counts, self-delimiting model encoding, and treatment of equivalent state-labelings remain unresolved.
- Compute cost may guide scheduling; whether to study a separate speed-weighted model prior remains an explicit ablation.
- License selection remains pending in the queue.

## Evidence so far

The [B1 prefix record](experiments/B1-enwik8-baseline.md) contains the first measured unigram coding cost and its provenance/limitations. Tests separately cover numerical bookkeeping and harness correctness. The log-base conversion and finite-tail bounds in [theory](theory.md) are algebraic statements under stated assumptions, not measured findings.
