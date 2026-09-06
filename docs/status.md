# Current status

Updated: 2026-09-06 (UTC).

## Stage

Byte harness ready; the first user-reported enwik8 prefix baseline is recorded. Full-corpus comparison and finite-state mixture experiments remain pending.

## Implemented

- Minimal dependency-free Rust library with declared MSRV and stable CI coverage.
- Stable log-weight normalization with zero-mass/error handling and numerical unit tests.
- Generic `Model<T>` / `Distribution<T>` traits, streaming byte evaluator, optional per-byte cost sink, and local-file CLI.
- Uniform-byte and Dirichlet-1/2 adaptive unigram baselines. See [harness](harness.md).
- Generic two-learner Bayesian `Mixture<A, B>` with posterior-odds updates from coding advantage; implemented and CI-validated on `feat/mixture-model`.
- Higher-is-better coding ratio (`baseline_cost / model_cost`), including the uniform-relative compression ratio, replaces bits-per-byte in the harness output on that branch. CI run 34035684830 passed Rust 1.85.1, stable 1.98.1, formatting, Clippy, tests, rustdoc, and documentation links.
- CI for formatting, Clippy, tests, rustdoc, and local Markdown file links; dependency-update configuration and contribution templates.
- Research context, proposed architecture, experiment protocols, and prioritized queue.
- GitHub description and all seven research topics configured; confirmed by user-provided CLI output.

The validation and remote setup outcome is recorded in the [bootstrap log](research-log.md). No finite-state learner, enumerator, scheduler, GPU backend, synthetic generator, or automatic run-manifest system is implemented yet. Dataset files remain local to the user.

## Immediate next step

**B1: complete the full-file byte-baseline comparison and input metadata.** The user has run the unigram on a million-byte prefix under two compilers; see the [result record](experiments/B1-enwik8-baseline.md). The request calls the benchmark WikiText, while the supplied paths are enwik8/enwik9, so preserve the actual dataset name. Q1 (finite-state family specification) remains next model-design work, independent of the harness.

## Open questions / blockers

- The exact previously discussed multiply-shift mapping is not recoverable from the available context. It must be specified explicitly before implementation.
- The model encoding, emission prior, and treatment of equivalent state-labelings are unresolved.
- Compute cost may guide scheduling; whether to study a separate speed-weighted model prior remains an explicit ablation.
- License selection remains pending in the queue.

## Evidence so far

The [B1 prefix record](experiments/B1-enwik8-baseline.md) contains the first measured unigram coding cost and its provenance/limitations. Tests separately cover numerical bookkeeping and harness correctness. The log-base conversion and finite-tail bounds in [theory](theory.md) are algebraic statements under stated assumptions, not measured findings.
