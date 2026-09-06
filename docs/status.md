# Current status

Updated: 2026-09-06 (UTC).

## Stage

Byte harness ready, before the first dataset benchmark. No empirical conclusions about KRAFT's predictive quality, efficiency, or GPU suitability have been established.

## Implemented

- Minimal dependency-free Rust library and pinned reference toolchain.
- Stable log-weight normalization with zero-mass/error handling and numerical unit tests.
- Generic `Model<T>` / `Distribution<T>` traits, streaming byte evaluator, optional per-byte cost sink, and local-file CLI.
- Uniform-byte and Dirichlet-1/2 adaptive unigram baselines. See [harness](harness.md).
- CI for formatting, Clippy, tests, rustdoc, and local Markdown file links; dependency-update configuration and contribution templates.
- Research context, proposed architecture, experiment protocols, and prioritized queue.
- GitHub description and all seven research topics configured; confirmed by user-provided CLI output.

The validation and remote setup outcome is recorded in the [bootstrap log](research-log.md). No finite-state learner, enumerator, scheduler, GPU backend, synthetic generator, or automatic run-manifest system is implemented yet. Dataset files remain local to the user.

## Immediate next step

**B1: run the byte baselines on the local enwik8 file**, starting with a short prefix; record the exact input and total coding costs. The request calls the benchmark WikiText, while the supplied paths are enwik8/enwik9, so preserve the actual dataset name. Q1 (finite-state family specification) remains next model-design work, independent of the harness.

## Open questions / blockers

- The exact previously discussed multiply-shift mapping is not recoverable from the available context. It must be specified explicitly before implementation.
- The model encoding, emission prior, and treatment of equivalent state-labelings are unresolved.
- Compute cost may guide scheduling; whether to study a separate speed-weighted model prior remains an explicit ablation.
- License selection remains pending in the queue.

## Evidence so far

The tests concern numerical bookkeeping and harness correctness, including analytic sequence likelihoods and scoring order. They are not dataset research experiments. The log-base conversion and finite-tail bounds in [theory](theory.md) are algebraic statements under stated assumptions, not measured findings.
