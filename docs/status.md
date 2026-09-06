# Current status

Updated: 2026-09-06 (UTC).

## Stage

Byte harness ready; the exact fixed-N partial-DFA posterior is implemented and measured on an enwik8 prefix. Persistent transition and emission histories substantially reduce physical state duplication, but exact posterior leaf count remains exponential.

## Implemented

- Minimal dependency-free Rust library pinned to Rust 1.98.1, which is also the declared minimum supported Rust version; CI additionally checks current stable.
- Stable log-weight normalization with zero-mass/error handling and numerical unit tests.
- Generic `Model<T>` / `Distribution<T>` traits, streaming byte evaluator, optional per-byte cost sink, and local-file CLI.
- Uniform-byte and Dirichlet-1/2 adaptive unigram baselines. See [harness](harness.md).
- Generic two-learner Bayesian `Mixture<A, B>` with posterior-odds updates from coding advantage; implemented and CI-validated on `feat/mixture-model`.
- Higher-is-better coding ratio (`baseline_cost / model_cost`), including the uniform-relative compression ratio, replaces bits-per-byte in the harness output on that branch. CI run 34035684830 validated the mixture before the toolchain-floor change. The current branch now targets Rust 1.98.1 plus stable; fresh CI validation is pending.
- CI for formatting, Clippy, tests, rustdoc, and local Markdown file links; dependency-update configuration and contribution templates.
- Research context, proposed architecture, experiment protocols, and prioritized queue.
- GitHub description and all seven research topics configured; confirmed by user-provided CLI output.
- Exact fixed-N Bayesian posterior over byte-input DFA transition tables with lazy transition instantiation, exact unused-label aggregation, Dirichlet-1/2 state emissions, and discovery/predictive state-label quotients.
- Persistent parent-linked arenas for transition assignments and emission observations. Component merging uses semantic content, not arena identity, and short-prefix tests compare both quotient modes against the prior vector-backed oracle.
- DFA diagnostics for logical assigned transitions, physical transition/emission arena nodes, payload estimate, Linux process RSS, elapsed time, exact posterior concentration, and higher-is-better coding ratios.

The validation and remote setup outcome is recorded in the [research log](research-log.md). No approximate scheduler, frontier/replay pruner, GPU backend, synthetic generator, or automatic run-manifest system is implemented. Dataset files remain local to the user.

## Immediate next step

**Choose the next exact-inference representation question.** [E0b](experiments/E0-persistent-dfa-state.md) shows that persistent histories reduce byte-22 payload by 76.41% and extend the N = 2 run to byte 26 / 9,289,728 components, after which leaf-wise compute and component count dominate. A weighted decision/arithmetic DAG is the proposed next research direction; it has not been implemented. B1's full-file baseline comparison remains independently pending.

## Open questions / blockers

- The exact previously discussed multiply-shift mapping is not recoverable from the available context. It must be specified explicitly before implementation.
- The exact fixed-N table experiment has a conditional uniform transition prior and an integrated emission law, but a self-delimiting cross-N model code/prior remains unresolved.
- Compute cost may guide scheduling; whether to study a separate speed-weighted model prior remains an explicit ablation.
- License selection remains pending in the queue.

## Evidence so far

The [B1 prefix record](experiments/B1-enwik8-baseline.md) contains the first measured unigram coding cost and its provenance/limitations. [E0](experiments/E0-partial-dfa-posterior.md) records the exact posterior and quotient experiment. [E0b](experiments/E0-persistent-dfa-state.md) records the persistent-state implementation, exactness checks, byte-by-byte growth, payload estimates, RSS samples, and timing limitations. The log-base conversion and finite-tail bounds in [theory](theory.md) remain algebraic statements under stated assumptions, not measured findings.
