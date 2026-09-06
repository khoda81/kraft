# Current status

Updated: 2026-09-06 (UTC).

## Stage

Bootstrap, before E0. No empirical conclusions about KRAFT's predictive quality, efficiency, or GPU suitability have been established.

## Implemented

- Minimal dependency-free Rust library and pinned reference toolchain.
- Stable log-weight normalization with zero-mass/error handling and numerical unit tests.
- CI for formatting, Clippy, tests, rustdoc, and local Markdown file links; dependency-update configuration and contribution templates.
- Research context, proposed architecture, experiment protocols, and prioritized queue.

The validation and remote setup outcome is recorded in the [bootstrap log](research-log.md). No finite-state learner, enumerator, scheduler, runner, GPU backend, or experiment dataset is implemented yet.

## Immediate next step

**Q1: specify the smallest exact finite-state model family.** Write its state-transition timing, probabilistic emission rule, code/parameter prior, start-state convention, and finite enumeration bounds. Hand-enumerate a one-state example before building E0.

## Open questions / blockers

- The exact previously discussed multiply-shift mapping is not recoverable from the available context. It must be specified explicitly before implementation.
- The model encoding, emission prior, and treatment of equivalent state-labelings are unresolved.
- Compute cost may guide scheduling; whether to study a separate speed-weighted model prior remains an explicit ablation.
- License selection and repository-admin metadata status are tracked in the queue/log.

## Evidence so far

The existing unit tests concern numerical bookkeeping only. They are not research experiments. The log-base conversion and finite-tail bounds in [theory](theory.md) are algebraic statements under stated assumptions, not measured findings.
