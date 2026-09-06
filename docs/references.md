# References and reading queue

## Project context

The 2026-09-04 conversation “Bayesian Program Mixtures” supplies the initial design direction. The [decision register](decisions.md) records what was recovered and what remains open. It is a source of project intent, not an external empirical result.

## Background reading queue

Locate and review primary sources before claiming a formal connection or inheriting their guarantees:

- Kraft inequality and prefix-free coding; distinguish finite categorical priors from actual program codes.
- Solomonoff induction and computability limits of universal mixtures.
- Levin search and runtime-aware search; compare its guarantees to the proposed scheduling heuristics.
- Bayesian sequence prediction / prequential coding.
- Probabilistic finite-state models and prior-preserving automaton canonicalization.

These items are leads, not a completed literature review. When adding a source, record its exact citation/link, the claim it supports, assumptions, and how KRAFT differs.

## Implementation references checked at bootstrap

- [Rust 1.85.0 and edition 2024](https://blog.rust-lang.org/2025/02/20/Rust-1.85.0/): edition/toolchain baseline.
- [actions/checkout releases](https://github.com/actions/checkout/releases): checkout release history; the workflow pins an immutable commit.
