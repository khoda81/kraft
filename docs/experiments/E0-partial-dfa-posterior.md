# E0 — Exact lazy DFA posterior growth

Status: implementation and oracle invariants validated on feat/partial-dfa-posterior. CI run 34040037792 passed formatting, Clippy, tests, rustdoc, and documentation links on Rust 1.98.1.

## Question

How quickly does the exact Bayesian posterior over small byte-input DFAs grow, and how many posterior components would need to be retained to stay within a requested forward KL bound?

This experiment is deliberately oracle-first. It keeps the exact canonical posterior and measures the smallest top-posterior truncation that would satisfy the bound. It does not yet discard components.

## Model

Fix a labeled DFA state count N. State zero is the start state. Every complete transition-table entry

    (state, byte) -> destination

has an independent uniform prior over the N labeled destination states.

A complete labeled transition table therefore has probability N^(-256N) conditional on N, but unvisited transition entries are never materialized. They remain marginalized.

Each state predicts the next byte with an integrated symmetric Dirichlet-1/2 categorical model:

    P(x = j | state s, history) = (n[s,j] + 1/2) / (n[s] + 128).

Prediction happens before the observation. Then the state's byte count is updated and the transition for the observed (state, byte) pair is followed.

## Canonical lazy branching

When an unassigned transition is required and k canonical states have been discovered:

- each existing canonical destination has prior branch mass 1/N;
- if k < N, all N-k unused labels are aggregated into one new-state branch with prior mass (N-k)/N.

The new state receives the next canonical index. This exactly sums state-label permutations that are still observationally symmetric.

Components with identical canonical transition assignments, current state, and emission sufficient statistics are merged by adding their probability mass in log space.

## Epsilon retention diagnostic

Let the exact posterior over canonical components be P. Sort components by posterior mass and retain the smallest set S with retained mass r satisfying

    r >= exp(-epsilon).

Conditioning the exact posterior on S gives Q. Then

    D_KL(Q || P) = -ln(r) <= epsilon.

The CLI reports the exact minimum retained component count for the requested epsilon. This is a diagnostic only: the oracle itself remains exact.

Hard pruning would require a frontier/replay mechanism to preserve the guarantee on future prefixes, because a component with small posterior now can recover relative posterior mass later. Implement that only if the oracle retention profile shows a useful reduction.

## Run

The dedicated binary has intentionally conservative defaults:

    cargo run --release --bin dfa-posterior -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8

This means:

    states = 2
    limit = 64 bytes
    epsilon = 0.01 nat
    max_components = 2,000,000
    report_every = 1 byte

A more explicit first sweep:

    cargo run --release --bin dfa-posterior -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
      --states 1 \
      --limit 64 \
      --epsilon 0.01

    cargo run --release --bin dfa-posterior -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
      --states 2 \
      --limit 64 \
      --epsilon 0.01 \
      --max-components 2000000

    cargo run --release --bin dfa-posterior -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
      --states 2 \
      --limit 64 \
      --epsilon 0.001 \
      --max-components 2000000

The one-state run must match the byte KT unigram exactly.

The binary stops before an observation whose unmerged branch count could exceed max-components. This is a conservative memory guard: merging can only reduce the realized next component count.

## Reported measurements

Each row reports:

- exact canonical posterior component count;
- minimal retained component count for epsilon;
- retained fraction of component count;
- retained posterior mass and resulting forward KL;
- posterior effective component count exp(H);
- largest component posterior mass;
- total explicit transition assignments;
- total nonzero emission counters;
- approximate component payload MB, excluding HashMap bucket overhead;
- coding ratio versus uniform, higher is better;
- coding ratio versus the byte KT baseline, higher is better;
- elapsed wall time.

The payload figure is intentionally labeled an estimate and is not a process RSS measurement.

## First decision gate

The key quantity is not merely exact component growth. It is the ratio

    retained_components / exact_components

at useful epsilon values.

If this rapidly becomes small, implementing a certified frontier/replay pruner is justified.

If it remains near one while exact components explode, raw DFA-table priors are computationally unattractive and the next model family should introduce structured transition descriptions such as product-shift or small programs.
