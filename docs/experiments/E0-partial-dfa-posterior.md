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

## Exact state-label quotients

When an unassigned transition is required and k states have been discovered:

- each existing destination has prior branch mass 1/N;
- if k < N, all N-k unused labels are aggregated into one new-state branch with prior mass (N-k)/N.

Two exact quotient modes are available.

discovery preserves the original first-discovery state identities. It removes unused-label permutations but continues to distinguish already discovered states by historical name.

predictive additionally forgets those historical names after every observation. It canonicalizes the complete future-relevant sufficient state — current state, per-state emission sufficient statistics, and assigned transition constraints — under every permutation of discovered state identities, with the current state distinguished. Isomorphic sufficient states are merged by summing their probability mass.

Both modes represent the same Bayesian posterior and therefore must have identical predictive probabilities and marginal evidence. predictive can only use the same or fewer explicit components. The current oracle brute-forces state permutations and limits predictive mode to N <= 8; this is intended for exact small-N experiments, not as the eventual graph-isomorphism implementation.

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
    quotient = discovery
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

- exact quotient posterior component count;
- number of children generated before exact merging on the latest update;
- number of generated children merged away exactly;
- minimal retained component count for epsilon;
- retained fraction of component count;
- retained posterior mass and resulting forward KL;
- posterior effective component count exp(H);
- largest component posterior mass;
- total explicit transition assignments;
- physical transition-arena and emission-arena node counts, plus nodes per component;
- total nonzero emission counters;
- approximate component payload MB, excluding HashMap bucket overhead;
- Linux process RSS when `/proc/self/status` is available;
- coding ratio versus uniform, higher is better;
- coding ratio versus the byte KT baseline, higher is better;
- elapsed wall time.

The payload figure is intentionally labeled an estimate and is distinct from process RSS.

## First decision gate

The key quantity is not merely exact component growth. It is the ratio

    retained_components / exact_components

at useful epsilon values.

If this rapidly becomes small, implementing a certified frontier/replay pruner is justified.

If it remains near one while exact components explode, raw DFA-table priors are computationally unattractive and the next model family should introduce structured transition descriptions such as product-shift or small programs.


## First discovery-quotient result

User-run result on the first enwik8 bytes, N = 2, discovery quotient, epsilon = 0.01 nat:

- reached 1,032,192 exact components after 22 bytes;
- the next observation had 2,064,384 prospective unmerged children and hit the 2,000,000 guard;
- 864,578 components (83.76%) were needed to retain mass 0.990049845681 and stay below 0.01 nat forward KL;
- approximate component payload was 457.310 MB, excluding HashMap bucket overhead;
- coding ratio versus KT was 0.992364462196.

At epsilon = 0.001 nat, 1,013,545 of 1,032,192 components (98.19%) were required.

This establishes the discovery quotient as the baseline. The predictive quotient ablation asks whether a substantial fraction of those million components are merely state-name/isomorphism redundancy. Because both quotients are exact, coding cost and evidence should remain unchanged; only representation size and runtime may differ.

## Predictive-quotient result and persistent follow-up

The first predictive-quotient run produced exactly the same 1,032,192 components at byte 22, with `merged_children_last = 0`. The predictive/evidence results matched discovery, so discovered-state naming was not the source of the observed component growth.

The representation follow-up is recorded separately in [E0b](E0-persistent-dfa-state.md). Persistent transition and emission histories preserve the exact model while reducing the byte-22 payload estimate from 457.310 MB to 107.872 MB. With the guard raised to ten million, both quotients reached byte 26 / 9,289,728 components before stopping ahead of byte 27. The posterior leaf count and leaf-wise CPU remain exponential.
