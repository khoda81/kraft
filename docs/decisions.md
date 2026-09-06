# Decision register

Date of initial capture: 2026-09-06. Earlier direction is reconstructed from the 2026-09-04 KRAFT discussion and the supplied project context; missing details remain open.

| ID | Status | Decision | Rationale / consequence |
| --- | --- | --- | --- |
| D001 | Agreed | Name: KRAFT | Reference to description-length priors and the Kraft inequality |
| D002 | Agreed | Rust, CPU first, with a future GPU path | Correctness and enumeration first; explicit packed state and batched kernels later |
| D003 | Agreed direction | Start with finite-state / small integer transition objects | Avoid VM overhead in the first investigation; exact tiny spaces are possible |
| D004 | Agreed direction | Search prioritization uses description length plus prediction loss | Posterior contribution is the organizing quantity; longer programs may enter later |
| D005 | Mathematical clarification | Uniform change of log base adds no scheduling degree of freedom | `u/c` is unchanged; a compute exponent is a separate policy knob |
| D006 | Proposed in bootstrap | Exact oracle before adaptive scheduling | Separates inference correctness from search-policy quality |
| D007 | Bootstrap implementation | Single dependency-free Rust crate; publication disabled | Avoid speculative workspace/framework complexity; split modules when implemented |
| D008 | Proposed in bootstrap | Binary synthetic sequences and probabilistic finite-state emissions for E0 | Small model spaces and analytic examples; exact emission scheme still needs Q1 |
| D009 | Bootstrap implementation | Research prose owns theory/results; code/config owns executable defaults | Reduce documentation drift |

## Open decisions

- O1: transition convention, emissions, start state, model bounds, and parameter priors.
- O2: actual self-delimiting code and model-length prior; how finite truncation is normalized.
- O3: labeled descriptions versus canonicalized models with aggregated prior mass.
- O4: multiply-shift formula, word width, overflow, state range, and structural capacity.
- O5: scheduling cost unit and policy; whether fixed speed-weighted priors merit a separate comparison.
- O6: exact shared-likelihood lower bounds for frontier pruning.
- O7: artifact backend and eventual experiment tracking integration.
- O8: license. No license grant was selected during bootstrap.

For changes, append a dated record stating evidence, alternatives, consequences, and superseded IDs. Do not silently rewrite an earlier accepted decision as though it had always been the plan.

## 2026-09-06 — D010: byte harness first (user-directed)

**Accepted:** expose `Model<T>::predict/observe` and `Distribution<T>::ln_prob`. Require `Model<u8>` for raw text evaluation. Minimize total coding cost on the sequential stream, scoring before every update. The user has local dataset files and supplied enwik8/enwik9 paths; do not label those results as WikiText.

This supersedes the bootstrap ordering that made the first runner depend on finite-state enumeration and D008's binary-only initial benchmark proposal. Binary toy examples and the exact oracle remain useful correctness work. Compute accounting remains relevant to later scheduling comparisons but is not a replacement for the initial coding-cost objective. Added uniform-byte and Dirichlet-1/2 unigram baselines as harness checks; no particular finite-state family is chosen by this change.

## 2026-09-06 — D011: current development pin, separate minimum support

The user committed Rust 1.98.1 as the development toolchain in `1fc543fef87e`. Preserve that version pin rather than replacing it with a moving `stable` channel. `rust-toolchain.toml` is authoritative for development and formatting. `Cargo.toml` separately declares minimum supported Rust; retaining that compatibility check does not force local development onto that compiler. CI reads both files and also checks current stable, avoiding another duplicated version pin in workflow YAML.


## 2026-09-06 — D012: exact labeled binary FSM oracle

**Accepted for Q1/E0:** the first finite-state oracle uses deterministic binary transitions with a fixed labeled state set and initial state zero. Before each bit, the current state emits a Bernoulli posterior predictive distribution with a Jeffreys Beta(1/2, 1/2) prior; after observing the bit, that state's count is updated and the transition indexed by the observed bit is followed. Raw bytes are factored MSB-first into eight such steps. Emission counts are mutable Bayesian inference state, not part of model description length.

For a fixed state count N, Q1 conditions on the uniform prior over all N^(2N) labeled transition tables. It deliberately does **not** yet define a prior across N, canonicalize state renamings, or claim that this is the final compact program representation; those remain Q5/Q6. Stable enumeration ranks the 2N transition targets as base-N digits in (state, bit) order.

The one-state hand check reduces to ordinary bitwise KT prediction. Starting from Beta(1/2, 1/2), the first two bits 1 then 0 have probabilities 1/2 and 1/4, so P(10)=1/8.

**Rationale:** this family is exactly enumerable (1, 16, 729, 65,536 tables for N=1..4), has proper full-support predictions, and provides a reference against which the later multiply-shift family can be measured.

## 2026-09-06 — D013: no dedicated development toolchain pin

The user's commit 3c6d09f removed rust-toolchain.toml after D011. CI now tests the Cargo.toml MSRV plus current stable and formats on stable. This supersedes D011's claim that rust-toolchain.toml is authoritative; no additional rationale is inferred.
