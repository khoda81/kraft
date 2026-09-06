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


## 2026-09-06 — D012: higher-is-better coding ratio

**Accepted:** replace bits-per-byte as a headline evaluation metric with coding ratio. For candidate model M against baseline B, define coding ratio as C_B / C_M, where both coding costs use the same observations and any common log unit. Higher is better: 1 means equal coding cost, values above 1 mean M codes better, and values below 1 mean worse. The ratio is invariant to log base. Against the uniform-byte model, this equals the ideal compression ratio for the fixed byte stream.

Total nats/bits remain valid absolute coding-cost reports; only the per-byte normalization is removed from the primary harness output.

## 2026-09-06 — D013: generic two-learner Bayesian mixture

**Accepted:** the generic two-model mixture stores log posterior odds ln(w_A / w_B). After observing x, update the odds by the likelihood ratio P_A(x) / P_B(x), equivalently add ln P_A(x) - ln P_B(x) = C_B(x) - C_A(x). The next prediction is the posterior-weighted mixture of the component predictive distributions. Both component learners observe every symbol. Equal prior odds are the default; explicit prior log odds are supported.

Exact zero posterior mass is absorbing. If both models assign zero probability to the same observation, the relative odds are left unchanged because that observation supplies no defined likelihood ratio.

## 2026-09-06 — D014: no dedicated development toolchain pin

The user's commit 3c6d09f removed rust-toolchain.toml after D011. CI therefore tests the Cargo.toml MSRV plus current stable and formats on stable. This supersedes D011's claim that rust-toolchain.toml is authoritative.


## 2026-09-06 — D015: Rust 1.98.1 is the development floor

**Accepted:** use Rust 1.98.1 everywhere for active development instead of retaining Rust 1.85 as a compatibility target. `Cargo.toml` now declares `rust-version = "1.98.1"`, `rust-toolchain.toml` pins 1.98.1 with rustfmt and Clippy, and CI tests the pinned toolchain plus current stable.

This supersedes D014 and the compatibility portion of D011. Historical experiment records that happened to use Rust 1.85 remain historical evidence and are not rewritten.
