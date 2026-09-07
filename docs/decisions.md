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

- O1: **Resolved by D018/D019 for the current byte models.** Predict before observe; fixed DFA starts in state 0; state emissions are Dirichlet-1/2 integrated predictors. Future model languages may reopen their own emission semantics.
- O2: proper description priors for each declared model language; finite inference must not silently renormalize away nonzero prior support.
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

## 2026-09-06 — D016: persistent histories preserve the exact fixed-N posterior

**Accepted for the exact oracle:** represent transition assignments and emission observations as immutable parent-linked nodes in append-only `u32`-indexed arenas. A posterior component stores history heads, lengths, semantic fingerprints, the current logical state, and the logical-to-storage state mapping. Branch children append only their new information and share the parent's physical history.

Arena node identity is not model identity. Compact fingerprints accelerate component-map lookup, but collisions are resolved by exact logical transition and emission-content comparison. Predictive state canonicalization reorders logical state mappings without rewriting historical arena nodes. Linear history lookup is accepted for the current short-prefix oracle; a cache requires measured justification.

This changes representation and diagnostics only. It does not change the conditional uniform transition prior, Dirichlet-1/2 emission law, prequential order, quotient semantics, posterior masses, or epsilon-retention calculation. Persistent emissions were included after a transition-only byte-22 measurement showed that cloned emission vectors had become the dominant estimated payload. No pruning or weighted decision DAG is part of this decision.

## 2026-09-06 — D017: exact N = 2 joint evidence by ADD weighted model counting

**Accepted as a second oracle:** for N = 2, treat each encountered transition-table destination as a uniform Boolean variable and represent the hidden state, sufficient emission counts, and integrated log likelihood as reduced ordered algebraic decision diagrams. Average the final likelihood function over its Boolean decisions to obtain exact joint evidence. This is the same labeled transition-table prior as the leaf oracle: variables irrelevant to a trajectory reduce away, reproducing the `2^(-m(s))` transition-constraint weight.

The new evaluator computes joint evidence and cumulative coding cost only; it does not replace the leaf engine when posterior components or sequential predictive distributions are required. Complete-table enumeration and leaf-oracle agreement are mandatory regressions. Node guards and garbage collection manage representation resources but do not prune model mass.

Use discovery quotient as the default leaf-oracle mode. Retain predictive quotient as an explicit option and regression test because it preserves evidence but showed no component reduction and added runtime on the measured prefix. For the ADD prototype, retain first-seen byte-pair variable ordering until variable-order experiments provide evidence for a replacement.

## 2026-09-07 — D018: causal prequential coding is the primary KRAFT score

**Accepted:** KRAFT is evaluated as an online codec. For every observation, form the predictive distribution from the already observed/decoded prefix, score the symbol, then update. The primary score is cumulative prequential coding cost. A structure selected using the whole evaluation stream and then scored on that same stream is a hindsight/oracle diagnostic, not the online score of the structure-learning algorithm.

Optimized batch evidence calculations are permitted only when proven equivalent to the literal causal `predict -> score -> observe` product for a prespecified model. The generic evaluator remains the semantic reference.

## 2026-09-07 — D019: Bayesian prior complexity is paid through prediction, not separate transmission

**Accepted:** KRAFT's actual model is the Bayesian mixture `M(x)=sum_h pi(h) P_h(x)`. Encoder and decoder share the prior and update it causally; no selected model description is transmitted after training. The KRAFT code length is `-ln M(x)`, equivalently the sum of online mixture log losses.

For any fixed candidate `h`, `-ln P_h(x) - ln pi(h)` is a valid single-hypothesis upper bound on the Bayesian-mixture code and has an MDL/two-part form. When one posterior mode dominates, Bayesian prequential cost approaches that value. The prior term is therefore an inference/model-identification penalty, not an extra payload added to the measured mixture code.

## 2026-09-07 — D020: model-space choices are latent; resource choices are inference knobs

**Accepted direction for the rewrite:** quantities that change which hypotheses exist or their prior probability belong inside the Bayesian model. For sparse DFAs this includes state count `N`, default topology, exception count `K`, exception keys, and destinations. Finite compute should determine only which unresolved mass is refined and how tight the current approximation/certificate is.

Hard search cutoffs such as `--states ...` and `--max-exceptions ...` remain valid for historical/oracle experiments but are not acceptable as the semantics of the eventual KRAFT mixture when the declared prior gives omitted structures nonzero mass.

## 2026-09-07 — D021: transition descriptions should reward short generators

**Accepted research direction:** a general DFA is expressive enough to represent fixed-order n-grams, but the present sparse-transition description makes shift-register context machines extremely expensive. Future model languages should assign short descriptions to generated transition functions such as shift registers, counters, latches, and compositions, with optional sparse overrides, rather than special-casing only literal transition tables. Recursive state-local predictors remain a separate extension.

## 2026-09-07 — D022: keep rewrite code minimal and invariant-driven

**Accepted:** keep the inference/model core compact and readable. Internal invariants should be expressed through types, ownership, private construction, and narrow APIs rather than repeated runtime validation of states produced only by KRAFT itself. Defensive checks remain appropriate at external-input boundaries.

The rewrite may make breaking or nuclear internal changes when they remove duplicated logic, stale abstractions, or semantic ambiguity. Readability and line count matter as engineering constraints, provided mathematical correctness and measured performance are preserved.
