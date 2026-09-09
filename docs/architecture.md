# Architecture

KRAFT separates the **Bayesian model** from the **inference policy**. The model is defined independently of how much compute is available; finite compute controls only how accurately and how quickly the online Bayesian mixture is approximated.

The canonical coding semantics are in [prequential.md](prequential.md).

## Core boundaries

| Boundary | Responsibility | Invariants |
| --- | --- | --- |
| Hypothesis language / prior | Define model descriptions and their proper prior | Search settings never change which hypotheses exist |
| Fixed predictor | Causal state transition and local predictive learning | `predict` uses only the observed prefix; `observe` happens after scoring |
| Bayesian mixture | Posterior reweighting and predictive marginalization | Prior belongs to inference; no separately transmitted model |
| Hypothesis region | Represent a disjoint subset of descriptions with known prior mass | Refinement preserves total prior mass exactly |
| Evidence bounds | Bound a region's contribution to the mixture | Valid refinement never loses unresolved mass |
| Search frontier | Store unresolved Bayesian mass and partial evaluations | Dormant regions retain their mass; they are not discarded |
| Scheduler | Choose which region/evaluation receives the next unit of compute | Scheduling changes speed, not the Bayesian target |
| Observation store | Permit exact replay for late-activated hypotheses | Replayed state equals causal state from the full prefix |
| Runner / harness | Measure online coding and compute | No future-data-selected structure is counted as an online prediction |

## Mathematical target

For descriptions `h` with prior `pi(h)` and causal fixed-model probabilities `P_h`, KRAFT targets

```math
M(x_{1:T})=\sum_h \pi(h)P_h(x_{1:T}),
```

whose prequential coding cost is

```math
-\ln M(x_{1:T})
=
-\sum_t \ln M(x_t\mid x_{<t}).
```

The posterior is useful internal state; the predictive mixture is the product being coded.

## Anytime inference

An unresolved frontier node represents a disjoint region `R` of the hypothesis space and bounds its joint mixture contribution

```math
L_R \le Z_R(x)=\sum_{h\in R}\pi(h)P_h(x) \le U_R.
```

Allowed refinement operations are:

1. **Partition:** replace `R` by disjoint children whose union is exactly `R` and whose prior masses sum exactly to the parent mass.
2. **Tighten:** keep the same region and improve its likelihood/evidence bounds, for example by scoring a concrete model on more of the observed prefix.
3. **Resolve:** compute `Z_R` exactly, whether by enumeration, dynamic programming, conjugacy, ADD/WMC, or another symbolic method.

No model-space parameter such as maximum state count, topology set, or maximum exception count should be an inference cutoff when the declared Bayesian family assigns nonzero mass beyond it. Resource knobs may control work budget, precision target, thread count, evaluation chunking, checkpoint cadence, or scheduling policy.

## Causal compute-bounded prediction

At byte `t`, the probability used by the codec may depend only on the fixed model/prior, the decoded prefix `x_<t`, deterministic inference state derived from that prefix, and the declared compute/precision policy.

Searching the complete file and then replaying it with structures discovered from future bytes is allowed only as an oracle analysis. It is not a KRAFT prequential run.

Late model/region activation therefore requires replay of the already observed prefix, or a proven equivalent sufficient-state reconstruction. Replay cost belongs to inference compute accounting.

## Fixed-DFA evidence shortcut

For a prespecified DFA with Dirichlet-1/2 state emissions, the deterministic causal trajectory partitions observations by state. Closed-form integrated Dirichlet evidence computed from the final state/byte counts is exactly equal to the product of sequential posterior-predictive probabilities. The optimized scorer may use this identity, but it must be regression-tested against the literal online evaluator.

Choosing that DFA after inspecting the complete stream changes the interpretation: its negative log evidence becomes a hindsight/oracle diagnostic. Adding negative log prior gives a valid upper bound on the Bayesian-mixture cost.

## Model-language direction

The current sparse-DFA family is a useful finite-state description language, not the endpoint. Ordinary n-grams are already representable by finite automata, but shift-register context dynamics are extremely expensive under a literal or sparse-transition-table description. Future hypothesis languages should therefore reward **short transition programs** rather than only small extensional tables.

A later language may include generated transition functions (shift registers, counters, latches, compositions) plus sparse overrides. State-local predictors may also be generalized beyond Dirichlet categoricals. Finite nested automata do not exceed finite-state computational power, but can provide exponentially shorter descriptions and useful parameter sharing; their Bayesian factorization is a separate research direction.

## CPU/GPU constraints

Keep a scalar `f64` reference and deterministic replay semantics. Optimize only behind regression tests that preserve prequential probabilities or exact joint evidence. Candidate models are independent enough for batched scoring, but individual DFA trajectories remain sequential in time; GPU work should follow measured bottlenecks rather than architecture preference.
