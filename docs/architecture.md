# Architecture proposal

The [byte harness](harness.md), generic model/distribution traits, two baselines, and numerical helper are implemented. The boundaries below describe the future finite-model mixture/search system; they do not block running the harness.

## CPU reference

| Boundary | Responsibility | Invariants |
| --- | --- | --- |
| Model / codec | Enumerate descriptions, transition tables, and priors | Stable model identity; explicit coding measure |
| Predictor state | State transitions and probabilistic emissions | Predict before observing; deterministic replay |
| Mixture | Log posterior updates and predictive marginalization | Normalized probabilities; no scheduler penalty in posterior |
| Search frontier | Enumerate/admit hypotheses and bound omitted mass | No overlap in subtree bounds; explicit coverage |
| Scheduler | Allocate evaluation work under a budget | Every evaluated/replayed symbol is charged |
| Runner | Existing byte CLI runs baselines; generators/manifests remain planned | Same byte streams across comparisons |

Begin with direct scalar enumeration and transparent data structures. Avoid a large trait hierarchy until two actual implementations need one. A learned proposal distribution is a later experiment, not a replacement for the first exhaustive reference.

## Finite-state oracle semantics (Q1)

D012 fixes the exact oracle. Each labeled state has integrated Bernoulli emissions with a Jeffreys Beta(1/2, 1/2) prior. Prediction happens before observation; the observed bit updates the current state's emission counts and then selects one of two deterministic outgoing transitions. State zero is the initial state. Raw bytes are processed MSB-first as eight internal bit steps, so the existing byte harness receives a normalized probability mass over 256 values.

For a fixed state count `N`, the first exact mixture is uniform over all `N^(2N)` labeled transition tables. State renamings and unreachable-state redundancy are intentionally retained until E2. Emission counts are mutable inference state, not description bits. A prior across state counts and a self-delimiting model code remain Q5 work.

The scalar implementation favors transparency over packed performance. Explicit operation counts remain independent of wall-clock timing and belong with later scheduler work.

## GPU migration constraints

- Store state and transition/emission data explicitly, with stable integer widths.
- Keep model-level execution batchable; prefer contiguous buffers when profiling motivates them.
- Separate host enumeration/scheduling from a future batched score/transition kernel.
- Define overflow behavior and deterministic reduction tolerances before cross-backend comparisons.
- Preserve the scalar `f64` reference for numerical validation; test any lower-precision path against it.

Do not select CUDA, wgpu, or another backend before E4 establishes where CPU time goes. Small irregular model sets may not amortize transfer/launch overhead; that is an empirical question.

## Late model admission

A newly discovered model needs likelihood and state for the entire observed prefix before receiving an exact posterior weight. Charge replay, or use a provably equivalent sufficient-state reconstruction. Predictions missed before admission cannot be retroactively counted as online predictions. Define deadlines and fallback predictions for exhausted compute budgets.
