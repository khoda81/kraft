# Architecture proposal

Only the numerical helper in `src/lib.rs` exists today. Everything below is a proposed implementation boundary, not an API commitment.

## CPU reference

| Boundary | Responsibility | Invariants |
| --- | --- | --- |
| Model / codec | Enumerate descriptions, transition tables, and priors | Stable model identity; explicit coding measure |
| Predictor state | State transitions and probabilistic emissions | Predict before observing; deterministic replay |
| Mixture | Log posterior updates and predictive marginalization | Normalized probabilities; no scheduler penalty in posterior |
| Search frontier | Enumerate/admit hypotheses and bound omitted mass | No overlap in subtree bounds; explicit coverage |
| Scheduler | Allocate evaluation work under a budget | Every evaluated/replayed symbol is charged |
| Runner | Generate data, run baselines, serialize records | Same streams and budget conventions across comparisons |

Begin with direct scalar enumeration and transparent data structures. Avoid a large trait hierarchy until two actual implementations need one. A learned proposal distribution is a later experiment, not a replacement for the first exhaustive reference.

## Finite-state semantics to settle in Q1

A candidate starting convention is: at time `t`, a state emits a probability for the next binary symbol; after the symbol is revealed, update its emission posterior (if learned) and transition using the observed symbol. A transition table alone is not a probabilistic predictor. Decide whether emissions are a fixed finite grid or integrated Beta-Bernoulli parameters. Specify whether state visits/emission counts are part of mutable inference state rather than description length.

Specify the initial state, reset boundaries, and whether descriptions include all states or only reachable ones. Implement explicit operation counts independently of wall-clock timing.

## GPU migration constraints

- Store state and transition/emission data explicitly, with stable integer widths.
- Keep model-level execution batchable; prefer contiguous buffers when profiling motivates them.
- Separate host enumeration/scheduling from a future batched score/transition kernel.
- Define overflow behavior and deterministic reduction tolerances before cross-backend comparisons.
- Preserve the scalar `f64` reference for numerical validation; test any lower-precision path against it.

Do not select CUDA, wgpu, or another backend before E4 establishes where CPU time goes. Small irregular model sets may not amortize transfer/launch overhead; that is an empirical question.

## Late model admission

A newly discovered model needs likelihood and state for the entire observed prefix before receiving an exact posterior weight. Charge replay, or use a provably equivalent sufficient-state reconstruction. Predictions missed before admission cannot be retroactively counted as online predictions. Define deadlines and fallback predictions for exhausted compute budgets.
