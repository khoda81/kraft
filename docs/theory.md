# Theory and research scope

## Question

Can a finite compute budget be allocated across small program hypotheses so that a tractable Bayesian mixture approaches exhaustive inference, with measurable prediction error and, where possible, a certified omitted-mass bound?

The agreed starting point is tiny finite-state objects rather than a literal VM. The longer-term computation ladder is finite automata → stack-augmented machines → a Turing-complete family. Those later stages are research directions, not delivered capabilities.

## Bayesian mixture and description length

For a prefix-free binary description of program `p`, let `L(p)` be its length in bits. The Kraft inequality gives `sum_p 2^{-L(p)} <= 1`. A finite experiment may normalize these masses over its declared model family; that conditional prior is not the full unbounded prior.

For an observed sequence `x_1:t`, define

```math
u_t(p)=2^{-L(p)}\prod_{i=1}^{t}P_p(x_i\mid x_{<i}),\qquad
C_t(p)=L(p)\ln 2-\sum_{i=1}^{t}\ln P_p(x_i\mid x_{<i}).
```

Thus `u_t(p) = exp(-C_t(p))`. Normalize `u_t` for posterior weights. Before observing the next symbol, predict

```math
P(x_{t+1}\mid x_{1:t})=\sum_p w_t(p)P_p(x_{t+1}\mid x_{1:t}).
```

The implementation stores natural-log weights. Report coding loss in bits by dividing natural-log loss by `ln 2`. A parameterized family must specify a proper parameter prior or code, not only a transition-table code.

## Compute weighting and the apparent log-base knob

If the scheduling score is literally posterior contribution per positive compute cost, then

```math
s_t(p)=u_t(p)/c_t(p),\qquad -\ln s_t(p)=C_t(p)+\ln c_t(p).
```

Changing the logarithm base for the entire score is a unit conversion and preserves rankings. Altering only the coefficient of the compute term yields `u_t / c_t^alpha`, a genuinely different policy. Costs require a defined unit and scope; zero-cost actions need an explicit convention.

**Scheduling is not inference.** Choosing when to evaluate a model with `u/c` does not justify replacing its posterior mass with `u/c`. A fixed prior proportional to `2^{-L}/c(p)` defines a different Bayesian model. A data-dependent, repeatedly recomputed cost penalty is not automatically that fixed-prior model. E3 must separate these interpretations.

Furthermore, posterior mass per cost is a proposed heuristic, not a proven optimal value-of-computation policy. Unknown candidate likelihoods and unknown runtimes require estimates or bounds before evaluation.

## Exact versus truncated inference

Let `S` be evaluated models, `Z_S = sum_{p in S} u_t(p) > 0`, and `R` the true omitted unnormalized posterior mass. Suppose a valid bound gives `R <= U`. The posterior conditioned on `S`, written `q`, satisfies

```math
D_{KL}(q\Vert w)=\ln(1+R/Z_S)\leq\ln(1+U/Z_S).
```

This is the **retained-to-full** KL direction on model identity. Reverse KL is infinite if the full posterior assigns positive mass to excluded models. Marginalizing to next-symbol predictions gives the same upper bound on `KL(Q_predictive || P_predictive)` by data processing, provided both use the same model-conditioned predictions. It does not certify reverse predictive KL or a bound on every individual symbol's log loss.

The omitted posterior fraction is at most `U/(Z_S+U)`; this also bounds total variation of the predictive distributions. To certify the stated KL at most `epsilon` nats, it suffices that `U <= expm1(epsilon) * Z_S`.

For a disjoint, prefix-free unresolved frontier `F`, each prefix `s` has prior subtree mass at most `2^{-|s|}`. If `ell_t(s)` is a proven lower bound on cumulative NLL for **every completion**, then

```math
U=\sum_{s\in F}2^{-|s|}e^{-\ell_t(s)}
```

is a valid tail bound. Discrete symbol likelihoods are at most one, so `ell=0` is a safe but potentially loose starting point. Arbitrary partial-program loss is not necessarily a shared lower bound. Densities for continuous observations need a different argument. Normalization conventions for prior masses must match between `U` and `Z_S`.

No general computable stopping certificate is claimed for unrestricted Turing programs. A finite declared class permits an exact oracle; useful certificates for an expanding or unbounded family remain a research problem.

## Representation and redundancy

State renaming, unreachable states, and different programs with identical predictions can consume search budget. Removing duplicate descriptions changes the effective function prior unless their prior masses are summed. E2 will explicitly distinguish description-level and function-level priors.

Multiply-shift transitions may offer compact arithmetic and batching, but their formula, overflow semantics, representational coverage, and induced prior are open. They must not be treated as equivalent to all DFA tables without proof or measurement.
