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

## Hidden trajectories and exact DFA weighted model counting

For a fixed N-state deterministic byte-input DFA and observed bytes `x_0:T-1`, introduce the hidden state trajectory `s_0:T` with fixed start `s_0 = 0`. A trajectory is compatible with at least one deterministic transition table exactly when

```math
s_i=s_j\ \text{and}\ x_i=x_j \quad\Longrightarrow\quad s_{i+1}=s_{j+1}.
```

Let `m(s)` be the number of distinct `(s_t, x_t)` pairs touched by the trajectory. Under the independent uniform destination prior, marginalizing all untouched transition entries gives transition-constraint weight `N^{-m(s)}`. With state/byte counts `n_{q,b}` and totals `n_q`, integrating the symmetric Dirichlet-1/2 emissions gives

```math
P(x\mid s)=\prod_{q=0}^{N-1}
\frac{\Gamma(128)}{\Gamma(n_q+128)}
\prod_{b=0}^{255}
\frac{\Gamma(n_{q,b}+1/2)}{\Gamma(1/2)}.
```

Therefore the exact fixed-N evidence can be written as a weighted trajectory count:

```math
P(x_{0:T-1})=
\sum_{s_1,\ldots,s_T}
\mathbf 1[\text{DFA-consistent}(s,x)]
N^{-m(s)}P(x\mid s).
```

For N = 2, each transition destination is a Boolean variable. Averaging a likelihood function over the uniform complete transition table is equivalent to the expression above: variables untouched on a particular path cancel under marginalization. This permits exact BDD/ADD, variable-elimination, tensor-contraction, or other weighted-model-counting representations without changing the Bayesian model.

For prequential Bayes prediction, the chain rule gives

```math
\sum_{t=0}^{T-1}-\ln P(x_t\mid x_{<t})=-\ln P(x_{0:T-1}).
```

Thus an exact joint-evidence evaluator is sufficient for cumulative coding-cost comparison even if it does not expose every sequential posterior mixture. It is not sufficient when the experiment needs posterior components, next-symbol probabilities for unobserved alternatives, or search-policy diagnostics.

## Proper prior over all finite labeled DFAs

KRAFT's first unbounded recurrent-DFA prior uses a unary code for the state count:

```math
P(N)=2^{-N},\qquad N=1,2,\ldots
```

This is already normalized because the state-count code can be read as `1^(N-1)0`.

Conditional on `N`, the start state is fixed to label zero and every one of the `256N` transition entries independently chooses a destination uniformly from the `N` labels:

```math
P(\delta\mid N)=N^{-256N}.
```

Therefore each complete labeled DFA has prior

```math
P(N,\delta)=2^{-N}N^{-256N}.
```

For fixed `N`, summing over all `N^(256N)` labeled transition tables gives `2^-N`; summing over every finite `N` gives one. State-emission probabilities are not point-estimated parameters: each state's byte categorical distribution is integrated under the symmetric Dirichlet-1/2 prior.

This is deliberately a description-level prior. State renamings, unreachable-state variants, and other distinct labeled descriptions may induce the same predictive function and retain their combined prior multiplicity. A future function-level quotient would need to sum those masses rather than simply delete duplicate descriptions.

### Exact finite prefix with a certified unbounded tail

The implementation evaluates `N=1..N_max` exactly using the lazy partial-DFA oracle. It does not renormalize the declared model prior and pretend larger DFAs do not exist.

The omitted state-count prior is exactly

```math
U_0=\sum_{N>N_{\max}}2^{-N}=2^{-N_{\max}}.
```

For any observed prefix, every likelihood is at most one, so the omitted unnormalized posterior mass remains bounded by `U_t <= U_0`.

If

```math
Z_t=\sum_{N=1}^{N_{\max}}2^{-N}P_N(x_{1:t})
```

is the exact joint mass of evaluated classes, then the posterior conditioned on evaluated classes has certified omitted-mass and forward-KL bounds

```math
\delta_t\le\frac{U_0}{Z_t+U_0},
\qquad
D_{KL}(Q_t\Vert P_t)\le\ln\left(1+\frac{U_0}{Z_t}\right).
```

The truncated predictive distribution is exact conditional on `N<=N_max`; the KL certificate quantifies how far that retained posterior can be from the full unbounded DFA posterior. The certificate can loosen as evidence shrinks, so larger state-count classes must eventually be opened if the tail becomes important.

### Description cost

For a complete labeled `N`-state transition table, the ideal negative-log prior cost is

```math
L(N,\delta)=N+256N\log_2N\quad\text{bits}.
```

The first term is the unary state-count code. The second is the conditional transition-table code. This cost is not a claim that a literal table is the best way to describe structured DFAs; later KRAFT priors can add short program descriptions for transition functions such as shift registers, latches, counters, or generated automata and Bayesian-mix them with the literal-table family.
