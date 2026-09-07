# E0e — Sparse-DFA hindsight search on full enwik8

Status: completed local heuristic/oracle run; informs the anytime-inference rewrite.

## Scope and interpretation

This experiment uses the complete evaluation corpus to choose sparse-DFA structures. Therefore candidate data costs are **hindsight/oracle fixed-model diagnostics**, not KRAFT's online prequential coding cost. For each fixed candidate, integrated Dirichlet-1/2 evidence is still exactly equal to its causal prequential probability.

For any candidate `h`, `C_h(x) - ln P(h)` is a rigorous single-hypothesis upper bound on the exact sparse-DFA Bayesian-mixture cost.

## Model family used by the run

- state count fixed to `N=8`
- default topology fixed to `next`
- sparse override count searched through the artificial bound `K<=256`
- beam width 12
- 12 proposed keys per parent and 8 destinations per key
- full 100 MB enwik8 structure score
- 1 MB proposal prefilter retaining 96 candidates
- full-score audit every 10 exception depths

These are heuristic search choices, not the desired final Bayesian semantics.

## Main result

The best discovered candidate again sits exactly at the maximum searched complexity:

```text
N = 8
topology = next
K = 256
oracle fixed-model data cost = 313,674,088.479826 nat
prior cost = 1,850.443972 bit
single-model mixture upper bound = 313,675,371.109848 nat
coding ratio vs KT, oracle fixed model = 1.122599196782
coding ratio vs KT, certified mixture lower bound = 1.122594606433
coding ratio vs uniform, certified mixture lower bound = 1.767807725822
```

Relative to the previous `K<=24` winner, the single-model mixture upper bound improves by about 12,723,944.90 nat, equivalent to about 2.295 MB of ideal code.

The final step from `K=255` to `K=256` still improves the joint/upper-bound cost by about 5,367.44 nat after paying the additional prior penalty. The complexity ceiling is therefore visibly binding; the run does not support treating `K=256` as a learned optimum.

## Prefilter audit

All 25 audits at depths 10, 20, ..., 250 retained 12/12 of the true full-corpus top-12 beam candidates.

The safety margin deteriorated at high complexity. At depth 250, the worst true top-12 candidate was ranked 95th by the 1 MB prefilter while the promotion cutoff was 96. Thus the 1 MB prefix remained a strong cheap signal, but a smaller fixed promotion cutoff would have changed the search near the interesting high-complexity regime.

## Interpretation

1. The sparse family continues to buy useful structure far beyond the original `K=24` boundary.
2. A fixed beam/prefilter remains a useful heuristic optimizer, but increasingly fragile promotion ranks make it unsuitable as the semantics of Bayesian inference.
3. The result strongly motivates an anytime frontier where rejected hypotheses remain dormant probability mass rather than disappearing.
4. `N`, topology, and `K` should move inside the Bayesian model; resource knobs should control only which unresolved mass is refined.

## Follow-up

Replace the bounded beam search with a prior-mass-preserving anytime mixture engine. Validate first on tiny exhaustive spaces, then reuse the fixed-DFA scorer as an exact leaf evaluator behind causal/prequential regression tests.
