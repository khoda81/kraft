# Current status

Updated: 2026-09-07 (UTC).

## Stage

The generic byte harness is causally prequential. Exact fixed-N DFA posterior oracles and the N=2 ADD weighted-model-counting evaluator remain available as correctness references. The current sparse-DFA program is a fast **hindsight/oracle structure search**, not yet the KRAFT online Bayesian codec.

A full enwik8 sparse-DFA search with `N=8`, `next` topology, and `K<=256` completed in about 19 minutes using the 1 MB prefilter. All 25 ten-depth audits retained 12/12 of the true full-corpus beam winners. The best candidate again landed exactly at the researcher-imposed `K=256` ceiling, and the joint-cost curve was still improving, motivating removal of semantic search cutoffs.

## Implemented

- Generic `Model<T>` / `Distribution<T>` traits and streaming `predict -> score -> observe` evaluator.
- Uniform-byte and Dirichlet-1/2 adaptive unigram baselines.
- Generic two-learner Bayesian mixture with exact posterior-odds updates.
- Higher-is-better coding ratio (`baseline_cost / model_cost`), with uniform-relative ratio equal to ideal compression ratio.
- Exact fixed-N partial-DFA Bayesian posterior with Dirichlet-1/2 emissions and state-label quotients.
- Persistent histories for the leaf oracle and exact N=2 ADD joint-evidence weighted model counting.
- Proper sparse-DFA description prior over state count, default topology, exception count, exception keys, and destinations.
- Heuristic sparse-DFA beam search with dense compiled transition scoring, multi-fidelity 1 MB proposal prefilter, periodic full-score audit, progress checkpoints, and automatic compressed artifact bundles.
- CI on Rust 1.98.1 and stable, plus documentation-link checks.

## Canonical evaluation interpretation

The primary KRAFT score is the causal Bayesian-mixture prequential cost defined in [prequential.md](prequential.md). For a prespecified fixed DFA, closed-form integrated Dirichlet evidence is exactly equal to its sequential prequential probability. If the DFA is selected using the complete evaluation corpus, its data cost is only an oracle/hindsight diagnostic. Adding negative log prior gives a valid single-hypothesis upper bound on the exact Bayesian-mixture cost.

Historical sparse-search results remain useful for model-language and search-quality analysis, but they are not retroactively relabeled as KRAFT prequential scores.

## Immediate next step

**Rewrite sparse-DFA inference around unresolved Bayesian mass.** The model prior should remain unbounded where declared; `N`, topology, and `K` become latent model structure rather than researcher-selected search dimensions. The engine should maintain disjoint hypothesis regions with exact prior mass plus lower/upper evidence bounds, and refine regions by partitioning, tightening, or exact symbolic resolution.

The rewrite is now runnable as an anytime **evidence-certificate engine**. The prior path covers unbounded state count, topology, exception count, uniform key subsets, and uniform non-default destinations down to concrete sparse DFAs. `sparse-dfa-anytime` refines the full prior by largest unresolved upper mass and reports bounds on `M(x_1:t)`, hence bounds on the exact Bayesian prequential code `-ln M(x_1:t)`. Concrete leaves use the fixed-DFA causal/evidence-equivalent scorer. Unmaterializable huge state counts remain explicit unresolved mass rather than disappearing.

## Model-language follow-up

Ordinary n-grams are finite-state models, but the current sparse transition-description language represents shift-register context dynamics very inefficiently. A later hypothesis language should include short generated transition programs such as shift registers, counters, and latches, optionally wrapped in sparse overrides. Recursive state-local predictors remain a separate research direction.

## Open questions

- Best region representation for unbounded state-count and exception-count tails.
- Strong but cheap likelihood bounds for partially specified transition programs.
- Scheduler objective: upper posterior mass first, upper-mass per estimated compute, or a more principled value-of-computation rule.
- How to extend the transition description language without special-casing n-grams.
- Description-level versus semantic/function-level prior aggregation.
- License selection.
