# Current status

Updated: 2026-09-07 (UTC).
 
## Latest measured progress (2026-09-08)

A new generated-state emission-partition posterior now implements causal `Model<u8>` with exact finite-family inference and no replay. With a frozen eight-byte-history generator it achieves 275720.910 nats on 100k enwik8 bytes and 2291568.906 nats on 1M bytes, improving on the best tested fixed-order KT n-gram by 1.781% and 4.912%, respectively. [E0g](experiments/E0-generated-partition-dfa.md) records the full comparison, prior, tests, and hashes.

This is a different declared family with shared emission parameters; it does not fix the original sparse posterior's tail certificate. History-only results demonstrate variable-context learning, not arbitrary recurrent transition discovery. Full-corpus evaluation, stronger smoothing controls, and useful non-context state generators remain open. The historical sparse-search status below still applies to that path.

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

### Data-directed sparse trajectory refinement

The sparse anytime engine no longer enumerates override keys in lexicographic description order. For fixed `N`, topology, and `K`, untouched keys are exchangeable under the uniform `K`-subset prior. The region stores only remaining keys, remaining overrides, and assignments for keys actually queried by the observed trajectory. When the prefix first reaches an undecided `(state, byte)` key, the exact conditional split is `(r-k)/r` for default and `k/r` for override; destination choice is refined only on the override branch.

This is exact Rao-Blackwellization / trajectory weighted model counting: all unqueried key identities and destinations remain analytically marginalized. Likelihood-bound evaluation was also reduced from quadratic prefix rescanning to linear-time sparse emission counts. Concrete DFA leaves are no longer part of this certificate path; when every transition queried by the prefix is forced, the entire remaining completion region is resolved symbolically.

### Anytime convergence diagnostics (2026-09-08)

The anytime CLI now reports precise code endpoints, interval width, endpoint gains, and interval-width improvement per refinement and per search second. `--diagnostics` adds unresolved upper mass by region category, ordinary and upper-mass-weighted forced-prefix length, and cumulative refinement counts. These shares describe upper bounds, not inferred posterior probabilities. Likelihood scan-byte counts include repeated bound replay but exclude transition-only scans. Empty and one-byte inputs stop when the frontier has no refinable work. See [diagnostic usage](harness.md#anytime-convergence-diagnostics).

The instrumentation preserves the prior and likelihood bounds. Local validation on 2026-09-08 passed formatting, clippy with warnings denied, locked tests (including short-input CLI termination), and documentation checks. The [1,000-byte diagnostic and scheduler ablation](experiments/E0-anytime-diagnostic.md) now completed: default bounds are 2383.656250–3631.549819 nats after 100,000 refinements, below the strongest tested fixed-order baseline's cost of 3661.633677 nats at the upper endpoint. This is evidence about the exact mixture, not a measured streaming approximate learner.

An opt-in exposed-tail-mass scheduler slightly improves the interval at equal refinement count but costs about 2.84 times as much elapsed time in this single diagnostic. The default remains unchanged. Stronger symbolic bounds and a reusable causal posterior remain necessary research work toward beating n-grams on substantial data.

The large-state representation also imposes a permanent coding-cost lower-endpoint ceiling of `-ln B(x) + ln(65536)` for prefixes of at least two bytes. See [bound limitations](theory.md#current-sparse-anytime-bound-limitations). Mass accounting is preserved, but arbitrarily tight convergence is not delivered. The relative contribution of loose bounds, prerequisite expansion, and replay overhead to the observed plateau still needs measurement.
