# Experiment plan

The byte harness is implemented and tested; B1 is **partially run**. The [initial enwik8 prefix record](B1-enwik8-baseline.md) captures two user-reported unigram runs with equal printed coding costs across Rust 1.85.0 and 1.98.1. Full-file uniform/unigram results and input metadata remain pending. See [harness](../harness.md) for commands and scoring rules.

The fixed-N partial-DFA [E0 oracle](E0-partial-dfa-posterior.md) and its [persistent-state follow-up](E0-persistent-dfa-state.md) are implemented and measured on the first 26 enwik8 bytes. The broader synthetic E0 campaign and E1–E6 remain planned. Queue IDs are in the [work queue](../queue.md). Shared rules live in [methodology](../methodology.md); create a separate record from the [template](../templates/experiment.md) before execution.

| ID | Question | Comparison / measurements | Gate |
| --- | --- | --- | --- |
| E0 | Is the finite-mixture implementation correct? **Partially run:** exact byte-DFA oracle, state-label quotients, and persistent representation are validated; planned synthetic generators remain pending. | One-state identity, vector-oracle parity, exact quotient agreement, enwik8 prefix growth | Agreement within declared `f64` tolerance; normalized predictions and correct timing |
| E1 | Does the description prior behave as specified? | Enumerated code lengths and Kraft sums; normalized finite prior; ordered search versus exhaustive posterior | Verify code injectivity/prefix-freeness or use an explicitly finite categorical prior; no accidental claim of a universal prior |
| E2 | How much work is representation redundancy costing? | Labeled tables versus canonical representatives with summed original mass; count unreachable/equivalent structures | Preserve oracle predictions when preserving prior mass; quantify runtime/memory tradeoff |
| E3 | Which scheduler best approximates the oracle at fixed compute? | Exhaustive/round-robin/mass/mass-per-cost; several geometric budgets; exact omitted mass, predictive loss, KL and certificate tightness | Valid certificates on every checked prefix; report quality-cost curves even if no scheduler wins |
| E4 | Where does CPU execution spend time? | Scalar baseline versus batched layout; separate enumeration, updates, replay, reductions | Numerical parity; repeated end-to-end timings identify a demonstrated bottleneck |
| E5 | Does GPU execution improve useful throughput? | Same encoded models/streams, CPU versus one chosen GPU backend | Parity within declared tolerance; report transfer/launch costs and crossover batch size |
| E6 | Does a richer program family justify its cost? | Explicit multiply-shift family, then possible stack augmentation; in-class and out-of-class data | Specify induced prior and model coverage; improve measured tradeoffs or record a negative result |

## First campaign: E0

**Hypothesis:** an explicit tiny finite-state mixture can serve as an independent reference for future search approximations.

1. Resolve Q1: transition/emission semantics, prior, and finite family.
2. Work out a one-state case by hand, including the first two observations.
3. Enumerate one- and two-state binary transition tables with start state fixed, subject to the finalized encoding. With labeled states this gives `n^(2n)` transition tables before emissions; emission choices add their own factor.
4. Compare online mixture prediction products to a separately computed sum of per-model full-sequence likelihoods.
5. Check reset behavior, zero-mass/error handling, symbol order, long-sequence numerical stability, and deterministic replay.

Proposed short inputs: empty sequence, single symbols, all-zero/all-one, alternating, and tiny noisy sequences. Exact lengths and tolerances belong in the committed test/run config when implemented. Initial budget: laptop-scale, no GPU; stop and reassess if even the tiny oracle becomes expensive. Correctness is the outcome, not predictive superiority.

## E3 design detail

Freeze the model family/prior from E0–E1. Compare schedulers without changing the target posterior. Use geometric compute budgets and paired streams; include the cost of catching up newly admitted models. Plot or tabulate prediction regret versus total work and end-to-end time. If studying `u/c` as a fixed prior, label that as a separate model-prior ablation, with its own exact reference.

Do not declare success from a favorable average alone: inspect worst checked certificate violation (must be zero within tolerance), per-stream regressions, and whether the policy actually reaches useful tail bounds.
