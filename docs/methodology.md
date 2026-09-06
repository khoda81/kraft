# Experimental methodology

## Evaluation contract

For each symbol: form a prediction from the previously observed prefix, score the revealed symbol, then update model weights and states. Hyperparameters chosen with the evaluation stream invalidate a clean held-out claim. Fix exploratory/tuning streams separately from final reporting streams.

The primary performance view is cumulative prequential negative log likelihood **versus total compute**. Report bits/symbol alongside cumulative bits. Compare at matched sequence lengths and budgets; do not hide search cost behind a fixed model count.

## Baselines

- Fair-coin predictor (one bit/symbol) as an implementation sanity check.
- Online Beta-Bernoulli predictor with its hyperprior stated explicitly.
- Short context/Markov predictors with a declared smoothing rule.
- Exact finite mixture over the identical model class and prior, wherever tractable.
- Best individual in-class model in hindsight as a labeled diagnostic, never an online competitor.
- For scheduling: exhaustive order, round robin, posterior-mass priority, and posterior-per-cost priority under identical accounting.

## Measurements

| Dimension | Required measurements |
| --- | --- |
| Prediction | Cumulative bits, bits/symbol, regret to exact mixture at matching prefixes |
| Inference | Exact retained/omitted mass when available; stated KL direction; predictive total variation |
| Search | Models considered/admitted, frontier size, prior mass coverage, duplicate fraction |
| Compute | Model-symbol updates, replay steps, proposal/enumeration work, wall time, peak memory |
| Reproducibility | Revision, configuration, stream/seed, hardware, compiler, output checksums |

Do not assume operation counts and hardware time are interchangeable. Report both. Include model construction, deduplication, search, and replay in end-to-end time; provide kernel-only time as a separate diagnostic. Compile before timing and disclose warm-up/repetition rules. Enforce a budget before costly actions, or report bounded overshoot explicitly.

## Data and uncertainty

Start with binary IID sources, periodic signals, noisy finite-state generators, and finite-memory dependencies. Later include deliberately out-of-class processes and nonstationary streams to reveal model mismatch. Fix lengths, rates, noise, seeds, and generator version in committed run configs before each campaign. Use analytic deterministic cases for correctness; for stochastic comparisons use at least 10 paired seeds initially, expanding only when uncertainty prevents a decision.

Report per-seed values and paired differences, then mean/median and a stated uncertainty interval. Distinguish seed variability from repeated timing variability. Do not select only favorable generators or runs. Failed, timed-out, and numerically invalid runs count and need explanations.

## Approximation checks

On tiny spaces compute the exact mixture and omitted mass at every evaluated prefix. Compare the algorithm's asserted certificate to actual retained-to-full KL and predictive total variation. Track `U/Z_S` as well as whether a threshold is reached. A loose but valid bound and an invalid tight bound are different outcomes. All-zero likelihood is an explicit failure condition unless the protocol defines a probabilistic fallback beforehand.

## Reproduction and retention

Each campaign uses the [experiment template](templates/experiment.md). Record the full command and config, Git revision/dirty patch, seed and data hashes, toolchain and hardware, metrics definitions, and output hashes. Keep concise tables and manifests in Git. Follow the [artifact policy](../artifacts/README.md) for larger outputs. No current experiment runner or remote artifact store is implied by this specification.

## Gates

Correctness gates precede performance claims. E0 establishes a reliable oracle; E1 validates model coding; E2 examines redundancy; E3 compares scheduling; E4 profiles the CPU; E5 considers GPU. Change this order only with an explicit decision and rationale.
