# Experimental methodology

## Evaluation contract

For each symbol: form a prediction from the previously observed prefix, score the revealed symbol, then update model weights and states. Hyperparameters chosen with the evaluation stream invalidate a clean held-out claim. Fix exploratory/tuning streams separately from final reporting streams.

The initial benchmark objective is **total prequential coding cost on the same raw byte stream**, as requested by the user. Report total nats, total bits, and bits per byte. Timing is secondary and does not alter this objective. When comparing future schedulers, additionally compare at matched compute budgets; do not hide search cost behind model count.

## Baselines

- Uniform-byte predictor (eight bits/byte), implemented as a sanity check.
- Adaptive byte unigram with symmetric Dirichlet-1/2 prior, implemented as the initial learning baseline.
- For future binary synthetic tests: fair-coin and Beta-Bernoulli predictors with the prior stated explicitly.
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

Start dataset evaluation with the user's local text file, scored as raw bytes through `Model<u8>`. The supplied files are enwik8/enwik9; a separately supplied WikiText file is a different benchmark. Record file hash, ordering, and any prefix limit. No shuffling, tokenization, or implicit reset is performed.

Retain binary IID sources, periodic signals, noisy finite-state generators, and finite-memory dependencies for planned oracle and model correctness experiments. Later include deliberately out-of-class processes and nonstationary streams to reveal model mismatch. Fix lengths, rates, noise, seeds, and generator version in committed run configs before each campaign. Use analytic deterministic cases for correctness; for stochastic comparisons use at least 10 paired seeds initially, expanding only when uncertainty prevents a decision.

Report per-seed values and paired differences, then mean/median and a stated uncertainty interval. Distinguish seed variability from repeated timing variability. Do not select only favorable generators or runs. Failed, timed-out, and numerically invalid runs count and need explanations.

## Approximation checks

On tiny spaces compute the exact mixture and omitted mass at every evaluated prefix. Compare the algorithm's asserted certificate to actual retained-to-full KL and predictive total variation. Track `U/Z_S` as well as whether a threshold is reached. A loose but valid bound and an invalid tight bound are different outcomes. All-zero likelihood is an explicit failure condition unless the protocol defines a probabilistic fallback beforehand.

## Reproduction and retention

Each campaign uses the [experiment template](templates/experiment.md). Record the full command and config, Git revision/dirty patch, seed and data hashes, toolchain and hardware, metrics definitions, and output hashes. Keep concise tables and manifests in Git. Follow the [artifact policy](../artifacts/README.md) for larger outputs. The byte CLI currently prints totals and optionally writes per-byte costs; automatic manifests and a remote artifact store are not configured.

## Gates

The user prioritized an independent byte harness and local text baselines before the finite-model work. Analytic harness checks precede dataset interpretation. B1 records the first local text baseline; it does not require E0. For mixture/search claims, E0 establishes a reliable oracle; E1 validates model coding; E2 examines redundancy; E3 compares scheduling; E4 profiles the CPU; E5 considers GPU. Change this order only with an explicit decision and rationale.
