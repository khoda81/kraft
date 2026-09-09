# E0h — Dynamic symbolic prediction groups

Status: complete
Written (UTC): 2026-09-09
Executed (UTC): 2026-09-09
Related queue/decision IDs: P1, D026

## Question and hypothesis

Can exact next-byte agreement share prediction and likelihood work while preserving
transition alternatives that disagree after later observations? Does saved
likelihood work translate into lower elapsed time, or is symbolic maintenance the
new bottleneck?

## Protocol frozen before execution

- Same fixed-N labeled-table prior as the partial-DFA oracle: independent uniform
  destinations, start state zero, independent Dirichlet-1/2 byte emissions.
- No state-count truncation claim, pruning, beam, or new emission tying assumption.
- Synthetic inputs: `abababababab`, `the cat sa`, and `abacaba`, with N=2 and N=3.
  Diagnostic CLI commands use `--compare-oracle`, 500,000 symbolic nodes and
  100,000 prospective oracle children as stop-on-exhaustion budgets.
- Check every observed prediction and prefix evidence within 1e-8 nats. Unit tests
  additionally compare all 256 predictions, including all length-five binary
  strings for N=1,2,3 and nonbinary destination/input cases.
- Compare prediction-group count, oracle component count, likelihood evaluations,
  count updates, symbolic operation/projection visits, and elapsed prediction plus
  update time. Diagnostic formatting is outside the reported model timings.
- Repeat each diagnostic three times; report median timings and all budget failures.
  No benchmark-driven tuning in this experiment.
- Stop after the declared cases. No large-corpus or asymptotic speed claim.

## Reproduce

Implementation: `src/models/predictive_dfa/`; CLI: `kraft infer dfa-grouped`.
Starting revision: `a78485b059052fed775873fd6b32b27bf92fffa4` on
`feat/partial-dfa-posterior`; changes are the commit containing this record.

```sh
printf 'abababababab' | cargo run --release --locked -- infer dfa-grouped --states 3 --compare-oracle --report-every 12
printf 'the cat sa' | cargo run --release --locked -- infer dfa-grouped --states 3 --compare-oracle --report-every 10
printf 'abacaba' | cargo run --release --locked -- infer dfa-grouped --states 3 --compare-oracle --report-every 7
```

Repeat with `--states 2`, or run
`python3 scripts/benchmark_prediction_groups.py` for all three repetitions of all
six cases. Inputs are the literal ASCII strings above, without a newline.
[All runs and input hashes](data/E0h-dynamic-prediction-groups.json) are committed.

Environment: Linux x86_64 shared VM, CPU only, release build with no explicit
warm-up. The available compiler was Ubuntu Rust/Cargo 1.91.1; the repository's
1.98.1 pin and minimum declaration were left unchanged. Local Cargo build/test/
clippy commands used `--ignore-rust-version`. Pinned-1.98.1 validation was not run.
Timings are short shared-VM measurements, not isolated performance estimates.

## Results

| States | Input | Final groups / oracle components | Likelihood evaluations: grouped / oracle | Grouped time, median ms | Oracle time, median ms |
| --- | --- | --- | --- | --- | --- |
| 2 | `abababababab` | 7 / 8 | 67 / 77 | 0.446 | 0.031 |
| 2 | `the cat sa` | 192 / 432 | 431 / 655 | 3.510 | 0.321 |
| 2 | `abacaba` | 9 / 18 | 41 / 57 | 0.296 | 0.024 |
| 3 | `abababababab` | 16 / 47 | 105 / 359 | 6.128 | 0.254 |
| 3 | `the cat sa` | 284 / 13892 | 531 / 9340 | 556.988 | 13.134 |
| 3 | `abacaba` | 17 / 140 | 50 / 223 | 4.395 | 0.122 |

All 18 runs completed within the declared budgets. Maximum observed prediction/
evidence disagreement was 1.421e-13 nats. The N=3 text case reduced update
likelihood evaluations 17.6-fold, but its median total model time increased from
13.134 ms to 556.988 ms (about 42-fold). Prediction groups are clearly smaller
than the structural alternatives; this implementation does not yet make total
compute follow that reduction.

Validation passed: 95 tests including doctests; all-target/all-feature clippy with
warnings denied; formatting; and relative documentation links. New tests cover
all-byte normalization and oracle equality, deterministic recurrence, nonbinary
state destinations, equal-probability/different-count merging then splitting and
remerging, garbage collection/chunk continuation, shared likelihood work, stdin,
and transactional budget failure. An initial state-major variable ordering hit
the 500,000-node limit in a wider unit-test case. Grouping variables by first byte
encounter resolved that failure before this protocol was executed. An interrupted
build also left a zero-length test executable; clearing only the crate's debug
artifacts and rebuilding resolved it.

## Interpretation and limitations

This backend represents complete labeled-table alternatives symbolically; it does
not enumerate and then bucket posterior components. Reduced decision diagrams
share identical subfunctions, and constant prediction regions marginalize the
remaining weight function without opening individual assignments. Equal rational
prediction vectors merge even when their underlying count totals differ. State
counts and transitions remain separate symbolic roots, preserving future splits.

The number of prediction groups alone is not the algorithm's complexity. Decision
diagram width, weight correlations, per-state symbolic updates, and projection
work can still grow exponentially. The current count-vector interning arena keeps
historical entries until its explicit budget is reached; diagram nodes are garbage
collected. Exact grouping is not a posterior-mass pruning certificate, an unbounded
state prior implementation, or proof of a speedup over the canonical leaf oracle.

## Follow-up

Prioritize the measured structural overhead: factor count vectors or weights more
finely, and account for state-label redundancy within the symbolic representation.
Approximate predictive regions remain a separate research step. Do not add lossy pruning until its
causal error accounting and treatment of deferred alternatives are specified.
