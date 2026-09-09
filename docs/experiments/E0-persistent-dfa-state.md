# E0b — Persistent state for the exact partial-DFA posterior

Status: complete for the N = 2 enwik8 prefix experiment

Written (UTC): 2026-09-06

Executed (UTC): 2026-09-06

Related: [E0 exact posterior](E0-partial-dfa-posterior.md), D016

## Question and hypothesis

How much of the exact partial-DFA posterior's memory growth is duplicated physical state rather than distinct posterior hypotheses?

The hypothesis was that child components usually differ from their parent by one transition assignment and one emission update. Persistent parent-linked histories should therefore reduce physical storage without changing the posterior component count, predictions, evidence, or retention diagnostics.

## Protocol frozen before execution

- Model family, prior, prediction/update order, and exact state-label quotients are unchanged from [E0](E0-partial-dfa-posterior.md).
- No pruning or approximation is performed. The component guard is checked before observation and remains a run-safety limit, not posterior truncation.
- Transition assignments are immutable eight-byte arena nodes containing a parent node ID and one edge. A branch appends one node and shares its parent's history.
- Emission observations use a second immutable arena. A branch shares its parent's count-update history and appends one `(state, byte)` observation before transition branching.
- Component fingerprints are compact accelerators only. Fingerprint collisions are resolved by comparing logical emission counts and transition contents through the arenas; equality never depends on arena node identity.
- Linear history scans are accepted because the measured prefix depth is small. No auxiliary transition cache is included in this experiment.
- Primary comparison point: the prior vector-backed result at byte 22, N = 2, discovery quotient, epsilon = 0.01 nat, with 1,032,192 components and a 457.310 MB payload estimate.
- Metrics retain their existing definitions. Coding ratios are higher-is-better baseline cost divided by model cost.
- Acceptance tolerance for the vector-oracle test is `1e-12` per sequential log probability and coding ratio, and `1e-11` for marginal log evidence after accumulation.

## Reproduce

Implementation base: commit `6240e7d7d758` on `feat/partial-dfa-posterior`, plus the working-copy change recorded by this experiment.

Input:

- local path: `../text-preq-encoding/preq-encoding/data/enwik/enwik8`
- size: 100,000,000 bytes
- SHA-256: `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`

Environment:

- Rust/Cargo 1.98.1, release mode with debug information
- Linux 7.2.2-1-cachyos x86-64
- AMD Ryzen 5 4600H, 6 cores / 12 threads
- 30 GiB physical memory and 30 GiB swap configured

Commands:

```sh
cargo test --release --locked

cargo run --release --locked --bin dfa-posterior -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --states 2 \
  --quotient discovery \
  --limit 64 \
  --epsilon 0.01 \
  --max-components 10000000

cargo run --release --locked --bin dfa-posterior -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --states 2 \
  --quotient predictive \
  --limit 64 \
  --epsilon 0.01 \
  --max-components 10000000
```

Local full outputs are under `artifacts/partial-dfa-persistent/`:

- `discovery-cap10m.tsv` and `discovery-cap10m.stderr`
- `predictive-cap10m.tsv` and `predictive-cap10m.stderr`
- `discovery-shared-cap2m.tsv` and `discovery-shared-cap2m.stderr`

These raw outputs are intentionally ignored. The concise result below is the durable repository record.

## Exactness validation

The release test suite retains the N = 1 identity against the byte KT unigram and both quotient tests. A vector-backed legacy oracle is compiled only in tests. For both discovery and predictive quotients on `mediawiki`, the persistent and vector representations have:

- equal sequential predictive log probabilities within `1e-12`;
- equal marginal log evidence within `1e-11`;
- identical posterior component counts at every prefix;
- identical epsilon-retained component count, mass, and KL diagnostics;
- equal uniform and KT coding ratios within `1e-12`.

Separate tests inspect parent node IDs directly and verify that sibling transition and emission histories share their prefix. No test or run applies approximate pruning.

## Results

### Byte-22 comparison with the vector-backed baseline

| Measurement | Vector-backed baseline | Persistent transitions only | Persistent transitions and emissions |
| --- | ---: | ---: | ---: |
| Posterior components | 1,032,192 | 1,032,192 | 1,032,192 |
| Logical assigned transitions | 20,840,448 | 20,840,448 | 20,840,448 |
| Transition arena nodes | not applicable | 2,064,382 | 2,064,382 |
| Emission arena nodes | not applicable | not applicable | 1,205,503 |
| Payload estimate | 457.310 MB | 323.879 MB | 107.872 MB |
| Process RSS | not recorded | 480.686 MB | 262.107 MB |
| Coding ratio versus KT | 0.992364462196 | 0.992364462196 | 0.992364462195 |

The final payload estimate is 4.24 times smaller, a 76.41% reduction from the vector-backed baseline. Each leaf semantically contains 20.19 assigned transitions, while the transition arena contains 2.00 nodes per component. Across leaves, 20.84 million logical edge records are represented by 2.06 million physical transition nodes. Similarly, 20.84 million nonzero per-component emission counters are represented by 1.21 million emission-update nodes.

The RSS values are Linux `VmRSS` samples from the process itself. The payload estimate includes component values plus arena capacities but excludes top-level `HashMap` bucket overhead, allocator metadata, and temporary diagnostic vectors. This explains why RSS is larger than the payload estimate.

### Larger-cap discovery run

| Prefix bytes | Components | Retained at 0.01 nat | Transition nodes | Emission nodes | Payload MB | RSS MB | Elapsed to row, s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | 1,032,192 | 864,578 | 2,064,382 | 1,205,503 | 107.872 | 261.968 | 5.380 |
| 23 | 2,064,384 | 1,728,515 | 4,128,766 | 2,237,695 | 215.745 | 482.173 | 12.172 |
| 24 | 4,128,768 | 3,455,969 | 8,257,534 | 4,302,079 | 431.489 | 868.139 | 28.665 |
| 25 | 6,193,152 | 4,422,832 | 12,386,302 | 8,430,847 | 714.342 | 966.459 | 56.183 |
| 26 | 9,289,728 | 6,210,474 | 18,579,454 | 14,623,999 | 1,071.514 | 1,809.519 | 109.419 |

The run stopped before byte 27 because its 10,838,016 prospective unmerged children exceeded the configured 10,000,000 guard. The full process ran for 134.675 seconds including the final leaf-wise guard scan. At byte 26:

- retained mass was 0.990049833830 and retained KL was 0.009999999918 nat;
- coding ratio versus uniform was 1.023471216924;
- coding ratio versus KT was 0.989985815888;
- the logical posterior contained 218,529,792 assigned-transition records across leaves.

### Predictive-quotient control

The predictive run reached the same byte-26 state and the same 9,289,728 components. Its evidence-derived outputs agree with discovery within floating summation tolerance: both coding ratios print identically to 12 decimals, and the retained mass differs by `3e-11`. `merged_children_last` remained zero.

The predictive row at byte 26 took 173.171 seconds and its full run including the final guard scan took 196.532 seconds, versus 109.419 and 134.675 seconds for discovery. Payload and RSS were effectively identical. The stronger quotient therefore provides no representation reduction on this prefix and adds canonicalization cost.

## Interpretation and limitations

The experiment supports the physical-sharing hypothesis. At the original stopping prefix, most stored transition and emission records were duplicated history: the exact posterior and all retained-mass diagnostics stayed fixed while the estimated payload fell by 76%. A ten-million-component cap then allowed four more bytes and roughly nine times as many posterior leaves at less than 2 GB measured RSS.

The experiment does not solve posterior growth. Component count remains exponential and dominates both compute and eventually memory. At byte 26, exact retention at 0.01 nat still needs 6.21 million of 9.29 million leaves. Prediction, transition lookup, semantic hashing/equality, the prospective guard scan, and retention sorting remain leaf-wise. The wall-time growth and the extra predictive-canonicalization cost make CPU/leaf count the next bottleneck before the machine's physical-memory limit in this run.

Arena node counts are allocated-node counts. An append-only arena could retain nodes from children later removed by exact merging; this prefix showed no last-step merges, and the reported counts should not be interpreted as a minimal DAG. The process RSS samples are single observations, not peak-RSS measurements. Timings are one local run per quotient and are not a controlled performance benchmark.

## Follow-up

Persistent structural sharing is complete for transition assignments and emission observations. Do not optimize the history scans until profiling shows they dominate. A weighted decision/arithmetic DAG that reuses computation across assignments was intentionally outside this experiment and is evaluated separately in [E0c](E0-symbolic-dfa-wmc.md).
