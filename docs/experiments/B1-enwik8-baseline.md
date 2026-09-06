# B1 — Initial enwik8 byte-unigram baseline

Status: partial — user-reported prefix results; full-file comparison pending.
Written (UTC): 2026-09-06.
Executed: user terminal output supplied on 2026-09-06; exact run timestamps unavailable.
Related: B1 in the [queue](../queue.md), D010/D011 in [decisions](../decisions.md).

## Question and protocol

Establish the adaptive byte-unigram coding cost on the first 1,000,000 bytes of the user's local enwik8 file. The baseline is `kt`: a symmetric Dirichlet-1/2 unigram over 256 byte values, predicting and scoring before each update. Read raw bytes from the beginning with no preprocessing, reset, or excluded warm-up. Each CLI run initializes a fresh model.

This record is reconstructed from user-provided terminal output, not an independently rerun or preregistered experiment. The measured quantity is ideal total coding cost, not compressed-file size.

## Reproduce

Implementation revision: `2837ebdb153d54499c9802362cddb4d3529723e4` (byte harness merge). The second run used `cargo +stable` with Rust 1.98.1. A local toolchain-file edit was subsequently committed as `1fc543fef87e` ("Upgrade toolchain version"); the shown edit changes the development pin, not the model implementation.

```sh
cargo +1.85.0 run --release -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 1000000

cargo +1.98.1 run --release -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 1000000
```

These commands make the compiler selection explicit. In the supplied transcript the first command used the then-pinned compiler without an override, and the second used `+stable` after its update to 1.98.1. No `--costs` output was requested.

Input: local file named enwik8, 100,000,000 bytes in the earlier listing; first 1,000,000 bytes evaluated. Input checksum, complete environment manifest, run-specific hardware/OS details, and raw artifact hashes were not supplied. Seed: not applicable (deterministic model and fixed byte order). Build: Cargo release profile. One timing observation per compiler; no controlled warm-up or repetition schedule.

## Results

Source: user-pasted CLI summaries. Both runs printed identical coding metrics to the displayed precision.

| Metric | Rust 1.85.0 | Rust 1.98.1 |
| --- | ---: | ---: |
| Bytes scored | 1,000,000 | 1,000,000 |
| Total nats | 3507734.978874585126 | 3507734.978874585126 |
| Total bits | 5060591.858775116503 | 5060591.858775116503 |
| Coding ratio vs uniform | 1.580842759751 | 1.580842759751 |
| Evaluation seconds | 0.019772 | 0.027971 |

The analytic uniform-byte reference for this prefix is 8,000,000 bits; a uniform corpus run has not been supplied. Dividing that reference cost by the measured unigram cost gives a coding ratio of approximately 1.58084. This is log-base invariant and, relative to uniform, is the ideal compression ratio for the fixed byte stream. It excludes coding overhead such as file length/EOF and is not a measured encoded-file ratio from an actual compressor.

## Interpretation and limitations

This establishes a first user-reported corpus-prefix baseline and agreement of printed losses across the two compiler runs. It does not establish bitwise cross-toolchain equivalence, full-corpus performance, or a compiler speed difference. The timings are short, unreplicated observations; machine load and cache conditions were not controlled. There is no uncertainty estimate from these two runs.

The local file has not been checksum-verified against a canonical dataset. Record it as the user's enwik8 input, not WikiText. No claims about finite-state mixtures follow from this unigram result.

## Follow-up

B1 remains open: record the input checksum and full-file uniform/unigram outputs with compiler and environment metadata. Only repeat timing measurements if runtime comparisons become a research question. Q1, the finite-state model design, remains independent future work.
