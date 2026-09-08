# Byte coding harness

The benchmark objective is total causal prequential coding cost on a local text file. See [prequential.md](prequential.md) for the canonical distinction between a fixed-model score, a hindsight/oracle score, and KRAFT's Bayesian-mixture score. The user requested WikiText; the supplied local paths identify enwik8/enwik9. These names must remain distinct in result records. The harness accepts either as raw bytes and does not fetch datasets.

## Interface

```rust
pub trait Distribution<T> {
    fn ln_prob(&self, observation: &T) -> f64;
}

pub trait Model<T> {
    fn predict(&self) -> impl Distribution<T>;
    fn observe(&mut self, observation: T);
}
```

The traits are generic; file evaluation requires `Model<u8>`. A prediction may borrow the model. For each byte the harness calls `predict`, evaluates `ln_prob`, drops the prediction, records the cost, then calls `observe`. There is no update before scoring, tokenization, UTF-8 decoding, newline normalization, BOS/EOS insertion, warm-up exclusion, or implicit reset. Invalid UTF-8 bytes are valid observations too. File order and bytes define the benchmark.

The primary objective is `sum_t -ln P(x_t | x_<t)`, in nats. For KRAFT itself, `P` must be the online Bayesian-mixture prediction produced from the decoded prefix and the declared inference policy. A fixed structure selected using future bytes may be analyzed with the same evaluator, but that result is an oracle diagnostic rather than KRAFT's prequential coding cost. The report also converts to total bits and reports the coding ratio against uniform, defined as `uniform_cost / model_cost`. Higher is better: uniform is 1, values above 1 beat uniform, and values below 1 are worse. Because it is a ratio of coding costs, the value is invariant to log base. Against uniform it is the ideal compression ratio for the fixed byte stream. All bytes, including the first, are scored. This is ideal coding length, not an actual compressed-file size; no arithmetic coder or EOF/length encoding is implemented.

## Run

From the repository root, after `git pull --ff-only`:

```sh
cargo run --release -- ../text-preq-encoding/preq-encoding/data/enwik/enwik8 --limit 1000000
```

Remove `--limit` to evaluate the full file. Use `--model uniform` for the eight-bits-per-byte sanity baseline; default `kt` is an adaptive byte unigram with a symmetric Dirichlet-1/2 prior over all 256 byte values. It starts uniform and predicts `(count[byte] + 1/2) / (observed_bytes + 128)`. Both baselines are deterministic and start fresh in each CLI invocation.

```sh
mkdir -p artifacts/enwik8
cargo run --release -- path/to/enwik8 --model kt --costs artifacts/enwik8/costs.csv
```

`--costs` writes `byte_offset,cost_nats` rows with zero-based offsets. It creates a new file and refuses to overwrite an existing path, including the dataset itself. The parent directory must exist. Per-byte text output can be large and slow; leave it off when only the total is needed. Errors return nonzero; any already-written cost file is partial and must not be treated as a completed run.

## Library usage

```rust
use kraft::{baselines::Kt, evaluate_with_costs};

let mut model = Kt::default();
let mut costs = Vec::new();
let report = evaluate_with_costs(&b"hello"[..], &mut model, |cost| {
    costs.push(cost);
    Ok(())
}).unwrap();
```

Use `evaluate(reader, &mut model)` for totals only. Pass `Read::take(limit)` to bound a prefix. The caller owns model lifetime; repeated library evaluations continue its state, but each returned total covers only that call. The CLI always creates a new model.

## Numerical and execution contract

- `ln_prob` returns natural-log probability mass. Models must normalize their distributions; the generic evaluator cannot enumerate arbitrary support to prove this.
- Zero probability yields infinite cost without clipping, while later observations continue. NaN/positive log probabilities fail before observing the affected byte.
- Empty input costs zero; coding ratio is undefined (`None` in Rust, `n/a` in CLI output).
- Totals use compensated summation. The evaluator buffers input and uses constant memory excluding the model and cost consumer.
- Read and cost-output errors propagate. Prior observations are not rolled back; the reader may have buffered ahead.
- `evaluation_seconds` measures reading, prediction, scoring, updates, and optional cost-output flushing. It excludes compilation, model construction, file opening, CSV header, and printing the report. It is not a complete search-compute budget.

## Evidence and next run

Correctness checks cover analytic uniform and unigram sequence probabilities, normalization, strict call ordering, borrowed predictions, raw bytes/UTF-8, prefix limits, continued state, empty input, zero probability, invalid probability values, and I/O failure propagation. Optimized joint-evidence shortcuts for structured models must additionally regress against a literal `predict -> score -> observe` implementation. CLI smoke checks exercise prefix limits and refusal to overwrite files.

The user has supplied two unigram runs on the first million bytes of local enwik8; see the [B1 record](experiments/B1-enwik8-baseline.md) for the results and missing metadata. No corpus run was performed in the agent environment. Record the code revision, exact input name/hash, command, evaluated prefix length, model, compiler/hardware, and output for subsequent benchmarks. The CLI currently prints a plain-text summary; automatic manifests, dataset hashing, and experiment tracking remain future runner work.

## Sparse-DFA anytime certificate experiment

The rewrite has a runnable certificate engine for the full declared sparse-DFA prior:

```bash
cargo run --release --locked --bin sparse-dfa-anytime -- \
  path/to/enwik8 \
  --limit 8 \
  --steps 10000 \
  --report-every 1000
```

It reports lower/upper bounds on the exact Bayesian mixture prequential cost of the supplied prefix. The current unresolved-region likelihood upper bound is deliberately conservative, so start with short prefixes while validating refinement behavior. This binary is not yet the bounded-compute streaming codec: it certifies the exact target `-ln M(prefix)` after seeing the prefix rather than emitting an approximate causal probability before each byte.

## Anytime convergence diagnostics

Run the same prefix and budget with detailed reporting:

```sh
cargo run --release --locked --bin sparse-dfa-anytime -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 1000 --steps 100000 --report-every 1000 --diagnostics
```

The main stdout table reports code endpoints and gap in nats to nine decimal places, coding ratios, gains since the previous report, and gap reduction per refinement and per search second. Decimal output is diagnostic precision, not a floating-point error certificate. `NA` means undefined (including initial rates, empty-input ratios, or rates across an infinite initial gap).

Detailed stderr tables report each unresolved region category's count, log upper mass, share of summed unresolved upper bounds, mean forced-prefix bytes, upper-mass-weighted mean forced-prefix bytes, and cumulative refinements. A forced prefix includes the emission immediately before the first undecided transition. Upper shares are **not posterior probabilities**. Compare shares with refinement counts: many refinements with persistently high upper mass and short forced prefixes suggest insufficient bound tightening.

`bound_scan_bytes` counts cumulative bytes visited by likelihood-bound evaluations, including root initialization and repeated replay; it excludes transition-only ambiguity scans and suffix preprocessing. `work_s` measures only the last search batch. `elapsed_s` starts before input loading and includes setup and earlier reporting, so reporting overhead is not hidden in total runtime. Reporting still scans the frontier; use a larger report interval for throughput measurements.

A separate table format replaces the old compact interval columns. The command stops when its budget is reached or no refinable regions remain; any opaque mass is still included in the final bounds.

`--exposed-tail-priority` enables an experimental scheduler: state-count and exception-count tails are ranked by the upper mass of the next exact-count child they expose. This changes scheduling utility only; all Bayesian region masses and bounds remain intact. The default still ranks full upper mass. In the [first diagnostic](experiments/E0-anytime-diagnostic.md), the alternative tightened slightly per refinement but took more time, so it is not a recommended throughput optimization.

## Causal generated-state partition posterior

```sh
cargo run --release --locked --bin partition-dfa -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 --limit 1000000 --depth 8
```

This command uses the shared `predict -> score -> observe` evaluator. It reports total nats, bits per byte, joint log evidence, allocated nodes, node updates, and elapsed seconds. `--depth` defines the finite byte-history partition prior, not a compute truncation of the sparse-DFA model. Root stop/split choices are marginalized exactly; all logs are natural. The state starts padded with symbol 256. [Theory](theory.md#generated-dfa-states-with-bayesian-emission-partitions) and [E0g](experiments/E0-generated-partition-dfa.md) explain parameter sharing and interpretation.
