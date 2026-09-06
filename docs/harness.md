# Byte coding harness

The first benchmark objective is total prequential coding cost on a local text file. The user requested WikiText; the supplied local paths identify enwik8/enwik9. These names must remain distinct in result records. The harness accepts either as raw bytes and does not fetch datasets.

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

The primary objective is `sum_t -ln P(x_t | x_<t)`, in nats. The report also converts to total bits and ideal coding bytes, and gives the coding ratio to the uniform-byte model. All bytes, including the first, are scored. This is ideal coding length, not an actual compressed-file size; no arithmetic coder or EOF/length encoding is implemented.

## Coding ratio and units

For models A and B evaluated on the same input, define `R(A/B) = C_A / C_B`. Smaller is better for A; 1 means equal coding cost. Express both costs in the same unit and any log-base conversion cancels. Report absolute costs in nat, bit, byte, KB, MB, etc., alongside this dimensionless ratio. KB/MB are decimal multiples of bytes; use KiB/MiB for binary multiples when needed.

The CLI reports `coding_ratio_to_uniform` using the analytic uniform cost. This is the ideal compressed/original size ratio: the uniform model costs exactly the raw input length when expressed in coding bytes. It excludes any actual encoder overhead. `bytes` in the report is the input count; `total_coding_bytes` is a fractional ideal coding cost.

To compare any two model reports in the library, use `a.coding_ratio(&b)`. It checks equal byte counts; the caller must ensure identical input/protocol. Zero denominator or infinity/infinity is undefined. An infinite numerator against finite positive cost yields infinity; finite/infinity yields zero. No extra model pass is needed for the uniform reference.

## Run

From the repository root, after `git pull --ff-only`:

```sh
cargo run --release -- ../text-preq-encoding/preq-encoding/data/enwik/enwik8 --limit 1000000
```

Remove `--limit` to evaluate the full file. Use `--model uniform` for the sanity baseline whose coding ratio to uniform is 1; default `kt` is an adaptive byte unigram with a symmetric Dirichlet-1/2 prior over all 256 byte values. It starts uniform and predicts `(count[byte] + 1/2) / (observed_bytes + 128)`. Both baselines are deterministic and start fresh in each CLI invocation.

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
- Empty input costs zero; coding ratio to uniform is undefined (`None` in Rust, `n/a` in CLI output).
- Totals use compensated summation. The evaluator buffers input and uses constant memory excluding the model and cost consumer.
- Read and cost-output errors propagate. Prior observations are not rolled back; the reader may have buffered ahead.
- `evaluation_seconds` measures reading, prediction, scoring, updates, and optional cost-output flushing. It excludes compilation, model construction, file opening, CSV header, and printing the report. It is not a complete search-compute budget.

## Evidence and next run

Correctness checks cover analytic uniform and unigram sequence probabilities, normalization, strict call ordering, borrowed predictions, raw bytes/UTF-8, prefix limits, continued state, empty input, zero probability, invalid probability values, and I/O failure propagation. CLI smoke checks exercise prefix limits and refusal to overwrite files.

The user has supplied two unigram runs on the first million bytes of local enwik8; see the [B1 record](experiments/B1-enwik8-baseline.md) for the results and missing metadata. No corpus run was performed in the agent environment. Record the code revision, exact input name/hash, command, evaluated prefix length, model, compiler/hardware, and output for subsequent benchmarks. The CLI currently prints a plain-text summary; automatic manifests, dataset hashing, and experiment tracking remain future runner work.
