# KRAFT

[![CI](https://github.com/khoda81/kraft/actions/workflows/ci.yml/badge.svg)](https://github.com/khoda81/kraft/actions/workflows/ci.yml)

Compute-aware Bayesian program mixtures, starting with finite-state predictors.

KRAFT explores how to search and maintain a Bayesian mixture over small programs under a finite compute budget. The initial benchmark scores local text as bytes; the first program-mixture target is a tiny, exhaustively enumerable finite-state model family on CPU in Rust. That exact reference will let us measure what adaptive search misses before expanding the model family or moving to GPU.

**Stage:** streaming byte harness implemented with uniform and adaptive unigram baselines. The first benchmark minimizes total coding cost on a local text stream; a first local enwik8 prefix result is recorded, while full-file results and model enumeration remain pending. See [current status](docs/status.md).

## Start here

- [Run the byte harness](docs/harness.md): interface, dataset command, and scoring rules.
- [Research index](docs/README.md): how to recover the full context.
- [Theory](docs/theory.md): objective, priors, scheduling, and approximation guarantees.
- [Experiment plan](docs/experiments/README.md): staged questions and evaluation gates.
- [Work queue](docs/queue.md): the next concrete tasks.
- [Research log](docs/research-log.md): what happened and what was learned.

## Development

Install Rust through [rustup](https://rustup.rs/). The checked-in toolchain pins the development compiler; CI also checks the minimum supported version declared in Cargo.toml and current stable.

```sh
cargo test --locked
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
python3 scripts/check_docs.py
```

The crate remains dependency-free. Run `cargo run --release -- path/to/enwik8 --limit 1000000` to score a million-byte prefix. See the [harness guide](docs/harness.md) for models and per-byte costs. Planned model/search components and GPU considerations are in [architecture](docs/architecture.md). [Contributing](CONTRIBUTING.md) describes how to preserve research context.

## Repository layout

| Path | Purpose |
| --- | --- |
| `src/` | Shared Rust implementation |
| `docs/` | Theory, decisions, protocols, results, and work queue |
| `scripts/` | Development and later experiment entry points |
| `artifacts/` | Ignored generated outputs; retention policy in its README |
| `.github/` | CI, dependency updates, and contribution templates |

The name is a nod to the Kraft inequality and description-length priors. No expanded acronym is prescribed.

License selection is pending; the crate is not configured for publication.
