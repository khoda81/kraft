# KRAFT

[![CI](https://github.com/khoda81/kraft/actions/workflows/ci.yml/badge.svg)](https://github.com/khoda81/kraft/actions/workflows/ci.yml)

Compute-aware Bayesian program mixtures, starting with finite-state predictors.

KRAFT explores how to search and maintain a Bayesian mixture over small programs under a finite compute budget. The initial benchmark scores local text as bytes; the first program-mixture target is a tiny, exhaustively enumerable finite-state model family on CPU in Rust. That exact reference will let us measure what adaptive search misses before expanding the model family or moving to GPU.

**Stage:** causal byte harness and exact DFA oracles are implemented; sparse-DFA heuristic search now has optimized scoring, a multi-fidelity audited prefilter, checkpoints, and compressed artifact bundles. The latest enwik8 `N=8` search again improved all the way to the artificial `K=256` boundary. The next major step is an anytime Bayesian rewrite where structural choices remain inside the prior and finite compute only controls approximation quality. See [current status](docs/status.md).

## Start here

- [Prequential objective](docs/prequential.md): the canonical KRAFT score and causal Bayesian coding semantics.
- [Run the byte harness](docs/harness.md): interface, dataset command, and scoring rules.
- [Research index](docs/README.md): how to recover the full context.
- [Theory](docs/theory.md): priors, scheduling, and approximation guarantees.
- [Experiment plan](docs/experiments/README.md): staged questions and evaluation gates.
- [Work queue](docs/queue.md): the next concrete tasks.
- [Research log](docs/research-log.md): what happened and what was learned.

## Development

Install Rust through [rustup](https://rustup.rs/). Development is pinned to Rust 1.98.1, which is also the crate's minimum supported Rust version. CI checks that pinned toolchain and current stable.

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
