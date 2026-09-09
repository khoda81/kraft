# Contributing

Read the [status](docs/status.md), [theory](docs/theory.md), and [queue](docs/queue.md). Pick the first unblocked task and keep its scope small.

Use the commands in the [README](README.md) before pushing. CI checks the pinned development toolchain, the minimum supported Rust version from Cargo.toml, and current stable, plus rustdoc and local Markdown file links. Formatting uses the pinned development toolchain. Use a feature branch and PR for subsequent changes; the initial repository bootstrap is committed directly to main.

A research change is complete when another person can reconstruct its question, protocol, implementation revision, outcome, and next step. Use the [experiment template](docs/templates/experiment.md); update status and queue; append substantive work to the research log. Negative and inconclusive results belong in the record too.

Implementation defaults belong in code or committed run configs. Link to them from prose instead of maintaining a second copy. Do not include credentials, personal data, or large generated outputs in commits. Before running expensive experiments, define and record the budget.

## Implementation style

Keep the core small enough to read end-to-end. Prefer types and private construction that make invalid internal states unrepresentable over runtime guards for states our own code should never create. Defensive validation belongs at external-input boundaries.

During the rewrite, freely delete or reshape old abstractions when that removes duplicated logic or semantic ambiguity. Do not preserve compatibility with experimental internals at the cost of a larger core. Optimize for short, direct code without sacrificing mathematical semantics or measured performance.
