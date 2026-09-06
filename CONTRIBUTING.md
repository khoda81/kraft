# Contributing

Read the [status](docs/status.md), [theory](docs/theory.md), and [queue](docs/queue.md). Pick the first unblocked task and keep its scope small.

Use the commands in the [README](README.md) before pushing. CI checks Rust at the pinned baseline and current stable, rustdoc, and local Markdown file links. Use a feature branch and PR for subsequent changes; the initial repository bootstrap is committed directly to main.

A research change is complete when another person can reconstruct its question, protocol, implementation revision, outcome, and next step. Use the [experiment template](docs/templates/experiment.md); update status and queue; append substantive work to the research log. Negative and inconclusive results belong in the record too.

Implementation defaults belong in code or committed run configs. Link to them from prose instead of maintaining a second copy. Do not include credentials, personal data, or large generated outputs in commits. Before running expensive experiments, define and record the budget.
