# Working on KRAFT

Read README.md, docs/status.md, docs/theory.md, and docs/queue.md before substantive work.

- Preserve the distinction between agreed direction, proposed designs, and measured evidence.
- Keep implementation details in code/configs; docs explain theory, methodology, decisions, and results.
- CPU Rust first. Preserve a possible GPU path with explicit state and batching; do not add a GPU stack or VM spec prematurely.
- Use log weights, specify log units, and distinguish scheduling utility from posterior mass.
- Evaluate prequentially: predict and score before observing/updating. Charge replay, search, and proposal work to the budget.
- Never claim a global tail certificate for an unbounded program space without a valid bound.
- For substantive work update docs/status.md, docs/queue.md, and the research log. Record changed assumptions in docs/decisions.md. Record experiments using docs/templates/experiment.md.
- Keep large outputs under ignored artifacts/; preserve manifests and concise results in docs/experiments/ with durable artifact references.
- Run cargo fmt --all -- --check, cargo clippy --locked --all-targets --all-features -- -D warnings, cargo test --locked, and python3 scripts/check_docs.py before pushing code.
- Do not silently select a license, publish a crate, or claim unrun experiments succeeded.
