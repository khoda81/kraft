# Work queue

Updated: 2026-09-06. Work from the first unblocked item. `[ ]` is pending, `[x]` is complete; no task is silently in progress.

## Now

- [x] Q0 — Bootstrap repository, Rust crate, CI definitions, and research notebook. Validation/deployment status: [log](research-log.md).
- [x] B0 — Implement the user's generic model/distribution interface, streaming byte coding evaluator, local-file CLI, and simple baselines. See [harness](harness.md).
- [x] M0 — Implement the generic two-learner Bayesian mixture and higher-is-better coding-ratio metric. CI run 34035684830 passed on Rust 1.85.1 and stable 1.98.1.
- [ ] B1 — Complete the enwik8 baseline campaign. **Partial:** the user's million-byte unigram runs on Rust 1.85.0 and 1.98.1 are [recorded](experiments/B1-enwik8-baseline.md). Remaining: input hash/environment metadata and full-file uniform/unigram results. **Depends on local dataset access.**
- [x] Q1 — Specify the exact tiny finite-state family. The fixed-N transition-table prior, prediction/update order, statewise Dirichlet-1/2 emissions, and exact state-label quotients are recorded in [E0](experiments/E0-partial-dfa-posterior.md). The N = 1 identity is the byte KT unigram. A cross-N self-delimiting code remains Q5 rather than part of this conditional fixed-N family.
- [x] Q2 — Implement the scalar exact partial-DFA mixture and independent E0 checks. Tests cover the N = 1 KT identity, normalized predictions, unused-label multiplicity, quotient evidence agreement, and vector-versus-persistent short-prefix parity.
- [x] Q2b — Replace per-component transition and emission clones with persistent parent-linked arenas; add physical-node, payload, RSS, and timing diagnostics. [E0b](experiments/E0-persistent-dfa-state.md) records a 76.41% byte-22 payload reduction and a byte-26 / 9.29-million-component run.
- [ ] Q3 — Extend the existing byte runner with automatic manifests and synthetic binary generators for E0. Byte evaluation/CLI is complete under B0; generators/manifests remain pending. **Depends on Q2 for E0 integration.** Done when a clean checkout reproduces a synthetic run with manifest and per-step metrics.
- [ ] Q4 — Complete the planned synthetic E0 campaign. **Partial:** the local enwik8 exact-growth, quotient, and persistent-state experiments are recorded in [E0](experiments/E0-partial-dfa-posterior.md) and [E0b](experiments/E0-persistent-dfa-state.md). Synthetic generators/manifests from Q3 remain pending.

## Next

- [ ] Q5 — Implement and audit model encoding/prior (E1). **Depends on Q1, Q4.** Check finite truncation normalization and coding claims.
- [ ] Q6 — Measure redundancy and prior-preserving canonicalization (E2). **Depends on Q5.**
- [ ] Q7 — Implement explicit budget accounting, late-admission replay, and scheduler baselines. **Depends on Q4–Q5.**
- [ ] Q8 — Implement finite frontier bounds and compare E3 to exact posterior/predictions. **Depends on Q7.**
- [ ] Q9 — Profile and benchmark CPU batching (E4). **Depends on Q8.**
- [ ] Q9b — If exact fixed-N inference remains a priority, specify a weighted decision/arithmetic DAG that can reuse computation across transition assignments. Preserve exact posterior semantics and compare against the persistent-leaf oracle before replacing it. **Depends on E0b; independent of approximate scheduling.**

## Later / gated

- [ ] Q10 — Recover or newly specify the multiply-shift transition family; analyze coverage/induced prior before experiments. May proceed independently as theory; implementation waits for the reference.
- [ ] Q11 — Choose and test one GPU backend (E5). **Depends on E4 showing a suitable workload.**
- [ ] Q12 — Investigate richer machines, learned proposals, and recursive learner search. **Depends on convincing finite-class evidence; no universal certificate assumed.**

## Housekeeping

- [ ] H1 — Choose a license before encouraging external reuse or publishing the crate.
- [x] H2 — Apply and verify GitHub description/topics. Completed by the user with `scripts/configure_repo.sh`; successful CLI output confirmed the description and all seven topics on 2026-09-06.
- [ ] H3 — Select a durable artifact backend before generating outputs too expensive to reproduce.

New tasks should name a concrete deliverable, dependency, and completion criterion. Reprioritization belongs in the research log; keep abandoned tasks with a short reason rather than deleting their history.
