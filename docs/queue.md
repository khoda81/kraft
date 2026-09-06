# Work queue

Updated: 2026-09-06. Work from the first unblocked item. `[ ]` is pending, `[x]` is complete; no task is silently in progress.

## Now

- [x] Q0 — Bootstrap repository, Rust crate, CI definitions, and research notebook. Validation/deployment status: [log](research-log.md).
- [ ] Q1 — Specify the exact tiny finite-state family. Deliverable: a decision record covering prediction/transition order, emission law, start state, finite bounds, prior, and a hand-computed one-state example. **No dependency.**
- [ ] Q2 — Implement scalar models and exhaustive mixture; add independent E0 checks. **Depends on Q1.** Done when hand calculations and sequence marginal likelihood identities pass.
- [ ] Q3 — Build minimal reproducible runner and binary generators. **Depends on Q2.** Done when a clean checkout reproduces a short run with manifest and per-step metrics.
- [ ] Q4 — Run and document E0; update status with evidence and failure cases. **Depends on Q2–Q3.**

## Next

- [ ] Q5 — Implement and audit model encoding/prior (E1). **Depends on Q1, Q4.** Check finite truncation normalization and coding claims.
- [ ] Q6 — Measure redundancy and prior-preserving canonicalization (E2). **Depends on Q5.**
- [ ] Q7 — Implement explicit budget accounting, late-admission replay, and scheduler baselines. **Depends on Q4–Q5.**
- [ ] Q8 — Implement finite frontier bounds and compare E3 to exact posterior/predictions. **Depends on Q7.**
- [ ] Q9 — Profile and benchmark CPU batching (E4). **Depends on Q8.**

## Later / gated

- [ ] Q10 — Recover or newly specify the multiply-shift transition family; analyze coverage/induced prior before experiments. May proceed independently as theory; implementation waits for the reference.
- [ ] Q11 — Choose and test one GPU backend (E5). **Depends on E4 showing a suitable workload.**
- [ ] Q12 — Investigate richer machines, learned proposals, and recursive learner search. **Depends on convincing finite-class evidence; no universal certificate assumed.**

## Housekeeping

- [ ] H1 — Choose a license before encouraging external reuse or publishing the crate.
- [ ] H2 — Apply and verify GitHub description/topics; use `scripts/configure_repo.sh` if admin APIs are unavailable in the agent environment.
- [ ] H3 — Select a durable artifact backend before generating outputs too expensive to reproduce.

New tasks should name a concrete deliverable, dependency, and completion criterion. Reprioritization belongs in the research log; keep abandoned tasks with a short reason rather than deleting their history.
