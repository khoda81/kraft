# E0f — Sparse posterior convergence against fixed-order contexts

Status: complete diagnostic; superiority objective remains open
Written (UTC): 2026-09-08
Executed (UTC): 2026-09-08
Related queue/decision IDs: A3, A7, D023

## Question and hypothesis

Does the current sparse posterior approach the fixed-order context costs at a small, reproducible budget? Upper mass concentrated in shallow structural regions would favor stronger bounds or a different representation before scheduler tuning.

## Protocol frozen before execution

- Existing unbounded sparse prior and Dirichlet-1/2 emissions; unchanged largest-upper-mass scheduling.
- First 1,000 raw bytes of local enwik8, no shuffle or reset. No tuning split; this is a diagnostic, not a final superiority evaluation.
- Compare prespecified byte-context orders 0 through 4 with Dirichlet-1/2 emissions and uniform bootstrap. Report all orders; the best in hindsight is a reference, not a selected online learner.
- Budget: 100,000 refinements, reporting every 10,000; external 60-second execution cap. Compilation excluded. One run, no timing uncertainty claim.
- Metrics: evidence-derived code interval in nats, width, upper mass by region type, forced-prefix depth, bound scan bytes, elapsed time. Search/replay included in execution time; joint-evidence bounds do not implement a streaming approximate codec.
- Success for this diagnostic is complete accounting, not beating the baseline. No numerical enclosure is claimed. Stop at budget or timeout and retain partial output.

## Reproduce

- Base revision: `6792906a`, with formatting and audit-documentation working-copy changes; inference semantics unchanged.
- Input: `../text-preq-encoding/preq-encoding/data/enwik/enwik8`, first 1,000 bytes.
- Build: `cargo build --release --locked --bin sparse-dfa-anytime --bin ngram-fit`.
- Run: `timeout 60s target/release/sparse-dfa-anytime ../text-preq-encoding/preq-encoding/data/enwik/enwik8 --limit 1000 --steps 100000 --report-every 10000 --diagnostics`.
- Baseline: `target/release/ngram-fit artifacts/anytime-diagnostic-20260908/prefix.bin --orders 0,1,2,3,4`.
- Artifact directory: `artifacts/anytime-diagnostic-20260908/`; local only, hashes and environment recorded below after execution.

## Results

Both runs completed 100,000 refinements without timeout. Initial lower endpoint: 2380.761450065 nats. Baseline nats: order 0 = 3661.633676705047; order 1 = 3732.683983498916; order 2 = 3967.360274910841; order 3 = 4145.283464149748; order 4 = 4300.779298322023.

| Metric | Upper mass default | Exposed tail mass |
| --- | ---: | ---: |
| Code lower endpoint, nats | 2383.656249622 | 2384.471971573 |
| Code upper endpoint, nats | 3631.549819007 | 3631.415556492 |
| Interval width, nats | 1247.893569385 | 1246.943584919 |
| Elapsed seconds | 0.171434963 | 0.487395048 |
| Bound scan bytes | 998088 | 3578412 |
| Unresolved regions | 99203 | 96704 |
| Resolved regions | 608 | 1993 |
| State-tail refinements | 70269 | 1027 |
| Key refinements | 12914 | 85721 |
| Destination refinements | 0 | 2044 |
| Key share of unresolved upper mass | 0.701929549 | 0.345958728 |
| Exception-tail share | 0.100289380 | 0.370554747 |
| Upper-weighted key forced bytes | 5.689 | 11.504 |

Environment: Rust 1.98.1 (48a229cea, 2026-09-01), Linux 7.2.2-1-cachyos, AMD Ryzen 5 4600H, CPU release build. Single timing sample per policy; no warm-up or peak-memory measurement. The ablation includes the experimental scheduler patch in `src/models/sparse_dfa_anytime.rs` and CLI switch in `src/bin/sparse-dfa-anytime.rs`.

Local artifact SHA-256:

```text
prefix.bin       0f35cdeee80ba4c570885c34ee901aa579441fb8ba97351568a81bacdfc241fd
progress.tsv     9c5b728fd5bf27934d1667a4ca536b5ace100e4c9c9ecde9e094eb6e1c0998c5
diagnostics.tsv  b89d326fe835beaaa76fc22502727be4a90e42ce2b933aa80437f2a76557aac0
ngram.txt        e6e11008b87e06191187bafa386f9d34851d5cab7ea20594d535b99cc1127b1d
exposed-progress.tsv     d8033a37d52c476977a5b2240726cec446ced9e51fea5cf47ac130b7e5589938
exposed-diagnostics.tsv  4de090af5cc0fb7dd384aa83fde196148b750e38b16bc43cfcf392c5dfe92dac
```

### Scheduler ablation protocol (frozen after the initial diagnostic)

Repeat the identical prefix, refinement budget, reporting interval, and timeout with `--exposed-tail-priority`. This experimental policy scores state-count and exception-count tails by the upper mass of their next exact-count child, leaving all other priorities unchanged. It preserves region evidence and prior mass. Include priority calculation in timed work. Compare both code endpoints, not just frontier size or throughput; retain the original default regardless of this single-prefix result. This is a response to the initial run's 70,269 state-tail refinements and zero destination refinements, not a prespecified confirmatory comparison.

## Interpretation and limitations

This prefix does not establish full-corpus quality or generalization. The sparse interval can be inconclusive even if every fixed-order baseline is measured exactly.

On this prefix, resolved mixture mass is already sufficient to put the exact sparse mixture's cost below every tested fixed-order model. The default upper endpoint is 30.083858 nats below the strongest baseline (order 0); the ablation gains only a further 0.134263 nats. This establishes neither the achieved cost of an approximate streaming learner nor superiority over richer/adaptively smoothed n-grams.

The ablation redirects work toward trajectories and slightly tightens both endpoints at equal refinement count, but takes about 2.84 times as long in these single samples and scans about 3.59 times as many emission bytes. It is not an established compute-efficiency improvement. Both intervals remain very wide. This supports prioritizing bound/representation changes; it does not isolate their causal effect or remove the large-state ceiling.

## Follow-up

Use the observed mass and depth breakdown to choose the next inference change toward the DFA posterior coding-cost objective.

Keep the default scheduler unchanged. Next address symbolic likelihood bounds and a reusable causal posterior, then compare on longer prefixes where n-gram context learning is effective. Do not generalize the 1,000-byte unigram-dominated result to full enwik8.
