# E0g — Causal posterior over generated-DFA emission partitions

Status: complete initial causal comparison; broader objective open
Written (UTC): 2026-09-08
Executed (UTC): 2026-09-08
Related queue/decision IDs: L0, L1, A8, D025

## Question and hypothesis

Can an exactly marginalized emission-partition prior on compact generated DFA states beat fixed-order byte contexts in causal coding cost on substantial prefixes? This directly tests a richer model language, not a tighter approximation to the existing sparse prior.

## Protocol frozen before execution

- State generator: eight-byte shift register, most recent byte first, padded initially with start symbol 256. All states and transitions are finite and deterministic.
- Prior: each feature-prefix node independently stops or splits with probability 1/2; at depth eight stopping is forced. A leaf ties emission parameters for all full DFA states below it. Each leaf has independent symmetric Dirichlet-1/2 byte emissions. Unvisited subtrees integrate to one.
- Evaluate raw enwik8 prefixes of 100,000 and 1,000,000 bytes, in order, from fresh priors. Freeze depth eight for both; no parameter selection between runs. The prefixes overlap and are not independent replications.
- Baselines: fixed byte-context orders 0–8, Dirichlet-1/2 and uniform startup from existing `ngram-fit`. Report every order and the hindsight-best reference. The learned partition uses a start-symbol state rather than uniform startup; identify this difference, and do not attribute a small gain to structure without checking its scale.
- Score every byte by `Model<u8>` predict, score, then observe. Compare cumulative nats to negative root log evidence. All construction, updates and traversal are timed; no search or replay is performed. Count updated nodes; report allocated nodes. Compilation excluded; one timing sample and no warm-up.
- Stop each execution at 60 seconds. Retain partial results/failures. No sweep or additional benchmark is implied by a favorable result.
- Correctness gates before scoring: normalized predictions, depth-zero KT identity, explicit enumeration of all five prunings of a binary depth-two state tree, evidence/causal-loss identity, and chunked continuation. Test tolerances 1e-11 nats on tiny inputs; dataset accumulated evidence difference below 1e-5 nats.
- Success means lower causal nats than every tested fixed order on both prefixes. This does not establish superiority over all n-gram smoothing schemes or the full-corpus goal.

## Reproduce

- Base: `6792906a` plus working-copy diagnostic changes and new partition model/binary.
- Build: `cargo build --release --locked --bin partition-dfa --bin ngram-fit`.
- Posterior: `timeout 60s target/release/partition-dfa ../text-preq-encoding/preq-encoding/data/enwik/enwik8 --limit 100000 --depth 8` (repeat with `--limit 1000000`).
- Baseline: materialize each exact prefix with `head -c`, then `timeout 60s target/release/ngram-fit PREFIX --orders 0,1,2,3,4,5,6,7,8`.
- Artifacts: `artifacts/generated-partition-20260908/`, local only; input and output hashes recorded after execution.
- Environment: Rust 1.98.1, Linux 7.2.2-1-cachyos, AMD Ryzen 5 4600H; CPU release mode. No GPU.

## Results

All four executions completed without timeout. Correctness gates and full formatting, clippy, locked tests passed. Cumulative nats and negative root log evidence agree to nine printed decimal places on both prefixes.

| Prefix bytes | Posterior nats | Best fixed order | Best fixed-order nats | Reduction | Posterior bits/byte | Posterior seconds | Nodes | Node updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100,000 | 275720.909695725 | 1 | 280721.028885158 | 1.781% | 3.977811891 | 0.208160 | 239075 | 900000 |
| 1,000,000 | 2291568.905672037 | 2 | 2409951.421783493 | 4.912% | 3.306035096 | 2.523407 | 1536677 | 9000000 |

All fixed-order baseline nats:

| Order | 100,000 bytes | 1,000,000 bytes |
| --- | ---: | ---: |
| 0 | 338946.426877148 | 3507734.978874592 |
| 1 | 280721.028885158 | 2715703.916378015 |
| 2 | 290116.196726168 | 2409951.421783493 |
| 3 | 344570.558123308 | 2598638.134982758 |
| 4 | 397470.257644716 | 3078291.910820271 |
| 5 | 436961.683237990 | 3581784.411067284 |
| 6 | 465369.867566203 | 4001910.082886613 |
| 7 | 485987.949115985 | 4335341.154744701 |
| 8 | 500761.238531004 | 4587297.112600601 |

Source and artifact hashes are in the [manifest](E0-generated-partition-dfa.sha256). Artifacts are local, reproducible outputs, not remotely archived. No peak-memory measurement was taken; node count is not a byte-memory estimate.

## Interpretation and limitations

This is an exact posterior for a new explicitly finite generated-state family with tied emissions. The history constructor makes it a variable-context Bayesian model, not evidence that arbitrary recurrent structure was learned. The generic partition engine accepts other finite-state feature generators. Maximum depth defines this prior and is not presented as a harmless compute cutoff on the old unbounded prior.

The improvement is 5,000.119189 nats on 100k bytes and 118,382.516111 nats on 1M bytes. The first eight uniform-coded bytes cost only 44.361420 nats, so directly saving startup symbols cannot explain gains of this size; altered early counts can also affect later predictions. Neither prefix was used to choose depth after the protocol was frozen. The original 1,000-byte diagnostic informed the model-family direction, so this is developmental evidence, not an untouched final test set.

This meets the frozen comparison gate against the repository's fixed-order KT n-grams. It does not show optimality, superiority over adaptive smoothing/backoff, or a full-enwik8 result. The new language overlaps the variable-context model class; arbitrary DFA transition learning and the original sparse posterior remain unsolved.

## Follow-up

Record both positive and negative results. Expand state generators and compare stronger n-gram baselines only after causal correctness and measured coding gains are established.

Next compare emission priors under matched smoothing controls and add compact non-context state generators if they improve measured causal cost. Keep full-corpus evaluation and resource accounting as outstanding gates; do not declare the user objective complete from these prefixes.
