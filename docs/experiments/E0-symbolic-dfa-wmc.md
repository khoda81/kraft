# E0c — Exact N = 2 DFA evidence by symbolic weighted model counting

Status: first exact ADD evaluator implemented and measured; byte 64 not reached

Written (UTC): 2026-09-06

Executed (UTC): 2026-09-06

Related: [persistent leaf oracle](E0-persistent-dfa-state.md), D017

## Question and hypothesis

Can the exact N = 2 DFA joint evidence be computed by sharing algebraic subproblems across transition assignments instead of enumerating posterior leaves?

For an observed byte prefix and Boolean transition-table variables, the state trajectory is a deterministic Boolean function. The hypothesis was that a reduced ordered algebraic decision diagram (ADD) over those variables could share equal state, count, and likelihood subfunctions and therefore run materially beyond the byte-26 leaf-oracle limit.

## Exact formulation

For each byte value encountered, introduce two Boolean variables: the destination of `(state 0, byte)` and `(state 1, byte)`. The start state is zero. The symbolic state recurrence is

```math
s_{t+1}=\operatorname{ite}(s_t,z_{1,x_t},z_{0,x_t}).
```

The evaluator maintains reduced ordered ADDs for the current state, the number of observations assigned to state one, and the state-one count for each observed byte. State-zero counts are their complements. It reconstructs the integrated Dirichlet-1/2 log likelihood from these sufficient statistics using log rising factorials, which is algebraically the same gamma-function expression in [E0](E0-partial-dfa-posterior.md).

The final log-likelihood ADD is averaged recursively over each Boolean decision node with prior probability 1/2. Variables that do not affect a trajectory disappear under reduction, so averaging the full introduced transition table is equivalent to the trajectory weight `2^(-m(s))`. No transition assignment, posterior leaf, pruning rule, or approximation is materialized.

The variable order assigns the two transition bits for a byte consecutively when that byte first appears. This is an initial order, not an optimized or dynamically reordered one.

## Protocol

- Compare against complete enumeration of all N = 2 transition tables on tiny sequences.
- Compare joint log evidence against the persistent discovery-quotient leaf oracle at every prefix through byte 22, with tolerance `1e-10` nat in the CLI and tighter unit-test tolerances on shorter sequences.
- Run the ADD alone toward 64 bytes with explicit ten-million and thirty-million allocated-node guards.
- Report transition variables, total allocated nodes, live nodes reachable from current roots, garbage collections, reclaimed nodes, payload estimate, Linux RSS, exact log evidence, and coding ratios.
- Node-limit failures leave the previous completed prefix as the result. They are resource guards, not posterior pruning.

## Reproduce

Base: local `feat/partial-dfa-posterior` commit `16cc360444df`, plus the current working-copy change.

Input:

- `../text-preq-encoding/preq-encoding/data/enwik/enwik8`
- 100,000,000 bytes
- SHA-256 `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`

Environment is the same local Rust 1.98.1 / Ryzen 5 4600H system recorded by [E0b](E0-persistent-dfa-state.md).

```sh
cargo run --release --locked --bin dfa-wmc -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 22 \
  --max-nodes 10000000 \
  --oracle-through 22

cargo run --release --locked --bin dfa-wmc -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 64 \
  --max-nodes 10000000

cargo run --release --locked --bin dfa-wmc -- \
  ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
  --limit 64 \
  --max-nodes 30000000
```

Local raw outputs are under `artifacts/dfa-wmc/`. They are ignored; this file is the concise durable result.

## Validation results

Unit tests agree with explicit full transition-table enumeration on empty, one-byte, repeated-byte, and three-distinct-byte sequences. They also agree with the leaf oracle on every prefix of `mediawiki`. A separate test compacts all live ADD roots and verifies unchanged evidence and successful continuation.

The release CLI compared every enwik8 prefix from 1 through 22. The largest absolute ADD-versus-leaf evidence difference was approximately `2.42e-12` nat. At byte 22 both report:

- log evidence `-119.714066808757` nats;
- coding ratio versus uniform `1.019044019058`;
- coding ratio versus KT `0.992364462196`.

This validates the joint-evidence identity used by the evaluator without requiring sequential posterior predictions from the ADD.

## Growth results

Final closed-form ADD representation, ten-million-node guard:

| Prefix | Transition variables | Allocated nodes | Live nodes | Payload MB | RSS MB | Elapsed s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 22 | 36 | 70,660 | 3,991 | 3.954 | 6.840 | 0.027 |
| 26 | 40 | 129,704 | 7,529 | 5.810 | 10.203 | 0.049 |
| 32 | 42 | 1,010,189 | 176,672 | 46.309 | 64.455 | 0.441 |
| 37 | 42 | 2,569,274 | 783,872 | 125.872 | 116.806 | 5.075 |
| 44 | 48 | 2,441,322 | 2,441,322 | 125.872 | 162.988 | 19.099 |

The ten-million run stopped while constructing byte 45. Its final failed-step workspace reached the node cap and approximately 505 MB RSS; byte 44 is the last completed exact prefix.

With a thirty-million-node guard, exact evaluation reached byte 46:

- log evidence `-232.926235590729` nats;
- coding ratio versus uniform `1.095102755596`;
- coding ratio versus KT `0.987847759546`;
- 48 transition variables;
- 5,612,035 live nodes after byte 46;
- approximately 280 MB RSS immediately after garbage collection;
- 42.864 seconds to the byte-46 row.

Construction of byte 47 exceeded the thirty-million-node guard even after garbage collection. The failed temporary workspace reached approximately 1.68 GB RSS and the full stopped run took 83.463 seconds.

For comparison, the persistent leaf oracle at byte 26 used 9,289,728 posterior components, 1,071.514 MB estimated payload, 1,809.519 MB RSS, and 109.419 seconds to the completed row. At the same byte the ADD used 7,529 live nodes, 5.810 MB estimated payload, roughly 10 MB RSS, and 0.049 seconds. These are single-run wall times, not a controlled performance benchmark, but the scale difference establishes genuine cross-leaf algebraic reuse.

## Interpretation and limitations

The weighted-model-counting formulation is correct for the tested prefixes and materially stronger than prefix sharing. It extends the exact enwik8 computation from byte 26 to byte 46 and compresses millions of leaf hypotheses into thousands of live nodes at byte 26.

It did not reach byte 64. The next limitation is ADD apply width and variable order. At byte 46 only 5.61 million nodes are live, but constructing the next integrated-likelihood function requires more than 24 million additional temporary nodes. The manager currently garbage-collects only between complete observations; it cannot compact intermediate factor sums within one ADD apply. First-seen byte-pair variable ordering is also unoptimized.

The allocated-node payload estimate includes node and uniqueness-table capacities but not all allocator/hash-table metadata. RSS is sampled, not peak-tracked. Floating terminals store log likelihoods; reduction uses exact `f64::to_bits` terminal identity, while evidence comparisons use tolerance. The mathematical model remains exact, but floating summation order differs slightly from the leaf oracle.

## Follow-up

Keep both engines:

- the persistent discovery-quotient leaf implementation is the short-prefix posterior oracle;
- `dfa-wmc` is the joint-evidence evaluator for symbolic experiments;
- predictive quotient remains a regression test and explicit diagnostic option, not the normal benchmark default.

The next surgical symbolic experiments are variable-order comparisons and garbage collection/factor scheduling inside closed-form likelihood construction. Byte 64 remains an unmet gate. A structured transition-description prior should be considered if those changes do not prevent essentially exponential ADD width.
