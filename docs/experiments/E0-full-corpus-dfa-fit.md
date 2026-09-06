# E0d — Full-corpus heuristic two-state DFA fit

Status: implementation added; local enwik8 result pending.

## Question

What coding performance does the fixed N = 2 DFA family appear to approach on the full 100 MB enwik8 corpus, before investing further in exact symbolic inference?

Exact weighted model counting currently reaches byte 46. For the complete N = 2 family there are 512 independent Boolean transition destinations, hence 2^512 complete labeled transition tables.

For any observed corpus x and any concrete candidate DFA h,

    P_mix(x) = 2^-512 sum_h P_h(x)

implies

    2^-512 P_h(x) <= P_mix(x).

Therefore

    C_mix(x) <= C_h(x) + 512 bits.

This bound holds for any candidate, regardless of whether the search found the global MAP table. The additive gap is exactly 64 bytes over the whole corpus.

Consequently, finding one strong full-corpus DFA gives a rigorous lower bound on the exact mixture's coding ratio:

    coding_ratio_uniform(mix)
      >= C_uniform / (C_candidate + 512 bits).

If the heuristic candidate is near MAP, the bound is also a close estimate of the mixture's asymptotic full-file performance.

## Candidate evaluator

A complete N = 2 table stores delta(0, byte), delta(1, byte) in {0,1} for all 256 byte values.

For a fixed table, one pass through the corpus determines the state trajectory and the two state-conditioned byte histograms. The integrated Dirichlet-1/2 evidence is then computed from those sufficient statistics.

A candidate evaluation therefore requires only one state bit, a 512-bit transition table, two 256-entry counters, one streaming pass over the bytes, and no per-byte logarithms.

## Search procedure

### 1. Full-corpus reset-DFA seed

First fit the restricted family

    delta(0,b) = delta(1,b) = g(b).

Then the next state depends only on the previous byte. This is a two-cluster partition of previous-byte bigram rows.

The full 256 x 256 bigram matrix is accumulated once over the entire corpus. Multi-start coordinate descent moves whole previous-byte rows between the two states using exact integrated-evidence deltas.

This stage is cheap and produces a strong structured seed plus an immediately valid full-corpus mixture bound.

### 2. General 512-bit CEM search

Cross-entropy-method search then operates over all 512 transition bits on a shorter prefix.

Each generation samples a population of complete DFA tables from independent Bernoulli transition-bit probabilities, scores all tables exactly on the search prefix, keeps the elite tables, updates each transition-bit probability toward its elite empirical frequency, and retains a hall of strong candidates.

The first CEM restart is biased around the reset-DFA seed. Later restarts begin from an uninformative 0.5 Bernoulli distribution.

This is heuristic optimization only. It is not an approximation to posterior mass.

### 3. Screening and full-file final scoring

The search hall is screened on a larger prefix, then only the best finalists are evaluated on the complete corpus.

The reset seed and one-state-equivalent zero table are always included in final full-file scoring.

## Reproduce

Input:

    ../text-preq-encoding/preq-encoding/data/enwik/enwik8

Default first run:

    cargo run --release --locked --bin dfa-fit -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8

Defaults:

    search_bytes = 500,000
    screen_bytes = 10,000,000
    population = 64
    generations = 20
    elite = 8
    CEM restarts = 2
    reset restarts = 32
    screen candidates = 32
    finalists = 8
    threads = available parallelism
    seed = 1

A more expensive follow-up after the default result:

    cargo run --release --locked --bin dfa-fit -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8 \
      --search-bytes 2000000 \
      --screen-bytes 20000000 \
      --population 128 \
      --generations 40 \
      --elite 16 \
      --cem-restarts 4 \
      --reset-restarts 64 \
      --screen-candidates 64 \
      --finalists 16

## Reported result

The binary prints exact byte-KT full-corpus cost, optimized reset-DFA full-corpus cost, CEM search progress, full-file finalist costs, coding ratios versus uniform and KT, the best complete transition table, the exact 512-bit transition-prior penalty, and certified exact-mixture bounds.

Higher coding ratio is better.

The transition table is printed as two hexadecimal rows. Within each hexadecimal nibble, byte offsets are stored least-significant bit first.

## Interpretation

This experiment answers a different question from E0c. E0c asks how far exact symbolic marginalization can proceed. E0d asks what the full two-state model family is capable of predicting on the complete corpus.

If even the strong heuristic DFA barely beats KT, then richer state count or richer program structure is likely more important than exact N = 2 inference.

If it materially beats KT, then the exact N = 2 mixture is guaranteed to be at least almost as good in cumulative coding cost: the candidate-to-mixture penalty is at most 512 bits total.

If CEM materially improves over the reset-DFA seed, recurrent two-state memory is useful beyond merely clustering the previous byte. If it does not, that suggests a structured byte-context model may capture most of the available N = 2 gain with a far shorter description.