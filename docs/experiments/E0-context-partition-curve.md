# E0e — Full-corpus previous-byte context curve

Status: implementation added; enwik8 result pending.

## Question

How much coding gain comes from increasing the number of learned previous-byte context classes, before adding recurrent DFA structure?

The model is

    state_(t+1) = g(x_t)

with g mapping each of the 256 byte values to one of N context states. Each state has the same integrated Dirichlet-1/2 next-byte model used elsewhere in KRAFT.

## Fitting

The entire 100 MB corpus is summarized as a 256 x 256 previous-byte/next-byte count matrix.

The fitter starts from 256 singleton previous-byte clusters and greedily agglomerates the pair whose merge loses the least exact integrated evidence. Pair merge scores are cached; only scores involving the newly merged cluster are recomputed. This produces one nested hierarchy and therefore a full N-state curve cheaply.

The hierarchy is a heuristic for intermediate N. The N=1 point is exactly the byte KT unigram. The N=256 point is exactly the full previous-byte context partition. Intermediate points are not claimed globally optimal.

Because the start state is fixed while cluster labels are otherwise arbitrary, the first observed byte is assigned to whichever active cluster gives the best exact one-observation evidence increment.

## Mixture bounds

For any concrete reset mapping g with N labels, a uniform prior over all N^256 reset mappings gives

    C_reset_mix <= C_g + 256 log2(N) bits.

The same mapping can be embedded as one table in the complete N-state DFA family. Under the uniform labeled transition-table prior,

    C_full_dfa_mix <= C_g + 256 N log2(N) bits.

These are valid candidate bounds even though the agglomerative fit is heuristic.

## Reproduce

    cargo run --release --locked --bin context-fit -- \
      ../text-preq-encoding/preq-encoding/data/enwik/enwik8

Default requested states:

    1,2,4,8,16,32,64,128,256

The output reports candidate total nats, coding ratio versus uniform and KT, reset-family prior cost and certified coding-ratio lower bound, and the corresponding full-DFA-family prior cost and bound.

## Decision use

The main artifact is the curve

    N -> coding ratio.

If gains saturate quickly, KRAFT should invest in variable-length/history structure rather than simply adding more previous-byte clusters. If gains continue strongly through large N, richer finite context partitions are a valuable next rung.