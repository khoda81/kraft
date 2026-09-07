# Prequential objective

This document is the canonical definition of KRAFT's evaluation target.

## Primary score

KRAFT is an online probabilistic learner. For observations (x_1,ldots,x_T), the model must predict each observation using only the previously decoded prefix:

```text
cost = 0
for x in data:
    prediction = model.predict()
    cost -= prediction.ln_prob(x)
    model.observe(x)
```

The primary coding cost is therefore

```math
C_{mathrm{preq}}(x_{1:T})
=
-sum_{t=1}^{T}ln P(x_tmid x_{<t}).
```

Prediction must happen before observation. No model may use the current symbol or future symbols when forming the probability used to code the current symbol.

This is an ideal code length. An arithmetic coder is not required for the benchmark.

## Bayesian mixture is the KRAFT model

Let (h) index model descriptions with a proper prior (pi(h)). Each fixed model supplies a causal predictive distribution and therefore a joint prequential probability

```math
P_h(x_{1:T})=prod_{t=1}^{T}P_h(x_tmid x_{<t}).
```

KRAFT's Bayesian mixture is

```math
M(x_{1:T})=sum_h pi(h)P_h(x_{1:T}).
```

The online mixture predictor is

```math
M(x_tmid x_{<t})
=
sum_h P(hmid x_{<t})P_h(x_tmid x_{<t}),
```

equivalently (M(x_{1:t})/M(x_{<t})). Hence

```math
C_{mathrm{KRAFT}}(x_{1:T})
=
-sum_tln M(x_tmid x_{<t})
=
-ln M(x_{1:T}).
```

This is the quantity KRAFT ultimately optimizes and reports as its own coding cost.

## The prior is not an extra transmitted model

Encoder and decoder begin from the same prior and perform the same causal Bayesian update after each decoded observation. No model description is separately transmitted.

For every fixed hypothesis (h),

```math
M(x)ge pi(h)P_h(x),
```

so

```math
C_{mathrm{KRAFT}}(x)
le
C_h(x)-lnpi(h).
```

The right-hand side is a valid single-hypothesis upper bound on the mixture cost and has the familiar two-part-MDL form. It is not an additional charge added to the actual mixture code.

The Bayesian mixture is a soft minimum over these description-plus-data costs:

```math
C_{mathrm{KRAFT}}(x)
=
-lnsum_h
expleft[-left(C_h(x)-lnpi(h)ight)ight].
```

When one posterior mode dominates, the mixture cost approaches that mode's data cost plus its negative log prior. Thus the model-identification cost appears as predictive regret while Bayes learns which description to trust.

## What is and is not a valid reported score

Three quantities must remain distinct.

1. **Prespecified fixed-model prequential cost**  
   (C_h(x)=-ln P_h(x)) is valid if (h) was fixed independently of the evaluated future observations.

2. **Hindsight fixed-model cost**  
   If the whole evaluation stream is searched to choose (h_*), then (C_{h_*}(x)) is an oracle diagnostic. It is not the online coding cost of the structure-learning algorithm.

3. **Bayesian mixture prequential cost**  
   (-ln M(x)) is the actual KRAFT target. The prior is part of the predictor and no separate model transmission is required.

The bound (C_{h_*}(x)-lnpi(h_*)) remains valid even when (h_*) was found in hindsight, because it upper-bounds the same Bayesian mixture probability. Search quality changes bound tightness, not validity.

## Batch evidence shortcuts

A batch computation is allowed only when it is mathematically identical to the causal prequential product for a prespecified model.

For example, a fixed DFA with Dirichlet-1/2 emissions may first collect the state/byte counts induced by its causal trajectory and then evaluate the integrated Dirichlet evidence in closed form. Conjugacy makes that joint evidence exactly equal to the product of the per-symbol posterior predictive probabilities.

Every such shortcut should have a regression test against a literal `predict -> score -> observe` reference implementation.

## Causal approximation and compute

Approximate inference is also part of the codec. At symbol (t), the probability used to encode (x_t) may depend only on:

- the common prior and model definition,
- the already decoded prefix (x_{<t}),
- deterministic inference state derived from that prefix,
- and the declared compute/precision policy.

It may not be selected by looking at (x_t) or future observations.

A search run over the entire corpus followed by replay with the discovered structures is therefore an oracle analysis, not a prequential KRAFT run.

The long-term compute-bounded objective is:

```math
	ext{given an inference budget }B,quad
	ext{minimize }C_{mathrm{preq}}.
```

As inference work increases, the approximate predictor should approach the exact Bayesian mixture without changing the declared hypothesis space.

## Metrics

Use total nats, total bits/bytes/KB/MB where useful, and coding ratio

```math
mathrm{ratio}(A	ext{ vs }B)=C_B/C_A,
```

with higher better. The ratio is invariant to log base. Against a uniform byte model it equals the ideal compression ratio.
