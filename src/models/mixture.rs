//! Bayesian mixtures of online predictive models.
//!
//! A two-model mixture stores the log posterior weight ratio between its
//! component learners. After observing a symbol, that ratio is multiplied by
//! the models' likelihood ratio; in log space this is addition of coding
//! advantage.

use crate::{Distribution, Model};

/// Exact Bayesian mixture of two online predictive models.
///
/// log_weight_ratio is ln(w_a / w_b). The default constructor starts with
/// equal prior weights, so the ratio is one and the log ratio is zero.
///
/// For an observed symbol x, Bayes updates the weight ratio as:
///
/// (w_a / w_b)' = (w_a / w_b) * P_a(x) / P_b(x).
///
/// Equivalently in log space:
///
/// log_weight_ratio += ln P_a(x) - ln P_b(x).
///
/// The next prediction is the posterior-weighted mixture of the two component
/// predictions. Component models are both updated on every observation.
#[derive(Debug, Clone)]
pub struct Mixture<A, B> {
    pub a: A,
    pub b: B,
    log_weight_ratio: f64,
}

impl<A, B> Mixture<A, B> {
    /// Construct a mixture with equal prior weight on both component models.
    pub fn new(a: A, b: B) -> Self {
        Self {
            a,
            b,
            log_weight_ratio: 0.0,
        }
    }

    /// Natural logarithm of the posterior weight ratio w_a / w_b.
    pub fn log_weight_ratio(&self) -> f64 {
        self.log_weight_ratio
    }

    /// Current posterior weight assigned to model A.
    pub fn weight_a(&self) -> f64 {
        sigmoid(self.log_weight_ratio)
    }

    /// Current posterior weight assigned to model B.
    pub fn weight_b(&self) -> f64 {
        sigmoid(-self.log_weight_ratio)
    }
}

/// Predictive distribution produced by a Mixture.
///
/// The component predictions are combined using the mixture's posterior model
/// weights at the moment predict was called.
#[derive(Debug, Clone)]
pub struct MixturePrediction<A, B> {
    a: A,
    b: B,
    log_weight_ratio: f64,
}

impl<A, B, T> Distribution<T> for MixturePrediction<A, B>
where
    A: Distribution<T>,
    B: Distribution<T>,
{
    fn ln_prob(&self, observation: &T) -> f64 {
        let a_ln_prob = self.a.ln_prob(observation);
        let b_ln_prob = self.b.ln_prob(observation);

        if self.log_weight_ratio == f64::INFINITY {
            return a_ln_prob;
        }
        if self.log_weight_ratio == f64::NEG_INFINITY {
            return b_ln_prob;
        }

        // If r = w_a / w_b, then
        // P(x) = (r P_a(x) + P_b(x)) / (r + 1).
        log_add_exp(a_ln_prob + self.log_weight_ratio, b_ln_prob)
            - log_add_exp(self.log_weight_ratio, 0.0)
    }
}

impl<A, B, T> Model<T> for Mixture<A, B>
where
    A: Model<T>,
    B: Model<T>,
    T: Clone,
{
    fn predict(&self) -> impl Distribution<T> {
        MixturePrediction {
            a: self.a.predict(),
            b: self.b.predict(),
            log_weight_ratio: self.log_weight_ratio,
        }
    }

    fn observe(&mut self, observation: T) {
        let (a_ln_prob, b_ln_prob) = {
            let a_prediction = self.a.predict();
            let b_prediction = self.b.predict();
            (
                a_prediction.ln_prob(&observation),
                b_prediction.ln_prob(&observation),
            )
        };

        // Exact posterior zero mass is absorbing. If both models assign zero
        // probability to an observation, there is no relative evidence, so keep
        // the existing odds rather than manufacturing NaN from (-inf)-(-inf).
        if self.log_weight_ratio.is_finite() {
            if a_ln_prob == f64::NEG_INFINITY && b_ln_prob == f64::NEG_INFINITY {
                // No relative evidence.
            } else if a_ln_prob == f64::NEG_INFINITY {
                self.log_weight_ratio = f64::NEG_INFINITY;
            } else if b_ln_prob == f64::NEG_INFINITY {
                self.log_weight_ratio = f64::INFINITY;
            } else {
                self.log_weight_ratio += a_ln_prob - b_ln_prob;
            }
        }

        self.a.observe(observation.clone());
        self.b.observe(observation);
    }
}

fn sigmoid(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

fn log_add_exp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    if a >= b {
        a + (b - a).exp().ln_1p()
    } else {
        b + (a - b).exp().ln_1p()
    }
}
