//! Minimal interfaces for sequential probabilistic prediction.

/// A probability mass distribution over observations.
///
/// Return natural-log probability, with `NEG_INFINITY` for zero probability.
/// Implementations must be normalized over their support. The evaluator checks
/// the observed log probability for NaN/positive values, not full normalization.
pub trait Distribution<T> {
    fn ln_prob(&self, observation: &T) -> f64;
}

/// An online model: predict from the observed prefix, then observe one new item.
///
/// Predictions may borrow model state. They are dropped before `observe`.
/// `predict` must not use future observations (including through interior state).
pub trait Model<T> {
    fn predict(&self) -> impl Distribution<T>;
    fn observe(&mut self, observation: T);
}
