//! KRAFT: compute-aware Bayesian program mixtures and prequential evaluation.

pub mod anytime;
pub mod baselines;
pub mod evaluate;
pub mod model;
pub mod models;

pub use evaluate::{Evaluation, evaluate, evaluate_with_costs};
pub use model::{Distribution, Model};

/// Why a set of log weights cannot define a categorical distribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightError {
    /// The model set is empty.
    Empty,
    /// NaN and positive infinity are not supported.
    Invalid,
    /// Every model has zero mass (negative-infinite log weight).
    ZeroMass,
}

/// Normalize natural-log weights using a shifted exponential sum.
///
/// Negative infinity denotes zero mass. At least one weight must be finite.
/// Extremely small probabilities may underflow to zero in `f64`.
///
/// ```
/// let weights = kraft::normalize_log_weights(&[0.0, 0.0]).unwrap();
/// assert_eq!(weights, vec![0.5, 0.5]);
/// ```
pub fn normalize_log_weights(log_weights: &[f64]) -> Result<Vec<f64>, WeightError> {
    if log_weights.is_empty() {
        return Err(WeightError::Empty);
    }
    if log_weights
        .iter()
        .any(|x| x.is_nan() || *x == f64::INFINITY)
    {
        return Err(WeightError::Invalid);
    }
    let maximum = log_weights
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    if maximum == f64::NEG_INFINITY {
        return Err(WeightError::ZeroMass);
    }
    let mut weights: Vec<_> = log_weights.iter().map(|x| (x - maximum).exp()).collect();
    let total: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= total;
    }
    Ok(weights)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recovers_known_bayes_posterior() {
        // Prior [1/4, 3/4], observed-data likelihood [4/5, 1/5].
        let weights =
            normalize_log_weights(&[(0.25_f64 * 0.8).ln(), (0.75_f64 * 0.2).ln()]).unwrap();
        assert!((weights[0] - 4.0 / 7.0).abs() < 1e-14);
        assert!((weights[1] - 3.0 / 7.0).abs() < 1e-14);
    }

    #[test]
    fn stable_under_large_common_log_offset() {
        let expected = normalize_log_weights(&[-1.0, -2.0, -3.0]).unwrap();
        let actual = normalize_log_weights(&[-10001.0, -10002.0, -10003.0]).unwrap();
        for (a, b) in actual.iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!((actual.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }

    #[test]
    fn zero_mass_models_are_allowed() {
        assert_eq!(
            normalize_log_weights(&[f64::NEG_INFINITY, -10000.0]),
            Ok(vec![0.0, 1.0])
        );
    }

    #[test]
    fn rejects_undefined_distributions() {
        assert_eq!(normalize_log_weights(&[]), Err(WeightError::Empty));
        assert_eq!(
            normalize_log_weights(&[f64::NEG_INFINITY]),
            Err(WeightError::ZeroMass)
        );
        for invalid in [f64::NAN, f64::INFINITY] {
            assert_eq!(
                normalize_log_weights(&[0.0, invalid]),
                Err(WeightError::Invalid)
            );
        }
    }
}
