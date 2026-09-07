//! Generic primitives for prior-mass-preserving anytime Bayesian inference.
//!
//! This module intentionally contains no scheduler. It represents the fixed
//! Bayesian target and evidence bounds; a scheduler may choose refinement order
//! without changing the meaning of the regions or their probability mass.

use std::{error::Error, fmt};

/// A disjoint subset of the declared hypothesis space with known prior mass.
pub trait PriorRegion {
    /// Natural logarithm of the region's total prior probability.
    fn ln_prior_mass(&self) -> f64;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundError {
    NaN,
    PositiveLogMass,
    Reversed,
}

impl fmt::Display for BoundError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NaN => write!(f, "evidence bounds cannot contain NaN"),
            Self::PositiveLogMass => write!(f, "joint probability mass cannot exceed one"),
            Self::Reversed => write!(f, "evidence lower bound exceeds upper bound"),
        }
    }
}

impl Error for BoundError {}

/// Lower and upper bounds on one region's joint Bayesian contribution.
///
/// Values are natural logarithms of probability mass. Negative infinity denotes
/// exact zero. For a region R these bounds mean
///
/// `exp(ln_lower) <= sum_{h in R} pi(h) P_h(x) <= exp(ln_upper)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LogEvidenceBounds {
    ln_lower: f64,
    ln_upper: f64,
}

impl LogEvidenceBounds {
    pub fn new(ln_lower: f64, ln_upper: f64) -> Result<Self, BoundError> {
        if ln_lower.is_nan() || ln_upper.is_nan() {
            return Err(BoundError::NaN);
        }
        if ln_lower > 0.0 || ln_upper > 0.0 {
            return Err(BoundError::PositiveLogMass);
        }
        if ln_lower > ln_upper {
            return Err(BoundError::Reversed);
        }
        Ok(Self {
            ln_lower,
            ln_upper,
        })
    }

    pub fn exact(ln_mass: f64) -> Result<Self, BoundError> {
        Self::new(ln_mass, ln_mass)
    }

    pub fn unresolved(ln_upper: f64) -> Result<Self, BoundError> {
        Self::new(f64::NEG_INFINITY, ln_upper)
    }

    pub fn ln_lower(self) -> f64 {
        self.ln_lower
    }

    pub fn ln_upper(self) -> f64 {
        self.ln_upper
    }

    pub fn is_exact(self) -> bool {
        self.ln_lower == self.ln_upper
    }
}

/// One unresolved or resolved Bayesian region on the frontier.
#[derive(Debug, Clone, PartialEq)]
pub struct FrontierNode<R> {
    pub region: R,
    pub evidence: LogEvidenceBounds,
}

/// The three semantics-preserving ways an inference engine may refine work.
#[derive(Debug, Clone, PartialEq)]
pub enum Refinement<R> {
    /// Replace one region by an exact disjoint partition of the same region.
    Partition(Vec<FrontierNode<R>>),
    /// Keep the same region but replace its evidence bounds with tighter ones.
    Tighten(FrontierNode<R>),
    /// Resolve the region's joint contribution exactly.
    Resolve { ln_joint_mass: f64 },
}

/// Aggregate disjoint region bounds into global mixture-evidence bounds.
pub fn aggregate_evidence<'a>(
    bounds: impl IntoIterator<Item = &'a LogEvidenceBounds>,
) -> Result<LogEvidenceBounds, BoundError> {
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for bound in bounds {
        lower = log_add_exp(lower, bound.ln_lower);
        upper = log_add_exp(upper, bound.ln_upper);
    }
    LogEvidenceBounds::new(lower, upper)
}

/// Stable `ln(exp(a) + exp(b))` with negative infinity as exact zero.
pub fn log_add_exp(a: f64, b: f64) -> f64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Clone, Copy, PartialEq)]
    struct ToyRegion(f64);

    impl PriorRegion for ToyRegion {
        fn ln_prior_mass(&self) -> f64 {
            self.0.ln()
        }
    }

    #[test]
    fn validates_probability_bounds() {
        assert_eq!(
            LogEvidenceBounds::new(-2.0, -3.0),
            Err(BoundError::Reversed)
        );
        assert_eq!(
            LogEvidenceBounds::new(-1.0, 0.1),
            Err(BoundError::PositiveLogMass)
        );
        assert!(LogEvidenceBounds::exact(f64::NEG_INFINITY).is_ok());
    }

    #[test]
    fn aggregates_disjoint_evidence_bounds() {
        let a = LogEvidenceBounds::new(0.1_f64.ln(), 0.2_f64.ln()).unwrap();
        let b = LogEvidenceBounds::new(0.3_f64.ln(), 0.4_f64.ln()).unwrap();
        let total = aggregate_evidence([&a, &b]).unwrap();
        assert!((total.ln_lower().exp() - 0.4).abs() < 1e-14);
        assert!((total.ln_upper().exp() - 0.6).abs() < 1e-14);
    }

    #[test]
    fn toy_partition_preserves_prior_mass() {
        let parent = ToyRegion(0.75);
        let children = [ToyRegion(0.5), ToyRegion(0.25)];
        let child_log_mass = children
            .iter()
            .fold(f64::NEG_INFINITY, |total, child| {
                log_add_exp(total, child.ln_prior_mass())
            });
        assert!((child_log_mass.exp() - parent.ln_prior_mass().exp()).abs() < 1e-14);
    }
}
