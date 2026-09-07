//! Generic primitives for prior-mass-preserving anytime Bayesian inference.
//!
//! This module intentionally contains no scheduler. It represents the fixed
//! Bayesian target and evidence bounds; a scheduler may choose refinement order
//! without changing the meaning of the regions or their probability mass.

/// A disjoint subset of the declared hypothesis space with known prior mass.
pub trait PriorRegion {
    /// Natural logarithm of the region's total prior probability.
    fn ln_prior_mass(&self) -> f64;
}

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
    pub fn exact(ln_mass: f64) -> Self {
        Self {
            ln_lower: ln_mass,
            ln_upper: ln_mass,
        }
    }

    pub fn unresolved(ln_upper: f64) -> Self {
        Self {
            ln_lower: f64::NEG_INFINITY,
            ln_upper,
        }
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
) -> LogEvidenceBounds {
    let mut lower = f64::NEG_INFINITY;
    let mut upper = f64::NEG_INFINITY;
    for bound in bounds {
        lower = log_add_exp(lower, bound.ln_lower);
        upper = log_add_exp(upper, bound.ln_upper);
    }
    LogEvidenceBounds {
        ln_lower: lower,
        ln_upper: upper,
    }
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

    #[test]
    fn aggregates_disjoint_evidence_bounds() {
        let a = LogEvidenceBounds {
            ln_lower: 0.1_f64.ln(),
            ln_upper: 0.2_f64.ln(),
        };
        let b = LogEvidenceBounds {
            ln_lower: 0.3_f64.ln(),
            ln_upper: 0.4_f64.ln(),
        };
        let total = aggregate_evidence([&a, &b]);
        assert!((total.ln_lower().exp() - 0.4).abs() < 1e-14);
        assert!((total.ln_upper().exp() - 0.6).abs() < 1e-14);
    }
}
