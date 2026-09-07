//! Prior regions for the unbounded sparse-DFA state count.

use std::num::NonZeroU64;

use crate::{anytime::PriorRegion, nat::PositiveNat};

/// One exact state count N.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCount(PositiveNat);

impl StateCount {
    pub fn one() -> Self {
        Self(PositiveNat::one())
    }

    pub fn successor(&self) -> Self {
        Self(self.0.successor())
    }

    pub fn value(&self) -> &PositiveNat {
        &self.0
    }
}

impl From<PositiveNat> for StateCount {
    fn from(value: PositiveNat) -> Self {
        Self(value)
    }
}

impl From<NonZeroU64> for StateCount {
    fn from(value: NonZeroU64) -> Self {
        Self(value.into())
    }
}

impl PriorRegion for StateCount {
    fn ln_prior_mass(&self) -> f64 {
        -self.0.ln() - self.0.successor().ln()
    }
}

/// The infinite region N >= min.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCountTail(StateCount);

impl StateCountTail {
    pub fn root() -> Self {
        Self(StateCount::one())
    }

    pub fn from_min(min: StateCount) -> Self {
        Self(min)
    }

    pub fn min(&self) -> &StateCount {
        &self.0
    }

    pub fn split(&self) -> (StateCount, Self) {
        let exact = self.0.clone();
        let next = exact.successor();
        (exact, Self(next))
    }
}

impl PriorRegion for StateCountTail {
    fn ln_prior_mass(&self) -> f64 {
        -self.0.0.ln()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::anytime::log_add_exp;

    #[test]
    fn root_is_the_whole_state_count_prior() {
        assert_eq!(StateCountTail::root().ln_prior_mass(), 0.0);
    }

    #[test]
    fn splitting_preserves_telescoping_prior_mass() {
        let mut tail = StateCountTail::root();
        for _ in 0..1000 {
            let parent = tail.ln_prior_mass();
            let (exact, next) = tail.split();
            let children = log_add_exp(exact.ln_prior_mass(), next.ln_prior_mass());
            assert!((children - parent).abs() < 1e-12);
            tail = next;
        }
    }

    #[test]
    fn state_count_is_not_limited_by_u64() {
        let max = PositiveNat::from(NonZeroU64::new(u64::MAX).unwrap());
        let count = StateCount::from(max.successor());
        assert_eq!(count.value().to_string(), "18446744073709551616");
    }
}
