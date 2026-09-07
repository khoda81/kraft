//! Exact prior regions for sparse-DFA structure.

use crate::{anytime::PriorRegion, models::sparse_dfa::DefaultTopology, nat::PositiveNat};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCount(PositiveNat);

impl StateCount {
    pub fn value(&self) -> &PositiveNat {
        &self.0
    }

    fn exception_normalizer(&self) -> PositiveNat {
        if self.0.is_one() {
            PositiveNat::one()
        } else {
            self.0.shifted(8).successor()
        }
    }

    pub fn partition_topologies(&self) -> [TopologyChoice; 3] {
        DefaultTopology::ALL.map(|topology| TopologyChoice {
            states: self.clone(),
            topology,
        })
    }
}

impl From<PositiveNat> for StateCount {
    fn from(value: PositiveNat) -> Self {
        Self(value)
    }
}

impl PriorRegion for StateCount {
    fn ln_prior_mass(&self) -> f64 {
        -self.0.ln() - self.0.successor().ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StateCountTail(PositiveNat);

impl StateCountTail {
    pub fn root() -> Self {
        Self(PositiveNat::one())
    }

    pub fn split(&self) -> (StateCount, Self) {
        (StateCount(self.0.clone()), Self(self.0.successor()))
    }
}

impl PriorRegion for StateCountTail {
    fn ln_prior_mass(&self) -> f64 {
        -self.0.ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TopologyChoice {
    states: StateCount,
    topology: DefaultTopology,
}

impl TopologyChoice {
    pub fn states(&self) -> &StateCount {
        &self.states
    }

    pub fn topology(&self) -> DefaultTopology {
        self.topology
    }

    pub fn exception_counts(&self) -> ExceptionCountTail {
        ExceptionCountTail {
            topology: self.clone(),
            next: PositiveNat::one(),
            remaining: self.states.exception_normalizer(),
        }
    }
}

impl PriorRegion for TopologyChoice {
    fn ln_prior_mass(&self) -> f64 {
        self.states.ln_prior_mass() - (3.0_f64).ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionCount {
    topology: TopologyChoice,
    plus_one: PositiveNat,
}

impl PriorRegion for ExceptionCount {
    fn ln_prior_mass(&self) -> f64 {
        let normalizer = self.topology.states.exception_normalizer();
        self.topology.ln_prior_mass() + normalizer.successor().ln()
            - normalizer.ln()
            - self.plus_one.ln()
            - self.plus_one.successor().ln()
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ExceptionCountTail {
    topology: TopologyChoice,
    next: PositiveNat,
    remaining: PositiveNat,
}

impl ExceptionCountTail {
    pub fn split(&self) -> (ExceptionCount, Option<Self>) {
        let exact = ExceptionCount {
            topology: self.topology.clone(),
            plus_one: self.next.clone(),
        };
        let tail = self.remaining.predecessor().map(|remaining| Self {
            topology: self.topology.clone(),
            next: self.next.successor(),
            remaining,
        });
        (exact, tail)
    }
}

impl PriorRegion for ExceptionCountTail {
    fn ln_prior_mass(&self) -> f64 {
        self.topology.ln_prior_mass() + self.remaining.ln()
            - self.next.ln()
            - self.topology.states.exception_normalizer().ln()
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use super::*;
    use crate::{anytime::log_add_exp, models::sparse_dfa::ln_exception_count_prior};

    fn states(n: u64) -> StateCount {
        StateCount::from(PositiveNat::from(NonZeroU64::new(n).unwrap()))
    }

    #[test]
    fn state_count_tail_preserves_mass_and_crosses_u64() {
        let mut tail = StateCountTail::root();
        for _ in 0..1000 {
            let parent = tail.ln_prior_mass();
            let (exact, next) = tail.split();
            assert!(
                (log_add_exp(exact.ln_prior_mass(), next.ln_prior_mass()) - parent).abs() < 1e-12
            );
            tail = next;
        }

        let max = PositiveNat::from(NonZeroU64::new(u64::MAX).unwrap());
        assert_eq!(max.successor().to_string(), "18446744073709551616");
    }

    #[test]
    fn topology_partition_preserves_state_mass() {
        for n in [1, 2, 8, 1024] {
            let state = states(n);
            let children = state
                .partition_topologies()
                .iter()
                .fold(f64::NEG_INFINITY, |sum, child| {
                    log_add_exp(sum, child.ln_prior_mass())
                });
            assert!((children - state.ln_prior_mass()).abs() < 1e-12);
        }
    }

    #[test]
    fn exception_tail_matches_existing_prior_and_preserves_mass() {
        for n in [1_u64, 2, 8] {
            let topology = states(n).partition_topologies()[0].clone();
            let topology_mass = topology.ln_prior_mass();
            let key_count = if n == 1 { 0 } else { n as usize * 256 };
            let mut tail = Some(topology.exception_counts());
            let mut k = 0;

            while let Some(current) = tail {
                let parent = current.ln_prior_mass();
                let (exact, next) = current.split();
                let children = next.as_ref().map_or(exact.ln_prior_mass(), |next| {
                    log_add_exp(exact.ln_prior_mass(), next.ln_prior_mass())
                });
                assert!((children - parent).abs() < 1e-12);
                assert!(
                    (exact.ln_prior_mass()
                        - topology_mass
                        - ln_exception_count_prior(key_count, k))
                    .abs()
                        < 1e-12
                );
                tail = next;
                k += 1;
            }

            assert_eq!(k, key_count + 1);
        }
    }
}
