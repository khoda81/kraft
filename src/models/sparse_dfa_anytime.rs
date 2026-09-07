//! Runnable anytime evidence bounds for the sparse-DFA Bayesian mixture.

use crate::{
    anytime::{FrontierNode, LogEvidenceBounds, PriorRegion, aggregate_evidence},
    models::{
        sparse_dfa::{DefaultTopology, SparseDfa, SparseOverride},
        sparse_dfa_region::{
            ExceptionCount, ExceptionCountTail, StateCount, StateCountTail, TopologyChoice,
        },
    },
};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Shape {
    states: u16,
    topology: DefaultTopology,
}

#[derive(Debug, Clone, PartialEq)]
struct KeySetRegion {
    shape: Shape,
    next: u32,
    remaining: u32,
    needed: u32,
    chosen: Vec<u32>,
    ln_prior_mass: f64,
}

impl KeySetRegion {
    fn new(count: &ExceptionCount) -> Option<Self> {
        let states = count.topology().states().to_u16()?;
        let needed = count.to_u32()?;
        let remaining = if states == 1 {
            0
        } else {
            u32::from(states) << 8
        };
        Some(Self {
            shape: Shape {
                states,
                topology: count.topology().topology(),
            },
            next: 0,
            remaining,
            needed,
            chosen: Vec::new(),
            ln_prior_mass: count.ln_prior_mass(),
        })
    }

    fn refine(mut self) -> Vec<SparseRegion> {
        if self.needed == 0 {
            return vec![destinations(self.shape, self.chosen, self.ln_prior_mass)];
        }
        if self.needed == self.remaining {
            self.chosen.extend(self.next..self.next + self.remaining);
            return vec![destinations(self.shape, self.chosen, self.ln_prior_mass)];
        }

        let ln_remaining = (self.remaining as f64).ln();
        let mut include = self.clone();
        include.chosen.push(self.next);
        include.next += 1;
        include.remaining -= 1;
        include.needed -= 1;
        include.ln_prior_mass += (self.needed as f64).ln() - ln_remaining;

        self.next += 1;
        self.remaining -= 1;
        self.ln_prior_mass += ((self.remaining + 1 - self.needed) as f64).ln() - ln_remaining;
        vec![SparseRegion::Keys(include), SparseRegion::Keys(self)]
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DestinationRegion {
    shape: Shape,
    keys: Vec<u32>,
    overrides: Vec<SparseOverride>,
    index: usize,
    start: u16,
    end: u16,
    ln_prior_mass: f64,
}

impl DestinationRegion {
    fn refine(mut self) -> Vec<SparseRegion> {
        let width = self.end - self.start;
        if width > 1 {
            let middle = self.start + width / 2;
            let mut left = self.clone();
            left.end = middle;
            left.ln_prior_mass += ((middle - self.start) as f64 / width as f64).ln();
            self.start = middle;
            self.ln_prior_mass += ((self.end - middle) as f64 / width as f64).ln();
            return vec![
                SparseRegion::Destinations(left),
                SparseRegion::Destinations(self),
            ];
        }

        let key = self.keys[self.index];
        let source = (key >> 8) as u16;
        let byte = key as u8;
        let default = self
            .shape
            .topology
            .default_destination(source, self.shape.states);
        let destination = if self.start < default {
            self.start
        } else {
            self.start + 1
        };
        self.overrides.push(SparseOverride {
            source,
            byte,
            destination,
        });
        self.index += 1;

        if self.index == self.keys.len() {
            vec![SparseRegion::Concrete(SparseDfa::from_valid_parts(
                self.shape.states,
                self.shape.topology,
                self.overrides,
            ))]
        } else {
            self.start = 0;
            self.end = self.shape.states - 1;
            vec![SparseRegion::Destinations(self)]
        }
    }
}

fn destinations(shape: Shape, keys: Vec<u32>, ln_prior_mass: f64) -> SparseRegion {
    if keys.is_empty() {
        SparseRegion::Concrete(SparseDfa::from_valid_parts(
            shape.states,
            shape.topology,
            Vec::new(),
        ))
    } else {
        SparseRegion::Destinations(DestinationRegion {
            shape,
            keys,
            overrides: Vec::new(),
            index: 0,
            start: 0,
            end: shape.states - 1,
            ln_prior_mass,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
enum SparseRegion {
    StatesTail(StateCountTail),
    States(StateCount),
    Topology(TopologyChoice),
    ExceptionTail(ExceptionCountTail),
    Opaque(ExceptionCount),
    Keys(KeySetRegion),
    Destinations(DestinationRegion),
    Concrete(SparseDfa),
}

impl PriorRegion for SparseRegion {
    fn ln_prior_mass(&self) -> f64 {
        match self {
            Self::StatesTail(region) => region.ln_prior_mass(),
            Self::States(region) => region.ln_prior_mass(),
            Self::Topology(region) => region.ln_prior_mass(),
            Self::ExceptionTail(region) => region.ln_prior_mass(),
            Self::Opaque(region) => region.ln_prior_mass(),
            Self::Keys(region) => region.ln_prior_mass,
            Self::Destinations(region) => region.ln_prior_mass,
            Self::Concrete(model) => model.ln_prior(),
        }
    }
}

impl SparseRegion {
    fn refinable(&self) -> bool {
        !matches!(self, Self::Concrete(_) | Self::Opaque(_))
    }

    fn refine(self) -> Vec<Self> {
        match self {
            Self::StatesTail(region) => {
                let (exact, tail) = region.split();
                vec![Self::States(exact), Self::StatesTail(tail)]
            }
            Self::States(region) => region
                .partition_topologies()
                .into_iter()
                .map(Self::Topology)
                .collect(),
            Self::Topology(region) => vec![Self::ExceptionTail(region.exception_counts())],
            Self::ExceptionTail(region) => {
                let (exact, tail) = region.split();
                let exact = KeySetRegion::new(&exact).map_or(Self::Opaque(exact), Self::Keys);
                let mut children = vec![exact];
                children.extend(tail.map(Self::ExceptionTail));
                children
            }
            Self::Keys(region) => region.refine(),
            Self::Destinations(region) => region.refine(),
            Self::Concrete(_) | Self::Opaque(_) => Vec::new(),
        }
    }
}

fn universal_ln_likelihood_upper(data: &[u8]) -> f64 {
    let mut counts = [0_u64; 256];
    data.iter()
        .map(|&byte| {
            let count = counts[usize::from(byte)];
            counts[usize::from(byte)] += 1;
            ((count as f64 + 0.5) / (count as f64 + 128.0)).ln()
        })
        .sum()
}

fn node(
    region: SparseRegion,
    data: &[u8],
    unresolved_ln_likelihood_upper: f64,
) -> FrontierNode<SparseRegion> {
    let evidence = match &region {
        SparseRegion::Concrete(model) => {
            LogEvidenceBounds::exact(model.ln_prior() + model.ln_evidence(data))
        }
        _ => LogEvidenceBounds::unresolved(
            region.ln_prior_mass() + unresolved_ln_likelihood_upper,
        ),
    };
    FrontierNode { region, evidence }
}

#[derive(Debug, Clone)]
pub struct SparseDfaAnytime {
    data: Vec<u8>,
    frontier: Vec<FrontierNode<SparseRegion>>,
    unresolved_ln_likelihood_upper: f64,
    steps: usize,
}

impl SparseDfaAnytime {
    pub fn new(data: &[u8]) -> Self {
        let unresolved_ln_likelihood_upper = universal_ln_likelihood_upper(data);
        Self {
            data: data.to_vec(),
            frontier: vec![node(
                SparseRegion::StatesTail(StateCountTail::root()),
                data,
                unresolved_ln_likelihood_upper,
            )],
            unresolved_ln_likelihood_upper,
            steps: 0,
        }
    }

    pub fn step(&mut self) -> bool {
        let Some(index) = self
            .frontier
            .iter()
            .enumerate()
            .filter(|(_, node)| node.region.refinable())
            .max_by(|(_, a), (_, b)| a.evidence.ln_upper().total_cmp(&b.evidence.ln_upper()))
            .map(|(index, _)| index)
        else {
            return false;
        };

        let parent = self.frontier.swap_remove(index);
        self.frontier.extend(
            parent
                .region
                .refine()
                .into_iter()
                .map(|region| {
                    node(
                        region,
                        &self.data,
                        self.unresolved_ln_likelihood_upper,
                    )
                }),
        );
        self.steps += 1;
        true
    }

    pub fn run(&mut self, steps: usize) {
        for _ in 0..steps {
            if !self.step() {
                break;
            }
        }
    }

    pub fn bounds(&self) -> LogEvidenceBounds {
        aggregate_evidence(self.frontier.iter().map(|node| &node.evidence))
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn regions(&self) -> usize {
        self.frontier.len()
    }

    pub fn resolved_models(&self) -> usize {
        self.frontier
            .iter()
            .filter(|node| node.evidence.is_exact())
            .count()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evidence_interval_tightens_monotonically() {
        let data = b"aba";
        let mut search = SparseDfaAnytime::new(data);
        let mut previous = search.bounds();
        for _ in 0..2000 {
            search.step();
            let current = search.bounds();
            assert!(current.ln_lower() + 1e-12 >= previous.ln_lower());
            assert!(current.ln_upper() <= previous.ln_upper() + 1e-12);
            previous = current;
        }
        assert!(search.resolved_models() > 0);
    }

    #[test]
    fn empty_sequence_keeps_unit_evidence_upper_bound() {
        let mut search = SparseDfaAnytime::new(&[]);
        for _ in 0..1000 {
            search.step();
            assert!(search.bounds().ln_upper().abs() < 1e-12);
        }
    }
}
