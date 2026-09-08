//! Runnable anytime evidence bounds for the sparse-DFA Bayesian mixture.

use std::{cmp::Ordering, collections::{BinaryHeap, HashMap}};

use crate::{
    anytime::{FrontierNode, LogEvidenceBounds, PriorRegion, log_add_exp},
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
    remaining: u32,
    needed: u32,
    assignments: Vec<(u32, u16)>,
    ln_prior_mass: f64,
}

impl KeySetRegion {
    fn new(count: &ExceptionCount) -> Option<Self> {
        let states = count.topology().states().to_u16()?;
        Some(Self {
            shape: Shape {
                states,
                topology: count.topology().topology(),
            },
            remaining: if states == 1 {
                0
            } else {
                u32::from(states) << 8
            },
            needed: count.to_u32()?,
            assignments: Vec::new(),
            ln_prior_mass: count.ln_prior_mass(),
        })
    }

    fn destination(&self, state: u16, byte: u8) -> Option<u16> {
        let key = (u32::from(state) << 8) | u32::from(byte);
        let default = self
            .shape
            .topology
            .default_destination(state, self.shape.states);
        match self.assignments.binary_search_by_key(&key, |&(key, _)| key) {
            Ok(index) => Some(self.assignments[index].1),
            Err(_) if self.needed == 0 => Some(default),
            Err(_) if self.needed == self.remaining && self.shape.states == 2 => {
                Some(1 - default)
            }
            Err(_) => None,
        }
    }

    fn first_ambiguous(&self, data: &[u8]) -> Option<(u16, u8)> {
        let mut state = 0_u16;
        for &byte in data.iter().take(data.len().saturating_sub(1)) {
            let Some(next) = self.destination(state, byte) else {
                return Some((state, byte));
            };
            state = next;
        }
        None
    }

    fn assign(&mut self, key: u32, destination: u16) {
        let index = self
            .assignments
            .binary_search_by_key(&key, |&(key, _)| key)
            .unwrap_err();
        self.assignments.insert(index, (key, destination));
    }

    fn refine(mut self, data: &[u8]) -> Vec<SparseRegion> {
        let Some((state, byte)) = self.first_ambiguous(data) else {
            return Vec::new();
        };
        let key = (u32::from(state) << 8) | u32::from(byte);
        let default = self
            .shape
            .topology
            .default_destination(state, self.shape.states);
        let remaining = self.remaining as f64;
        let needed = self.needed;

        let mut children = Vec::new();
        if needed < self.remaining {
            let mut default_branch = self.clone();
            default_branch.ln_prior_mass += ((self.remaining - needed) as f64 / remaining).ln();
            default_branch.remaining -= 1;
            default_branch.assign(key, default);
            children.push(SparseRegion::Keys(default_branch));
        }

        if needed > 0 {
            self.ln_prior_mass += (needed as f64 / remaining).ln();
            self.remaining -= 1;
            self.needed -= 1;
            if self.shape.states == 2 {
                self.assign(key, 1 - default);
                children.push(SparseRegion::Keys(self));
            } else {
                let end = self.shape.states - 1;
                children.push(SparseRegion::Destination(DestinationRegion {
                    keys: self,
                    key,
                    start: 0,
                    end,
                }));
            }
        }
        children
    }
}

#[derive(Debug, Clone, PartialEq)]
struct DestinationRegion {
    keys: KeySetRegion,
    key: u32,
    start: u16,
    end: u16,
}

impl DestinationRegion {
    fn ln_prior_mass(&self) -> f64 {
        self.keys.ln_prior_mass
            + ((self.end - self.start) as f64 / (self.keys.shape.states - 1) as f64).ln()
    }

    fn destination(&self, state: u16, byte: u8) -> Option<u16> {
        let key = (u32::from(state) << 8) | u32::from(byte);
        if key != self.key {
            return self.keys.destination(state, byte);
        }
        (self.end - self.start == 1).then(|| {
            let default = self
                .keys
                .shape
                .topology
                .default_destination(state, self.keys.shape.states);
            if self.start < default {
                self.start
            } else {
                self.start + 1
            }
        })
    }

    fn refine(mut self) -> Vec<SparseRegion> {
        if self.end - self.start > 1 {
            let middle = self.start + (self.end - self.start) / 2;
            let mut left = self.clone();
            left.end = middle;
            self.start = middle;
            return vec![
                SparseRegion::Destination(left),
                SparseRegion::Destination(self),
            ];
        }

        let source = (self.key >> 8) as u16;
        let byte = self.key as u8;
        let destination = self.destination(source, byte).unwrap();
        self.keys.ln_prior_mass = self.ln_prior_mass();
        self.keys.assign(self.key, destination);
        vec![SparseRegion::Keys(self.keys)]
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
    Destination(DestinationRegion),
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
            Self::Destination(region) => region.ln_prior_mass(),
            Self::Concrete(model) => model.ln_prior(),
        }
    }
}

impl SparseRegion {
    fn forced_destination(&self, state: u16, byte: u8) -> Option<u16> {
        match self {
            Self::States(region) if region.to_u16() == Some(1) => Some(0),
            Self::Topology(region) if region.states().to_u16() == Some(1) => Some(0),
            Self::Keys(region) => region.destination(state, byte),
            Self::Destination(region) => region.destination(state, byte),
            Self::Concrete(model) => Some(model.destination(state, byte)),
            _ => None,
        }
    }

    fn ln_likelihood_bound(&self, data: &[u8], suffix_upper: &[f64]) -> (f64, bool) {
        let mut totals = HashMap::<u16, u32>::new();
        let mut counts = HashMap::<(u16, u8), u32>::new();
        let mut state = 0_u16;
        let mut ln_likelihood = 0.0;

        for (i, &byte) in data.iter().enumerate() {
            let total = *totals.get(&state).unwrap_or(&0);
            let matching = *counts.get(&(state, byte)).unwrap_or(&0);
            ln_likelihood += ((matching as f64 + 0.5) / (total as f64 + 128.0)).ln();
            *totals.entry(state).or_default() += 1;
            *counts.entry((state, byte)).or_default() += 1;

            if i + 1 == data.len() {
                return (ln_likelihood, true);
            }
            let Some(next) = self.forced_destination(state, byte) else {
                return (ln_likelihood + suffix_upper[i + 1], false);
            };
            state = next;
        }

        (0.0, true)
    }

    fn refinable(&self) -> bool {
        !matches!(self, Self::Concrete(_) | Self::Opaque(_))
    }

    fn refine(self, data: &[u8]) -> Vec<Self> {
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
            Self::Keys(region) => region.refine(data),
            Self::Destination(region) => region.refine(),
            Self::Concrete(_) | Self::Opaque(_) => Vec::new(),
        }
    }
}

fn universal_suffix_upper(data: &[u8]) -> Vec<f64> {
    let mut counts = [0_u64; 256];
    let mut suffix = vec![0.0; data.len() + 1];
    for (i, &byte) in data.iter().enumerate() {
        let count = counts[usize::from(byte)];
        counts[usize::from(byte)] += 1;
        suffix[i] = ((count as f64 + 0.5) / (count as f64 + 128.0)).ln();
    }
    for i in (0..data.len()).rev() {
        suffix[i] += suffix[i + 1];
    }
    suffix
}

fn node(region: SparseRegion, data: &[u8], suffix_upper: &[f64]) -> FrontierNode<SparseRegion> {
    let ln_prior = region.ln_prior_mass();
    let evidence = match &region {
        SparseRegion::Concrete(model) => {
            LogEvidenceBounds::exact(model.ln_prior() + model.ln_evidence(data))
        }
        _ => {
            let (ln_likelihood, exact) = region.ln_likelihood_bound(data, suffix_upper);
            if exact {
                LogEvidenceBounds::exact(ln_prior + ln_likelihood)
            } else {
                LogEvidenceBounds::unresolved(ln_prior + ln_likelihood)
            }
        }
    };
    FrontierNode { region, evidence }
}

#[derive(Debug, Clone)]
struct WorkItem {
    node: FrontierNode<SparseRegion>,
    order: usize,
}

impl PartialEq for WorkItem {
    fn eq(&self, other: &Self) -> bool {
        self.order == other.order
    }
}

impl Eq for WorkItem {}

impl PartialOrd for WorkItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for WorkItem {
    fn cmp(&self, other: &Self) -> Ordering {
        self.node
            .evidence
            .ln_upper()
            .total_cmp(&other.node.evidence.ln_upper())
            .then_with(|| other.order.cmp(&self.order))
    }
}

#[derive(Debug, Clone)]
pub struct SparseDfaAnytime {
    data: Vec<u8>,
    suffix_upper: Vec<f64>,
    work: BinaryHeap<WorkItem>,
    opaque: Vec<FrontierNode<SparseRegion>>,
    resolved_ln_mass: f64,
    next_order: usize,
    steps: usize,
    resolved_regions: usize,
}

impl SparseDfaAnytime {
    pub fn new(data: &[u8]) -> Self {
        let suffix_upper = universal_suffix_upper(data);
        let root = node(
            SparseRegion::StatesTail(StateCountTail::root()),
            data,
            &suffix_upper,
        );
        let mut search = Self {
            data: data.to_vec(),
            suffix_upper,
            work: BinaryHeap::new(),
            opaque: Vec::new(),
            resolved_ln_mass: f64::NEG_INFINITY,
            next_order: 0,
            steps: 0,
            resolved_regions: 0,
        };
        search.push(root);
        search
    }

    fn push(&mut self, node: FrontierNode<SparseRegion>) {
        if node.evidence.is_exact() {
            self.resolved_ln_mass = log_add_exp(self.resolved_ln_mass, node.evidence.ln_lower());
            self.resolved_regions += 1;
        } else if node.region.refinable() {
            self.work.push(WorkItem {
                node,
                order: self.next_order,
            });
            self.next_order += 1;
        } else {
            self.opaque.push(node);
        }
    }

    pub fn step(&mut self) -> bool {
        let Some(parent) = self.work.pop() else {
            return false;
        };

        for region in parent.node.region.refine(&self.data) {
            self.push(node(region, &self.data, &self.suffix_upper));
        }
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
        let mut lower = self.resolved_ln_mass;
        let mut upper = self.resolved_ln_mass;
        for bound in self
            .work
            .iter()
            .map(|item| &item.node.evidence)
            .chain(self.opaque.iter().map(|node| &node.evidence))
        {
            lower = log_add_exp(lower, bound.ln_lower());
            upper = log_add_exp(upper, bound.ln_upper());
        }
        LogEvidenceBounds::new_internal(lower, upper)
    }

    pub fn steps(&self) -> usize {
        self.steps
    }

    pub fn regions(&self) -> usize {
        self.work.len() + self.opaque.len()
    }

    pub fn resolved_regions(&self) -> usize {
        self.resolved_regions
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn n2_stay_k1_region() -> KeySetRegion {
        let (_, tail) = StateCountTail::root().split();
        let (states, _) = tail.split();
        let topology = states.partition_topologies()[0].clone();
        let (_, tail) = topology.exception_counts().split();
        let (count, _) = tail.unwrap().split();
        KeySetRegion::new(&count).unwrap()
    }

    fn exact_k1_mass(region: &KeySetRegion, data: &[u8]) -> f64 {
        (0..512_u32)
            .filter(|key| region.assignments.binary_search_by_key(key, |&(key, _)| key).is_err())
            .fold(f64::NEG_INFINITY, |mass, key| {
                let source = (key >> 8) as u16;
                let byte = key as u8;
                let model = SparseDfa::from_valid_parts(
                    2,
                    DefaultTopology::Stay,
                    vec![SparseOverride {
                        source,
                        byte,
                        destination: 1 - source,
                    }],
                );
                log_add_exp(mass, model.ln_prior() + model.ln_evidence(data))
            })
    }

    fn default_child(region: KeySetRegion, data: &[u8]) -> KeySetRegion {
        region
            .refine(data)
            .into_iter()
            .find_map(|child| match child {
                SparseRegion::Keys(region)
                    if region.assignments.last().is_some_and(|&(key, destination)| {
                        let source = (key >> 8) as u16;
                        destination
                            == region
                                .shape
                                .topology
                                .default_destination(source, region.shape.states)
                    }) =>
                {
                    Some(region)
                }
                _ => None,
            })
            .unwrap()
    }

    #[test]
    fn data_directed_key_bound_contains_exhaustive_mass_and_resolves_irrelevant_tail() {
        let data = b"aba";
        let suffix = universal_suffix_upper(data);

        let region = n2_stay_k1_region();
        let exact_mass = exact_k1_mass(&region, data);
        let bound = node(SparseRegion::Keys(region.clone()), data, &suffix).evidence;
        assert!(!bound.is_exact());
        assert!(exact_mass <= bound.ln_upper() + 1e-12);

        let region = default_child(default_child(region, data), data);
        let exact_mass = exact_k1_mass(&region, data);
        let bound = node(SparseRegion::Keys(region), data, &suffix).evidence;
        assert!(bound.is_exact());
        assert!((bound.ln_lower() - exact_mass).abs() < 1e-12);
    }

    #[test]
    fn universal_suffix_bound_has_correct_offsets() {
        let data = b"abca";
        let suffix = universal_suffix_upper(data);
        let mut counts = [0_u64; 256];
        let mut terms = Vec::new();
        for &byte in data {
            let count = counts[usize::from(byte)];
            counts[usize::from(byte)] += 1;
            terms.push(((count as f64 + 0.5) / (count as f64 + 128.0)).ln());
        }
        for i in 0..=data.len() {
            let expected: f64 = terms[i..].iter().sum();
            assert!((suffix[i] - expected).abs() < 1e-12);
        }
    }

    #[test]
    fn universal_bound_dominates_fixed_dfas() {
        let data = b"<mediawi";
        let upper = universal_suffix_upper(data)[0];
        for topology in DefaultTopology::ALL {
            let model = SparseDfa::empty(4, topology).unwrap();
            assert!(model.ln_evidence(data) <= upper + 1e-12);
        }
    }

    #[test]
    fn evidence_interval_tightens_monotonically() {
        let mut search = SparseDfaAnytime::new(b"aba");
        let mut previous = search.bounds();
        for _ in 0..2000 {
            search.step();
            let current = search.bounds();
            assert!(current.ln_lower() + 1e-12 >= previous.ln_lower());
            assert!(current.ln_upper() <= previous.ln_upper() + 1e-12);
            previous = current;
        }
        assert!(search.resolved_regions() > 0);
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
