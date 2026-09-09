//! Runnable anytime evidence bounds for the sparse-DFA Bayesian mixture.

use std::{
    cmp::Ordering,
    collections::{BinaryHeap, HashMap},
};

use crate::{
    anytime::{FrontierNode, LogEvidenceBounds, PriorRegion, log_add_exp},
    models::{
        sparse_dfa::DefaultTopology,
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
            Err(_) if self.needed == self.remaining && self.shape.states == 2 => Some(1 - default),
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
            _ => None,
        }
    }

    fn ln_likelihood_bound(&self, data: &[u8], suffix_upper: &[f64]) -> (f64, bool, usize) {
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
                return (ln_likelihood, true, i + 1);
            }
            let Some(next) = self.forced_destination(state, byte) else {
                return (ln_likelihood + suffix_upper[i + 1], false, i + 1);
            };
            state = next;
        }

        (0.0, true, 0)
    }

    fn kind(&self) -> usize {
        match self {
            Self::StatesTail(_) => 0,
            Self::States(_) => 1,
            Self::Topology(_) => 2,
            Self::ExceptionTail(_) => 3,
            Self::Keys(_) => 4,
            Self::Destination(_) => 5,
            Self::Opaque(_) => 6,
        }
    }

    fn refinable(&self) -> bool {
        !matches!(self, Self::Opaque(_))
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
            Self::Opaque(_) => Vec::new(),
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

fn evaluated_node(
    region: SparseRegion,
    data: &[u8],
    suffix_upper: &[f64],
) -> (FrontierNode<SparseRegion>, usize) {
    let ln_prior = region.ln_prior_mass();
    let (ln_likelihood, exact, depth) = region.ln_likelihood_bound(data, suffix_upper);
    let evidence = if exact {
        LogEvidenceBounds::exact(ln_prior + ln_likelihood)
    } else {
        LogEvidenceBounds::unresolved(ln_prior + ln_likelihood)
    };
    (FrontierNode { region, evidence }, depth)
}

#[cfg(test)]
fn node(region: SparseRegion, data: &[u8], suffix_upper: &[f64]) -> FrontierNode<SparseRegion> {
    evaluated_node(region, data, suffix_upper).0
}

#[derive(Debug, Clone)]
struct WorkItem {
    node: FrontierNode<SparseRegion>,
    ln_priority: f64,
    order: usize,
    forced_prefix_bytes: usize,
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
        self.ln_priority
            .total_cmp(&other.ln_priority)
            .then_with(|| other.order.cmp(&self.order))
    }
}

/// One unresolved region category. Shares refer to upper bounds, not posterior mass.
#[derive(Debug, Clone)]
pub struct RegionDiagnostics {
    pub kind: &'static str,
    pub regions: usize,
    pub ln_upper: f64,
    pub upper_share: f64,
    pub mean_forced_prefix_bytes: f64,
    pub upper_weighted_forced_prefix_bytes: f64,
    pub refinements: usize,
}

#[derive(Debug, Clone)]
pub struct AnytimeDiagnostics {
    pub bounds: LogEvidenceBounds,
    pub ln_unresolved_upper: f64,
    /// Bytes evaluated in likelihood-bound scans, including repeated replay.
    /// Excludes transition-only scans in first_ambiguous and reporting work.
    pub bound_scan_bytes: u64,
    pub categories: Vec<RegionDiagnostics>,
}

const REGION_KINDS: [&str; 7] = [
    "states_tail",
    "states",
    "topology",
    "exception_tail",
    "keys",
    "destination",
    "opaque",
];

/// Scheduling changes work order only, never Bayesian region mass.
#[derive(Debug, Clone, Copy, Default)]
pub enum Schedule {
    #[default]
    UpperMass,
    /// Rank a tail by the upper mass of the next exact count it exposes.
    ExposedTailMass,
}

impl Schedule {
    fn priority(self, node: &FrontierNode<SparseRegion>) -> f64 {
        let adjustment = match (self, &node.region) {
            (Self::ExposedTailMass, SparseRegion::StatesTail(tail)) => {
                tail.split().0.ln_prior_mass() - tail.ln_prior_mass()
            }
            (Self::ExposedTailMass, SparseRegion::ExceptionTail(tail)) => {
                tail.split().0.ln_prior_mass() - tail.ln_prior_mass()
            }
            _ => 0.0,
        };
        node.evidence.ln_upper() + adjustment
    }
}

#[derive(Debug, Clone)]
pub struct SparseDfaAnytime {
    schedule: Schedule,
    data: Vec<u8>,
    suffix_upper: Vec<f64>,
    work: BinaryHeap<WorkItem>,
    opaque: Vec<FrontierNode<SparseRegion>>,
    resolved_ln_mass: f64,
    next_order: usize,
    steps: usize,
    resolved_regions: usize,
    refinements: [usize; 7],
    bound_scan_bytes: u64,
}

impl SparseDfaAnytime {
    pub fn new(data: &[u8]) -> Self {
        Self::with_schedule(data, Schedule::UpperMass)
    }

    pub fn with_schedule(data: &[u8], schedule: Schedule) -> Self {
        let suffix_upper = universal_suffix_upper(data);
        let (root, depth) = evaluated_node(
            SparseRegion::StatesTail(StateCountTail::root()),
            data,
            &suffix_upper,
        );
        let mut search = Self {
            schedule,
            data: data.to_vec(),
            suffix_upper,
            work: BinaryHeap::new(),
            opaque: Vec::new(),
            resolved_ln_mass: f64::NEG_INFINITY,
            next_order: 0,
            steps: 0,
            resolved_regions: 0,
            refinements: [0; 7],
            bound_scan_bytes: 0,
        };
        search.push(root, depth);
        search
    }

    fn push(&mut self, node: FrontierNode<SparseRegion>, depth: usize) {
        self.bound_scan_bytes += depth as u64;
        if node.evidence.is_exact() {
            self.resolved_ln_mass = log_add_exp(self.resolved_ln_mass, node.evidence.ln_lower());
            self.resolved_regions += 1;
        } else if node.region.refinable() {
            self.work.push(WorkItem {
                ln_priority: self.schedule.priority(&node),
                node,
                order: self.next_order,
                forced_prefix_bytes: depth,
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

        self.refinements[parent.node.region.kind()] += 1;
        for region in parent.node.region.refine(&self.data) {
            let (child, depth) = evaluated_node(region, &self.data, &self.suffix_upper);
            self.push(child, depth);
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
        let mut upper = self.resolved_ln_mass;
        for node in self.work.iter().map(|item| &item.node).chain(&self.opaque) {
            upper = log_add_exp(upper, node.evidence.ln_upper());
        }
        LogEvidenceBounds::new_internal(self.resolved_ln_mass, upper)
    }

    pub fn has_work(&self) -> bool {
        !self.work.is_empty()
    }

    /// Aggregate cached metadata without replaying any trajectories.
    pub fn diagnostics(&self) -> AnytimeDiagnostics {
        let mut categories: Vec<_> = REGION_KINDS
            .iter()
            .enumerate()
            .map(|(i, &kind)| RegionDiagnostics {
                kind,
                regions: 0,
                ln_upper: f64::NEG_INFINITY,
                upper_share: 0.0,
                mean_forced_prefix_bytes: 0.0,
                upper_weighted_forced_prefix_bytes: 0.0,
                refinements: self.refinements[i],
            })
            .collect();
        let mut total_upper = self.resolved_ln_mass;
        for (node, depth) in self
            .work
            .iter()
            .map(|item| (&item.node, item.forced_prefix_bytes))
            // Opaque regions always stop after the first emission.
            .chain(self.opaque.iter().map(|node| (node, 1)))
        {
            let category = &mut categories[node.region.kind()];
            let upper = node.evidence.ln_upper();
            total_upper = log_add_exp(total_upper, upper);
            let combined = log_add_exp(category.ln_upper, upper);
            category.upper_weighted_forced_prefix_bytes = (category.ln_upper - combined).exp()
                * category.upper_weighted_forced_prefix_bytes
                + (upper - combined).exp() * depth as f64;
            category.ln_upper = combined;
            category.regions += 1;
            category.mean_forced_prefix_bytes += depth as f64;
        }
        let unresolved = categories
            .iter()
            .fold(f64::NEG_INFINITY, |sum, c| log_add_exp(sum, c.ln_upper));
        for category in &mut categories {
            if category.regions != 0 {
                category.mean_forced_prefix_bytes /= category.regions as f64;
                category.upper_share = (category.ln_upper - unresolved).exp();
            }
        }
        AnytimeDiagnostics {
            bounds: LogEvidenceBounds::new_internal(self.resolved_ln_mass, total_upper),
            ln_unresolved_upper: unresolved,
            bound_scan_bytes: self.bound_scan_bytes,
            categories,
        }
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
    use crate::models::sparse_dfa::{SparseDfa, SparseOverride};

    #[test]
    fn diagnostics_account_for_frontier_and_work_without_changing_search() {
        let mut search = SparseDfaAnytime::new(b"abacaba");
        search.run(500);
        let before = search.bounds();
        let diagnostic = search.diagnostics();
        assert_eq!(before, diagnostic.bounds);
        assert_eq!(
            diagnostic
                .categories
                .iter()
                .map(|c| c.regions)
                .sum::<usize>(),
            search.regions()
        );
        assert_eq!(
            diagnostic
                .categories
                .iter()
                .map(|c| c.refinements)
                .sum::<usize>(),
            search.steps()
        );
        let shares: f64 = diagnostic.categories.iter().map(|c| c.upper_share).sum();
        assert!((shares - 1.0).abs() < 1e-12);
        assert!(diagnostic.bound_scan_bytes >= search.steps() as u64);
        for category in diagnostic.categories.iter().filter(|c| c.regions > 0) {
            assert!((1.0..=7.0).contains(&category.mean_forced_prefix_bytes));
            assert!(category.upper_weighted_forced_prefix_bytes >= 1.0 - 1e-12);
            assert!(category.upper_weighted_forced_prefix_bytes <= 7.0 + 1e-12);
        }
        assert_eq!(search.bounds(), before);
    }

    #[test]
    fn short_inputs_are_resolved_without_refinement() {
        for data in [&b""[..], &b"a"[..]] {
            let mut search = SparseDfaAnytime::new(data);
            assert!(!search.has_work());
            assert!(!search.step());
            assert_eq!(search.steps(), 0);
            let diagnostic = search.diagnostics();
            assert!(diagnostic.bounds.is_exact());
            assert_eq!(diagnostic.ln_unresolved_upper, f64::NEG_INFINITY);
            assert_eq!(diagnostic.bound_scan_bytes, data.len() as u64);
            assert!(diagnostic.categories.iter().all(|c| c.upper_share == 0.0));
        }
    }

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
            .filter(|key| {
                region
                    .assignments
                    .binary_search_by_key(key, |&(key, _)| key)
                    .is_err()
            })
            .fold(f64::NEG_INFINITY, |mass, key| {
                let source = (key >> 8) as u16;
                let byte = key as u8;
                let model = SparseDfa::new(
                    2,
                    DefaultTopology::Stay,
                    vec![SparseOverride {
                        source,
                        byte,
                        destination: 1 - source,
                    }],
                )
                .unwrap();
                log_add_exp(mass, model.ln_prior() + model.ln_evidence(data))
            })
    }

    fn default_child(region: KeySetRegion, data: &[u8]) -> KeySetRegion {
        region
            .refine(data)
            .into_iter()
            .find_map(|child| match child {
                SparseRegion::Keys(region)
                    if region
                        .assignments
                        .last()
                        .is_some_and(|&(key, destination)| {
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
        for schedule in [Schedule::UpperMass, Schedule::ExposedTailMass] {
            let mut search = SparseDfaAnytime::with_schedule(b"aba", schedule);
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
    }

    #[test]
    fn both_schedules_resolve_the_same_exhaustive_submixture() {
        let data = b"ababa";
        let region = n2_stay_k1_region();
        let expected = exact_k1_mass(&region, data);
        for schedule in [Schedule::UpperMass, Schedule::ExposedTailMass] {
            let mut search = SparseDfaAnytime::with_schedule(data, schedule);
            search.work.clear();
            let (root, depth) = evaluated_node(
                SparseRegion::Keys(region.clone()),
                data,
                &search.suffix_upper,
            );
            search.push(root, depth);
            search.run(1000);
            assert!(!search.has_work());
            let bounds = search.bounds();
            assert!(bounds.is_exact());
            assert!((bounds.ln_lower() - expected).abs() < 1e-12);
        }
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
