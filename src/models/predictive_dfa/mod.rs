//! Dynamic exact prediction groups for the fixed-N DFA prior.
//!
//! Transition alternatives, state, and state-local emission counts are symbolic
//! multi-way decision diagrams. Equal *whole next-byte distributions* share one
//! likelihood evaluation. Conditioning preserves the hidden alternatives, so
//! groups can split on the next prediction and merge again later. No component
//! list, complete-table enumeration, future-data lookahead, or pruning is used.
//!
//! This is the same labeled-table prior as `ExactPartialDfaMixture`, not a new
//! model family. Exact symbolic width can still grow exponentially. Limits stop
//! with an error instead of silently dropping probability mass.

mod diagram;

use crate::{Distribution, Model};
use diagram::{Diagram, Id, Value};
use std::{
    collections::{BinaryHeap, HashMap, hash_map::Entry},
    error::Error,
    fmt,
};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GroupedDfaError {
    InvalidStates,
    NodeLimit(usize),
    CountsLimit(usize),
    ObservationLimit,
}

impl fmt::Display for GroupedDfaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStates => write!(f, "grouped DFA requires 1..=256 states"),
            Self::NodeLimit(n) => write!(
                f,
                "symbolic node budget {n} exhausted; posterior was not pruned"
            ),
            Self::CountsLimit(n) => write!(
                f,
                "emission-statistics budget {n} exhausted; posterior was not pruned"
            ),
            Self::ObservationLimit => write!(f, "grouped DFA observation counter exhausted"),
        }
    }
}
impl Error for GroupedDfaError {}

#[derive(Clone, Debug, Default, Eq, PartialEq, Hash)]
struct Counts(Vec<(u8, u32)>);

impl Counts {
    fn incremented(&self, byte: u8) -> Self {
        let mut next = self.clone();
        match next.0.binary_search_by_key(&byte, |&(b, _)| b) {
            Ok(i) => next.0[i].1 += 1,
            Err(i) => next.0.insert(i, (byte, 1)),
        }
        next
    }
    fn prediction(&self) -> PredictionKey {
        let mut numerators = [1_u64; 256];
        for &(b, n) in &self.0 {
            numerators[usize::from(b)] += 2 * u64::from(n);
        }
        let divisor = numerators.iter().copied().reduce(gcd).unwrap();
        let baseline = numerators[0] / divisor;
        PredictionKey {
            baseline,
            denominator: numerators.iter().sum::<u64>() / divisor,
            exceptions: numerators
                .iter()
                .enumerate()
                .filter_map(|(b, &n)| {
                    let n = n / divisor;
                    (n != baseline).then_some((b as u8, n))
                })
                .collect(),
        }
    }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        (a, b) = (b, a % b);
    }
    a
}

/// Canonical rational vector, so equal predictions with different count totals
/// merge too (e.g. all-zero counts and equal positive counts for every byte).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
struct PredictionKey {
    baseline: u64,
    exceptions: Vec<(u8, u64)>,
    denominator: u64,
}
impl PredictionKey {
    fn ln_prob(&self, byte: u8) -> f64 {
        let numerator = self
            .exceptions
            .binary_search_by_key(&byte, |&(b, _)| b)
            .map_or(self.baseline, |i| self.exceptions[i].1);
        (numerator as f64 / self.denominator as f64).ln()
    }
}

#[derive(Clone, Debug)]
struct CountArena {
    values: Vec<Counts>,
    unique: HashMap<Counts, u32>,
    limit: usize,
}
impl CountArena {
    fn intern(&mut self, counts: Counts) -> Result<u32, GroupedDfaError> {
        if let Some(&id) = self.unique.get(&counts) {
            return Ok(id);
        }
        if self.values.len() >= self.limit {
            return Err(GroupedDfaError::CountsLimit(self.limit));
        }
        let id = u32::try_from(self.values.len())
            .map_err(|_| GroupedDfaError::CountsLimit(self.limit))?;
        self.values.push(counts.clone());
        self.unique.insert(counts, id);
        Ok(id)
    }
    fn rollback(&mut self, checkpoint: usize) {
        self.values.truncate(checkpoint);
        self.unique.retain(|_, id| (*id as usize) < checkpoint);
    }
}

#[derive(Clone, Debug)]
struct Group {
    prediction: PredictionKey,
    ln_mass: f64,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GroupedDfaDiagnostics {
    pub observations: u32,
    pub states: u16,
    pub prediction_groups: usize,
    /// Full-vector scalar likelihood evaluations on observed bytes, once per group.
    pub likelihood_evaluations: u64,
    /// Distinct current count vectors advanced over all symbolic alternatives.
    pub count_updates: u64,
    /// Diagram operation cache misses, including work in failed budget attempts.
    pub diagram_visits: u64,
    /// Distinct (prediction region, weight function) pairs visited during grouping.
    pub projection_visits: u64,
    pub allocated_nodes: usize,
    pub count_vectors: usize,
    pub garbage_collections: u64,
    pub ln_evidence: f64,
}

#[derive(Clone, Debug)]
pub struct GroupedDfa {
    manager: Diagram,
    counts: CountArena,
    state: Id,
    state_counts: Vec<Id>,
    active_counts: Id,
    log_weight: Id,
    prediction_root: Id,
    groups: Vec<Group>,
    ln_evidence: f64,
    observations: u32,
    transition_variables: [Option<u32>; 256],
    next_variable: u32,
    likelihood_evaluations: u64,
    count_updates: u64,
    projection_visits: u64,
    garbage_collections: u64,
}

impl GroupedDfa {
    /// Exact fixed-N prior. `max_nodes` also bounds interned count-vector entries;
    /// neither budget changes the model's prior or enables lossy pruning.
    pub fn new(states: u16, max_nodes: usize) -> Result<Self, GroupedDfaError> {
        if !(1..=256).contains(&states) {
            return Err(GroupedDfaError::InvalidStates);
        }
        let mut manager = Diagram::new(states, max_nodes);
        let zero = manager.leaf(Value::Index(0))?;
        let log_weight = manager.leaf(Value::log(0.0))?;
        let mut counts = CountArena {
            values: Vec::new(),
            unique: HashMap::new(),
            limit: max_nodes,
        };
        counts.intern(Counts::default())?;
        Ok(Self {
            manager,
            counts,
            state: zero,
            state_counts: vec![zero; usize::from(states)],
            active_counts: zero,
            log_weight,
            prediction_root: zero,
            groups: vec![Group {
                prediction: Counts::default().prediction(),
                ln_mass: 0.0,
            }],
            ln_evidence: 0.0,
            observations: 0,
            transition_variables: [None; 256],
            next_variable: 0,
            likelihood_evaluations: 0,
            count_updates: 0,
            projection_visits: 0,
            garbage_collections: 0,
        })
    }

    pub fn ln_evidence(&self) -> f64 {
        self.ln_evidence
    }

    pub fn diagnostics(&self) -> GroupedDfaDiagnostics {
        GroupedDfaDiagnostics {
            observations: self.observations,
            states: self.manager.arity,
            prediction_groups: self.groups.len(),
            likelihood_evaluations: self.likelihood_evaluations,
            count_updates: self.count_updates,
            diagram_visits: self.manager.visits,
            projection_visits: self.projection_visits,
            allocated_nodes: self.manager.nodes.len(),
            count_vectors: self.counts.values.len(),
            garbage_collections: self.garbage_collections,
            ln_evidence: self.ln_evidence,
        }
    }

    /// Fallible, transactional update for budget-aware callers. On failure the
    /// previous prediction/evidence remain usable; the byte was not consumed.
    pub fn try_observe(&mut self, byte: u8) -> Result<(), GroupedDfaError> {
        if self.observations == u32::MAX {
            return Err(GroupedDfaError::ObservationLimit);
        }
        if self.manager.nodes.len() > self.manager.limit / 2 {
            self.collect_garbage()?;
        }
        match self.observe_once(byte) {
            Err(GroupedDfaError::NodeLimit(_)) => {
                // A completed update can leave unreachable scratch nodes below
                // the usual collection threshold. Retry only if collection helps.
                let before = self.manager.nodes.len();
                self.collect_garbage()?;
                if self.manager.nodes.len() < before {
                    self.observe_once(byte)
                } else {
                    Err(GroupedDfaError::NodeLimit(self.manager.limit))
                }
            }
            result => result,
        }
    }

    fn observe_once(&mut self, byte: u8) -> Result<(), GroupedDfaError> {
        let node_checkpoint = self.manager.nodes.len();
        let count_checkpoint = self.counts.values.len();
        let visits_checkpoint = self.projection_visits;
        // Equal predictions share likelihood work even when their sufficient
        // count vectors or transition rules differ.
        let likelihoods: Vec<_> = self
            .groups
            .iter()
            .map(|g| g.prediction.ln_prob(byte))
            .collect();
        let mut count_updates = 0;
        let variable_base =
            self.transition_variables[usize::from(byte)].unwrap_or(self.next_variable);
        let result = (|| {
            let likelihood = self.manager.map(self.prediction_root, |v| {
                Ok(Value::log(likelihoods[v.index() as usize]))
            })?;
            let log_weight = self.manager.add_logs(self.log_weight, likelihood)?;
            let counts = &mut self.counts;
            let incremented = self.manager.map(self.active_counts, |v| {
                count_updates += 1;
                let updated = counts.values[v.index() as usize].incremented(byte);
                Ok(Value::Index(counts.intern(updated)?))
            })?;
            let mut state_counts = Vec::with_capacity(self.state_counts.len());
            for (state, &old_counts) in self.state_counts.iter().enumerate() {
                let condition = self.manager.map(self.state, |v| {
                    Ok(Value::Index(u32::from(v.index() as usize == state)))
                })?;
                state_counts.push(self.manager.select(condition, &[old_counts, incremented])?);
            }
            let destinations = (0..u32::from(self.manager.arity))
                .map(|state| self.manager.variable(variable_base + state))
                .collect::<Result<Vec<_>, _>>()?;
            let state = self.manager.select(self.state, &destinations)?;
            let active_counts = self.manager.select(state, &state_counts)?;
            let (prediction_root, groups) = self.group_predictions(active_counts, log_weight)?;
            let ln_evidence = log_sum(groups.iter().map(|g| g.ln_mass));
            Ok((
                state,
                state_counts,
                active_counts,
                log_weight,
                prediction_root,
                groups,
                ln_evidence,
            ))
        })();
        match result {
            Ok((
                state,
                state_counts,
                active_counts,
                log_weight,
                prediction_root,
                groups,
                ln_evidence,
            )) => {
                self.likelihood_evaluations += likelihoods.len() as u64;
                self.count_updates += count_updates;
                self.state = state;
                self.state_counts = state_counts;
                self.active_counts = active_counts;
                self.log_weight = log_weight;
                self.prediction_root = prediction_root;
                self.groups = groups;
                self.ln_evidence = ln_evidence;
                if self.transition_variables[usize::from(byte)].is_none() {
                    self.transition_variables[usize::from(byte)] = Some(variable_base);
                    self.next_variable += u32::from(self.manager.arity);
                }
                self.observations += 1;
                Ok(())
            }
            Err(error) => {
                self.manager.rollback(node_checkpoint);
                self.counts.rollback(count_checkpoint);
                // Failed grouping work is still charged to the diagnostics.
                debug_assert!(self.projection_visits >= visits_checkpoint);
                Err(error)
            }
        }
    }

    fn group_predictions(
        &mut self,
        active_counts: Id,
        log_weight: Id,
    ) -> Result<(Id, Vec<Group>), GroupedDfaError> {
        let mut keys = HashMap::new();
        let mut groups = Vec::new();
        let counts = &self.counts;
        let root = self.manager.map(active_counts, |value| {
            let key = counts.values[value.index() as usize].prediction();
            let next_id = groups.len() as u32;
            let id = *keys.entry(key.clone()).or_insert_with(|| {
                groups.push(Group {
                    prediction: key,
                    ln_mass: f64::NEG_INFINITY,
                });
                next_id
            });
            Ok(Value::Index(id))
        })?;

        // Forward mass propagation over pairs of DAG nodes. Every edge strictly
        // decreases the sum of node IDs, so all incoming mass arrives before a
        // pair is processed. Constant prediction regions integrate the remaining
        // weight DAG immediately instead of opening hidden alternatives.
        let mut pending = BinaryHeap::new();
        let mut reach = HashMap::new();
        let mut expectation_cache = HashMap::new();
        pending.push((u64::from(root) + u64::from(log_weight), root, log_weight));
        reach.insert((root, log_weight), 0.0);
        let ln_arity = f64::from(self.manager.arity).ln();
        while let Some((_, prediction, weight)) = pending.pop() {
            self.projection_visits += 1;
            let ln_reach = reach.remove(&(prediction, weight)).unwrap();
            if let Some(value) = self.manager.value(prediction) {
                let mass = ln_reach + self.manager.log_expectation(weight, &mut expectation_cache);
                let group = &mut groups[value.index() as usize];
                group.ln_mass = log_add(group.ln_mass, mass);
                continue;
            }
            let variable = self
                .manager
                .top(prediction)
                .into_iter()
                .chain(self.manager.top(weight))
                .min()
                .unwrap();
            for destination in 0..usize::from(self.manager.arity) {
                let p = self.manager.cofactor(prediction, variable, destination);
                let w = self.manager.cofactor(weight, variable, destination);
                match reach.entry((p, w)) {
                    Entry::Occupied(mut entry) => {
                        *entry.get_mut() = log_add(*entry.get(), ln_reach - ln_arity)
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(ln_reach - ln_arity);
                        pending.push((u64::from(p) + u64::from(w), p, w));
                    }
                }
            }
        }
        Ok((root, groups))
    }

    fn collect_garbage(&mut self) -> Result<(), GroupedDfaError> {
        let mut roots = vec![
            self.state,
            self.active_counts,
            self.log_weight,
            self.prediction_root,
        ];
        roots.extend_from_slice(&self.state_counts);
        let (manager, roots) = self.manager.compact(&roots)?;
        self.manager = manager;
        self.state = roots[0];
        self.active_counts = roots[1];
        self.log_weight = roots[2];
        self.prediction_root = roots[3];
        self.state_counts.copy_from_slice(&roots[4..]);
        self.garbage_collections += 1;
        Ok(())
    }
}

struct Prediction<'a>(&'a GroupedDfa);
impl Distribution<u8> for Prediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        log_sum(
            self.0
                .groups
                .iter()
                .map(|g| g.ln_mass + g.prediction.ln_prob(*byte)),
        ) - self.0.ln_evidence
    }
}
impl Model<u8> for GroupedDfa {
    fn predict(&self) -> impl Distribution<u8> {
        Prediction(self)
    }
    fn observe(&mut self, byte: u8) {
        self.try_observe(byte)
            .expect("grouped DFA budget exhausted; use try_observe for fallible updates");
    }
}

fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    a.max(b) + (-(a - b).abs()).exp().ln_1p()
}
fn log_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(f64::NEG_INFINITY, log_add)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{baselines::Kt, models::partial_dfa::ExactPartialDfaMixture};

    fn probabilities(model: &impl Model<u8>) -> Vec<f64> {
        (0..=255)
            .map(|b| model.predict().ln_prob(&b).exp())
            .collect()
    }

    fn compare(states: u16, data: &[u8]) {
        let mut grouped = GroupedDfa::new(states, 500_000).unwrap();
        let mut oracle = ExactPartialDfaMixture::new(states).unwrap();
        let mut cost = 0.0;
        for &byte in data {
            let expected = probabilities(&oracle);
            let actual = probabilities(&grouped);
            assert!((actual.iter().sum::<f64>() - 1.0).abs() < 1e-11);
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 1e-11);
            }
            cost -= grouped.predict().ln_prob(&byte);
            grouped
                .try_observe(byte)
                .unwrap_or_else(|error| panic!("states={states} data={data:?}: {error}"));
            oracle.observe(byte);
            assert!((grouped.ln_evidence() - oracle.ln_evidence()).abs() < 1e-10);
            assert!((cost + grouped.ln_evidence()).abs() < 1e-10);
        }
        for (a, b) in probabilities(&grouped).iter().zip(probabilities(&oracle)) {
            assert!((a - b).abs() < 1e-11);
        }
    }

    #[test]
    fn every_binary_prefix_matches_the_independent_leaf_oracle() {
        for states in 1..=3 {
            for bits in 0..32 {
                let data: Vec<_> = (0..5).map(|shift| b'a' + ((bits >> shift) & 1)).collect();
                compare(states, &data);
            }
        }
    }

    #[test]
    fn recurrence_unseen_bytes_and_nonbinary_destinations_match() {
        for states in [2, 3, 4] {
            for data in [
                b"abababa".as_slice(),
                b"abbabba",
                b"abcdef",
                &[0, 255, 0, 128, 255, 1],
            ] {
                compare(states, data);
            }
        }
    }

    #[test]
    fn one_state_is_kt_including_all_byte_values() {
        let mut grouped = GroupedDfa::new(1, 100_000).unwrap();
        let mut kt = Kt::default();
        for b in (0..=255).chain(0..=255) {
            assert!((grouped.predict().ln_prob(&b) - kt.predict().ln_prob(&b)).abs() < 1e-12);
            grouped.try_observe(b).unwrap();
            kt.observe(b);
        }
        assert_eq!(grouped.diagnostics().prediction_groups, 1);
        assert_eq!(grouped.diagnostics().likelihood_evaluations, 512);
    }

    #[test]
    fn equal_predictions_merge_despite_different_counts_then_split() {
        let mut model = GroupedDfa::new(2, 10_000).unwrap();
        let empty = model.counts.intern(Counts::default()).unwrap();
        let balanced = model
            .counts
            .intern(Counts((0..=255).map(|b| (b, 1)).collect()))
            .unwrap();
        assert_eq!(
            model.counts.values[empty as usize].prediction(),
            model.counts.values[balanced as usize].prediction()
        );
        let selector = model.manager.variable(0).unwrap();
        let a = model.manager.leaf(Value::Index(empty)).unwrap();
        let b = model.manager.leaf(Value::Index(balanced)).unwrap();
        let active = model.manager.select(selector, &[a, b]).unwrap();
        let (_, groups) = model.group_predictions(active, model.log_weight).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(groups[0].ln_mass.abs() < 1e-14);
        let ca = model
            .counts
            .intern(Counts::default().incremented(b'a'))
            .unwrap();
        let cb = model
            .counts
            .intern(model.counts.values[balanced as usize].incremented(b'a'))
            .unwrap();
        let a = model.manager.leaf(Value::Index(ca)).unwrap();
        let b = model.manager.leaf(Value::Index(cb)).unwrap();
        let active = model.manager.select(selector, &[a, b]).unwrap();
        let (_, groups) = model.group_predictions(active, model.log_weight).unwrap();
        assert_eq!(groups.len(), 2);
        for g in groups {
            assert!((g.ln_mass.exp() - 0.5).abs() < 1e-14);
        }
        // Complete one observation of every byte. Both alternatives become
        // uniform again, although their counts still differ by one everywhere.
        let mut active = active;
        for byte in (0..=255).filter(|&byte| byte != b'a') {
            let counts = &mut model.counts;
            active = model
                .manager
                .map(active, |v| {
                    let updated = counts.values[v.index() as usize].incremented(byte);
                    Ok(Value::Index(counts.intern(updated)?))
                })
                .unwrap();
        }
        let (prediction, groups) = model.group_predictions(active, model.log_weight).unwrap();
        assert_eq!(groups.len(), 1);
        assert!(model.manager.value(prediction).is_some());
        assert!(model.manager.value(active).is_none());
        assert!(model.manager.value(selector).is_none());
    }

    #[test]
    fn prediction_groups_preserve_posterior_mass_and_save_likelihood_work() {
        let mut model = GroupedDfa::new(3, 500_000).unwrap();
        let mut oracle = ExactPartialDfaMixture::new(3).unwrap();
        let mut group_work = 0;
        let mut component_work = 0;
        for &b in b"abacaba" {
            group_work += model.groups.len() as u64;
            component_work += oracle.component_count() as u64;
            model.try_observe(b).unwrap();
            oracle.observe(b);
            assert!(
                (log_sum(model.groups.iter().map(|g| g.ln_mass)) - oracle.ln_evidence()).abs()
                    < 1e-10
            );
        }
        assert_eq!(model.diagnostics().likelihood_evaluations, group_work);
        assert!(group_work < component_work);
    }

    #[test]
    fn collection_and_chunking_preserve_predictions_and_updates() {
        let mut whole = GroupedDfa::new(2, 500_000).unwrap();
        let mut chunks = whole.clone();
        for &b in b"abacaba" {
            whole.try_observe(b).unwrap();
        }
        for chunk in [b"ab".as_slice(), b"aca", b"ba"] {
            for &b in chunk {
                chunks.try_observe(b).unwrap();
            }
            let before = probabilities(&chunks);
            chunks.collect_garbage().unwrap();
            assert_eq!(before, probabilities(&chunks));
        }
        assert_eq!(probabilities(&whole), probabilities(&chunks));
        whole.try_observe(255).unwrap();
        chunks.try_observe(255).unwrap();
        assert_eq!(probabilities(&whole), probabilities(&chunks));
    }

    #[test]
    fn budget_failure_leaves_the_previous_posterior_intact_and_retryable() {
        let mut model = GroupedDfa::new(2, 3).unwrap();
        let before = probabilities(&model);
        assert!(matches!(
            model.try_observe(b'a'),
            Err(GroupedDfaError::NodeLimit(3))
        ));
        assert_eq!(model.observations, 0);
        assert_eq!(model.ln_evidence(), 0.0);
        assert_eq!(probabilities(&model), before);
        model.manager.limit = 100_000;
        model.counts.limit = 100_000;
        model.try_observe(b'a').unwrap();
        let mut reference = GroupedDfa::new(2, 100_000).unwrap();
        reference.try_observe(b'a').unwrap();
        assert_eq!(probabilities(&model), probabilities(&reference));
    }

    #[test]
    fn invalid_construction_is_reported() {
        assert!(matches!(
            GroupedDfa::new(0, 100),
            Err(GroupedDfaError::InvalidStates)
        ));
        assert!(matches!(
            GroupedDfa::new(257, 100),
            Err(GroupedDfaError::InvalidStates)
        ));
        assert!(matches!(
            GroupedDfa::new(2, 0),
            Err(GroupedDfaError::NodeLimit(0))
        ));
    }
}
