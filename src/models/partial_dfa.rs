//! Exact Bayesian posterior over lazily instantiated finite-state transition tables.
//!
//! For a fixed maximum state count N, every labeled transition
//! (state, byte) -> destination has an independent uniform prior over N labels.
//! Unused labels are aggregated by first discovery. Optionally, KRAFT then takes
//! a stronger exact quotient: after each observation, the entire future-relevant
//! sufficient machine state is canonicalized under permutations of discovered
//! state identities. Histories that differ only by irrelevant state names then
//! share one posterior component.
//!
//! Each state predicts bytes with an integrated Dirichlet-1/2 categorical model.
//! Unvisited transition-table entries remain marginalized rather than materialized.

use std::{
    collections::{HashMap, hash_map::Entry},
    error::Error,
    fmt,
    mem::size_of,
};

use crate::{Distribution, Model};

const ALPHABET_SIZE: f64 = 256.0;
const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = ALPHABET_SIZE * JEFFREYS_ALPHA;
const MAX_PREDICTIVE_CANONICAL_STATES: u16 = 8;

/// Construction errors for an exact partial-DFA mixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartialDfaError {
    /// At least one state is required.
    ZeroStates,
    /// The compact edge representation supports at most 256 states.
    TooManyStates,
    /// Brute-force predictive canonicalization is intentionally limited.
    TooManyStatesForPredictiveQuotient,
}

impl fmt::Display for PartialDfaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStates => write!(f, "a DFA mixture needs at least one state"),
            Self::TooManyStates => write!(f, "partial DFA oracle supports at most 256 states"),
            Self::TooManyStatesForPredictiveQuotient => write!(
                f,
                "predictive DFA quotient currently supports at most {MAX_PREDICTIVE_CANONICAL_STATES} states"
            ),
        }
    }
}

impl Error for PartialDfaError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Edge {
    key: u16,
    destination: u16,
}

impl Edge {
    fn new(state: u16, byte: u8, destination: u16) -> Self {
        debug_assert!(state < 256);
        Self {
            key: (state << 8) | u16::from(byte),
            destination,
        }
    }

    fn key(state: u16, byte: u8) -> u16 {
        debug_assert!(state < 256);
        (state << 8) | u16::from(byte)
    }
}

/// Sparse sufficient statistics for one state's Dirichlet-1/2 byte predictor.
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct StateCounts {
    total: u32,
    counts: Vec<(u8, u32)>,
}

impl StateCounts {
    fn count(&self, byte: u8) -> u32 {
        self.counts
            .binary_search_by_key(&byte, |&(value, _)| value)
            .map_or(0, |index| self.counts[index].1)
    }

    fn ln_prob(&self, byte: u8) -> f64 {
        let numerator = f64::from(self.count(byte)) + JEFFREYS_ALPHA;
        let denominator = f64::from(self.total) + JEFFREYS_TOTAL;
        (numerator / denominator).ln()
    }

    fn observe(&mut self, byte: u8) {
        self.total = self
            .total
            .checked_add(1)
            .expect("partial DFA state observation count overflow");
        match self.counts.binary_search_by_key(&byte, |&(value, _)| value) {
            Ok(index) => {
                self.counts[index].1 = self.counts[index]
                    .1
                    .checked_add(1)
                    .expect("partial DFA byte count overflow");
            }
            Err(index) => self.counts.insert(index, (byte, 1)),
        }
    }
}

/// Canonical sufficient state of one aggregated family of labeled DFAs.
///
/// emissions.len() is the number of canonical states discovered so far.
/// All not-yet-discovered state labels are symmetric and remain aggregated.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
struct Component {
    current_state: u16,
    edges: Vec<Edge>,
    emissions: Vec<StateCounts>,
}

impl Component {
    fn initial() -> Self {
        Self {
            current_state: 0,
            edges: Vec::new(),
            emissions: vec![StateCounts::default()],
        }
    }

    fn discovered_states(&self) -> u16 {
        self.emissions.len() as u16
    }

    fn ln_prob(&self, byte: u8) -> f64 {
        self.emissions[usize::from(self.current_state)].ln_prob(byte)
    }

    fn observe_emission(&mut self, byte: u8) {
        self.emissions[usize::from(self.current_state)].observe(byte);
    }

    fn transition(&self, state: u16, byte: u8) -> Option<u16> {
        let key = Edge::key(state, byte);
        self.edges
            .binary_search_by_key(&key, |edge| edge.key)
            .ok()
            .map(|index| self.edges[index].destination)
    }

    fn assign_transition(&mut self, state: u16, byte: u8, destination: u16) {
        let edge = Edge::new(state, byte, destination);
        match self.edges.binary_search_by_key(&edge.key, |item| item.key) {
            Ok(_) => panic!("attempted to reassign an existing DFA transition"),
            Err(index) => self.edges.insert(index, edge),
        }
    }

    fn payload_bytes_estimate(&self) -> usize {
        let mut bytes = size_of::<Self>();
        bytes += self.edges.capacity() * size_of::<Edge>();
        bytes += self.emissions.capacity() * size_of::<StateCounts>();
        for state in &self.emissions {
            bytes += state.counts.capacity() * size_of::<(u8, u32)>();
        }
        bytes
    }

    /// Canonicalize the entire future-relevant sufficient state under arbitrary
    /// permutations of discovered state identities.
    ///
    /// The historical start-state label is not part of the sufficient state for
    /// future prediction. The current state is distinguished and always mapped to
    /// canonical state zero; all remaining discovered states are permuted and the
    /// lexicographically minimal representation is selected.
    fn canonicalized_predictive(&self) -> Self {
        let state_count = self.emissions.len();
        if state_count <= 1 {
            let mut canonical = self.clone();
            canonical.current_state = 0;
            return canonical;
        }

        let current = usize::from(self.current_state);
        let mut remaining: Vec<u16> = (0..state_count as u16)
            .filter(|&state| usize::from(state) != current)
            .collect();
        let mut best: Option<Self> = None;

        visit_permutations(&mut remaining, 0, &mut |order| {
            let mut mapping = vec![0_u16; state_count];
            mapping[current] = 0;
            for (index, &old_state) in order.iter().enumerate() {
                mapping[usize::from(old_state)] = (index + 1) as u16;
            }

            let mut emissions = vec![StateCounts::default(); state_count];
            for (old_state, counts) in self.emissions.iter().enumerate() {
                emissions[usize::from(mapping[old_state])] = counts.clone();
            }

            let mut edges = Vec::with_capacity(self.edges.len());
            for edge in &self.edges {
                let old_source = usize::from(edge.key >> 8);
                let byte = edge.key as u8;
                let old_destination = usize::from(edge.destination);
                edges.push(Edge::new(
                    mapping[old_source],
                    byte,
                    mapping[old_destination],
                ));
            }
            edges.sort_unstable();

            let candidate = Self {
                current_state: 0,
                edges,
                emissions,
            };
            match &best {
                None => best = Some(candidate),
                Some(existing) if candidate < *existing => best = Some(candidate),
                Some(_) => {}
            }
        });

        best.expect("at least one state permutation exists")
    }
}

fn visit_permutations(values: &mut [u16], start: usize, callback: &mut impl FnMut(&[u16])) {
    if start == values.len() {
        callback(values);
        return;
    }
    for index in start..values.len() {
        values.swap(start, index);
        visit_permutations(values, start + 1, callback);
        values.swap(start, index);
    }
}

/// Exact state-label quotient used by the partial-DFA oracle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfaQuotient {
    /// Only aggregate labels that have never been discovered.
    Discovery,
    /// Also quotient the complete predictive sufficient state by state renaming.
    Predictive,
}

/// Exact posterior diagnostics after an observed prefix.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PartialDfaDiagnostics {
    /// Number of distinct canonical sufficient states in the exact posterior.
    pub components: usize,
    /// Natural-log marginal probability of the observed prefix under this class.
    pub ln_evidence: f64,
    /// Shannon entropy of the exact posterior over canonical components, in nats.
    pub posterior_entropy_nats: f64,
    /// exp(posterior entropy), a useful effective posterior component count.
    pub effective_components: f64,
    /// Largest exact posterior mass of any one canonical component.
    pub top_component_mass: f64,
    /// Minimal number of highest-mass components needed for the requested KL bound.
    pub retained_components: usize,
    /// Exact posterior mass of that minimal retained set.
    pub retained_mass: f64,
    /// D_KL(Q || P) after conditioning the posterior on that retained set.
    pub retained_kl_nats: f64,
    /// Total explicitly assigned transitions across all canonical components.
    pub assigned_transitions: usize,
    /// Total nonzero state-byte counters across all canonical components.
    pub nonzero_emission_counts: usize,
    /// Approximate component payload bytes, excluding HashMap bucket overhead.
    pub payload_bytes_estimate: usize,
    /// Number of children generated by the most recent observation before merging.
    pub generated_children_last: usize,
    /// Children eliminated by exact sufficient-state merging on the last update.
    pub merged_children_last: usize,
}

/// Exact fixed-N Bayesian mixture over all labeled byte-input DFAs.
///
/// The complete transition table has N^(256N) labeled possibilities. KRAFT never
/// materializes that table. Instead, transition entries are instantiated only
/// when the observed stream first requires them. Canonical state discovery
/// aggregates the unused state labels exactly, including their prior multiplicity.
///
/// This is an oracle implementation: it keeps the full canonical posterior. Its
/// retention diagnostics answer how many components a top-posterior truncation
/// would need to keep for a requested forward KL bound, without actually pruning.
#[derive(Debug, Clone)]
pub struct ExactPartialDfaMixture {
    state_count: u16,
    quotient: DfaQuotient,
    components: HashMap<Component, f64>,
    generated_children_last: usize,
}

impl ExactPartialDfaMixture {
    /// Create the exact posterior for all labeled state_count-state DFAs.
    ///
    /// The start state is fixed to label zero. Every transition-table entry has
    /// an independent uniform prior over all state_count destination labels.
    /// All complete DFAs therefore have equal prior probability conditional on N.
    pub fn new(state_count: u16) -> Result<Self, PartialDfaError> {
        Self::with_quotient(state_count, DfaQuotient::Predictive)
    }

    /// Create the oracle with an explicit exact state-label quotient.
    pub fn with_quotient(state_count: u16, quotient: DfaQuotient) -> Result<Self, PartialDfaError> {
        if state_count == 0 {
            return Err(PartialDfaError::ZeroStates);
        }
        if state_count > 256 {
            return Err(PartialDfaError::TooManyStates);
        }
        if quotient == DfaQuotient::Predictive && state_count > MAX_PREDICTIVE_CANONICAL_STATES {
            return Err(PartialDfaError::TooManyStatesForPredictiveQuotient);
        }
        let mut components = HashMap::new();
        components.insert(Component::initial(), 0.0);
        Ok(Self {
            state_count,
            quotient,
            components,
            generated_children_last: 0,
        })
    }

    pub fn quotient(&self) -> DfaQuotient {
        self.quotient
    }

    pub fn state_count(&self) -> u16 {
        self.state_count
    }

    pub fn component_count(&self) -> usize {
        self.components.len()
    }

    /// Exact natural-log marginal likelihood of the observed prefix.
    pub fn ln_evidence(&self) -> f64 {
        log_sum_exp(self.components.values().copied())
    }

    /// Number of unmerged children that the next observation would create.
    ///
    /// This is an upper bound on the next exact component count because children
    /// with identical sufficient state are merged after branching.
    pub fn prospective_child_count(&self, byte: u8) -> usize {
        self.components
            .keys()
            .map(|component| {
                if component
                    .transition(component.current_state, byte)
                    .is_some()
                {
                    1
                } else {
                    usize::from(self.state_count)
                        .min(usize::from(component.discovered_states()) + 1)
                }
            })
            .sum()
    }

    /// Exact diagnostics and the smallest top-mass truncation satisfying epsilon.
    ///
    /// If retained posterior mass is r, conditioning on that retained set gives
    /// D_KL(Q || P) = -ln(r). Therefore epsilon requires r >= exp(-epsilon).
    pub fn diagnostics(&self, epsilon_nats: f64) -> PartialDfaDiagnostics {
        assert!(
            epsilon_nats.is_finite() && epsilon_nats >= 0.0,
            "epsilon must be a finite nonnegative KL bound"
        );

        let ln_evidence = self.ln_evidence();
        let mut weights: Vec<f64> = self
            .components
            .values()
            .map(|ln_mass| (*ln_mass - ln_evidence).exp())
            .collect();
        weights.sort_by(|a, b| b.total_cmp(a));

        let posterior_entropy_nats = weights
            .iter()
            .filter(|&&weight| weight > 0.0)
            .map(|&weight| -weight * weight.ln())
            .sum::<f64>();
        let effective_components = posterior_entropy_nats.exp();
        let top_component_mass = weights.first().copied().unwrap_or(0.0);

        let target_mass = (-epsilon_nats).exp();
        let mut retained_mass = 0.0;
        let mut retained_components = 0;
        for weight in &weights {
            if retained_mass >= target_mass {
                break;
            }
            retained_mass += *weight;
            retained_components += 1;
        }
        if retained_components == weights.len() && target_mass >= 1.0 {
            retained_mass = 1.0;
        }
        retained_mass = retained_mass.min(1.0);
        let retained_kl_nats = -retained_mass.ln();

        let assigned_transitions = self
            .components
            .keys()
            .map(|component| component.edges.len())
            .sum();
        let nonzero_emission_counts = self
            .components
            .keys()
            .flat_map(|component| &component.emissions)
            .map(|state| state.counts.len())
            .sum();
        let payload_bytes_estimate = self
            .components
            .keys()
            .map(Component::payload_bytes_estimate)
            .sum::<usize>()
            + self.components.len() * size_of::<f64>();

        PartialDfaDiagnostics {
            components: self.components.len(),
            ln_evidence,
            posterior_entropy_nats,
            effective_components,
            top_component_mass,
            retained_components,
            retained_mass,
            retained_kl_nats,
            assigned_transitions,
            nonzero_emission_counts,
            payload_bytes_estimate,
            generated_children_last: self.generated_children_last,
            merged_children_last: self
                .generated_children_last
                .saturating_sub(self.components.len()),
        }
    }

    fn observe_exact(&mut self, byte: u8) {
        let old_components = std::mem::take(&mut self.components);
        let mut next = HashMap::new();
        let ln_state_count = f64::from(self.state_count).ln();
        let mut generated_children = 0_usize;

        for (mut component, ln_mass) in old_components {
            let source = component.current_state;
            let ln_likelihood = component.ln_prob(byte);
            component.observe_emission(byte);
            let ln_base_mass = ln_mass + ln_likelihood;

            if let Some(destination) = component.transition(source, byte) {
                component.current_state = destination;
                generated_children += 1;
                self.insert_component(&mut next, component, ln_base_mass);
                continue;
            }

            let discovered = component.discovered_states();

            for destination in 0..discovered {
                let mut child = component.clone();
                child.assign_transition(source, byte, destination);
                child.current_state = destination;
                generated_children += 1;
                self.insert_component(&mut next, child, ln_base_mass - ln_state_count);
            }

            if discovered < self.state_count {
                let unused_labels = self.state_count - discovered;
                let mut child = component;
                let destination = discovered;
                child.assign_transition(source, byte, destination);
                child.current_state = destination;
                child.emissions.push(StateCounts::default());
                let ln_branch = f64::from(unused_labels).ln() - ln_state_count;
                generated_children += 1;
                self.insert_component(&mut next, child, ln_base_mass + ln_branch);
            }
        }

        self.generated_children_last = generated_children;
        self.components = next;
    }

    fn insert_component(
        &self,
        map: &mut HashMap<Component, f64>,
        component: Component,
        ln_mass: f64,
    ) {
        let component = match self.quotient {
            DfaQuotient::Discovery => component,
            DfaQuotient::Predictive => component.canonicalized_predictive(),
        };
        insert_log_mass(map, component, ln_mass);
    }
}

fn insert_log_mass(map: &mut HashMap<Component, f64>, component: Component, ln_mass: f64) {
    match map.entry(component) {
        Entry::Occupied(mut entry) => {
            *entry.get_mut() = log_add_exp(*entry.get(), ln_mass);
        }
        Entry::Vacant(entry) => {
            entry.insert(ln_mass);
        }
    }
}

struct ExactPartialDfaPrediction<'a> {
    mixture: &'a ExactPartialDfaMixture,
    ln_evidence: f64,
}

impl Distribution<u8> for ExactPartialDfaPrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        log_sum_exp(
            self.mixture
                .components
                .iter()
                .map(|(component, ln_mass)| *ln_mass + component.ln_prob(*byte)),
        ) - self.ln_evidence
    }
}

impl Model<u8> for ExactPartialDfaMixture {
    fn predict(&self) -> impl Distribution<u8> {
        ExactPartialDfaPrediction {
            mixture: self,
            ln_evidence: self.ln_evidence(),
        }
    }

    fn observe(&mut self, observation: u8) {
        self.observe_exact(observation);
    }
}

fn log_sum_exp(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(f64::NEG_INFINITY, log_add_exp)
}

fn log_add_exp(a: f64, b: f64) -> f64 {
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
    use crate::{baselines::Kt, evaluate};

    #[test]
    fn one_state_class_is_exactly_the_byte_kt_unigram() {
        let data = b"KRAFT posterior";
        let exact = evaluate(&data[..], &mut ExactPartialDfaMixture::new(1).unwrap()).unwrap();
        let kt = evaluate(&data[..], &mut Kt::default()).unwrap();
        assert!((exact.total_nats - kt.total_nats).abs() < 1e-12);
    }

    #[test]
    fn unused_labels_are_aggregated_with_exact_multiplicity() {
        let mut mixture = ExactPartialDfaMixture::new(3).unwrap();
        mixture.observe(b'A');

        assert_eq!(mixture.component_count(), 2);
        let ln_evidence = mixture.ln_evidence();
        let mut weights: Vec<_> = mixture
            .components
            .values()
            .map(|mass| (*mass - ln_evidence).exp())
            .collect();
        weights.sort_by(|a, b| a.total_cmp(b));
        assert!((weights[0] - 1.0 / 3.0).abs() < 1e-14);
        assert!((weights[1] - 2.0 / 3.0).abs() < 1e-14);
    }

    #[test]
    fn retention_profile_matches_the_exact_forward_kl_identity() {
        let mut mixture = ExactPartialDfaMixture::new(3).unwrap();
        mixture.observe(b'A');

        let epsilon = -(2.0_f64 / 3.0).ln();
        let diagnostics = mixture.diagnostics(epsilon + 1e-14);
        assert_eq!(diagnostics.retained_components, 1);
        assert!((diagnostics.retained_mass - 2.0 / 3.0).abs() < 1e-14);
        assert!((diagnostics.retained_kl_nats - epsilon).abs() < 1e-14);
    }

    #[test]
    fn predictions_remain_normalized_after_branching() {
        let mut mixture = ExactPartialDfaMixture::new(2).unwrap();
        for byte in b"ABACA" {
            let mass: f64 = (0..=255)
                .map(|candidate| mixture.predict().ln_prob(&candidate).exp())
                .sum();
            assert!((mass - 1.0).abs() < 1e-11);
            mixture.observe(*byte);
        }
    }

    #[test]
    fn prospective_child_count_is_a_safe_preallocation_bound() {
        let mut mixture = ExactPartialDfaMixture::new(2).unwrap();
        assert_eq!(mixture.prospective_child_count(b'A'), 2);
        mixture.observe(b'A');
        assert!(mixture.prospective_child_count(b'B') >= mixture.component_count());
    }

    #[test]
    fn predictive_canonicalization_forgets_irrelevant_state_names() {
        let left = Component {
            current_state: 0,
            edges: vec![Edge::new(0, b'x', 1)],
            emissions: vec![
                StateCounts {
                    total: 1,
                    counts: vec![(b'A', 1)],
                },
                StateCounts {
                    total: 1,
                    counts: vec![(b'B', 1)],
                },
            ],
        };
        let right = Component {
            current_state: 1,
            edges: vec![Edge::new(1, b'x', 0)],
            emissions: vec![
                StateCounts {
                    total: 1,
                    counts: vec![(b'B', 1)],
                },
                StateCounts {
                    total: 1,
                    counts: vec![(b'A', 1)],
                },
            ],
        };
        assert_ne!(left, right);
        assert_eq!(
            left.canonicalized_predictive(),
            right.canonicalized_predictive()
        );
    }

    #[test]
    fn predictive_and_discovery_quotients_have_identical_evidence() {
        let mut discovery =
            ExactPartialDfaMixture::with_quotient(2, DfaQuotient::Discovery).unwrap();
        let mut predictive =
            ExactPartialDfaMixture::with_quotient(2, DfaQuotient::Predictive).unwrap();

        for &byte in b"mediawiki" {
            let discovery_ln_prob = discovery.predict().ln_prob(&byte);
            let predictive_ln_prob = predictive.predict().ln_prob(&byte);
            assert!((discovery_ln_prob - predictive_ln_prob).abs() < 1e-12);
            discovery.observe(byte);
            predictive.observe(byte);
            assert!(
                predictive.component_count() <= discovery.component_count(),
                "predictive quotient cannot create more exact components"
            );
            assert!((discovery.ln_evidence() - predictive.ln_evidence()).abs() < 1e-12);
        }
    }
}
