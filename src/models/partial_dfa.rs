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
//! Assigned transitions and emission observations live in persistent parent-linked
//! arenas, so posterior branches share their complete physical history.

use std::{
    cmp::{Ordering, Reverse},
    collections::{BinaryHeap, HashMap, hash_map::Entry},
    error::Error,
    fmt,
    hash::{DefaultHasher, Hash, Hasher},
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

type TransitionNodeId = u32;
const NO_TRANSITION_NODE: TransitionNodeId = TransitionNodeId::MAX;

/// One immutable transition assignment in an append-only persistent arena.
///
/// A component stores only the newest node ID. Branches append one node and
/// retain the parent's ID, so their complete historical transition maps share
/// storage. `parent` uses a sentinel instead of `Option<u32>` to keep each node
/// at eight bytes.
#[derive(Debug, Clone, Copy)]
struct TransitionNode {
    parent: TransitionNodeId,
    edge: Edge,
}

#[derive(Debug, Clone, Default)]
struct TransitionArena {
    nodes: Vec<TransitionNode>,
}

impl TransitionArena {
    fn append(&mut self, parent: TransitionNodeId, edge: Edge) -> TransitionNodeId {
        let id = TransitionNodeId::try_from(self.nodes.len())
            .expect("partial DFA transition arena exhausted u32 node IDs");
        assert_ne!(
            id, NO_TRANSITION_NODE,
            "reserved transition node ID reached"
        );
        self.nodes.push(TransitionNode { parent, edge });
        id
    }

    fn iter(&self, head: TransitionNodeId) -> TransitionIter<'_> {
        TransitionIter {
            arena: self,
            next: head,
        }
    }

    fn payload_bytes_estimate(&self) -> usize {
        self.nodes.capacity() * size_of::<TransitionNode>()
    }
}

struct TransitionIter<'a> {
    arena: &'a TransitionArena,
    next: TransitionNodeId,
}

impl Iterator for TransitionIter<'_> {
    type Item = Edge;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == NO_TRANSITION_NODE {
            return None;
        }
        let node = self.arena.nodes[usize::try_from(self.next).expect("u32 fits usize")];
        self.next = node.parent;
        Some(node.edge)
    }
}

fn edge_semantic_hash(edge: Edge) -> u64 {
    let mut hasher = DefaultHasher::new();
    edge.hash(&mut hasher);
    hasher.finish()
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

#[derive(Debug, Clone, Copy)]
struct TransitionHistory {
    head: TransitionNodeId,
    len: u32,
    semantic_hash: u64,
}

type EmissionNodeId = u32;
const NO_EMISSION_NODE: EmissionNodeId = EmissionNodeId::MAX;

#[derive(Debug, Clone, Copy)]
struct EmissionNode {
    parent: EmissionNodeId,
    storage_state: u16,
    byte: u8,
}

#[derive(Debug, Clone, Default)]
struct EmissionArena {
    nodes: Vec<EmissionNode>,
}

impl EmissionArena {
    fn append(&mut self, parent: EmissionNodeId, storage_state: u16, byte: u8) -> EmissionNodeId {
        let id = EmissionNodeId::try_from(self.nodes.len())
            .expect("partial DFA emission arena exhausted u32 node IDs");
        assert_ne!(id, NO_EMISSION_NODE, "reserved emission node ID reached");
        self.nodes.push(EmissionNode {
            parent,
            storage_state,
            byte,
        });
        id
    }

    fn iter(&self, head: EmissionNodeId) -> EmissionIter<'_> {
        EmissionIter {
            arena: self,
            next: head,
        }
    }

    fn payload_bytes_estimate(&self) -> usize {
        self.nodes.capacity() * size_of::<EmissionNode>()
    }
}

struct EmissionIter<'a> {
    arena: &'a EmissionArena,
    next: EmissionNodeId,
}

impl Iterator for EmissionIter<'_> {
    type Item = EmissionNode;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next == NO_EMISSION_NODE {
            return None;
        }
        let node = self.arena.nodes[usize::try_from(self.next).expect("u32 fits usize")];
        self.next = node.parent;
        Some(node)
    }
}

fn emission_semantic_hash(state: u16, byte: u8) -> u64 {
    let mut hasher = DefaultHasher::new();
    (state, byte).hash(&mut hasher);
    hasher.finish()
}

#[derive(Debug, Clone, Copy)]
struct EmissionHistory {
    head: EmissionNodeId,
    len: u32,
    semantic_hash: u64,
}

impl Default for EmissionHistory {
    fn default() -> Self {
        Self {
            head: NO_EMISSION_NODE,
            len: 0,
            semantic_hash: 0,
        }
    }
}

/// Logical-state order mapped to stable arena labels.
///
/// The common oracle case (at most eight states) fits inline in one `u64`.
/// Larger discovery-only mixtures fall back to a boxed slice.
#[derive(Debug, Clone)]
enum StateLabels {
    Packed { labels: u64, len: u8 },
    Large(Box<[u16]>),
}

impl StateLabels {
    fn initial() -> Self {
        Self::Packed { labels: 0, len: 1 }
    }

    fn len(&self) -> usize {
        match self {
            Self::Packed { len, .. } => usize::from(*len),
            Self::Large(labels) => labels.len(),
        }
    }

    fn get(&self, logical_state: u16) -> u16 {
        let index = usize::from(logical_state);
        match self {
            Self::Packed { labels, len } => {
                assert!(index < usize::from(*len));
                ((labels >> (index * 8)) & 0xff) as u16
            }
            Self::Large(labels) => labels[index],
        }
    }

    fn push(&mut self, storage_label: u16) {
        match self {
            Self::Packed { labels, len } if *len < 8 => {
                debug_assert!(storage_label <= u16::from(u8::MAX));
                *labels |= u64::from(storage_label) << (usize::from(*len) * 8);
                *len += 1;
            }
            Self::Packed { labels, len } => {
                let mut expanded: Vec<u16> = (0..usize::from(*len))
                    .map(|index| ((*labels >> (index * 8)) & 0xff) as u16)
                    .collect();
                expanded.push(storage_label);
                *self = Self::Large(expanded.into_boxed_slice());
            }
            Self::Large(labels) => {
                let mut expanded = labels.to_vec();
                expanded.push(storage_label);
                *labels = expanded.into_boxed_slice();
            }
        }
    }

    fn reordered(&self, old_to_new: &[u16]) -> Self {
        let mut reordered = vec![0_u16; self.len()];
        for (old_state, &new_state) in old_to_new.iter().enumerate() {
            reordered[usize::from(new_state)] = self.get(old_state as u16);
        }
        if reordered.len() <= 8 {
            let labels = reordered
                .iter()
                .enumerate()
                .fold(0_u64, |packed, (index, &label)| {
                    packed | (u64::from(label) << (index * 8))
                });
            Self::Packed {
                labels,
                len: reordered.len() as u8,
            }
        } else {
            Self::Large(reordered.into_boxed_slice())
        }
    }

    fn payload_bytes_estimate(&self) -> usize {
        match self {
            Self::Packed { .. } => 0,
            Self::Large(labels) => std::mem::size_of_val(&**labels),
        }
    }
}

impl Default for TransitionHistory {
    fn default() -> Self {
        Self {
            head: NO_TRANSITION_NODE,
            len: 0,
            semantic_hash: 0,
        }
    }
}

impl StateCounts {
    #[cfg(test)]
    fn count(&self, byte: u8) -> u32 {
        self.counts
            .binary_search_by_key(&byte, |&(value, _)| value)
            .map_or(0, |index| self.counts[index].1)
    }

    #[cfg(test)]
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
/// state_labels.len() is the number of canonical states discovered so far.
/// All not-yet-discovered state labels are symmetric and remain aggregated.
#[derive(Debug, Clone)]
struct Component {
    current_state: u16,
    transitions: TransitionHistory,
    emissions: EmissionHistory,
    state_labels: StateLabels,
}

impl Component {
    fn initial() -> Self {
        Self {
            current_state: 0,
            transitions: TransitionHistory::default(),
            emissions: EmissionHistory::default(),
            state_labels: StateLabels::initial(),
        }
    }

    fn discovered_states(&self) -> u16 {
        self.state_labels.len() as u16
    }

    fn ln_prob(&self, arena: &EmissionArena, byte: u8) -> f64 {
        let storage_state = self.state_labels.get(self.current_state);
        let mut total = 0_u32;
        let mut count = 0_u32;
        for update in arena.iter(self.emissions.head) {
            if update.storage_state == storage_state {
                total += 1;
                if update.byte == byte {
                    count += 1;
                }
            }
        }
        let numerator = f64::from(count) + JEFFREYS_ALPHA;
        let denominator = f64::from(total) + JEFFREYS_TOTAL;
        (numerator / denominator).ln()
    }

    fn observe_emission(&mut self, arena: &mut EmissionArena, byte: u8) {
        let storage_state = self.state_labels.get(self.current_state);
        self.emissions.head = arena.append(self.emissions.head, storage_state, byte);
        self.emissions.len = self
            .emissions
            .len
            .checked_add(1)
            .expect("partial DFA emission history length overflow");
        self.emissions.semantic_hash = self
            .emissions
            .semantic_hash
            .wrapping_add(emission_semantic_hash(self.current_state, byte));
    }

    fn logical_state_for_storage_label(&self, storage_label: u16) -> u16 {
        (0..self.discovered_states())
            .position(|state| self.state_labels.get(state) == storage_label)
            .and_then(|state| u16::try_from(state).ok())
            .expect("transition references a discovered storage label")
    }

    fn transition(&self, arena: &TransitionArena, state: u16, byte: u8) -> Option<u16> {
        let storage_source = self.state_labels.get(state);
        let key = Edge::key(storage_source, byte);
        arena
            .iter(self.transitions.head)
            .find(|edge| edge.key == key)
            .map(|edge| self.logical_state_for_storage_label(edge.destination))
    }

    fn assign_transition(
        &mut self,
        arena: &mut TransitionArena,
        state: u16,
        byte: u8,
        destination: u16,
    ) {
        debug_assert!(self.transition(arena, state, byte).is_none());
        let semantic_edge = Edge::new(state, byte, destination);
        let storage_edge = Edge::new(
            self.state_labels.get(state),
            byte,
            self.state_labels.get(destination),
        );
        self.transitions.head = arena.append(self.transitions.head, storage_edge);
        self.transitions.len = self
            .transitions
            .len
            .checked_add(1)
            .expect("partial DFA transition history length overflow");
        self.transitions.semantic_hash ^= edge_semantic_hash(semantic_edge);
    }

    fn payload_bytes_estimate(&self) -> usize {
        size_of::<Self>() + self.state_labels.payload_bytes_estimate()
    }

    fn semantic_fingerprint(&self) -> u64 {
        let mut hasher = DefaultHasher::new();
        self.current_state.hash(&mut hasher);
        self.transitions.len.hash(&mut hasher);
        self.transitions.semantic_hash.hash(&mut hasher);
        self.emissions.len.hash(&mut hasher);
        self.emissions.semantic_hash.hash(&mut hasher);
        self.discovered_states().hash(&mut hasher);
        hasher.finish()
    }

    fn emission_counts(&self, arena: &EmissionArena) -> Vec<StateCounts> {
        let mut counts = vec![StateCounts::default(); usize::from(self.discovered_states())];
        for update in arena.iter(self.emissions.head) {
            let logical_state = self.logical_state_for_storage_label(update.storage_state);
            counts[usize::from(logical_state)].observe(update.byte);
        }
        counts
    }

    fn nonzero_emission_count(&self, arena: &EmissionArena) -> usize {
        self.emission_counts(arena)
            .iter()
            .map(|state| state.counts.len())
            .sum()
    }

    fn semantically_eq(
        &self,
        other: &Self,
        transition_arena: &TransitionArena,
        emission_arena: &EmissionArena,
    ) -> bool {
        if self.current_state != other.current_state
            || self.transitions.len != other.transitions.len
            || self.transitions.semantic_hash != other.transitions.semantic_hash
            || self.emissions.len != other.emissions.len
            || self.emissions.semantic_hash != other.emissions.semantic_hash
            || self.discovered_states() != other.discovered_states()
            || self.emission_counts(emission_arena) != other.emission_counts(emission_arena)
        {
            return false;
        }

        transition_arena.iter(self.transitions.head).all(|edge| {
            let source = self.logical_state_for_storage_label(edge.key >> 8);
            let byte = edge.key as u8;
            let destination = self.logical_state_for_storage_label(edge.destination);
            other.transition(transition_arena, source, byte) == Some(destination)
        })
    }

    fn logical_edges(&self, arena: &TransitionArena) -> Vec<Edge> {
        let mut edges: Vec<_> = arena
            .iter(self.transitions.head)
            .map(|edge| {
                Edge::new(
                    self.logical_state_for_storage_label(edge.key >> 8),
                    edge.key as u8,
                    self.logical_state_for_storage_label(edge.destination),
                )
            })
            .collect();
        edges.sort_unstable();
        edges
    }

    /// Canonicalize the entire future-relevant sufficient state under arbitrary
    /// permutations of discovered state identities.
    ///
    /// The historical start-state label is not part of the sufficient state for
    /// future prediction. The current state is distinguished and always mapped to
    /// canonical state zero; all remaining discovered states are permuted and the
    /// lexicographically minimal representation is selected.
    fn canonicalize_predictive(
        &mut self,
        transition_arena: &TransitionArena,
        emission_arena: &EmissionArena,
    ) {
        let state_count = self.state_labels.len();
        if state_count <= 1 {
            self.current_state = 0;
            return;
        }

        let current = usize::from(self.current_state);
        let logical_edges = self.logical_edges(transition_arena);
        let logical_emissions = self.emission_counts(emission_arena);
        let mut remaining: Vec<u16> = (0..state_count as u16)
            .filter(|&state| usize::from(state) != current)
            .collect();
        let mut best: Option<(Vec<StateCounts>, Vec<Edge>, Vec<u16>)> = None;

        visit_permutations(&mut remaining, 0, &mut |order| {
            let mut mapping = vec![0_u16; state_count];
            mapping[current] = 0;
            for (index, &old_state) in order.iter().enumerate() {
                mapping[usize::from(old_state)] = (index + 1) as u16;
            }

            let mut emissions = vec![StateCounts::default(); state_count];
            for (old_state, emission) in logical_emissions.iter().enumerate() {
                emissions[usize::from(mapping[old_state])] = emission.clone();
            }

            let mut edges = Vec::with_capacity(logical_edges.len());
            for edge in &logical_edges {
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

            let candidate = (emissions, edges, mapping);
            match &best {
                None => best = Some(candidate),
                Some(existing) if (&candidate.0, &candidate.1) < (&existing.0, &existing.1) => {
                    best = Some(candidate);
                }
                Some(_) => {}
            }
        });

        let (_, canonical_edges, mapping) = best.expect("at least one state permutation exists");
        self.emissions.semantic_hash = emission_arena
            .iter(self.emissions.head)
            .map(|update| {
                let old_state = self.logical_state_for_storage_label(update.storage_state);
                emission_semantic_hash(mapping[usize::from(old_state)], update.byte)
            })
            .fold(0_u64, u64::wrapping_add);
        self.state_labels = self.state_labels.reordered(&mapping);
        self.current_state = 0;
        self.transitions.semantic_hash = canonical_edges
            .into_iter()
            .fold(0, |hash, edge| hash ^ edge_semantic_hash(edge));
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
    /// Immutable transition nodes allocated in the shared arena.
    pub transition_arena_nodes: usize,
    /// Shared arena nodes divided by posterior component count.
    pub transition_nodes_per_component: f64,
    /// Immutable emission-update nodes allocated in the shared arena.
    pub emission_arena_nodes: usize,
    /// Shared emission nodes divided by posterior component count.
    pub emission_nodes_per_component: f64,
    /// Total nonzero state-byte counters across all canonical components.
    pub nonzero_emission_counts: usize,
    /// Approximate component and arena payload bytes, excluding HashMap buckets.
    pub payload_bytes_estimate: usize,
    /// Number of children generated by the most recent observation before merging.
    pub generated_children_last: usize,
    /// Children eliminated by exact sufficient-state merging on the last update.
    pub merged_children_last: usize,
}

#[derive(Debug, Clone)]
struct WeightedComponent {
    component: Component,
    ln_mass: f64,
}

/// Exact semantic component map with compact non-owning hash keys.
///
/// A 64-bit fingerprint selects the first slot. In the unlikely event of a
/// collision, successive slots are probed and full semantic equality is checked
/// through the transition arena. This preserves exactness without storing a
/// duplicated transition vector in every hash-map key.
#[derive(Debug, Clone, Default)]
struct ComponentMap {
    entries: HashMap<(u64, u32), WeightedComponent>,
}

impl ComponentMap {
    fn singleton(component: Component, ln_mass: f64) -> Self {
        let mut entries = HashMap::new();
        entries.insert(
            (component.semantic_fingerprint(), 0),
            WeightedComponent { component, ln_mass },
        );
        Self { entries }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn iter(&self) -> impl Iterator<Item = (&Component, &f64)> {
        self.entries
            .values()
            .map(|weighted| (&weighted.component, &weighted.ln_mass))
    }

    fn into_values(self) -> impl Iterator<Item = WeightedComponent> {
        self.entries.into_values()
    }

    fn insert(
        &mut self,
        transition_arena: &TransitionArena,
        emission_arena: &EmissionArena,
        component: Component,
        ln_mass: f64,
    ) {
        let fingerprint = component.semantic_fingerprint();
        self.insert_with_fingerprint(
            transition_arena,
            emission_arena,
            component,
            ln_mass,
            fingerprint,
        );
    }

    fn insert_with_fingerprint(
        &mut self,
        transition_arena: &TransitionArena,
        emission_arena: &EmissionArena,
        component: Component,
        ln_mass: f64,
        fingerprint: u64,
    ) {
        let mut collision_index = 0_u32;
        loop {
            match self.entries.entry((fingerprint, collision_index)) {
                Entry::Occupied(mut entry) => {
                    if entry.get().component.semantically_eq(
                        &component,
                        transition_arena,
                        emission_arena,
                    ) {
                        entry.get_mut().ln_mass = log_add_exp(entry.get().ln_mass, ln_mass);
                        return;
                    }
                    collision_index = collision_index
                        .checked_add(1)
                        .expect("partial DFA component fingerprint collision index overflow");
                }
                Entry::Vacant(entry) => {
                    entry.insert(WeightedComponent { component, ln_mass });
                    return;
                }
            }
        }
    }
}

/// One assigned transition in an inspectable posterior component.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DfaAssignedTransition {
    pub source: u16,
    pub byte: u8,
    pub destination: u16,
}

/// Sparse emission counts for one logical DFA state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DfaStateEmissionCounts {
    pub state: u16,
    pub total: u32,
    pub counts: Vec<(u8, u32)>,
}

/// An inspectable exact posterior component for one fixed-N DFA class.
///
/// This is an aggregate sufficient-state hypothesis, not a fully specified
/// transition table: unobserved transitions remain marginalized.
#[derive(Debug, Clone, PartialEq)]
pub struct DfaPosteriorComponent {
    /// Posterior mass conditional on this fixed state-count class.
    pub conditional_posterior_mass: f64,
    /// Natural-log posterior mass conditional on this fixed state-count class.
    pub conditional_ln_posterior: f64,
    pub current_state: u16,
    pub discovered_states: u16,
    pub assigned_transitions: Vec<DfaAssignedTransition>,
    pub emissions: Vec<DfaStateEmissionCounts>,
}

#[derive(Clone, Copy)]
struct TopComponentRef<'a> {
    ln_mass: f64,
    component: &'a Component,
}

impl PartialEq for TopComponentRef<'_> {
    fn eq(&self, other: &Self) -> bool {
        self.ln_mass.total_cmp(&other.ln_mass) == Ordering::Equal
    }
}

impl Eq for TopComponentRef<'_> {}

impl PartialOrd for TopComponentRef<'_> {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TopComponentRef<'_> {
    fn cmp(&self, other: &Self) -> Ordering {
        self.ln_mass.total_cmp(&other.ln_mass)
    }
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
    transition_arena: TransitionArena,
    emission_arena: EmissionArena,
    components: ComponentMap,
    generated_children_last: usize,
}

impl ExactPartialDfaMixture {
    /// Create the exact posterior for all labeled state_count-state DFAs.
    ///
    /// The start state is fixed to label zero. Every transition-table entry has
    /// an independent uniform prior over all state_count destination labels.
    /// All complete DFAs therefore have equal prior probability conditional on N.
    pub fn new(state_count: u16) -> Result<Self, PartialDfaError> {
        Self::with_quotient(state_count, DfaQuotient::Discovery)
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
        let components = ComponentMap::singleton(Component::initial(), 0.0);
        Ok(Self {
            state_count,
            quotient,
            transition_arena: TransitionArena::default(),
            emission_arena: EmissionArena::default(),
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
        log_sum_exp(self.components.iter().map(|(_, ln_mass)| *ln_mass))
    }

    /// Highest-mass exact sufficient-state components in this fixed-N class.
    ///
    /// The returned masses are conditional on this state-count class. Unseen
    /// transition entries remain marginalized inside each returned component.
    pub fn top_components(&self, limit: usize) -> Vec<DfaPosteriorComponent> {
        if limit == 0 || self.components.len() == 0 {
            return Vec::new();
        }

        let ln_evidence = self.ln_evidence();
        let mut heap: BinaryHeap<Reverse<TopComponentRef<'_>>> =
            BinaryHeap::with_capacity(limit.min(self.components.len()));

        for (component, &ln_mass) in self.components.iter() {
            let candidate = TopComponentRef { ln_mass, component };
            if heap.len() < limit {
                heap.push(Reverse(candidate));
            } else if heap
                .peek()
                .is_some_and(|smallest| ln_mass > smallest.0.ln_mass)
            {
                heap.pop();
                heap.push(Reverse(candidate));
            }
        }

        let mut selected: Vec<_> = heap.into_iter().map(|entry| entry.0).collect();
        selected.sort_by(|left, right| right.ln_mass.total_cmp(&left.ln_mass));

        selected
            .into_iter()
            .map(|selected| {
                let component = selected.component;
                let assigned_transitions = component
                    .logical_edges(&self.transition_arena)
                    .into_iter()
                    .map(|edge| DfaAssignedTransition {
                        source: edge.key >> 8,
                        byte: edge.key as u8,
                        destination: edge.destination,
                    })
                    .collect();
                let emissions = component
                    .emission_counts(&self.emission_arena)
                    .into_iter()
                    .enumerate()
                    .map(|(state, counts)| DfaStateEmissionCounts {
                        state: state as u16,
                        total: counts.total,
                        counts: counts.counts,
                    })
                    .collect();
                let conditional_ln_posterior = selected.ln_mass - ln_evidence;
                DfaPosteriorComponent {
                    conditional_posterior_mass: conditional_ln_posterior.exp(),
                    conditional_ln_posterior,
                    current_state: component.current_state,
                    discovered_states: component.discovered_states(),
                    assigned_transitions,
                    emissions,
                }
            })
            .collect()
    }

    /// Number of unmerged children that the next observation would create.
    ///
    /// This is an upper bound on the next exact component count because children
    /// with identical sufficient state are merged after branching.
    pub fn prospective_child_count(&self, byte: u8) -> usize {
        self.components
            .iter()
            .map(|(component, _)| component)
            .map(|component| {
                if component
                    .transition(&self.transition_arena, component.current_state, byte)
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
            .iter()
            .map(|(_, ln_mass)| (*ln_mass - ln_evidence).exp())
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
            .iter()
            .map(|(component, _)| component.transitions.len as usize)
            .sum();
        let transition_arena_nodes = self.transition_arena.nodes.len();
        let transition_nodes_per_component =
            transition_arena_nodes as f64 / self.components.len() as f64;
        let emission_arena_nodes = self.emission_arena.nodes.len();
        let emission_nodes_per_component =
            emission_arena_nodes as f64 / self.components.len() as f64;
        let nonzero_emission_counts = self
            .components
            .iter()
            .map(|(component, _)| component.nonzero_emission_count(&self.emission_arena))
            .sum();
        let payload_bytes_estimate = self
            .components
            .iter()
            .map(|(component, _)| component.payload_bytes_estimate())
            .sum::<usize>()
            + self.components.len() * size_of::<f64>()
            + self.transition_arena.payload_bytes_estimate()
            + self.emission_arena.payload_bytes_estimate();

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
            transition_arena_nodes,
            transition_nodes_per_component,
            emission_arena_nodes,
            emission_nodes_per_component,
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
        let mut next = ComponentMap::default();
        let ln_state_count = f64::from(self.state_count).ln();
        let mut generated_children = 0_usize;

        for weighted in old_components.into_values() {
            let mut component = weighted.component;
            let ln_mass = weighted.ln_mass;
            let source = component.current_state;
            let ln_likelihood = component.ln_prob(&self.emission_arena, byte);
            component.observe_emission(&mut self.emission_arena, byte);
            let ln_base_mass = ln_mass + ln_likelihood;

            if let Some(destination) = component.transition(&self.transition_arena, source, byte) {
                component.current_state = destination;
                generated_children += 1;
                insert_component(
                    self.quotient,
                    &self.transition_arena,
                    &self.emission_arena,
                    &mut next,
                    component,
                    ln_base_mass,
                );
                continue;
            }

            let discovered = component.discovered_states();

            for destination in 0..discovered {
                let mut child = component.clone();
                child.assign_transition(&mut self.transition_arena, source, byte, destination);
                child.current_state = destination;
                generated_children += 1;
                insert_component(
                    self.quotient,
                    &self.transition_arena,
                    &self.emission_arena,
                    &mut next,
                    child,
                    ln_base_mass - ln_state_count,
                );
            }

            if discovered < self.state_count {
                let unused_labels = self.state_count - discovered;
                let mut child = component;
                let destination = discovered;
                child.state_labels.push(discovered);
                child.assign_transition(&mut self.transition_arena, source, byte, destination);
                child.current_state = destination;
                let ln_branch = f64::from(unused_labels).ln() - ln_state_count;
                generated_children += 1;
                insert_component(
                    self.quotient,
                    &self.transition_arena,
                    &self.emission_arena,
                    &mut next,
                    child,
                    ln_base_mass + ln_branch,
                );
            }
        }

        self.generated_children_last = generated_children;
        self.components = next;
    }
}

fn insert_component(
    quotient: DfaQuotient,
    transition_arena: &TransitionArena,
    emission_arena: &EmissionArena,
    map: &mut ComponentMap,
    mut component: Component,
    ln_mass: f64,
) {
    if quotient == DfaQuotient::Predictive {
        component.canonicalize_predictive(transition_arena, emission_arena);
    }
    map.insert(transition_arena, emission_arena, component, ln_mass);
}

struct ExactPartialDfaPrediction<'a> {
    mixture: &'a ExactPartialDfaMixture,
    ln_evidence: f64,
}

impl Distribution<u8> for ExactPartialDfaPrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        log_sum_exp(self.mixture.components.iter().map(|(component, ln_mass)| {
            *ln_mass + component.ln_prob(&self.mixture.emission_arena, *byte)
        })) - self.ln_evidence
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

    /// Vector-backed implementation retained only as an independent short-prefix
    /// oracle for the persistent representation.
    #[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
    struct LegacyComponent {
        current_state: u16,
        edges: Vec<Edge>,
        emissions: Vec<StateCounts>,
    }

    impl LegacyComponent {
        fn initial() -> Self {
            Self {
                current_state: 0,
                edges: Vec::new(),
                emissions: vec![StateCounts::default()],
            }
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
            let index = self
                .edges
                .binary_search_by_key(&edge.key, |item| item.key)
                .expect_err("legacy oracle cannot reassign a transition");
            self.edges.insert(index, edge);
        }

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
                let mut edges: Vec<_> = self
                    .edges
                    .iter()
                    .map(|edge| {
                        Edge::new(
                            mapping[usize::from(edge.key >> 8)],
                            edge.key as u8,
                            mapping[usize::from(edge.destination)],
                        )
                    })
                    .collect();
                edges.sort_unstable();
                let candidate = Self {
                    current_state: 0,
                    edges,
                    emissions,
                };
                if best.as_ref().is_none_or(|existing| candidate < *existing) {
                    best = Some(candidate);
                }
            });
            best.expect("at least one legacy state permutation exists")
        }
    }

    struct LegacyMixture {
        state_count: u16,
        quotient: DfaQuotient,
        components: HashMap<LegacyComponent, f64>,
    }

    impl LegacyMixture {
        fn new(state_count: u16, quotient: DfaQuotient) -> Self {
            Self {
                state_count,
                quotient,
                components: HashMap::from([(LegacyComponent::initial(), 0.0)]),
            }
        }

        fn ln_evidence(&self) -> f64 {
            log_sum_exp(self.components.values().copied())
        }

        fn ln_prob(&self, byte: u8) -> f64 {
            log_sum_exp(self.components.iter().map(|(component, ln_mass)| {
                *ln_mass + component.emissions[usize::from(component.current_state)].ln_prob(byte)
            })) - self.ln_evidence()
        }

        fn observe(&mut self, byte: u8) {
            let old_components = std::mem::take(&mut self.components);
            let mut next = HashMap::new();
            let ln_state_count = f64::from(self.state_count).ln();
            for (mut component, ln_mass) in old_components {
                let source = component.current_state;
                let ln_likelihood = component.emissions[usize::from(source)].ln_prob(byte);
                component.emissions[usize::from(source)].observe(byte);
                let ln_base_mass = ln_mass + ln_likelihood;

                if let Some(destination) = component.transition(source, byte) {
                    component.current_state = destination;
                    self.insert(&mut next, component, ln_base_mass);
                    continue;
                }

                let discovered = component.emissions.len() as u16;
                for destination in 0..discovered {
                    let mut child = component.clone();
                    child.assign_transition(source, byte, destination);
                    child.current_state = destination;
                    self.insert(&mut next, child, ln_base_mass - ln_state_count);
                }
                if discovered < self.state_count {
                    let unused_labels = self.state_count - discovered;
                    let mut child = component;
                    child.assign_transition(source, byte, discovered);
                    child.current_state = discovered;
                    child.emissions.push(StateCounts::default());
                    self.insert(
                        &mut next,
                        child,
                        ln_base_mass + f64::from(unused_labels).ln() - ln_state_count,
                    );
                }
            }
            self.components = next;
        }

        fn insert(
            &self,
            map: &mut HashMap<LegacyComponent, f64>,
            component: LegacyComponent,
            ln_mass: f64,
        ) {
            let component = match self.quotient {
                DfaQuotient::Discovery => component,
                DfaQuotient::Predictive => component.canonicalized_predictive(),
            };
            match map.entry(component) {
                Entry::Occupied(mut entry) => {
                    *entry.get_mut() = log_add_exp(*entry.get(), ln_mass);
                }
                Entry::Vacant(entry) => {
                    entry.insert(ln_mass);
                }
            }
        }

        fn retention(&self, epsilon_nats: f64) -> (usize, f64, f64) {
            let ln_evidence = self.ln_evidence();
            let mut weights: Vec<_> = self
                .components
                .values()
                .map(|ln_mass| (*ln_mass - ln_evidence).exp())
                .collect();
            weights.sort_by(|a, b| b.total_cmp(a));
            let target_mass = (-epsilon_nats).exp();
            let mut retained_mass = 0.0;
            let mut retained_components = 0;
            for weight in weights {
                if retained_mass >= target_mass {
                    break;
                }
                retained_mass += weight;
                retained_components += 1;
            }
            retained_mass = retained_mass.min(1.0);
            (retained_components, retained_mass, -retained_mass.ln())
        }
    }

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
            .iter()
            .map(|(_, mass)| (*mass - ln_evidence).exp())
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
        let mut transition_arena = TransitionArena::default();
        let mut emission_arena = EmissionArena::default();

        let mut left = Component::initial();
        left.state_labels.push(1);
        left.observe_emission(&mut emission_arena, b'A');
        left.current_state = 1;
        left.observe_emission(&mut emission_arena, b'B');
        left.current_state = 0;
        left.assign_transition(&mut transition_arena, 0, b'x', 1);

        let mut right = Component::initial();
        right.state_labels.push(1);
        right.observe_emission(&mut emission_arena, b'B');
        right.current_state = 1;
        right.observe_emission(&mut emission_arena, b'A');
        right.assign_transition(&mut transition_arena, 1, b'x', 0);

        assert!(!left.semantically_eq(&right, &transition_arena, &emission_arena));
        left.canonicalize_predictive(&transition_arena, &emission_arena);
        right.canonicalize_predictive(&transition_arena, &emission_arena);
        assert!(left.semantically_eq(&right, &transition_arena, &emission_arena));
    }

    #[test]
    fn persistent_branches_share_their_transition_prefix() {
        let mut arena = TransitionArena::default();
        let mut parent = Component::initial();
        parent.state_labels.push(1);
        parent.assign_transition(&mut arena, 0, b'a', 1);
        let shared_head = parent.transitions.head;

        let mut left = parent.clone();
        let mut right = parent;
        left.assign_transition(&mut arena, 1, b'b', 0);
        right.assign_transition(&mut arena, 1, b'b', 1);

        assert_eq!(
            arena.nodes[usize::try_from(left.transitions.head).unwrap()].parent,
            shared_head
        );
        assert_eq!(
            arena.nodes[usize::try_from(right.transitions.head).unwrap()].parent,
            shared_head
        );
        assert_eq!(arena.nodes.len(), 3);
        assert_eq!(left.transitions.len + right.transitions.len, 4);
    }

    #[test]
    fn persistent_branches_share_their_emission_prefix() {
        let mut arena = EmissionArena::default();
        let mut parent = Component::initial();
        parent.observe_emission(&mut arena, b'a');
        let shared_head = parent.emissions.head;

        let mut left = parent.clone();
        let mut right = parent;
        left.observe_emission(&mut arena, b'b');
        right.observe_emission(&mut arena, b'c');

        assert_eq!(
            arena.nodes[usize::try_from(left.emissions.head).unwrap()].parent,
            shared_head
        );
        assert_eq!(
            arena.nodes[usize::try_from(right.emissions.head).unwrap()].parent,
            shared_head
        );
        assert_eq!(arena.nodes.len(), 3);
        assert_eq!(left.emissions.len + right.emissions.len, 4);
    }

    #[test]
    fn component_fingerprint_collisions_use_semantic_equality() {
        let transition_arena = TransitionArena::default();
        let emission_arena = EmissionArena::default();
        let mut left = Component::initial();
        left.state_labels.push(1);
        let mut right = left.clone();
        right.current_state = 1;
        let mut components = ComponentMap::default();

        components.insert_with_fingerprint(
            &transition_arena,
            &emission_arena,
            left.clone(),
            -1.0,
            7,
        );
        components.insert_with_fingerprint(&transition_arena, &emission_arena, right, -2.0, 7);
        components.insert_with_fingerprint(&transition_arena, &emission_arena, left, -3.0, 7);

        assert_eq!(components.len(), 2);
        let left_mass = components
            .entries
            .get(&(7, 0))
            .expect("first collision slot exists")
            .ln_mass;
        assert!((left_mass - log_add_exp(-1.0, -3.0)).abs() < 1e-15);
    }

    #[test]
    fn persistent_representation_matches_vector_oracle_exactly() {
        let data = b"mediawiki";
        let epsilon = 0.01;

        for quotient in [DfaQuotient::Discovery, DfaQuotient::Predictive] {
            let mut persistent = ExactPartialDfaMixture::with_quotient(2, quotient).unwrap();
            let mut legacy = LegacyMixture::new(2, quotient);
            let mut kt = Kt::default();
            let mut persistent_nats = 0.0;
            let mut legacy_nats = 0.0;
            let mut kt_nats = 0.0;

            for &byte in data {
                let persistent_ln_prob = persistent.predict().ln_prob(&byte);
                let legacy_ln_prob = legacy.ln_prob(byte);
                assert!((persistent_ln_prob - legacy_ln_prob).abs() < 1e-12);
                persistent_nats -= persistent_ln_prob;
                legacy_nats -= legacy_ln_prob;
                kt_nats -= kt.predict().ln_prob(&byte);

                persistent.observe(byte);
                legacy.observe(byte);
                kt.observe(byte);

                assert_eq!(persistent.component_count(), legacy.components.len());
                assert!((persistent.ln_evidence() - legacy.ln_evidence()).abs() < 1e-11);

                let diagnostics = persistent.diagnostics(epsilon);
                let (retained_components, retained_mass, retained_kl_nats) =
                    legacy.retention(epsilon);
                assert_eq!(diagnostics.retained_components, retained_components);
                assert!((diagnostics.retained_mass - retained_mass).abs() < 1e-12);
                assert!((diagnostics.retained_kl_nats - retained_kl_nats).abs() < 1e-12);
            }

            assert!((persistent_nats - legacy_nats).abs() < 1e-12);
            assert!((persistent.ln_evidence() + persistent_nats).abs() < 1e-11);
            let uniform_nats = data.len() as f64 * ALPHABET_SIZE.ln();
            assert!((uniform_nats / persistent_nats - uniform_nats / legacy_nats).abs() < 1e-12);
            assert!((kt_nats / persistent_nats - kt_nats / legacy_nats).abs() < 1e-12);
        }
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
