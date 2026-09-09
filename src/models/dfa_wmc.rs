//! Exact N=2 DFA evidence by algebraic decision-diagram evaluation.
//!
//! Each transition-table entry is a Boolean variable. The hidden state,
//! state-one visit counts, per-byte state-one counts, and integrated emission log
//! likelihood are reduced ordered algebraic decision diagrams (ADDs) over those
//! variables. Identical algebraic subfunctions are hash-consed, so the evaluator
//! sums over transition assignments without materializing posterior leaves.
//! Averaging the final log-likelihood ADD over its Boolean variables gives the
//! exact joint marginal likelihood under the uniform transition-table prior.

use std::{
    collections::{HashMap, HashSet},
    error::Error,
    fmt,
    mem::size_of,
};

const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = 128.0;
const LN_2: f64 = std::f64::consts::LN_2;

type NodeId = u32;

#[derive(Debug, Clone, Copy)]
enum AddNode {
    Terminal(u64),
    Branch {
        variable: u16,
        low: NodeId,
        high: NodeId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BranchKey {
    variable: u16,
    low: NodeId,
    high: NodeId,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfaWmcError {
    NodeLimit { limit: usize },
    NodeIdExhausted,
}

impl fmt::Display for DfaWmcError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NodeLimit { limit } => {
                write!(formatter, "ADD node limit {limit} would be exceeded")
            }
            Self::NodeIdExhausted => write!(formatter, "ADD exhausted its u32 node ID space"),
        }
    }
}

impl Error for DfaWmcError {}

#[derive(Debug, Clone)]
struct AddManager {
    nodes: Vec<AddNode>,
    terminals: HashMap<u64, NodeId>,
    branches: HashMap<BranchKey, NodeId>,
    max_nodes: usize,
    zero: NodeId,
    one: NodeId,
}

impl AddManager {
    fn new(max_nodes: usize) -> Result<Self, DfaWmcError> {
        let mut manager = Self {
            nodes: Vec::new(),
            terminals: HashMap::new(),
            branches: HashMap::new(),
            max_nodes,
            zero: 0,
            one: 0,
        };
        manager.zero = manager.terminal(0.0)?;
        manager.one = manager.terminal(1.0)?;
        Ok(manager)
    }

    fn push(&mut self, node: AddNode) -> Result<NodeId, DfaWmcError> {
        if self.nodes.len() >= self.max_nodes {
            return Err(DfaWmcError::NodeLimit {
                limit: self.max_nodes,
            });
        }
        let id = NodeId::try_from(self.nodes.len()).map_err(|_| DfaWmcError::NodeIdExhausted)?;
        self.nodes.push(node);
        Ok(id)
    }

    fn terminal(&mut self, value: f64) -> Result<NodeId, DfaWmcError> {
        debug_assert!(value.is_finite());
        let normalized = if value == 0.0 { 0.0 } else { value };
        let bits = normalized.to_bits();
        if let Some(&id) = self.terminals.get(&bits) {
            return Ok(id);
        }
        let id = self.push(AddNode::Terminal(bits))?;
        self.terminals.insert(bits, id);
        Ok(id)
    }

    fn branch(&mut self, variable: u16, low: NodeId, high: NodeId) -> Result<NodeId, DfaWmcError> {
        if low == high {
            return Ok(low);
        }
        debug_assert!(self.top_variable(low).is_none_or(|child| variable < child));
        debug_assert!(self.top_variable(high).is_none_or(|child| variable < child));
        let key = BranchKey {
            variable,
            low,
            high,
        };
        if let Some(&id) = self.branches.get(&key) {
            return Ok(id);
        }
        let id = self.push(AddNode::Branch {
            variable,
            low,
            high,
        })?;
        self.branches.insert(key, id);
        Ok(id)
    }

    fn variable(&mut self, variable: u16) -> Result<NodeId, DfaWmcError> {
        self.branch(variable, self.zero, self.one)
    }

    fn top_variable(&self, node: NodeId) -> Option<u16> {
        match self.nodes[usize::try_from(node).expect("u32 fits usize")] {
            AddNode::Terminal(_) => None,
            AddNode::Branch { variable, .. } => Some(variable),
        }
    }

    fn cofactors(&self, node: NodeId, variable: u16) -> (NodeId, NodeId) {
        match self.nodes[usize::try_from(node).expect("u32 fits usize")] {
            AddNode::Branch {
                variable: node_variable,
                low,
                high,
            } if node_variable == variable => (low, high),
            _ => (node, node),
        }
    }

    fn terminal_value(&self, node: NodeId) -> Option<f64> {
        match self.nodes[usize::try_from(node).expect("u32 fits usize")] {
            AddNode::Terminal(bits) => Some(f64::from_bits(bits)),
            AddNode::Branch { .. } => None,
        }
    }

    fn add(&mut self, left: NodeId, right: NodeId) -> Result<NodeId, DfaWmcError> {
        let mut memo = HashMap::new();
        self.add_recursive(left, right, &mut memo)
    }

    fn add_recursive(
        &mut self,
        left: NodeId,
        right: NodeId,
        memo: &mut HashMap<(NodeId, NodeId), NodeId>,
    ) -> Result<NodeId, DfaWmcError> {
        let key = if left <= right {
            (left, right)
        } else {
            (right, left)
        };
        if let Some(&result) = memo.get(&key) {
            return Ok(result);
        }
        if left == self.zero {
            return Ok(right);
        }
        if right == self.zero {
            return Ok(left);
        }
        if let (Some(left_value), Some(right_value)) =
            (self.terminal_value(left), self.terminal_value(right))
        {
            return self.terminal(left_value + right_value);
        }

        let variable = self
            .top_variable(left)
            .into_iter()
            .chain(self.top_variable(right))
            .min()
            .expect("at least one ADD operand is nonterminal");
        let (left_low, left_high) = self.cofactors(left, variable);
        let (right_low, right_high) = self.cofactors(right, variable);
        let low = self.add_recursive(left_low, right_low, memo)?;
        let high = self.add_recursive(left_high, right_high, memo)?;
        let result = self.branch(variable, low, high)?;
        memo.insert(key, result);
        Ok(result)
    }

    fn if_then_else(
        &mut self,
        condition: NodeId,
        when_true: NodeId,
        when_false: NodeId,
    ) -> Result<NodeId, DfaWmcError> {
        let mut memo = HashMap::new();
        self.ite_recursive(condition, when_true, when_false, &mut memo)
    }

    fn ite_recursive(
        &mut self,
        condition: NodeId,
        when_true: NodeId,
        when_false: NodeId,
        memo: &mut HashMap<(NodeId, NodeId, NodeId), NodeId>,
    ) -> Result<NodeId, DfaWmcError> {
        if when_true == when_false {
            return Ok(when_true);
        }
        if let Some(value) = self.terminal_value(condition) {
            debug_assert!(value == 0.0 || value == 1.0);
            return Ok(if value == 1.0 { when_true } else { when_false });
        }
        let key = (condition, when_true, when_false);
        if let Some(&result) = memo.get(&key) {
            return Ok(result);
        }
        let variable = self
            .top_variable(condition)
            .into_iter()
            .chain(self.top_variable(when_true))
            .chain(self.top_variable(when_false))
            .min()
            .expect("nonterminal ITE has a top variable");
        let (condition_low, condition_high) = self.cofactors(condition, variable);
        let (true_low, true_high) = self.cofactors(when_true, variable);
        let (false_low, false_high) = self.cofactors(when_false, variable);
        let low = self.ite_recursive(condition_low, true_low, false_low, memo)?;
        let high = self.ite_recursive(condition_high, true_high, false_high, memo)?;
        let result = self.branch(variable, low, high)?;
        memo.insert(key, result);
        Ok(result)
    }

    fn map_terminals(
        &mut self,
        root: NodeId,
        transform: impl Fn(f64) -> f64,
    ) -> Result<NodeId, DfaWmcError> {
        let mut memo = HashMap::new();
        self.map_terminals_recursive(root, &transform, &mut memo)
    }

    fn map_terminals_recursive(
        &mut self,
        node: NodeId,
        transform: &impl Fn(f64) -> f64,
        memo: &mut HashMap<NodeId, NodeId>,
    ) -> Result<NodeId, DfaWmcError> {
        if let Some(&result) = memo.get(&node) {
            return Ok(result);
        }
        let result = match self.nodes[usize::try_from(node).expect("u32 fits usize")] {
            AddNode::Terminal(bits) => self.terminal(transform(f64::from_bits(bits)))?,
            AddNode::Branch {
                variable,
                low,
                high,
            } => {
                let low = self.map_terminals_recursive(low, transform, memo)?;
                let high = self.map_terminals_recursive(high, transform, memo)?;
                self.branch(variable, low, high)?
            }
        };
        memo.insert(node, result);
        Ok(result)
    }

    fn integrated_log_likelihood(
        &mut self,
        state_one_total: NodeId,
        state_one_bytes: &[NodeId; 256],
        observations: u32,
        byte_observations: &[u32; 256],
    ) -> Result<NodeId, DfaWmcError> {
        let mut result = self.zero;
        for byte in 0..256 {
            let occurrences = byte_observations[byte];
            if occurrences == 0 {
                continue;
            }
            let state_one = self.map_terminals(state_one_bytes[byte], |count| {
                log_rising_factorial(count as u32, JEFFREYS_ALPHA)
            })?;
            let state_zero = self.map_terminals(state_one_bytes[byte], |count| {
                log_rising_factorial(occurrences - count as u32, JEFFREYS_ALPHA)
            })?;
            result = self.add(result, state_one)?;
            result = self.add(result, state_zero)?;
        }

        let state_one_denominator = self.map_terminals(state_one_total, |count| {
            -log_rising_factorial(count as u32, JEFFREYS_TOTAL)
        })?;
        let state_zero_denominator = self.map_terminals(state_one_total, |count| {
            -log_rising_factorial(observations - count as u32, JEFFREYS_TOTAL)
        })?;
        result = self.add(result, state_one_denominator)?;
        self.add(result, state_zero_denominator)
    }

    fn log_expectation(&self, root: NodeId) -> f64 {
        let mut memo = HashMap::new();
        self.log_expectation_recursive(root, &mut memo)
    }

    fn log_expectation_recursive(&self, node: NodeId, memo: &mut HashMap<NodeId, f64>) -> f64 {
        if let Some(&value) = memo.get(&node) {
            return value;
        }
        let value = match self.nodes[usize::try_from(node).expect("u32 fits usize")] {
            AddNode::Terminal(bits) => f64::from_bits(bits),
            AddNode::Branch { low, high, .. } => {
                log_add_exp(
                    self.log_expectation_recursive(low, memo),
                    self.log_expectation_recursive(high, memo),
                ) - LN_2
            }
        };
        memo.insert(node, value);
        value
    }

    fn reachable_nodes(&self, roots: impl IntoIterator<Item = NodeId>) -> usize {
        let mut seen = HashSet::new();
        let mut pending: Vec<_> = roots.into_iter().collect();
        while let Some(node) = pending.pop() {
            if !seen.insert(node) {
                continue;
            }
            if let AddNode::Branch { low, high, .. } =
                self.nodes[usize::try_from(node).expect("u32 fits usize")]
            {
                pending.push(low);
                pending.push(high);
            }
        }
        seen.len()
    }

    fn payload_bytes_estimate(&self) -> usize {
        self.nodes.capacity() * size_of::<AddNode>()
            + self.terminals.capacity() * (size_of::<u64>() + size_of::<NodeId>())
            + self.branches.capacity() * (size_of::<BranchKey>() + size_of::<NodeId>())
    }

    fn copy_node(
        &self,
        node: NodeId,
        destination: &mut Self,
        mapping: &mut [Option<NodeId>],
    ) -> Result<NodeId, DfaWmcError> {
        let index = usize::try_from(node).expect("u32 fits usize");
        if let Some(mapped) = mapping[index] {
            return Ok(mapped);
        }
        let mapped = match self.nodes[index] {
            AddNode::Terminal(bits) => destination.terminal(f64::from_bits(bits))?,
            AddNode::Branch {
                variable,
                low,
                high,
            } => {
                let low = self.copy_node(low, destination, mapping)?;
                let high = self.copy_node(high, destination, mapping)?;
                destination.branch(variable, low, high)?
            }
        };
        mapping[index] = Some(mapped);
        Ok(mapped)
    }
}

/// Diagnostics for the exact two-state ADD evaluator.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DfaWmcDiagnostics {
    pub observations: u32,
    pub distinct_input_bytes: u16,
    pub transition_variables: u16,
    pub allocated_nodes: usize,
    pub live_nodes: usize,
    pub garbage_collections: u32,
    pub reclaimed_nodes: usize,
    pub payload_bytes_estimate: usize,
    pub ln_evidence: f64,
}

/// Exact joint-evidence evaluator for the N=2 byte-input DFA family.
#[derive(Debug, Clone)]
pub struct ExactDfaWmc2 {
    manager: AddManager,
    state: NodeId,
    state_one_total: NodeId,
    state_one_bytes: [NodeId; 256],
    log_weight: NodeId,
    transition_variables: [Option<(NodeId, NodeId)>; 256],
    byte_observations: [u32; 256],
    observations: u32,
    next_variable: u16,
    garbage_collections: u32,
    reclaimed_nodes: usize,
}

impl ExactDfaWmc2 {
    pub fn new(max_nodes: usize) -> Result<Self, DfaWmcError> {
        let manager = AddManager::new(max_nodes)?;
        let zero = manager.zero;
        Ok(Self {
            manager,
            state: zero,
            state_one_total: zero,
            state_one_bytes: [zero; 256],
            log_weight: zero,
            transition_variables: [None; 256],
            byte_observations: [0; 256],
            observations: 0,
            next_variable: 0,
            garbage_collections: 0,
            reclaimed_nodes: 0,
        })
    }

    pub fn observe(&mut self, byte: u8) -> Result<(), DfaWmcError> {
        match self.observe_once(byte) {
            Ok(()) => {}
            Err(DfaWmcError::NodeLimit { .. }) => {
                self.collect_garbage()?;
                self.observe_once(byte)?;
            }
            Err(error) => return Err(error),
        }
        if self.manager.nodes.len() > self.manager.max_nodes / 2 {
            self.collect_garbage()?;
        }
        Ok(())
    }

    fn observe_once(&mut self, byte: u8) -> Result<(), DfaWmcError> {
        let byte_index = usize::from(byte);
        let state_one_total = self.manager.add(self.state_one_total, self.state)?;
        let state_one_byte = self
            .manager
            .add(self.state_one_bytes[byte_index], self.state)?;
        let mut state_one_bytes = self.state_one_bytes;
        state_one_bytes[byte_index] = state_one_byte;
        let mut byte_observations = self.byte_observations;
        byte_observations[byte_index] = byte_observations[byte_index]
            .checked_add(1)
            .expect("DFA WMC byte observation count overflow");
        let observations = self
            .observations
            .checked_add(1)
            .expect("DFA WMC observation count overflow");
        let log_weight = self.manager.integrated_log_likelihood(
            state_one_total,
            &state_one_bytes,
            observations,
            &byte_observations,
        )?;

        let (state_zero_destination, state_one_destination) =
            match self.transition_variables[byte_index] {
                Some(variables) => variables,
                None => {
                    let state_zero_variable = self.next_variable;
                    let state_one_variable = state_zero_variable
                        .checked_add(1)
                        .expect("N=2 DFA transition variable count fits u16");
                    let next_variable = state_one_variable
                        .checked_add(1)
                        .expect("N=2 DFA transition variable count fits u16");
                    let state_zero_destination = self.manager.variable(state_zero_variable)?;
                    let state_one_destination = self.manager.variable(state_one_variable)?;
                    self.next_variable = next_variable;
                    self.transition_variables[byte_index] =
                        Some((state_zero_destination, state_one_destination));
                    (state_zero_destination, state_one_destination)
                }
            };
        let state =
            self.manager
                .if_then_else(self.state, state_one_destination, state_zero_destination)?;

        self.log_weight = log_weight;
        self.state_one_total = state_one_total;
        self.state_one_bytes = state_one_bytes;
        self.state = state;
        self.byte_observations = byte_observations;
        self.observations = observations;
        Ok(())
    }

    fn collect_garbage(&mut self) -> Result<(), DfaWmcError> {
        let max_nodes = self.manager.max_nodes;
        let old_manager = std::mem::replace(&mut self.manager, AddManager::new(max_nodes)?);
        let old_nodes = old_manager.nodes.len();
        let mut mapping = vec![None; old_nodes];

        self.state = old_manager.copy_node(self.state, &mut self.manager, &mut mapping)?;
        self.state_one_total =
            old_manager.copy_node(self.state_one_total, &mut self.manager, &mut mapping)?;
        for node in &mut self.state_one_bytes {
            *node = old_manager.copy_node(*node, &mut self.manager, &mut mapping)?;
        }
        self.log_weight =
            old_manager.copy_node(self.log_weight, &mut self.manager, &mut mapping)?;
        for (zero, one) in self.transition_variables.iter_mut().flatten() {
            *zero = old_manager.copy_node(*zero, &mut self.manager, &mut mapping)?;
            *one = old_manager.copy_node(*one, &mut self.manager, &mut mapping)?;
        }

        self.garbage_collections = self
            .garbage_collections
            .checked_add(1)
            .expect("DFA WMC garbage collection count overflow");
        self.reclaimed_nodes += old_nodes.saturating_sub(self.manager.nodes.len());
        Ok(())
    }

    pub fn ln_evidence(&self) -> f64 {
        self.manager.log_expectation(self.log_weight)
    }

    pub fn diagnostics(&self) -> DfaWmcDiagnostics {
        let roots = std::iter::once(self.state)
            .chain(std::iter::once(self.state_one_total))
            .chain(self.state_one_bytes.iter().copied())
            .chain(std::iter::once(self.log_weight))
            .chain(
                self.transition_variables
                    .iter()
                    .flatten()
                    .flat_map(|&(zero, one)| [zero, one]),
            );
        DfaWmcDiagnostics {
            observations: self.observations,
            distinct_input_bytes: self.next_variable / 2,
            transition_variables: self.next_variable,
            allocated_nodes: self.manager.nodes.len(),
            live_nodes: self.manager.reachable_nodes(roots),
            garbage_collections: self.garbage_collections,
            reclaimed_nodes: self.reclaimed_nodes,
            payload_bytes_estimate: self.manager.payload_bytes_estimate(),
            ln_evidence: self.ln_evidence(),
        }
    }
}

fn log_add_exp(left: f64, right: f64) -> f64 {
    if left >= right {
        left + (right - left).exp().ln_1p()
    } else {
        right + (left - right).exp().ln_1p()
    }
}

fn log_rising_factorial(count: u32, offset: f64) -> f64 {
    (0..count)
        .map(|index| (f64::from(index) + offset).ln())
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Distribution, Model, models::partial_dfa::ExactPartialDfaMixture};

    fn brute_force_ln_evidence(data: &[u8]) -> f64 {
        let mut byte_variables = [None; 256];
        let mut distinct = 0_u32;
        for &byte in data {
            if byte_variables[usize::from(byte)].is_none() {
                byte_variables[usize::from(byte)] = Some(distinct);
                distinct += 1;
            }
        }
        let variable_count = 2 * distinct;
        assert!(variable_count < usize::BITS);
        let mut ln_sum = f64::NEG_INFINITY;
        for assignment in 0..(1_usize << variable_count) {
            let mut state = 0_usize;
            let mut totals = [0_u32; 2];
            let mut counts = [[0_u32; 256]; 2];
            let mut ln_likelihood = 0.0;
            for &byte in data {
                ln_likelihood += (f64::from(counts[state][usize::from(byte)]) + JEFFREYS_ALPHA)
                    .ln()
                    - (f64::from(totals[state]) + JEFFREYS_TOTAL).ln();
                totals[state] += 1;
                counts[state][usize::from(byte)] += 1;
                let byte_variable = byte_variables[usize::from(byte)].unwrap();
                let variable = 2 * byte_variable + state as u32;
                state = (assignment >> variable) & 1;
            }
            ln_sum = log_add_exp(ln_sum, ln_likelihood);
        }
        ln_sum - f64::from(variable_count) * LN_2
    }

    #[test]
    fn first_observation_is_uniform() {
        let mut wmc = ExactDfaWmc2::new(10_000).unwrap();
        wmc.observe(b'x').unwrap();
        assert!((wmc.ln_evidence() + 256.0_f64.ln()).abs() < 1e-14);
    }

    #[test]
    fn add_evidence_matches_the_exact_leaf_oracle() {
        let mut wmc = ExactDfaWmc2::new(1_000_000).unwrap();
        let mut oracle = ExactPartialDfaMixture::with_quotient(
            2,
            crate::models::partial_dfa::DfaQuotient::Discovery,
        )
        .unwrap();
        let mut sequential_ln_evidence = 0.0;

        for &byte in b"mediawiki" {
            sequential_ln_evidence += oracle.predict().ln_prob(&byte);
            oracle.observe(byte);
            wmc.observe(byte).unwrap();
            assert!((wmc.ln_evidence() - oracle.ln_evidence()).abs() < 1e-11);
            assert!((wmc.ln_evidence() - sequential_ln_evidence).abs() < 1e-11);
        }
    }

    #[test]
    fn add_evidence_matches_complete_transition_table_enumeration() {
        for data in [b"".as_slice(), b"a", b"aba", b"abba", b"abcab"] {
            let mut wmc = ExactDfaWmc2::new(1_000_000).unwrap();
            for &byte in data {
                wmc.observe(byte).unwrap();
            }
            assert!((wmc.ln_evidence() - brute_force_ln_evidence(data)).abs() < 1e-12);
        }
    }

    #[test]
    fn garbage_collection_preserves_all_roots_and_evidence() {
        let mut wmc = ExactDfaWmc2::new(1_000_000).unwrap();
        for &byte in b"mediawiki" {
            wmc.observe(byte).unwrap();
        }
        let before = wmc.ln_evidence();
        let allocated_before = wmc.manager.nodes.len();
        wmc.collect_garbage().unwrap();
        assert!((wmc.ln_evidence() - before).abs() < 1e-14);
        assert!(wmc.manager.nodes.len() < allocated_before);
        wmc.observe(b' ').unwrap();
    }

    #[test]
    fn node_limit_is_reported() {
        let mut wmc = ExactDfaWmc2::new(3).unwrap();
        assert_eq!(wmc.observe(b'x'), Err(DfaWmcError::NodeLimit { limit: 3 }));
    }
}
