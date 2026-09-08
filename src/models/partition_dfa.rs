//! Exact Bayesian emission partitions of a generated finite-state machine.
//!
//! At each feature prefix choose stop/split with prior 1/2 each; at maximum
//! depth stop is forced. A stop ties all descendant DFA states to one
//! Dirichlet-1/2 byte distribution. Unvisited subtrees have evidence one.
//! All evidence and posterior weights use natural logarithms.

use std::{collections::HashMap, f64::consts::LN_2};

use crate::{Distribution, Model, anytime::log_add_exp};

/// A deterministic finite-state generator, independent of emission inference.
/// Feature length and alphabets must remain fixed and finite. `observe` alone
/// advances state; `features` describes the state before the next observation.
pub trait StateProgram {
    fn features(&self) -> &[u16];
    fn observe(&mut self, byte: u8);
}

/// Shift-register state, most recent byte first, padded with start symbol 256.
#[derive(Clone, Debug)]
pub struct ByteHistory {
    state: Vec<u16>,
}

impl ByteHistory {
    pub fn new(depth: usize) -> Self {
        Self {
            state: vec![256; depth],
        }
    }
}

impl StateProgram for ByteHistory {
    fn features(&self) -> &[u16] {
        &self.state
    }

    fn observe(&mut self, byte: u8) {
        if !self.state.is_empty() {
            self.state.rotate_right(1);
            self.state[0] = u16::from(byte);
        }
    }
}

#[derive(Clone, Debug, Default)]
struct Node {
    counts: Vec<(u8, u64)>,
    total: u64,
    ln_stop: f64,
    ln_split: f64,
    ln_evidence: f64,
}

impl Node {
    fn ln_prob(&self, byte: u8) -> f64 {
        let count = self
            .counts
            .binary_search_by_key(&byte, |&(b, _)| b)
            .map_or(0, |index| self.counts[index].1);
        ((count as f64 + 0.5) / (self.total as f64 + 128.0)).ln()
    }

    fn observe(&mut self, byte: u8) {
        self.ln_stop += self.ln_prob(byte);
        match self.counts.binary_search_by_key(&byte, |&(b, _)| b) {
            Ok(index) => self.counts[index].1 += 1,
            Err(index) => self.counts.insert(index, (byte, 1)),
        }
        self.total += 1;
    }
}

/// Exact finite-family posterior, not an approximation to the sparse-DFA prior.
#[derive(Clone, Debug)]
pub struct PartitionDfa<P> {
    program: P,
    nodes: Vec<Node>,
    children: HashMap<(usize, u16), usize>,
    path: Vec<usize>,
    node_updates: u64,
}

impl<P: StateProgram> PartitionDfa<P> {
    pub fn new(program: P) -> Self {
        Self {
            program,
            nodes: vec![Node::default()],
            children: HashMap::new(),
            path: Vec::new(),
            node_updates: 0,
        }
    }

    pub fn ln_evidence(&self) -> f64 {
        self.nodes[0].ln_evidence
    }
    pub fn nodes(&self) -> usize {
        self.nodes.len()
    }
    pub fn node_updates(&self) -> u64 {
        self.node_updates
    }

    fn prediction(&self, byte: u8) -> f64 {
        // Descend only through materialized state prefixes. Every unvisited
        // subtree predicts uniformly, regardless of its remaining depth.
        let features = self.program.features();
        let mut node_id = 0;
        let mut path = Vec::with_capacity(features.len() + 1);
        path.push(0);
        for &feature in features {
            let Some(&child) = self.children.get(&(node_id, feature)) else {
                break;
            };
            node_id = child;
            path.push(child);
        }
        let mut prediction = -8.0 * LN_2;
        for (depth, &id) in path.iter().enumerate().rev() {
            let node = &self.nodes[id];
            let stop = node.ln_prob(byte);
            prediction = if depth == features.len() {
                stop
            } else {
                log_add_exp(node.ln_stop + stop, node.ln_split + prediction)
                    - log_add_exp(node.ln_stop, node.ln_split)
            };
        }
        prediction
    }
}

struct Prediction<'a, P>(&'a PartitionDfa<P>);

impl<P: StateProgram> Distribution<u8> for Prediction<'_, P> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        self.0.prediction(*byte)
    }
}

impl<P: StateProgram> Model<u8> for PartitionDfa<P> {
    fn predict(&self) -> impl Distribution<u8> {
        Prediction(self)
    }

    fn observe(&mut self, byte: u8) {
        self.path.clear();
        self.path.push(0);
        let mut node_id = 0;
        for &feature in self.program.features() {
            node_id = *self.children.entry((node_id, feature)).or_insert_with(|| {
                let id = self.nodes.len();
                self.nodes.push(Node::default());
                id
            });
            self.path.push(node_id);
        }
        let mut child_delta = 0.0;
        let leaf_depth = self.path.len() - 1;
        for (depth, &id) in self.path.iter().enumerate().rev() {
            let node = &mut self.nodes[id];
            let previous = node.ln_evidence;
            node.observe(byte);
            node.ln_split += child_delta;
            node.ln_evidence = if depth == leaf_depth {
                node.ln_stop
            } else {
                log_add_exp(node.ln_stop, node.ln_split) - LN_2
            };
            child_delta = node.ln_evidence - previous;
            self.node_updates += 1;
        }
        self.program.observe(byte);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{baselines::Kt, evaluate};

    #[test]
    fn zero_depth_is_kt_and_all_predictions_are_normalized() {
        let mut model = PartitionDfa::new(ByteHistory::new(0));
        let mut kt = Kt::default();
        for &byte in b"banana bandana" {
            for candidate in 0..=255 {
                assert!(
                    (model.predict().ln_prob(&candidate) - kt.predict().ln_prob(&candidate)).abs()
                        < 1e-12
                );
            }
            model.observe(byte);
            kt.observe(byte);
        }
        for depth in [1, 3, 8] {
            let mut model = PartitionDfa::new(ByteHistory::new(depth));
            for &byte in b"abracadabra abracadabra" {
                let sum: f64 = (0..=255).map(|b| model.predict().ln_prob(&b).exp()).sum();
                assert!((sum - 1.0).abs() < 1e-12);
                let before = model.ln_evidence();
                let prediction = model.predict().ln_prob(&byte);
                model.observe(byte);
                assert!((model.ln_evidence() - before - prediction).abs() < 1e-12);
            }
        }
    }

    #[test]
    fn one_level_matches_independent_explicit_two_model_mixture() {
        let data = b"abracadabra abracadabra";
        let mut global = Kt::default();
        let mut conditional: HashMap<u16, Kt> = HashMap::new();
        let (mut ln_global, mut ln_conditional) = (0.0, 0.0);
        let mut previous = 256;
        let mut model = PartitionDfa::new(ByteHistory::new(1));
        for &byte in data {
            ln_global += global.predict().ln_prob(&byte);
            global.observe(byte);
            let child = conditional.entry(previous).or_default();
            ln_conditional += child.predict().ln_prob(&byte);
            child.observe(byte);
            model.observe(byte);
            let expected = log_add_exp(ln_global, ln_conditional) - LN_2;
            assert!((model.ln_evidence() - expected).abs() < 1e-11);
            previous = u16::from(byte);
        }
    }

    #[test]
    fn streaming_chunks_preserve_posterior_and_evidence() {
        let data = b"some repeated text some repeated text";
        let mut whole = PartitionDfa::new(ByteHistory::new(4));
        let expected = evaluate(&data[..], &mut whole).unwrap().total_nats;
        let mut chunked = PartitionDfa::new(ByteHistory::new(4));
        let a = evaluate(&data[..11], &mut chunked).unwrap().total_nats;
        let b = evaluate(&data[11..], &mut chunked).unwrap().total_nats;
        assert!((a + b - expected).abs() < 1e-11);
        assert!((expected + whole.ln_evidence()).abs() < 1e-11);
        assert_eq!(whole.node_updates(), 5 * data.len() as u64);
    }

    #[test]
    fn two_level_binary_state_matches_all_five_partition_descriptions() {
        struct Bits([u16; 2]);
        impl StateProgram for Bits {
            fn features(&self) -> &[u16] {
                &self.0
            }
            fn observe(&mut self, byte: u8) {
                self.0 = [u16::from(byte & 1), self.0[0]];
            }
        }
        let mut posterior = PartitionDfa::new(Bits([0, 0]));
        let mut states = Bits([0, 0]);
        let mut leaves: Vec<HashMap<usize, Kt>> = (0..5).map(|_| HashMap::new()).collect();
        let mut masses = [-LN_2, -3.0 * LN_2, -3.0 * LN_2, -3.0 * LN_2, -3.0 * LN_2];
        for &byte in b"aabbabababbaaabbbab" {
            for description in 0..5 {
                let [first, second] = states.0.map(usize::from);
                let key = if description == 0 {
                    0
                } else if ((description - 1) >> first) & 1 == 0 {
                    1 + first
                } else {
                    3 + 2 * first + second
                };
                let leaf = leaves[description].entry(key).or_default();
                masses[description] += leaf.predict().ln_prob(&byte);
                leaf.observe(byte);
            }
            posterior.observe(byte);
            states.observe(byte);
            let exact = masses.iter().copied().fold(f64::NEG_INFINITY, log_add_exp);
            assert!((posterior.ln_evidence() - exact).abs() < 1e-11);
        }
    }
}
