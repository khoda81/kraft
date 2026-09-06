//! Exact scalar finite-state predictors used as the first KRAFT oracle.
//!
//! The model family is deliberately simple: a labeled binary transition table
//! chooses the next state after each observed bit, while each state's emission
//! probability is integrated with a Jeffreys Beta(1/2, 1/2) prior. Raw bytes
//! are processed most-significant bit first.

use std::{error::Error, fmt};

use crate::{Distribution, Model};

/// Errors constructing a finite-state predictor or exact finite mixture.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FsmError {
    EmptyTable,
    TooManyStates { state_count: usize },
    InvalidTarget {
        state: usize,
        bit: u8,
        target: u16,
        state_count: u16,
    },
    TableCountOverflow { state_count: u16 },
    TooManyModels { required: u128, limit: usize },
    EmptyMixture,
}

impl fmt::Display for FsmError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyTable => write!(formatter, "an FSM needs at least one state"),
            Self::TooManyStates { state_count } => {
                write!(formatter, "{state_count} states do not fit in a u16 state id")
            }
            Self::InvalidTarget {
                state,
                bit,
                target,
                state_count,
            } => write!(
                formatter,
                "transition ({state}, {bit}) -> {target} is outside 0..{state_count}"
            ),
            Self::TableCountOverflow { state_count } => write!(
                formatter,
                "the number of labeled {state_count}-state transition tables exceeds u128"
            ),
            Self::TooManyModels { required, limit } => write!(
                formatter,
                "exact mixture needs {required} models, above the requested limit {limit}"
            ),
            Self::EmptyMixture => write!(formatter, "an exact mixture needs at least one model"),
        }
    }
}

impl Error for FsmError {}

/// A deterministic binary transition table.
///
/// State labels are significant in this first oracle. The table has two targets
/// per state, indexed by the observed bit. State zero is always the initial state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TransitionTable {
    transitions: Vec<[u16; 2]>,
}

impl TransitionTable {
    pub fn new(transitions: Vec<[u16; 2]>) -> Result<Self, FsmError> {
        if transitions.is_empty() {
            return Err(FsmError::EmptyTable);
        }
        let state_count =
            u16::try_from(transitions.len()).map_err(|_| FsmError::TooManyStates {
                state_count: transitions.len(),
            })?;
        for (state, targets) in transitions.iter().enumerate() {
            for (bit, &target) in targets.iter().enumerate() {
                if target >= state_count {
                    return Err(FsmError::InvalidTarget {
                        state,
                        bit: bit as u8,
                        target,
                        state_count,
                    });
                }
            }
        }
        Ok(Self { transitions })
    }

    pub fn state_count(&self) -> u16 {
        self.transitions.len() as u16
    }

    pub fn next(&self, state: u16, bit: u8) -> u16 {
        debug_assert!(state < self.state_count());
        debug_assert!(bit <= 1);
        self.transitions[usize::from(state)][usize::from(bit)]
    }

    /// Decode a labeled table from its base-N rank.
    pub fn from_rank(state_count: u16, mut rank: u128) -> Option<Self> {
        let table_count = labeled_table_count(state_count)?;
        if state_count == 0 || rank >= table_count {
            return None;
        }
        let radix = u128::from(state_count);
        let mut transitions = vec![[0_u16; 2]; usize::from(state_count)];
        for targets in &mut transitions {
            for target in targets {
                *target = (rank % radix) as u16;
                rank /= radix;
            }
        }
        Some(Self { transitions })
    }

    /// Return the stable base-N rank when it fits in u128.
    pub fn rank(&self) -> Option<u128> {
        let radix = u128::from(self.state_count());
        let digits = self.transitions.len() * 2;
        let mut rank = 0_u128;
        let mut factor = 1_u128;
        let mut index = 0_usize;
        for targets in &self.transitions {
            for &target in targets {
                rank = rank.checked_add(factor.checked_mul(u128::from(target))?)?;
                index += 1;
                if index != digits {
                    factor = factor.checked_mul(radix)?;
                }
            }
        }
        Some(rank)
    }
}

/// Number of labeled deterministic binary transition tables with N states.
///
/// There are N^(2N) tables. Zero states is represented as an empty family.
pub fn labeled_table_count(state_count: u16) -> Option<u128> {
    if state_count == 0 {
        return Some(0);
    }
    u128::from(state_count).checked_pow(2 * u32::from(state_count))
}

/// A binary finite-state predictor with integrated per-state Bernoulli emissions.
///
/// Before each bit, the current state predicts with the posterior predictive
/// distribution from a Beta(1/2, 1/2) prior and the zero/one counts previously
/// observed in that state. The bit is then counted and selects the next state.
/// Byte probabilities factor eight bit predictions from MSB to LSB.
#[derive(Debug, Clone)]
pub struct BinaryKtFsm {
    table: TransitionTable,
    state: u16,
    counts: Vec<[u64; 2]>,
}

impl BinaryKtFsm {
    pub fn new(table: TransitionTable) -> Self {
        let state_count = usize::from(table.state_count());
        Self {
            table,
            state: 0,
            counts: vec![[0_u64; 2]; state_count],
        }
    }

    pub fn from_rank(state_count: u16, rank: u128) -> Option<Self> {
        TransitionTable::from_rank(state_count, rank).map(Self::new)
    }

    pub fn table(&self) -> &TransitionTable {
        &self.table
    }

    pub fn state(&self) -> u16 {
        self.state
    }

    pub fn counts(&self) -> &[[u64; 2]] {
        &self.counts
    }

    /// Exact natural-log probability of one byte under the current prefix state.
    pub fn byte_ln_prob(&self, byte: u8) -> f64 {
        let mut visited_states = [0_u16; 8];
        let mut visited_bits = [0_u8; 8];
        let mut visited = 0_usize;
        let mut state = self.state;
        let mut total_ln_probability = 0.0;

        for shift in (0..8).rev() {
            let bit = (byte >> shift) & 1;
            let bit_index = usize::from(bit);
            let state_index = usize::from(state);

            let mut extra_total = 0_u8;
            let mut extra_bit = 0_u8;
            for index in 0..visited {
                if visited_states[index] == state {
                    extra_total += 1;
                    if visited_bits[index] == bit {
                        extra_bit += 1;
                    }
                }
            }

            let base = self.counts[state_index];
            let numerator = base[bit_index] as f64 + f64::from(extra_bit) + 0.5;
            let denominator =
                base[0] as f64 + base[1] as f64 + f64::from(extra_total) + 1.0;
            total_ln_probability += (numerator / denominator).ln();

            visited_states[visited] = state;
            visited_bits[visited] = bit;
            visited += 1;
            state = self.table.next(state, bit);
        }

        total_ln_probability
    }

    fn observe_byte(&mut self, byte: u8) {
        for shift in (0..8).rev() {
            let bit = (byte >> shift) & 1;
            let state_index = usize::from(self.state);
            let bit_index = usize::from(bit);
            self.counts[state_index][bit_index] = self.counts[state_index][bit_index]
                .checked_add(1)
                .expect("finite-state emission count overflow");
            self.state = self.table.next(self.state, bit);
        }
    }
}

struct BinaryKtFsmPrediction<'a>(&'a BinaryKtFsm);

impl Distribution<u8> for BinaryKtFsmPrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        self.0.byte_ln_prob(*byte)
    }
}

impl Model<u8> for BinaryKtFsm {
    fn predict(&self) -> impl Distribution<u8> {
        BinaryKtFsmPrediction(self)
    }

    fn observe(&mut self, byte: u8) {
        self.observe_byte(byte);
    }
}

/// Exact Bayesian mixture over a finite list of FSMs with a uniform model prior.
#[derive(Debug, Clone)]
pub struct ExactFsmMixture {
    models: Vec<BinaryKtFsm>,
    log_weights: Vec<f64>,
}

impl ExactFsmMixture {
    pub fn uniform(models: Vec<BinaryKtFsm>) -> Result<Self, FsmError> {
        if models.is_empty() {
            return Err(FsmError::EmptyMixture);
        }
        let log_prior = -(models.len() as f64).ln();
        let log_weights = vec![log_prior; models.len()];
        Ok(Self {
            models,
            log_weights,
        })
    }

    /// Enumerate all N^(2N) labeled transition tables, guarded by max_models.
    pub fn all_labeled(state_count: u16, max_models: usize) -> Result<Self, FsmError> {
        if state_count == 0 {
            return Err(FsmError::EmptyTable);
        }
        let required = labeled_table_count(state_count)
            .ok_or(FsmError::TableCountOverflow { state_count })?;
        if required > max_models as u128 {
            return Err(FsmError::TooManyModels {
                required,
                limit: max_models,
            });
        }
        let mut models = Vec::with_capacity(required as usize);
        for rank in 0..required {
            models.push(
                BinaryKtFsm::from_rank(state_count, rank)
                    .expect("rank is inside the previously checked table count"),
            );
        }
        Self::uniform(models)
    }

    pub fn model_count(&self) -> usize {
        self.models.len()
    }

    /// Natural-log posterior weights, normalized up to floating-point error.
    pub fn posterior_log_weights(&self) -> &[f64] {
        &self.log_weights
    }

    pub fn posterior_weights(&self) -> Vec<f64> {
        self.log_weights.iter().map(|weight| weight.exp()).collect()
    }

    fn byte_ln_prob(&self, byte: u8) -> f64 {
        let numerator = log_sum_exp(
            self.models
                .iter()
                .zip(&self.log_weights)
                .map(|(model, weight)| *weight + model.byte_ln_prob(byte)),
        );
        numerator - log_sum_exp(self.log_weights.iter().copied())
    }
}

fn log_sum_exp(values: impl IntoIterator<Item = f64>) -> f64 {
    values
        .into_iter()
        .fold(f64::NEG_INFINITY, |accumulator, value| {
            if accumulator == f64::NEG_INFINITY {
                value
            } else if value == f64::NEG_INFINITY {
                accumulator
            } else if accumulator >= value {
                accumulator + (value - accumulator).exp().ln_1p()
            } else {
                value + (accumulator - value).exp().ln_1p()
            }
        })
}

struct ExactFsmMixturePrediction<'a>(&'a ExactFsmMixture);

impl Distribution<u8> for ExactFsmMixturePrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        self.0.byte_ln_prob(*byte)
    }
}

impl Model<u8> for ExactFsmMixture {
    fn predict(&self) -> impl Distribution<u8> {
        ExactFsmMixturePrediction(self)
    }

    fn observe(&mut self, byte: u8) {
        for (model, log_weight) in self.models.iter().zip(&mut self.log_weights) {
            *log_weight += model.byte_ln_prob(byte);
        }
        let normalizer = log_sum_exp(self.log_weights.iter().copied());
        for log_weight in &mut self.log_weights {
            *log_weight -= normalizer;
        }
        for model in &mut self.models {
            model.observe_byte(byte);
        }
    }
}
