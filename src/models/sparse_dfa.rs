//! Sparse recurrent byte-input DFAs with cheap default topology.
//!
//! A model has N states and one of three implicit transition skeletons:
//!
//! - stay:  d(s) = s
//! - next:  d(s) = min(s + 1, N - 1)
//! - cycle: d(s) = (s + 1) mod N
//!
//! A sparse set of byte-specific overrides replaces d(s). This makes persistent
//! recurrent state cheap while charging only for the exceptional transition
//! structure.
//!
//! The description prior is proper:
//!
//! P(N) = 1 / (N(N+1)), N >= 1
//! P(topology) = 1/3
//! P(K | N) proportional to 1 / ((K+1)(K+2)), 0 <= K <= 256N
//! P(keys | K,N) = 1 / choose(256N, K)
//! P(destinations | keys,N) = (N-1)^(-K)
//!
//! Destinations equal to the implicit default are excluded, so every stored
//! override changes the machine. For N=1 only K=0 exists.

use std::{error::Error, fmt};

const ALPHABET: usize = 256;
const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = 128.0;
const LOG_TWO_PI_HALF: f64 = 0.918_938_533_204_672_7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum DefaultTopology {
    Stay,
    Next,
    Cycle,
}

impl DefaultTopology {
    pub const ALL: [Self; 3] = [Self::Stay, Self::Next, Self::Cycle];

    pub fn as_str(self) -> &'static str {
        match self {
            Self::Stay => "stay",
            Self::Next => "next",
            Self::Cycle => "cycle",
        }
    }

    pub fn parse(value: &str) -> Option<Self> {
        match value {
            "stay" => Some(Self::Stay),
            "next" => Some(Self::Next),
            "cycle" => Some(Self::Cycle),
            _ => None,
        }
    }

    pub fn default_destination(self, state: u16, states: u16) -> u16 {
        debug_assert!(states > 0);
        debug_assert!(state < states);
        match self {
            Self::Stay => state,
            Self::Next => state.saturating_add(1).min(states - 1),
            Self::Cycle => {
                if state + 1 == states {
                    0
                } else {
                    state + 1
                }
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SparseOverride {
    pub source: u16,
    pub byte: u8,
    pub destination: u16,
}

impl SparseOverride {
    fn key(self) -> u32 {
        (u32::from(self.source) << 8) | u32::from(self.byte)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SparseDfaError {
    ZeroStates,
    TooManyStates,
    InvalidSource,
    InvalidDestination,
    DuplicateKey,
    RedundantOverride,
    TooManyOverrides,
}

impl fmt::Display for SparseDfaError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroStates => write!(f, "sparse DFA needs at least one state"),
            Self::TooManyStates => write!(f, "sparse DFA supports at most 65535 states"),
            Self::InvalidSource => write!(f, "override source is outside the state range"),
            Self::InvalidDestination => {
                write!(f, "override destination is outside the state range")
            }
            Self::DuplicateKey => write!(f, "sparse DFA has duplicate (state, byte) overrides"),
            Self::RedundantOverride => {
                write!(f, "override destination equals the implicit default")
            }
            Self::TooManyOverrides => write!(f, "override count exceeds the transition-key count"),
        }
    }
}

impl Error for SparseDfaError {}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct SparseDfa {
    states: u16,
    topology: DefaultTopology,
    overrides: Vec<SparseOverride>,
}

impl SparseDfa {
    pub fn new(
        states: u16,
        topology: DefaultTopology,
        mut overrides: Vec<SparseOverride>,
    ) -> Result<Self, SparseDfaError> {
        if states == 0 {
            return Err(SparseDfaError::ZeroStates);
        }

        let maximum_keys = usize::from(states) * ALPHABET;
        if overrides.len() > maximum_keys {
            return Err(SparseDfaError::TooManyOverrides);
        }

        overrides.sort_unstable_by_key(|edge| edge.key());
        let mut previous_key = None;
        for edge in &overrides {
            if edge.source >= states {
                return Err(SparseDfaError::InvalidSource);
            }
            if edge.destination >= states {
                return Err(SparseDfaError::InvalidDestination);
            }
            if edge.destination == topology.default_destination(edge.source, states) {
                return Err(SparseDfaError::RedundantOverride);
            }
            if previous_key == Some(edge.key()) {
                return Err(SparseDfaError::DuplicateKey);
            }
            previous_key = Some(edge.key());
        }

        Ok(Self {
            states,
            topology,
            overrides,
        })
    }

    pub fn empty(states: u16, topology: DefaultTopology) -> Result<Self, SparseDfaError> {
        Self::new(states, topology, Vec::new())
    }

    pub fn states(&self) -> u16 {
        self.states
    }

    pub fn topology(&self) -> DefaultTopology {
        self.topology
    }

    pub fn overrides(&self) -> &[SparseOverride] {
        &self.overrides
    }

    pub fn default_destination(&self, state: u16) -> u16 {
        self.topology.default_destination(state, self.states)
    }

    #[inline]
    pub fn destination(&self, state: u16, byte: u8) -> u16 {
        let key = (u32::from(state) << 8) | u32::from(byte);
        match self.overrides.binary_search_by_key(&key, |edge| edge.key()) {
            Ok(index) => self.overrides[index].destination,
            Err(_) => self.default_destination(state),
        }
    }

    pub fn with_added_override(&self, edge: SparseOverride) -> Result<Self, SparseDfaError> {
        let mut overrides = self.overrides.clone();
        overrides.push(edge);
        Self::new(self.states, self.topology, overrides)
    }

    pub fn without_override(&self, index: usize) -> Self {
        let mut overrides = self.overrides.clone();
        overrides.remove(index);
        Self {
            states: self.states,
            topology: self.topology,
            overrides,
        }
    }

    /// Natural-log prior probability of this complete sparse description.
    pub fn ln_prior(&self) -> f64 {
        let states = usize::from(self.states);
        let k = self.overrides.len();

        let ln_state_prior = -((states as f64).ln() + ((states + 1) as f64).ln());
        let ln_topology_prior = -(3.0_f64).ln();

        if states == 1 {
            debug_assert_eq!(k, 0);
            return ln_state_prior + ln_topology_prior;
        }

        let keys = states * ALPHABET;
        let ln_k_prior = ln_exception_count_prior(keys, k);
        let ln_key_prior = -ln_binomial(keys, k);
        let ln_destination_prior = -(k as f64) * ((states - 1) as f64).ln();

        ln_state_prior + ln_topology_prior + ln_k_prior + ln_key_prior + ln_destination_prior
    }

    pub fn prior_bits(&self) -> f64 {
        -self.ln_prior() / std::f64::consts::LN_2
    }

    /// Exact integrated Dirichlet-1/2 evidence of a byte sequence.
    pub fn ln_evidence(&self, data: &[u8]) -> f64 {
        self.score(data).ln_evidence
    }

    /// Score plus trajectory statistics useful to sparse-structure search.
    ///
    /// Scoring is the hot loop of the heuristic search, so compile the sparse
    /// transition program into a dense lookup table once per candidate. The
    /// same (state, byte) histogram is both the Bayesian sufficient statistic
    /// and the transition-visit statistic used by the search; keep only one
    /// counter array and derive state totals after the trajectory scan.
    pub fn score(&self, data: &[u8]) -> SparseDfaScore {
        let states = usize::from(self.states);
        let key_count = states * ALPHABET;

        let mut transitions = vec![0_u16; key_count];
        for state_index in 0..states {
            let state = state_index as u16;
            let default = self.default_destination(state);
            transitions[state_index * ALPHABET..(state_index + 1) * ALPHABET].fill(default);
        }
        for &edge in &self.overrides {
            transitions[edge.key() as usize] = edge.destination;
        }

        let mut visited_keys = vec![0_u64; key_count];
        let mut state = 0_u16;

        for &byte in data {
            let key = usize::from(state) * ALPHABET + usize::from(byte);
            visited_keys[key] += 1;
            state = transitions[key];
        }

        let state_counts = visited_keys.as_chunks::<ALPHABET>().0;
        let state_totals = state_counts
            .iter()
            .map(|state_counts| state_counts.iter().sum())
            .collect::<Vec<u64>>();

        let ln_evidence = state_counts
            .iter()
            .zip(&state_totals)
            .filter(|(_, total)| **total != 0)
            .map(|(state_counts, total)| state_ln_evidence(state_counts, *total))
            .sum();

        SparseDfaScore {
            ln_evidence,
            final_state: state,
            state_totals,
            visited_keys,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SparseDfaScore {
    pub ln_evidence: f64,
    pub final_state: u16,
    pub state_totals: Vec<u64>,
    /// Visit count of each (state,byte) key in row-major state*256+byte order.
    pub visited_keys: Vec<u64>,
}

pub fn ln_state_count_prior(states: u16) -> Result<f64, SparseDfaError> {
    if states == 0 {
        return Err(SparseDfaError::ZeroStates);
    }
    let states = f64::from(states);
    Ok(-(states.ln() + (states + 1.0).ln()))
}

/// Proper exception-count prior truncated to 0..=key_count.
///
/// Start from w_k = 1/((k+1)(k+2)); the finite normalizer is
/// sum_{k=0}^M w_k = (M+1)/(M+2).
pub fn ln_exception_count_prior(key_count: usize, exceptions: usize) -> f64 {
    assert!(exceptions <= key_count);
    let k = exceptions as f64;
    let m = key_count as f64;
    -((k + 1.0).ln() + (k + 2.0).ln()) - ((m + 1.0) / (m + 2.0)).ln()
}

fn ln_binomial(n: usize, k: usize) -> f64 {
    let k = k.min(n - k);
    (0..k)
        .map(|index| ((n - index) as f64).ln() - ((index + 1) as f64).ln())
        .sum()
}

fn ln_gamma(value: f64) -> f64 {
    debug_assert!(value >= 0.5);
    const COEFFICIENTS: [f64; 9] = [
        0.999_999_999_999_809_9,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_1,
        -176.615_029_162_140_6,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_572e-6,
        1.505_632_735_149_311_6e-7,
    ];

    let shifted = value - 1.0;
    let mut series = COEFFICIENTS[0];
    for (index, coefficient) in COEFFICIENTS.iter().enumerate().skip(1) {
        series += coefficient / (shifted + index as f64);
    }
    let t = shifted + 7.5;
    LOG_TWO_PI_HALF + (shifted + 0.5) * t.ln() - t + series.ln()
}

fn state_ln_evidence(counts: &[u64], total: u64) -> f64 {
    let mut result = ln_gamma(JEFFREYS_TOTAL) - ln_gamma(total as f64 + JEFFREYS_TOTAL);
    let prior = ln_gamma(JEFFREYS_ALPHA);
    for &count in counts {
        if count != 0 {
            result += ln_gamma(count as f64 + JEFFREYS_ALPHA) - prior;
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_count_prior_telescopes() {
        for maximum in [1_u16, 2, 3, 8, 64, 1024] {
            let explicit: f64 = (1..=maximum)
                .map(|states| ln_state_count_prior(states).unwrap().exp())
                .sum();
            let tail = 1.0 / (f64::from(maximum) + 1.0);
            assert!((explicit + tail - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn exception_count_prior_is_normalized() {
        for key_count in [1_usize, 2, 7, 256, 1024] {
            let total: f64 = (0..=key_count)
                .map(|k| ln_exception_count_prior(key_count, k).exp())
                .sum();
            assert!((total - 1.0).abs() < 1e-12);
        }
    }

    #[test]
    fn default_topologies_have_expected_transitions() {
        let stay = SparseDfa::empty(4, DefaultTopology::Stay).unwrap();
        let next = SparseDfa::empty(4, DefaultTopology::Next).unwrap();
        let cycle = SparseDfa::empty(4, DefaultTopology::Cycle).unwrap();

        assert_eq!(stay.destination(2, b'x'), 2);
        assert_eq!(next.destination(0, b'x'), 1);
        assert_eq!(next.destination(3, b'x'), 3);
        assert_eq!(cycle.destination(3, b'x'), 0);
    }

    #[test]
    fn sparse_override_replaces_only_its_key() {
        let dfa = SparseDfa::new(
            3,
            DefaultTopology::Stay,
            vec![SparseOverride {
                source: 0,
                byte: b'q',
                destination: 1,
            }],
        )
        .unwrap();

        assert_eq!(dfa.destination(0, b'q'), 1);
        assert_eq!(dfa.destination(0, b'x'), 0);
        assert_eq!(dfa.destination(1, b'q'), 1);
    }

    #[test]
    fn optimized_score_matches_sparse_lookup_reference() {
        let model = SparseDfa::new(
            5,
            DefaultTopology::Cycle,
            vec![
                SparseOverride {
                    source: 0,
                    byte: b'a',
                    destination: 3,
                },
                SparseOverride {
                    source: 3,
                    byte: b'b',
                    destination: 1,
                },
                SparseOverride {
                    source: 4,
                    byte: b' ',
                    destination: 2,
                },
            ],
        )
        .unwrap();
        let data = b"abracadabra abracadabra\n<xml>abba</xml>";

        let optimized = model.score(data);

        let mut visited_keys = vec![0_u64; usize::from(model.states()) * ALPHABET];
        let mut state_totals = vec![0_u64; usize::from(model.states())];
        let mut state = 0_u16;
        for &byte in data {
            let key = usize::from(state) * ALPHABET + usize::from(byte);
            visited_keys[key] += 1;
            state_totals[usize::from(state)] += 1;
            state = model.destination(state, byte);
        }
        let ln_evidence = visited_keys
            .as_chunks::<ALPHABET>()
            .0
            .iter()
            .zip(&state_totals)
            .filter(|(_, total)| **total != 0)
            .map(|(counts, total)| state_ln_evidence(counts, *total))
            .sum::<f64>();

        assert_eq!(optimized.final_state, state);
        assert_eq!(optimized.state_totals, state_totals);
        assert_eq!(optimized.visited_keys, visited_keys);
        assert!((optimized.ln_evidence - ln_evidence).abs() < 1e-12);
    }

    #[test]
    fn q_latch_has_persistent_state() {
        let dfa = SparseDfa::new(
            2,
            DefaultTopology::Stay,
            vec![SparseOverride {
                source: 0,
                byte: b'q',
                destination: 1,
            }],
        )
        .unwrap();
        let score = dfa.score(b"abqxxxxxxxx");

        assert_eq!(score.final_state, 1);
        assert_eq!(score.state_totals, vec![3, 8]);
    }

    #[test]
    fn redundant_and_duplicate_overrides_are_rejected() {
        assert_eq!(
            SparseDfa::new(
                2,
                DefaultTopology::Stay,
                vec![SparseOverride {
                    source: 0,
                    byte: b'a',
                    destination: 0,
                }],
            )
            .unwrap_err(),
            SparseDfaError::RedundantOverride
        );

        assert_eq!(
            SparseDfa::new(
                3,
                DefaultTopology::Stay,
                vec![
                    SparseOverride {
                        source: 0,
                        byte: b'a',
                        destination: 1,
                    },
                    SparseOverride {
                        source: 0,
                        byte: b'a',
                        destination: 2,
                    },
                ],
            )
            .unwrap_err(),
            SparseDfaError::DuplicateKey
        );
    }

    #[test]
    fn one_state_stay_is_byte_kt_evidence() {
        let data = b"banana";
        let dfa = SparseDfa::empty(1, DefaultTopology::Stay).unwrap();
        let mut counts = [0_u64; ALPHABET];
        for &byte in data {
            counts[usize::from(byte)] += 1;
        }
        let expected = state_ln_evidence(&counts, data.len() as u64);
        assert!((dfa.ln_evidence(data) - expected).abs() < 1e-12);
    }
}
