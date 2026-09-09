//! Proper Bayesian prior/posterior over recurrent byte-input DFAs.
//!
//! The prior is defined on all finite labeled DFAs:
//!
//! ```text
//! P(N) = 2^-N, N >= 1
//! P(delta | N) = N^(-256N)
//! ```
//!
//! where the start state is fixed to label zero and every transition destination
//! is independently uniform over the N labels. Each state's byte-emission
//! probabilities are integrated under the symmetric Dirichlet-1/2 prior.
//!
//! ExactDfaPriorPosterior evaluates the finite prefix N=1..=max_states exactly
//! using the lazy partial-DFA oracle. The omitted state-count tail remains part of
//! the declared unbounded prior. Since data likelihood is at most one, its
//! unnormalized posterior mass is always bounded by
//!
//! ```text
//! sum_{N>max_states} 2^-N = 2^-max_states.
//! ```
//!
//! This yields a certified one-sided KL bound between the retained posterior
//! conditioned on N<=max_states and the full unbounded DFA posterior.

use std::{error::Error, fmt};

use crate::{Distribution, Model};

use super::partial_dfa::{
    DfaPosteriorComponent, DfaQuotient, ExactPartialDfaMixture, PartialDfaError,
};

const LN_2: f64 = std::f64::consts::LN_2;

/// Construction errors for the unbounded DFA prior with an exact finite prefix.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DfaPriorError {
    /// At least one state-count class must be evaluated.
    ZeroMaxStates,
    /// One of the fixed-N exact oracles could not be constructed.
    PartialDfa(PartialDfaError),
}

impl fmt::Display for DfaPriorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroMaxStates => write!(f, "DFA prior needs max_states >= 1"),
            Self::PartialDfa(error) => write!(f, "{error}"),
        }
    }
}

impl Error for DfaPriorError {}

impl From<PartialDfaError> for DfaPriorError {
    fn from(error: PartialDfaError) -> Self {
        Self::PartialDfa(error)
    }
}

/// Posterior summary for one exact fixed-state-count class.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DfaStateCountPosterior {
    /// Number of labeled DFA states in this class.
    pub states: u16,
    /// Natural-log prior mass ln P(N) = -N ln 2.
    pub ln_prior: f64,
    /// Exact natural-log evidence conditional on this state count.
    pub ln_evidence: f64,
    /// Posterior mass among the explicitly evaluated state-count classes.
    pub retained_posterior_mass: f64,
    /// Number of canonical transition/emission posterior components in this class.
    pub components: usize,
}

/// One high-mass sufficient-state component across the evaluated DFA classes.
#[derive(Debug, Clone, PartialEq)]
pub struct DfaPriorPosteriorComponent {
    pub states: u16,
    /// Posterior mass conditioned on N<=max_states.
    pub retained_posterior_mass: f64,
    /// Posterior mass conditioned on this fixed-N class.
    pub within_class_posterior_mass: f64,
    pub component: DfaPosteriorComponent,
}

/// Diagnostics for the exact evaluated prefix of the unbounded DFA prior.
#[derive(Debug, Clone, PartialEq)]
pub struct DfaPriorDiagnostics {
    /// Largest explicitly evaluated state count.
    pub max_states: u16,
    /// Natural-log total joint mass of all evaluated classes.
    pub ln_active_joint_mass: f64,
    /// Exact prior mass of all omitted N > max_states classes.
    pub omitted_prior_mass_upper: f64,
    /// Natural-log upper bound on omitted unnormalized posterior/joint mass.
    pub ln_omitted_joint_mass_upper: f64,
    /// Certified upper bound on omitted posterior mass after the observed prefix.
    pub omitted_posterior_mass_upper: f64,
    /// Certified upper bound on D_KL(Q_retained || P_full), in nats.
    pub retained_to_full_kl_upper_nats: f64,
    /// Exact total canonical component count across evaluated classes.
    pub components: usize,
    /// State-count posterior conditioned on the evaluated prefix of classes.
    pub state_counts: Vec<DfaStateCountPosterior>,
}

/// Exact finite-prefix inference under a proper prior over all finite DFAs.
///
/// The Bayesian model itself is unbounded in N. This implementation evaluates
/// 1..=max_states exactly and retains a certified bound for all larger N.
#[derive(Debug, Clone)]
pub struct ExactDfaPriorPosterior {
    quotient: DfaQuotient,
    classes: Vec<ExactPartialDfaMixture>,
    global_byte_counts: [u64; 256],
    ln_likelihood_upper: f64,
}

impl ExactDfaPriorPosterior {
    /// Evaluate state-count classes 1..=max_states with the discovery quotient.
    pub fn new(max_states: u16) -> Result<Self, DfaPriorError> {
        Self::with_quotient(max_states, DfaQuotient::Discovery)
    }

    /// Evaluate state-count classes 1..=max_states with an explicit exact quotient.
    pub fn with_quotient(max_states: u16, quotient: DfaQuotient) -> Result<Self, DfaPriorError> {
        if max_states == 0 {
            return Err(DfaPriorError::ZeroMaxStates);
        }

        let mut classes = Vec::with_capacity(usize::from(max_states));
        for states in 1..=max_states {
            classes.push(ExactPartialDfaMixture::with_quotient(states, quotient)?);
        }

        Ok(Self {
            quotient,
            classes,
            global_byte_counts: [0; 256],
            ln_likelihood_upper: 0.0,
        })
    }

    /// Largest state-count class evaluated exactly.
    pub fn max_states(&self) -> u16 {
        u16::try_from(self.classes.len()).expect("max state count is bounded by u16")
    }

    /// Exact quotient used inside every fixed-N oracle.
    pub fn quotient(&self) -> DfaQuotient {
        self.quotient
    }

    /// Unary state-count prior: P(N) = 2^-N.
    pub fn ln_state_count_prior(states: u16) -> f64 {
        -f64::from(states) * LN_2
    }

    /// Exact prior mass of all state-count classes larger than max_states.
    pub fn omitted_prior_mass_upper(&self) -> f64 {
        (-f64::from(self.max_states()) * LN_2).exp()
    }

    /// Exact natural-log joint mass of the evaluated classes.
    pub fn ln_active_joint_mass(&self) -> f64 {
        log_sum_exp(
            self.classes
                .iter()
                .map(|class| Self::ln_state_count_prior(class.state_count()) + class.ln_evidence()),
        )
    }

    /// Total exact canonical components over all evaluated state-count classes.
    pub fn component_count(&self) -> usize {
        self.classes
            .iter()
            .map(ExactPartialDfaMixture::component_count)
            .sum()
    }

    /// Number of unmerged children the next observation would generate in total.
    pub fn prospective_child_count(&self, byte: u8) -> usize {
        self.classes
            .iter()
            .map(|class| class.prospective_child_count(byte))
            .sum()
    }

    /// Highest-mass exact sufficient-state components across evaluated classes.
    ///
    /// Masses are posterior probabilities conditioned on N<=max_states.
    pub fn top_components(&self, limit: usize) -> Vec<DfaPriorPosteriorComponent> {
        if limit == 0 {
            return Vec::new();
        }

        let ln_active_joint_mass = self.ln_active_joint_mass();
        let mut candidates = Vec::new();

        for class in &self.classes {
            let states = class.state_count();
            let ln_class_joint = Self::ln_state_count_prior(states) + class.ln_evidence();
            let class_posterior_mass = (ln_class_joint - ln_active_joint_mass).exp();

            for component in class.top_components(limit) {
                candidates.push(DfaPriorPosteriorComponent {
                    states,
                    retained_posterior_mass: class_posterior_mass
                        * component.conditional_posterior_mass,
                    within_class_posterior_mass: component.conditional_posterior_mass,
                    component,
                });
            }
        }

        candidates.sort_by(|left, right| {
            right
                .retained_posterior_mass
                .total_cmp(&left.retained_posterior_mass)
        });
        candidates.truncate(limit);
        candidates
    }

    /// Per-state-count prospective child counts.
    pub fn prospective_child_counts(&self, byte: u8) -> Vec<(u16, usize)> {
        self.classes
            .iter()
            .map(|class| (class.state_count(), class.prospective_child_count(byte)))
            .collect()
    }

    /// Exact posterior on evaluated classes plus a certified omitted-tail bound.
    pub fn diagnostics(&self) -> DfaPriorDiagnostics {
        let ln_active_joint_mass = self.ln_active_joint_mass();
        let ln_tail_prior = -f64::from(self.max_states()) * LN_2;
        let ln_omitted_joint_mass_upper = ln_tail_prior + self.ln_likelihood_upper;
        let ln_ratio = ln_omitted_joint_mass_upper - ln_active_joint_mass;

        let retained_to_full_kl_upper_nats = softplus(ln_ratio);
        let omitted_posterior_mass_upper = logistic(ln_ratio);
        let omitted_prior_mass_upper = ln_tail_prior.exp();

        let state_counts = self
            .classes
            .iter()
            .map(|class| {
                let states = class.state_count();
                let ln_prior = Self::ln_state_count_prior(states);
                let ln_evidence = class.ln_evidence();
                DfaStateCountPosterior {
                    states,
                    ln_prior,
                    ln_evidence,
                    retained_posterior_mass: (ln_prior + ln_evidence - ln_active_joint_mass).exp(),
                    components: class.component_count(),
                }
            })
            .collect();

        DfaPriorDiagnostics {
            max_states: self.max_states(),
            ln_active_joint_mass,
            omitted_prior_mass_upper,
            ln_omitted_joint_mass_upper,
            omitted_posterior_mass_upper,
            retained_to_full_kl_upper_nats,
            components: self.component_count(),
            state_counts,
        }
    }

    fn observe_exact(&mut self, byte: u8) {
        let global_count = self.global_byte_counts[usize::from(byte)];
        let numerator = global_count as f64 + 0.5;
        let denominator = global_count as f64 + 128.0;
        self.ln_likelihood_upper += (numerator / denominator).ln();
        self.global_byte_counts[usize::from(byte)] = global_count
            .checked_add(1)
            .expect("global DFA-prior byte count overflow");

        for class in &mut self.classes {
            class.observe(byte);
        }
    }
}

struct ExactDfaPriorPrediction<'a> {
    posterior: &'a ExactDfaPriorPosterior,
    ln_active_joint_mass: f64,
}

impl Distribution<u8> for ExactDfaPriorPrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        log_sum_exp(self.posterior.classes.iter().map(|class| {
            ExactDfaPriorPosterior::ln_state_count_prior(class.state_count())
                + class.ln_evidence()
                + class.predict().ln_prob(byte)
        })) - self.ln_active_joint_mass
    }
}

impl Model<u8> for ExactDfaPriorPosterior {
    fn predict(&self) -> impl Distribution<u8> {
        ExactDfaPriorPrediction {
            posterior: self,
            ln_active_joint_mass: self.ln_active_joint_mass(),
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

fn softplus(value: f64) -> f64 {
    if value > 0.0 {
        value + (-value).exp().ln_1p()
    } else {
        value.exp().ln_1p()
    }
}

fn logistic(value: f64) -> f64 {
    if value >= 0.0 {
        1.0 / (1.0 + (-value).exp())
    } else {
        let exp = value.exp();
        exp / (1.0 + exp)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::baselines::Kt;

    #[test]
    fn unary_state_count_prior_is_proper_with_exact_tail() {
        for maximum in 1..=16 {
            let explicit: f64 = (1..=maximum)
                .map(|states| ExactDfaPriorPosterior::ln_state_count_prior(states).exp())
                .sum();
            let tail = (-f64::from(maximum) * LN_2).exp();
            assert!((explicit + tail - 1.0).abs() < 1e-15);
        }
    }

    #[test]
    fn initial_state_count_posterior_matches_truncated_unary_prior() {
        let posterior = ExactDfaPriorPosterior::new(3).unwrap();
        let diagnostics = posterior.diagnostics();
        let masses: Vec<_> = diagnostics
            .state_counts
            .iter()
            .map(|class| class.retained_posterior_mass)
            .collect();

        assert!((masses[0] - 4.0 / 7.0).abs() < 1e-14);
        assert!((masses[1] - 2.0 / 7.0).abs() < 1e-14);
        assert!((masses[2] - 1.0 / 7.0).abs() < 1e-14);
        assert!((diagnostics.omitted_prior_mass_upper - 1.0 / 8.0).abs() < 1e-14);
        assert!((diagnostics.retained_to_full_kl_upper_nats - (8.0_f64 / 7.0).ln()).abs() < 1e-14);
    }

    #[test]
    fn one_state_retained_class_is_exactly_kt() {
        let data = b"KRAFT";
        let mut dfa = ExactDfaPriorPosterior::new(1).unwrap();
        let mut kt = Kt::default();

        for &byte in data {
            assert!((dfa.predict().ln_prob(&byte) - kt.predict().ln_prob(&byte)).abs() < 1e-12);
            dfa.observe(byte);
            kt.observe(byte);
        }
    }

    #[test]
    fn truncated_predictive_distribution_is_normalized() {
        let mut posterior = ExactDfaPriorPosterior::new(3).unwrap();
        for &byte in b"ABACA" {
            let mass: f64 = (0..=255)
                .map(|candidate| posterior.predict().ln_prob(&candidate).exp())
                .sum();
            assert!((mass - 1.0).abs() < 1e-11);
            posterior.observe(byte);
        }
    }

    #[test]
    fn cumulative_prediction_matches_active_joint_evidence_ratio() {
        let mut posterior = ExactDfaPriorPosterior::new(2).unwrap();
        let initial_ln_mass = posterior.ln_active_joint_mass();
        let mut total_nats = 0.0;

        for &byte in b"mediawiki" {
            total_nats -= posterior.predict().ln_prob(&byte);
            posterior.observe(byte);
        }

        let evidence_nats = initial_ln_mass - posterior.ln_active_joint_mass();
        assert!((total_nats - evidence_nats).abs() < 1e-10);
    }

    #[test]
    fn posterior_tail_certificate_is_finite_and_valid_at_start() {
        let posterior = ExactDfaPriorPosterior::new(4).unwrap();
        let diagnostics = posterior.diagnostics();
        assert!((diagnostics.omitted_prior_mass_upper - 1.0 / 16.0).abs() < 1e-14);
        assert!((diagnostics.ln_omitted_joint_mass_upper - (1.0_f64 / 16.0).ln()).abs() < 1e-14);
        assert!(diagnostics.omitted_posterior_mass_upper > 0.0);
        assert!(diagnostics.omitted_posterior_mass_upper < 1.0);
        assert!(diagnostics.retained_to_full_kl_upper_nats > 0.0);
    }

    #[test]
    fn universal_emission_bound_tracks_the_first_observation_exactly() {
        let mut posterior = ExactDfaPriorPosterior::new(3).unwrap();
        posterior.observe(b'A');
        let diagnostics = posterior.diagnostics();
        let expected_tail_joint = (1.0_f64 / 8.0) * (1.0 / 256.0);
        assert!((diagnostics.ln_omitted_joint_mass_upper - expected_tail_joint.ln()).abs() < 1e-14);
    }
}
