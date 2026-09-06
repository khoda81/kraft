use kraft::{
    Distribution, Model,
    models::mixture::Mixture,
};

#[derive(Debug, Clone, Copy)]
struct Bernoulli {
    p_one: f64,
}

#[derive(Debug, Clone, Copy)]
struct BernoulliPrediction {
    p_one: f64,
}

impl Distribution<u8> for BernoulliPrediction {
    fn ln_prob(&self, observation: &u8) -> f64 {
        match observation {
            0 => (1.0 - self.p_one).ln(),
            1 => self.p_one.ln(),
            _ => f64::NEG_INFINITY,
        }
    }
}

impl Model<u8> for Bernoulli {
    fn predict(&self) -> impl Distribution<u8> {
        BernoulliPrediction { p_one: self.p_one }
    }

    fn observe(&mut self, _: u8) {}
}

#[test]
fn starts_with_equal_prior_and_averages_predictions() {
    let mixture = Mixture::new(Bernoulli { p_one: 0.8 }, Bernoulli { p_one: 0.2 });
    assert!((mixture.weight_a() - 0.5).abs() < 1e-15);
    assert!((mixture.weight_b() - 0.5).abs() < 1e-15);
    assert!(mixture.log_weight_ratio().abs() < 1e-15);

    let probability = mixture.predict().ln_prob(&1).exp();
    assert!((probability - 0.5).abs() < 1e-15);
}

#[test]
fn coding_advantage_multiplies_the_posterior_weight_ratio() {
    let mut mixture = Mixture::new(Bernoulli { p_one: 0.8 }, Bernoulli { p_one: 0.2 });

    mixture.observe(1);

    assert!((mixture.log_weight_ratio() - 4.0_f64.ln()).abs() < 1e-15);
    assert!((mixture.weight_a() - 0.8).abs() < 1e-15);
    assert!((mixture.weight_b() - 0.2).abs() < 1e-15);

    let next_probability = mixture.predict().ln_prob(&1).exp();
    assert!((next_probability - 0.68).abs() < 1e-15);
}

#[test]
fn sequential_prediction_matches_exact_bayesian_model_average() {
    let mut mixture = Mixture::new(Bernoulli { p_one: 0.8 }, Bernoulli { p_one: 0.2 });
    let mut sequence_ln_prob = 0.0;

    for observation in [1, 1] {
        sequence_ln_prob += mixture.predict().ln_prob(&observation);
        mixture.observe(observation);
    }

    // 1/2 * 0.8^2 + 1/2 * 0.2^2 = 0.34.
    assert!((sequence_ln_prob.exp() - 0.34).abs() < 1e-15);
}

#[test]
fn impossible_observation_for_one_model_gives_the_other_full_weight() {
    let mut mixture = Mixture::new(Bernoulli { p_one: 1.0 }, Bernoulli { p_one: 0.5 });

    mixture.observe(0);

    assert_eq!(mixture.log_weight_ratio(), f64::NEG_INFINITY);
    assert_eq!(mixture.weight_a(), 0.0);
    assert_eq!(mixture.weight_b(), 1.0);
    assert!((mixture.predict().ln_prob(&1).exp() - 0.5).abs() < 1e-15);
}
