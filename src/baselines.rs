//! Small byte baselines for checking and exercising the harness.
use crate::{Distribution, Model};

/// Fixed uniform distribution over all 256 byte values: coding ratio 1 to itself.
#[derive(Debug, Default, Clone, Copy)]
pub struct Uniform;

impl Distribution<u8> for Uniform {
    fn ln_prob(&self, _: &u8) -> f64 {
        -8.0 * std::f64::consts::LN_2
    }
}

impl Model<u8> for Uniform {
    fn predict(&self) -> impl Distribution<u8> {
        *self
    }

    fn observe(&mut self, _: u8) {}
}

/// Online byte unigram with symmetric Dirichlet(1/2, ..., 1/2) prior.
///
/// Predictive probability: `(count[byte] + 0.5) / (total + 128)`.
/// Starts from scratch and has full support; counts are never reset implicitly.
#[derive(Debug, Clone)]
pub struct Kt {
    counts: [u64; 256],
    total: u64,
}

impl Default for Kt {
    fn default() -> Self {
        Self {
            counts: [0; 256],
            total: 0,
        }
    }
}

struct KtPrediction<'a> {
    counts: &'a [u64; 256],
    total: u64,
}

impl Distribution<u8> for KtPrediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        ((self.counts[*byte as usize] as f64 + 0.5) / (self.total as f64 + 128.0)).ln()
    }
}

impl Model<u8> for Kt {
    fn predict(&self) -> impl Distribution<u8> {
        KtPrediction {
            counts: &self.counts,
            total: self.total,
        }
    }

    fn observe(&mut self, byte: u8) {
        self.total = self
            .total
            .checked_add(1)
            .expect("KT observation count overflow");
        self.counts[byte as usize] += 1;
    }
}
