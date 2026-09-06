//! Streaming raw-byte prequential evaluation.
use std::io::{self, Read};

use crate::{Distribution, Model};

/// Ideal coding cost; no actual arithmetic-coded file is produced.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Evaluation {
    pub bytes: u64,
    pub total_nats: f64,
}

impl Evaluation {
    pub fn total_bits(&self) -> f64 {
        self.total_nats / std::f64::consts::LN_2
    }

    /// Coding ratio against the uniform byte model; higher is better.
    ///
    /// The ratio is uniform coding cost divided by this model's coding cost.
    /// It is therefore independent of logarithm base. A value of 1.0 matches
    /// uniform coding, values above 1.0 compress better, and values below 1.0
    /// are worse. Empty input has no defined ratio.
    pub fn coding_ratio_uniform(&self) -> Option<f64> {
        (self.bytes != 0).then(|| {
            let uniform_nats = self.bytes as f64 * 8.0 * std::f64::consts::LN_2;
            uniform_nats / self.total_nats
        })
    }

    /// Coding ratio against another evaluation of the same byte count.
    ///
    /// This returns baseline_cost / self_cost, so higher means this evaluation
    /// codes the data better than the baseline. Different or empty byte counts
    /// are not comparable and return None.
    pub fn coding_ratio_against(&self, baseline: &Self) -> Option<f64> {
        (self.bytes != 0 && self.bytes == baseline.bytes)
            .then(|| baseline.total_nats / self.total_nats)
    }
}

/// Evaluate a raw byte stream without retaining per-byte costs.
///
/// To evaluate a prefix, pass `reader.take(limit)`. The model is not reset: a
/// caller may continue an existing model across calls, but each report covers
/// only the bytes consumed by that call.
pub fn evaluate(reader: impl Read, model: &mut impl Model<u8>) -> io::Result<Evaluation> {
    evaluate_with_costs(reader, model, |_| Ok(()))
}

/// Predict, score, send the cost in nats to `on_cost`, then observe each byte.
///
/// The callback can collect a Vec or stream costs to disk. Memory usage is
/// constant apart from model/callback storage. Zero probability gives infinite
/// cost, which remains infinite for the rest of the evaluation; no clipping.
/// NaN or positive log probabilities return `InvalidData` before observing that
/// byte. Read/callback errors propagate; earlier observations are not rolled back.
/// Reading is buffered and may consume bytes beyond the last scored byte on error.
pub fn evaluate_with_costs(
    mut reader: impl Read,
    model: &mut impl Model<u8>,
    mut on_cost: impl FnMut(f64) -> io::Result<()>,
) -> io::Result<Evaluation> {
    let mut report = Evaluation::default();
    let mut correction = 0.0;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let length = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(length) => length,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };
        for &byte in &buffer[..length] {
            let log_probability = {
                let prediction = model.predict();
                prediction.ln_prob(&byte)
            };
            if log_probability.is_nan() || log_probability > 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid ln_prob {log_probability} at byte {}", report.bytes),
                ));
            }
            let cost = -log_probability;
            on_cost(cost)?;
            // Compensated summation for long streams; avoid inf - inf becoming NaN.
            if cost.is_infinite() || report.total_nats.is_infinite() {
                report.total_nats = f64::INFINITY;
                correction = 0.0;
            } else {
                let adjusted = cost - correction;
                let next = report.total_nats + adjusted;
                correction = (next - report.total_nats) - adjusted;
                report.total_nats = next;
            }
            model.observe(byte);
            report.bytes += 1;
        }
    }
    Ok(report)
}
