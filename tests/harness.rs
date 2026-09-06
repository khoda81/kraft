use kraft::{
    Distribution, Evaluation, Model,
    baselines::{Kt, Uniform},
    evaluate, evaluate_with_costs,
};
use std::{
    cell::Cell,
    io::{self, Read},
};

#[test]
fn raw_bytes_and_utf8_have_exact_uniform_cost() {
    let bytes: Vec<_> = (0..=255).chain("سلام 🦀\n".bytes()).collect();
    let result = evaluate(bytes.as_slice(), &mut Uniform).unwrap();
    assert_eq!(result.bytes, bytes.len() as u64);
    assert!((result.total_bits() - 8.0 * bytes.len() as f64).abs() < 1e-10);
    assert!((result.uniform_coding_ratio().unwrap() - 1.0).abs() < 1e-12);
    assert!((result.total_bytes() - bytes.len() as f64).abs() < 1e-10);
}

#[test]
fn empty_input_has_no_coding_ratio() {
    let report = evaluate(&b""[..], &mut Uniform).unwrap();
    assert_eq!(report.bytes, 0);
    assert_eq!(report.total_nats, 0.0);
    assert_eq!(report.uniform_coding_ratio(), None);
}

#[test]
fn kt_matches_independent_sequence_probability() {
    // Dirichlet integral for AAB: (1/2)(3/2)(1/2) / (128*129*130).
    let expected = -(0.5_f64 * 1.5 * 0.5 / (128.0 * 129.0 * 130.0)).ln();
    let mut costs = Vec::new();
    let report = evaluate_with_costs(&b"AAB"[..], &mut Kt::default(), |cost| {
        costs.push(cost);
        Ok(())
    })
    .unwrap();
    assert!((report.total_nats - expected).abs() < 1e-12);
    assert!((costs[0] - 256.0_f64.ln()).abs() < 1e-12);
    assert!((costs.iter().sum::<f64>() - expected).abs() < 1e-12);
}

struct Ordered {
    seen: usize,
    phase: Cell<u8>,
}
struct Prediction<'a>(&'a Ordered);
impl Distribution<u8> for Prediction<'_> {
    fn ln_prob(&self, byte: &u8) -> f64 {
        assert_eq!(self.0.phase.replace(2), 1);
        assert_eq!(*byte as usize, self.0.seen);
        0.0
    }
}
impl Model<u8> for Ordered {
    fn predict(&self) -> impl Distribution<u8> {
        assert_eq!(self.phase.replace(1), 0);
        Prediction(self)
    }
    fn observe(&mut self, byte: u8) {
        assert_eq!(self.phase.replace(0), 2);
        assert_eq!(byte as usize, self.seen);
        self.seen += 1;
    }
}

#[test]
fn prediction_can_borrow_and_is_scored_before_observation() {
    let mut model = Ordered {
        seen: 0,
        phase: Cell::new(0),
    };
    let report = evaluate(&[0, 1, 2][..], &mut model).unwrap();
    assert_eq!(report.bytes, 3);
    assert_eq!(model.seen, 3);
}

struct BadModel {
    value: f64,
    seen: usize,
}
struct Fixed(f64);
impl Distribution<u8> for Fixed {
    fn ln_prob(&self, _: &u8) -> f64 {
        self.0
    }
}
impl Model<u8> for BadModel {
    fn predict(&self) -> impl Distribution<u8> {
        Fixed(self.value)
    }
    fn observe(&mut self, _: u8) {
        self.seen += 1;
        self.value = 0.0;
    }
}

#[test]
fn zero_probability_remains_infinite_but_invalid_values_fail_before_observe() {
    let mut model = BadModel {
        value: f64::NEG_INFINITY,
        seen: 0,
    };
    let report = evaluate(&b"ab"[..], &mut model).unwrap();
    assert_eq!(report.total_nats, f64::INFINITY);
    assert_eq!(model.seen, 2);
    for value in [f64::NAN, f64::INFINITY, 0.1] {
        let mut model = BadModel { value, seen: 0 };
        assert_eq!(
            evaluate(&b"a"[..], &mut model).unwrap_err().kind(),
            io::ErrorKind::InvalidData
        );
        assert_eq!(model.seen, 0);
    }
}

struct InterruptedOnce(bool);
impl Read for InterruptedOnce {
    fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
        if std::mem::replace(&mut self.0, false) {
            Err(io::ErrorKind::Interrupted.into())
        } else {
            Err(io::ErrorKind::PermissionDenied.into())
        }
    }
}

#[test]
fn read_and_cost_sink_errors_propagate() {
    assert_eq!(
        evaluate(InterruptedOnce(true), &mut Uniform)
            .unwrap_err()
            .kind(),
        io::ErrorKind::PermissionDenied
    );
    let mut model = BadModel {
        value: 0.0,
        seen: 0,
    };
    let error = evaluate_with_costs(&b"a"[..], &mut model, |_| {
        Err(io::ErrorKind::BrokenPipe.into())
    })
    .unwrap_err();
    assert_eq!(error.kind(), io::ErrorKind::BrokenPipe);
    assert_eq!(model.seen, 0);
}

#[test]
fn prefix_limits_and_separate_calls_preserve_exact_model_state() {
    let bytes = b"ABACAB";
    let whole = evaluate(&bytes[..], &mut Kt::default()).unwrap();
    let mut model = Kt::default();
    let first = evaluate((&bytes[..]).take(3), &mut model).unwrap();
    let second = evaluate(&bytes[3..], &mut model).unwrap();
    assert_eq!(first.bytes, 3);
    assert!((first.total_nats + second.total_nats - whole.total_nats).abs() < 1e-12);
}

#[test]
fn kt_distribution_is_normalized_before_and_after_learning() {
    let mut model = Kt::default();
    for byte in [0, 255, 255, 1] {
        let mass: f64 = (0..=255).map(|b| model.predict().ln_prob(&b).exp()).sum();
        assert!((mass - 1.0).abs() < 1e-12);
        model.observe(byte);
    }
}

#[test]
fn coding_ratio_compares_models_and_is_unit_invariant() {
    let input = &b"AAAABABA"[..];
    let model = evaluate(input, &mut Kt::default()).unwrap();
    let reference = evaluate(input, &mut Uniform).unwrap();
    let ratio = model.coding_ratio(&reference).unwrap();
    assert!((ratio - model.total_bits() / reference.total_bits()).abs() < 1e-12);
    assert!((ratio - model.total_bytes() / reference.total_bytes()).abs() < 1e-12);
    assert!((ratio - model.uniform_coding_ratio().unwrap()).abs() < 1e-12);
    assert!((ratio * reference.coding_ratio(&model).unwrap() - 1.0).abs() < 1e-12);
}

#[test]
fn coding_ratio_handles_undefined_and_infinite_costs() {
    let finite = Evaluation {
        bytes: 3,
        total_nats: 4.0,
    };
    let zero = Evaluation {
        bytes: 3,
        total_nats: 0.0,
    };
    let infinite = Evaluation {
        bytes: 3,
        total_nats: f64::INFINITY,
    };
    assert_eq!(finite.coding_ratio(&zero), None);
    assert_eq!(zero.coding_ratio(&finite), Some(0.0));
    assert_eq!(infinite.coding_ratio(&infinite), None);
    assert_eq!(infinite.coding_ratio(&finite), Some(f64::INFINITY));
    assert_eq!(finite.coding_ratio(&infinite), Some(0.0));
    assert_eq!(
        finite.coding_ratio(&Evaluation { bytes: 4, ..finite }),
        None
    );
    for value in [-1.0, f64::NEG_INFINITY, f64::NAN] {
        let invalid = Evaluation {
            total_nats: value,
            ..finite
        };
        assert_eq!(finite.coding_ratio(&invalid), None);
        assert_eq!(invalid.coding_ratio(&finite), None);
    }
}
