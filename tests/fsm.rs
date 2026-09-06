use kraft::{
    Distribution, Model, evaluate,
    fsm::{BinaryKtFsm, ExactFsmMixture, FsmError, TransitionTable, labeled_table_count},
};

#[test]
fn labeled_table_counts_and_ranks_are_exact() {
    assert_eq!(labeled_table_count(0), Some(0));
    assert_eq!(labeled_table_count(1), Some(1));
    assert_eq!(labeled_table_count(2), Some(16));
    assert_eq!(labeled_table_count(3), Some(729));
    assert_eq!(labeled_table_count(4), Some(65_536));

    for state_count in 1..=4 {
        let count = labeled_table_count(state_count).unwrap();
        for rank in [0, count / 2, count - 1] {
            let table = TransitionTable::from_rank(state_count, rank).unwrap();
            assert_eq!(table.rank(), Some(rank));
        }
    }
}

#[test]
fn transition_tables_validate_targets() {
    assert_eq!(TransitionTable::new(Vec::new()), Err(FsmError::EmptyTable));
    assert!(matches!(
        TransitionTable::new(vec![[0, 2], [1, 0]]),
        Err(FsmError::InvalidTarget {
            state: 0,
            bit: 1,
            target: 2,
            state_count: 2,
        })
    ));
}

#[test]
fn one_state_first_byte_matches_beta_bernoulli_integral() {
    let model = BinaryKtFsm::from_rank(1, 0).unwrap();
    let probability = model.byte_ln_prob(0x80).exp();
    assert!((probability - 0.013_092_041_015_625).abs() < 1e-15);
}

#[test]
fn individual_fsm_distribution_is_normalized_before_and_after_learning() {
    let mut model = BinaryKtFsm::from_rank(2, 7).unwrap();
    for byte in [0x00, 0x80, 0xff, 0x55] {
        let mass: f64 = (0..=255)
            .map(|candidate| model.predict().ln_prob(&candidate).exp())
            .sum();
        assert!((mass - 1.0).abs() < 1e-12);
        model.observe(byte);
    }
}

#[test]
fn one_state_exact_mixture_matches_the_single_model() {
    let bytes = b"KRAFT";
    let single = evaluate(&bytes[..], &mut BinaryKtFsm::from_rank(1, 0).unwrap()).unwrap();
    let mixture =
        evaluate(&bytes[..], &mut ExactFsmMixture::all_labeled(1, 1).unwrap()).unwrap();
    assert!((single.total_nats - mixture.total_nats).abs() < 1e-12);
}

#[test]
fn two_state_exact_mixture_is_normalized_and_updates_posterior() {
    let mut mixture = ExactFsmMixture::all_labeled(2, 16).unwrap();
    assert_eq!(mixture.model_count(), 16);

    let initial: f64 = mixture.posterior_weights().iter().sum();
    assert!((initial - 1.0).abs() < 1e-12);

    for byte in [0xaa, 0x55] {
        let mass: f64 = (0..=255)
            .map(|candidate| mixture.predict().ln_prob(&candidate).exp())
            .sum();
        assert!((mass - 1.0).abs() < 1e-12);
        mixture.observe(byte);
        let posterior: f64 = mixture.posterior_weights().iter().sum();
        assert!((posterior - 1.0).abs() < 1e-12);
    }
}

#[test]
fn exact_enumeration_requires_an_explicit_memory_guard() {
    assert_eq!(
        ExactFsmMixture::all_labeled(4, 65_535).unwrap_err(),
        FsmError::TooManyModels {
            required: 65_536,
            limit: 65_535,
        }
    );
}
