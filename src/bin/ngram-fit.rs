//! Exact full-corpus fixed-order byte-context evaluator.
//!
//! For order k, the predictor state is exactly the previous k bytes:
//!
//!     s_t = (x_{t-k}, ..., x_{t-1})
//!
//! and the transition is the tiny shift-register program "drop oldest, append
//! observed byte". Per-context byte probabilities are integrated exactly under
//! the symmetric Dirichlet-1/2 prior.
//!
//! Only observed contexts are materialized. Orders through seven are supported;
//! the formal state space at order seven is 256^7 = 2^56 states.

use std::{
    env,
    ffi::OsString,
    fs,
    io,
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

const ALPHABET: usize = 256;
const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = 128.0;
const LN_2: f64 = std::f64::consts::LN_2;
const LOG_TWO_PI_HALF: f64 = 0.918_938_533_204_672_7;
const GAMMA_CACHE_MAX: usize = 65_536;

const HELP: &str = "Usage: ngram-fit <file> [--orders LIST]

Exactly evaluate fixed-order byte-context predictors on the full corpus.

For order k:
  state_t = previous k bytes
  transition = shift left by one byte and append x_t
  emission = per-context integrated Dirichlet-1/2 categorical model

The first k bytes are encoded uniformly because a full k-byte history does not
yet exist. That bootstrap cost is at most 7 bytes and is reported separately.

Defaults:
  --orders 0,1,2,3,4

Supported orders: 0..=7.

Implementation:
  order 0: 256-entry histogram
  order 1: dense 2-byte histogram
  order 2: dense 3-byte histogram (~64 MB)
  order 3: sorted packed u32 4-grams (~400 MB for enwik8)
  order 4..7: sorted packed u64 n-grams (~800 MB for enwik8)

Higher coding ratio is better.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    orders: Vec<usize>,
}

#[derive(Debug)]
struct Score {
    order: usize,
    total_nats: f64,
    bootstrap_nats: f64,
    observed_contexts: u64,
    observed_ngrams: u64,
    seconds: f64,
}

struct GammaCache {
    half: Vec<f64>,
    total: Vec<f64>,
    prior_half: f64,
    prior_total: f64,
}

impl GammaCache {
    fn new() -> Self {
        let mut half = Vec::with_capacity(GAMMA_CACHE_MAX + 1);
        let mut total = Vec::with_capacity(GAMMA_CACHE_MAX + 1);
        for count in 0..=GAMMA_CACHE_MAX {
            half.push(ln_gamma(count as f64 + JEFFREYS_ALPHA));
            total.push(ln_gamma(count as f64 + JEFFREYS_TOTAL));
        }
        Self {
            prior_half: half[0],
            prior_total: total[0],
            half,
            total,
        }
    }

    #[inline]
    fn half(&self, count: u64) -> f64 {
        if count <= GAMMA_CACHE_MAX as u64 {
            self.half[count as usize]
        } else {
            ln_gamma(count as f64 + JEFFREYS_ALPHA)
        }
    }

    #[inline]
    fn total(&self, count: u64) -> f64 {
        if count <= GAMMA_CACHE_MAX as u64 {
            self.total[count as usize]
        } else {
            ln_gamma(count as f64 + JEFFREYS_TOTAL)
        }
    }

    #[inline]
    fn context_evidence(&self, total: u64, symbol_terms: f64) -> f64 {
        self.prior_total - self.total(total) + symbol_terms
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_orders(text: &str) -> io::Result<Vec<usize>> {
    let mut orders = Vec::new();
    for part in text.split(',') {
        let value: usize = part
            .trim()
            .parse()
            .map_err(|_| invalid("--orders must be a comma-separated list of integers"))?;
        if value > 7 {
            return Err(invalid("each --orders value must be in 0..=7"));
        }
        orders.push(value);
    }
    orders.sort_unstable();
    orders.dedup();
    if orders.is_empty() {
        return Err(invalid("--orders must contain at least one value"));
    }
    Ok(orders)
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut orders = vec![0, 1, 2, 3, 4];
    let mut positional = false;

    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }
        if !positional && arg == "--orders" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --orders"))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid("invalid UTF-8 value for --orders"))?;
            orders = parse_orders(text)?;
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        orders,
    }))
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

fn bootstrap_nats(order: usize, data_len: usize) -> f64 {
    order.min(data_len) as f64 * 8.0 * LN_2
}

fn score_order_zero(data: &[u8], gamma: &GammaCache) -> Score {
    let started = Instant::now();
    let mut counts = [0_u64; ALPHABET];
    for &byte in data {
        counts[usize::from(byte)] += 1;
    }
    let symbol_terms = counts
        .iter()
        .filter(|&&count| count != 0)
        .map(|&count| gamma.half(count) - gamma.prior_half)
        .sum::<f64>();
    let ln_evidence = gamma.context_evidence(data.len() as u64, symbol_terms);
    Score {
        order: 0,
        total_nats: -ln_evidence,
        bootstrap_nats: 0.0,
        observed_contexts: 1,
        observed_ngrams: counts.iter().filter(|&&count| count != 0).count() as u64,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn score_order_one(data: &[u8], gamma: &GammaCache) -> Score {
    let started = Instant::now();
    if data.len() <= 1 {
        return Score {
            order: 1,
            total_nats: bootstrap_nats(1, data.len()),
            bootstrap_nats: bootstrap_nats(1, data.len()),
            observed_contexts: 0,
            observed_ngrams: 0,
            seconds: started.elapsed().as_secs_f64(),
        };
    }

    let mut counts = vec![0_u32; 1 << 16];
    for pair in data.windows(2) {
        let key = (usize::from(pair[0]) << 8) | usize::from(pair[1]);
        counts[key] += 1;
    }

    let mut ln_evidence = 0.0;
    let mut observed_contexts = 0_u64;
    let mut observed_ngrams = 0_u64;
    for context in 0..ALPHABET {
        let row = &counts[context << 8..(context + 1) << 8];
        let total: u64 = row.iter().map(|&count| u64::from(count)).sum();
        if total == 0 {
            continue;
        }
        observed_contexts += 1;
        let mut symbol_terms = 0.0;
        for &count in row {
            if count != 0 {
                observed_ngrams += 1;
                symbol_terms += gamma.half(u64::from(count)) - gamma.prior_half;
            }
        }
        ln_evidence += gamma.context_evidence(total, symbol_terms);
    }

    let bootstrap = bootstrap_nats(1, data.len());
    Score {
        order: 1,
        total_nats: -ln_evidence + bootstrap,
        bootstrap_nats: bootstrap,
        observed_contexts,
        observed_ngrams,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn score_order_two(data: &[u8], gamma: &GammaCache) -> Score {
    let started = Instant::now();
    if data.len() <= 2 {
        let bootstrap = bootstrap_nats(2, data.len());
        return Score {
            order: 2,
            total_nats: bootstrap,
            bootstrap_nats: bootstrap,
            observed_contexts: 0,
            observed_ngrams: 0,
            seconds: started.elapsed().as_secs_f64(),
        };
    }

    let mut counts = vec![0_u32; 1 << 24];
    let mut rolling = 0_u32;
    for (index, &byte) in data.iter().enumerate() {
        rolling = ((rolling << 8) | u32::from(byte)) & 0x00ff_ffff;
        if index >= 2 {
            counts[rolling as usize] += 1;
        }
    }

    let mut ln_evidence = 0.0;
    let mut observed_contexts = 0_u64;
    let mut observed_ngrams = 0_u64;
    for context in 0..(1_usize << 16) {
        let row = &counts[context << 8..(context + 1) << 8];
        let mut total = 0_u64;
        let mut symbol_terms = 0.0;
        for &count in row {
            if count != 0 {
                observed_ngrams += 1;
                total += u64::from(count);
                symbol_terms += gamma.half(u64::from(count)) - gamma.prior_half;
            }
        }
        if total != 0 {
            observed_contexts += 1;
            ln_evidence += gamma.context_evidence(total, symbol_terms);
        }
    }

    let bootstrap = bootstrap_nats(2, data.len());
    Score {
        order: 2,
        total_nats: -ln_evidence + bootstrap,
        bootstrap_nats: bootstrap,
        observed_contexts,
        observed_ngrams,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn finalize_sorted_context(
    gamma: &GammaCache,
    context_total: u64,
    symbol_terms: f64,
    ln_evidence: &mut f64,
    observed_contexts: &mut u64,
) {
    if context_total != 0 {
        *ln_evidence += gamma.context_evidence(context_total, symbol_terms);
        *observed_contexts += 1;
    }
}

fn score_order_three(data: &[u8], gamma: &GammaCache) -> Score {
    let started = Instant::now();
    if data.len() <= 3 {
        let bootstrap = bootstrap_nats(3, data.len());
        return Score {
            order: 3,
            total_nats: bootstrap,
            bootstrap_nats: bootstrap,
            observed_contexts: 0,
            observed_ngrams: 0,
            seconds: started.elapsed().as_secs_f64(),
        };
    }

    let mut keys = Vec::with_capacity(data.len() - 3);
    let mut rolling = 0_u32;
    for (index, &byte) in data.iter().enumerate() {
        rolling = (rolling << 8) | u32::from(byte);
        if index >= 3 {
            keys.push(rolling);
        }
    }
    keys.sort_unstable();

    let mut ln_evidence = 0.0;
    let mut observed_contexts = 0_u64;
    let mut observed_ngrams = 0_u64;
    let mut context_total = 0_u64;
    let mut symbol_terms = 0.0;
    let mut previous_context = None;
    let mut index = 0_usize;

    while index < keys.len() {
        let key = keys[index];
        let context = key >> 8;
        let mut end = index + 1;
        while end < keys.len() && keys[end] == key {
            end += 1;
        }
        let count = (end - index) as u64;

        if previous_context != Some(context) {
            finalize_sorted_context(
                gamma,
                context_total,
                symbol_terms,
                &mut ln_evidence,
                &mut observed_contexts,
            );
            previous_context = Some(context);
            context_total = 0;
            symbol_terms = 0.0;
        }

        context_total += count;
        symbol_terms += gamma.half(count) - gamma.prior_half;
        observed_ngrams += 1;
        index = end;
    }
    finalize_sorted_context(
        gamma,
        context_total,
        symbol_terms,
        &mut ln_evidence,
        &mut observed_contexts,
    );

    let bootstrap = bootstrap_nats(3, data.len());
    Score {
        order: 3,
        total_nats: -ln_evidence + bootstrap,
        bootstrap_nats: bootstrap,
        observed_contexts,
        observed_ngrams,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn order_mask(order: usize) -> u64 {
    let width_bits = 8 * (order + 1);
    if width_bits == 64 {
        u64::MAX
    } else {
        (1_u64 << width_bits) - 1
    }
}

fn score_order_u64(data: &[u8], order: usize, gamma: &GammaCache) -> Score {
    debug_assert!((4..=7).contains(&order));
    let started = Instant::now();
    if data.len() <= order {
        let bootstrap = bootstrap_nats(order, data.len());
        return Score {
            order,
            total_nats: bootstrap,
            bootstrap_nats: bootstrap,
            observed_contexts: 0,
            observed_ngrams: 0,
            seconds: started.elapsed().as_secs_f64(),
        };
    }

    let mut keys = Vec::with_capacity(data.len() - order);
    let mut rolling = 0_u64;
    let mask = order_mask(order);
    for (index, &byte) in data.iter().enumerate() {
        rolling = ((rolling << 8) | u64::from(byte)) & mask;
        if index >= order {
            keys.push(rolling);
        }
    }
    keys.sort_unstable();

    let mut ln_evidence = 0.0;
    let mut observed_contexts = 0_u64;
    let mut observed_ngrams = 0_u64;
    let mut context_total = 0_u64;
    let mut symbol_terms = 0.0;
    let mut previous_context = None;
    let mut index = 0_usize;

    while index < keys.len() {
        let key = keys[index];
        let context = key >> 8;
        let mut end = index + 1;
        while end < keys.len() && keys[end] == key {
            end += 1;
        }
        let count = (end - index) as u64;

        if previous_context != Some(context) {
            finalize_sorted_context(
                gamma,
                context_total,
                symbol_terms,
                &mut ln_evidence,
                &mut observed_contexts,
            );
            previous_context = Some(context);
            context_total = 0;
            symbol_terms = 0.0;
        }

        context_total += count;
        symbol_terms += gamma.half(count) - gamma.prior_half;
        observed_ngrams += 1;
        index = end;
    }
    finalize_sorted_context(
        gamma,
        context_total,
        symbol_terms,
        &mut ln_evidence,
        &mut observed_contexts,
    );

    let bootstrap = bootstrap_nats(order, data.len());
    Score {
        order,
        total_nats: -ln_evidence + bootstrap,
        bootstrap_nats: bootstrap,
        observed_contexts,
        observed_ngrams,
        seconds: started.elapsed().as_secs_f64(),
    }
}

fn score(data: &[u8], order: usize, gamma: &GammaCache) -> Score {
    match order {
        0 => score_order_zero(data, gamma),
        1 => score_order_one(data, gamma),
        2 => score_order_two(data, gamma),
        3 => score_order_three(data, gamma),
        4..=7 => score_order_u64(data, order, gamma),
        _ => unreachable!("orders are validated by the CLI"),
    }
}

fn possible_contexts(order: usize) -> u64 {
    if order == 0 {
        1
    } else {
        1_u64 << (8 * order)
    }
}

fn coding_ratio(reference_nats: f64, candidate_nats: f64) -> f64 {
    reference_nats / candidate_nats
}

fn run(args: &Args) -> io::Result<()> {
    let started = Instant::now();
    let data = fs::read(&args.path)?;
    if data.is_empty() {
        return Err(invalid("input file must not be empty"));
    }
    let gamma = GammaCache::new();
    let uniform_nats = data.len() as f64 * 8.0 * LN_2;
    let kt = score_order_zero(&data, &gamma);

    println!("input: {:?}", args.path);
    println!("bytes: {}", data.len());
    println!("kt_total_nats: {:.12}", kt.total_nats);
    println!(
        "kt_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, kt.total_nats)
    );
    println!();
    println!(
        "order\tpossible_contexts\tobserved_contexts\tobserved_ngrams\tbootstrap_nats\ttotal_nats\tcoding_ratio_uniform\tcoding_ratio_kt\tseconds"
    );

    for &order in &args.orders {
        let result = if order == 0 {
            Score {
                order: kt.order,
                total_nats: kt.total_nats,
                bootstrap_nats: kt.bootstrap_nats,
                observed_contexts: kt.observed_contexts,
                observed_ngrams: kt.observed_ngrams,
                seconds: kt.seconds,
            }
        } else {
            score(&data, order, &gamma)
        };
        println!(
            "{}\t{}\t{}\t{}\t{:.6}\t{:.12}\t{:.12}\t{:.12}\t{:.6}",
            result.order,
            possible_contexts(result.order),
            result.observed_contexts,
            result.observed_ngrams,
            result.bootstrap_nats,
            result.total_nats,
            coding_ratio(uniform_nats, result.total_nats),
            coding_ratio(kt.total_nats, result.total_nats),
            result.seconds,
        );
    }

    println!("evaluation_seconds: {:.6}", started.elapsed().as_secs_f64());
    Ok(())
}

fn main() -> ExitCode {
    let result = parse(env::args_os().skip(1)).and_then(|args| match args {
        None => {
            println!("{HELP}");
            Ok(())
        }
        Some(args) => run(&args),
    });

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("ngram-fit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prequential_nats(data: &[u8], order: usize) -> f64 {
        use std::collections::HashMap;

        let mut counts: HashMap<(Vec<u8>, u8), u64> = HashMap::new();
        let mut totals: HashMap<Vec<u8>, u64> = HashMap::new();
        let mut nats = bootstrap_nats(order, data.len());

        for index in order..data.len() {
            let context = data[index - order..index].to_vec();
            let byte = data[index];
            let count = *counts.get(&(context.clone(), byte)).unwrap_or(&0);
            let total = *totals.get(&context).unwrap_or(&0);
            nats -= ((count as f64 + JEFFREYS_ALPHA) / (total as f64 + JEFFREYS_TOTAL)).ln();
            *counts.entry((context.clone(), byte)).or_insert(0) += 1;
            *totals.entry(context).or_insert(0) += 1;
        }
        nats
    }

    #[test]
    fn all_implementations_match_prequential_oracle() {
        let data = b"abracadabra abracadabra";
        let gamma = GammaCache::new();
        for order in 0..=7 {
            let integrated = score(data, order, &gamma).total_nats;
            let sequential = prequential_nats(data, order);
            assert!(
                (integrated - sequential).abs() < 1e-9,
                "order {order}: integrated={integrated}, sequential={sequential}"
            );
        }
    }

    #[test]
    fn order_zero_matches_byte_kt_formula() {
        let data = b"KRAFT";
        let gamma = GammaCache::new();
        let result = score_order_zero(data, &gamma);
        assert!((result.total_nats - prequential_nats(data, 0)).abs() < 1e-10);
    }

    #[test]
    fn possible_state_counts_are_shift_register_sizes() {
        assert_eq!(possible_contexts(0), 1);
        assert_eq!(possible_contexts(1), 256);
        assert_eq!(possible_contexts(2), 65_536);
        assert_eq!(possible_contexts(3), 16_777_216);
        assert_eq!(possible_contexts(7), 1_u64 << 56);
    }
}
