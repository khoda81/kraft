//! Full-corpus byte-context partition curve.
//!
//! A reset context model uses
//!
//!     state_{t+1} = g(x_t)
//!
//! so the state for the next prediction is a learned cluster of the previous
//! byte. This binary greedily agglomerates the 256 previous-byte bigram rows
//! under exact integrated Dirichlet-1/2 evidence, producing a deterministic
//! full-corpus curve from one context through all 256 byte contexts.

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

const HELP: &str = "Usage: context-fit <file> [--states LIST]

Fit full-corpus reset/context models:
  state_(t+1) = g(x_t)

The 256 previous-byte bigram rows are agglomeratively clustered using exact
Dirichlet-1/2 integrated evidence. One hierarchy gives all requested state counts.

Defaults:
  --states 1,2,4,8,16,32,64,128,256

For each N the output reports the candidate coding cost plus two rigorous mixture
bounds:
  reset mixture: candidate + 256*log2(N) bits
  full N-state DFA mixture: candidate + 256*N*log2(N) bits

Higher coding ratio is better.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    states: Vec<usize>,
}

#[derive(Debug, Clone)]
struct Cluster {
    active: bool,
    counts: [u64; ALPHABET],
    total: u64,
    evidence: f64,
}

impl Cluster {
    fn empty() -> Self {
        Self {
            active: false,
            counts: [0; ALPHABET],
            total: 0,
            evidence: 0.0,
        }
    }
}

#[derive(Debug)]
struct BigramStats {
    rows: Vec<[u64; ALPHABET]>,
    row_totals: [u64; ALPHABET],
    unigram: [u64; ALPHABET],
    first: u8,
}

#[derive(Debug, Clone, Copy)]
struct CurvePoint {
    states: usize,
    total_nats: f64,
    start_cluster_predictive_ln: f64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_states(text: &str) -> io::Result<Vec<usize>> {
    let mut states = Vec::new();
    for part in text.split(',') {
        let value: usize = part
            .trim()
            .parse()
            .map_err(|_| invalid("--states must be a comma-separated list of integers"))?;
        if !(1..=ALPHABET).contains(&value) {
            return Err(invalid("each --states value must be in 1..=256"));
        }
        states.push(value);
    }
    states.sort_unstable();
    states.dedup();
    if states.is_empty() {
        return Err(invalid("--states must contain at least one value"));
    }
    Ok(states)
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut states = vec![1, 2, 4, 8, 16, 32, 64, 128, 256];
    let mut positional = false;

    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }
        if !positional && arg == "--states" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --states"))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid("invalid UTF-8 value for --states"))?;
            states = parse_states(text)?;
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        states,
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

fn state_ln_evidence(counts: &[u64; ALPHABET], total: u64) -> f64 {
    let mut result = ln_gamma(JEFFREYS_TOTAL) - ln_gamma(total as f64 + JEFFREYS_TOTAL);
    let prior = ln_gamma(JEFFREYS_ALPHA);
    for &count in counts {
        if count != 0 {
            result += ln_gamma(count as f64 + JEFFREYS_ALPHA) - prior;
        }
    }
    result
}

fn merged_ln_evidence(left: &Cluster, right: &Cluster) -> f64 {
    let total = left.total + right.total;
    let mut result = ln_gamma(JEFFREYS_TOTAL) - ln_gamma(total as f64 + JEFFREYS_TOTAL);
    let prior = ln_gamma(JEFFREYS_ALPHA);
    for byte in 0..ALPHABET {
        let count = left.counts[byte] + right.counts[byte];
        if count != 0 {
            result += ln_gamma(count as f64 + JEFFREYS_ALPHA) - prior;
        }
    }
    result
}

fn bigram_stats(data: &[u8]) -> BigramStats {
    let mut rows = vec![[0_u64; ALPHABET]; ALPHABET];
    let mut row_totals = [0_u64; ALPHABET];
    let mut unigram = [0_u64; ALPHABET];
    for &byte in data {
        unigram[usize::from(byte)] += 1;
    }
    for pair in data.windows(2) {
        let previous = usize::from(pair[0]);
        let next = usize::from(pair[1]);
        rows[previous][next] += 1;
        row_totals[previous] += 1;
    }
    BigramStats {
        rows,
        row_totals,
        unigram,
        first: data[0],
    }
}

fn partition_evidence(clusters: &[Cluster]) -> (f64, f64) {
    let base = clusters
        .iter()
        .filter(|cluster| cluster.active)
        .map(|cluster| cluster.evidence)
        .sum::<f64>();

    // The first observed byte is emitted from fixed start state zero. Since state
    // labels are otherwise arbitrary in a reset model, choose which active
    // cluster is called state zero. Adding one observation changes integrated
    // evidence by exactly its current posterior-predictive log probability.
    let first_byte = clusters
        .iter()
        .filter(|cluster| cluster.active)
        .map(|cluster| {
            ((cluster.counts[usize::from(0_u8)] as f64 + JEFFREYS_ALPHA)
                / (cluster.total as f64 + JEFFREYS_TOTAL))
                .ln()
        })
        .fold(f64::NEG_INFINITY, f64::max);

    (base + first_byte, first_byte)
}

fn fit_curve(stats: &BigramStats, requested: &[usize]) -> Vec<CurvePoint> {
    let mut clusters = vec![Cluster::empty(); ALPHABET];
    for previous in 0..ALPHABET {
        let counts = stats.rows[previous];
        let total = stats.row_totals[previous];
        clusters[previous] = Cluster {
            active: true,
            counts,
            total,
            evidence: state_ln_evidence(&counts, total),
        };
    }

    let mut merge_scores = vec![f64::NEG_INFINITY; ALPHABET * ALPHABET];
    let score_index = |left: usize, right: usize| left * ALPHABET + right;
    for left in 0..ALPHABET {
        for right in (left + 1)..ALPHABET {
            merge_scores[score_index(left, right)] =
                merged_ln_evidence(&clusters[left], &clusters[right])
                    - clusters[left].evidence
                    - clusters[right].evidence;
        }
    }

    let requested_set: std::collections::HashSet<_> = requested.iter().copied().collect();
    let mut points = Vec::new();
    let mut active_count = ALPHABET;

    loop {
        if requested_set.contains(&active_count) {
            let base = clusters
                .iter()
                .filter(|cluster| cluster.active)
                .map(|cluster| cluster.evidence)
                .sum::<f64>();
            let first_byte = stats.first;
            let best_first = clusters
                .iter()
                .filter(|cluster| cluster.active)
                .map(|cluster| {
                    ((cluster.counts[usize::from(first_byte)] as f64 + JEFFREYS_ALPHA)
                        / (cluster.total as f64 + JEFFREYS_TOTAL))
                        .ln()
                })
                .fold(f64::NEG_INFINITY, f64::max);
            points.push(CurvePoint {
                states: active_count,
                total_nats: -(base + best_first),
                start_cluster_predictive_ln: best_first,
            });
        }
        if active_count == 1 {
            break;
        }

        let mut best_pair = None;
        let mut best_delta = f64::NEG_INFINITY;
        for left in 0..ALPHABET {
            if !clusters[left].active {
                continue;
            }
            for right in (left + 1)..ALPHABET {
                if !clusters[right].active {
                    continue;
                }
                let delta = merge_scores[score_index(left, right)];
                if delta > best_delta {
                    best_delta = delta;
                    best_pair = Some((left, right));
                }
            }
        }
        let (left, right) = best_pair.expect("two active clusters remain");
        let merged_evidence = clusters[left].evidence + clusters[right].evidence + best_delta;
        let right_counts = clusters[right].counts;
        for (target, amount) in clusters[left].counts.iter_mut().zip(right_counts) {
            *target += amount;
        }
        clusters[left].total += clusters[right].total;
        clusters[left].evidence = merged_evidence;
        clusters[right].active = false;
        active_count -= 1;

        for other in 0..ALPHABET {
            if other == left || !clusters[other].active {
                continue;
            }
            let (a, b) = if left < other {
                (left, other)
            } else {
                (other, left)
            };
            merge_scores[score_index(a, b)] =
                merged_ln_evidence(&clusters[a], &clusters[b])
                    - clusters[a].evidence
                    - clusters[b].evidence;
        }
    }

    points.sort_by_key(|point| point.states);
    points
}

fn coding_ratio(reference_nats: f64, candidate_nats: f64) -> f64 {
    reference_nats / candidate_nats
}

fn prior_bits_reset(states: usize) -> f64 {
    if states <= 1 {
        0.0
    } else {
        ALPHABET as f64 * (states as f64).log2()
    }
}

fn prior_bits_full_dfa(states: usize) -> f64 {
    states as f64 * prior_bits_reset(states)
}

fn run(args: &Args) -> io::Result<()> {
    let started = Instant::now();
    let data = fs::read(&args.path)?;
    if data.is_empty() {
        return Err(invalid("input file must not be empty"));
    }
    let stats = bigram_stats(&data);
    let kt_nats = -state_ln_evidence(&stats.unigram, data.len() as u64);
    let uniform_nats = data.len() as f64 * 8.0 * LN_2;
    let points = fit_curve(&stats, &args.states);

    println!("input: {:?}", args.path);
    println!("bytes: {}", data.len());
    println!("kt_total_nats: {kt_nats:.12}");
    println!(
        "kt_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, kt_nats)
    );
    println!();
    println!(
        "states\ttotal_nats\tcoding_ratio_uniform\tcoding_ratio_kt\treset_prior_bits\treset_mix_ratio_uniform_lower\tfull_dfa_prior_bits\tfull_dfa_mix_ratio_uniform_lower"
    );
    for point in points {
        let reset_prior = prior_bits_reset(point.states);
        let full_prior = prior_bits_full_dfa(point.states);
        let reset_bound = point.total_nats + reset_prior * LN_2;
        let full_bound = point.total_nats + full_prior * LN_2;
        println!(
            "{}\t{:.12}\t{:.12}\t{:.12}\t{:.3}\t{:.12}\t{:.3}\t{:.12}",
            point.states,
            point.total_nats,
            coding_ratio(uniform_nats, point.total_nats),
            coding_ratio(kt_nats, point.total_nats),
            reset_prior,
            coding_ratio(uniform_nats, reset_bound),
            full_prior,
            coding_ratio(uniform_nats, full_bound),
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
            eprintln!("context-fit: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn one_context_matches_byte_kt() {
        let data = b"mediawiki";
        let stats = bigram_stats(data);
        let points = fit_curve(&stats, &[1]);
        let kt = -state_ln_evidence(&stats.unigram, data.len() as u64);
        assert!((points[0].total_nats - kt).abs() < 1e-10);
    }

    #[test]
    fn full_byte_context_matches_direct_bigram_partition() {
        let data = b"abracadabra abracadabra";
        let stats = bigram_stats(data);
        let points = fit_curve(&stats, &[ALPHABET]);
        let base = stats
            .rows
            .iter()
            .zip(stats.row_totals)
            .map(|(counts, total)| state_ln_evidence(counts, total))
            .sum::<f64>();
        let best_first = stats
            .rows
            .iter()
            .zip(stats.row_totals)
            .map(|(counts, total)| {
                ((counts[usize::from(stats.first)] as f64 + JEFFREYS_ALPHA)
                    / (total as f64 + JEFFREYS_TOTAL))
                    .ln()
            })
            .fold(f64::NEG_INFINITY, f64::max);
        assert!((points[0].total_nats + base + best_first).abs() < 1e-10);
    }

    #[test]
    fn prior_costs_match_declared_families() {
        assert_eq!(prior_bits_reset(1), 0.0);
        assert_eq!(prior_bits_reset(2), 256.0);
        assert_eq!(prior_bits_full_dfa(2), 512.0);
        assert_eq!(prior_bits_reset(4), 512.0);
        assert_eq!(prior_bits_full_dfa(4), 2048.0);
    }
}
