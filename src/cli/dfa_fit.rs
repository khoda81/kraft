//! Heuristic full-corpus search for a high-evidence two-state DFA.
//!
//! This binary does not approximate the Bayesian mixture directly. It searches
//! for one strong complete transition table. Because all 2^512 labeled N=2
//! transition tables have equal prior mass, any candidate h certifies
//!
//!     C_mix <= C_h + 512 bits.
//!
//! The 512-bit gap is only 64 bytes over the entire corpus.

use std::{
    cmp::Ordering, collections::HashSet, ffi::OsString, fs, io, path::PathBuf,
    thread, time::Instant,
};

const ALPHABET: usize = 256;
const TABLE_BITS: usize = 2 * ALPHABET;
const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = 128.0;
const LN_2: f64 = std::f64::consts::LN_2;
const LOG_TWO_PI_HALF: f64 = 0.918_938_533_204_672_7;

const HELP: &str = "Usage: dfa-fit <file> [options]

Heuristically search for a strong complete N=2 DFA on a large byte corpus.
The final candidates are scored on the entire file with exact integrated
Dirichlet-1/2 emission evidence.

Any returned candidate gives a rigorous exact-mixture coding bound:
  mixture_cost <= candidate_cost + 512 bits

Defaults:
  --search-bytes 500000
  --screen-bytes 10000000
  --population 64
  --generations 20
  --elite 8
  --cem-restarts 2
  --reset-restarts 32
  --screen-candidates 32
  --finalists 8
  --threads <available parallelism>
  --seed 1

The search has three stages:
  1. fit a full-corpus reset DFA where delta(0,b)=delta(1,b)=g(b);
  2. CEM search over all 512 transition bits on --search-bytes;
  3. screen candidates on --screen-bytes, then score finalists on the full file.

This is a heuristic MAP search. It does not claim to find the globally best DFA.";

#[derive(Debug, Clone)]
struct Args {
    path: PathBuf,
    search_bytes: usize,
    screen_bytes: usize,
    population: usize,
    generations: usize,
    elite: usize,
    cem_restarts: usize,
    reset_restarts: usize,
    screen_candidates: usize,
    finalists: usize,
    threads: usize,
    seed: u64,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct Dfa2 {
    transition: [u8; TABLE_BITS],
}

impl Dfa2 {
    fn zero() -> Self {
        Self {
            transition: [0; TABLE_BITS],
        }
    }

    fn reset(assignment: &[u8; ALPHABET]) -> Self {
        let mut transition = [0_u8; TABLE_BITS];
        transition[..ALPHABET].copy_from_slice(assignment);
        transition[ALPHABET..].copy_from_slice(assignment);
        Self { transition }
    }

    #[inline]
    fn destination(&self, state: usize, byte: u8) -> usize {
        usize::from(self.transition[state * ALPHABET + usize::from(byte)])
    }

    fn hex_row(&self, state: usize) -> String {
        let row = &self.transition[state * ALPHABET..(state + 1) * ALPHABET];
        let mut output = String::with_capacity(ALPHABET / 4);
        for nibble in row.as_chunks::<4>().0 {
            let value = nibble[0] | (nibble[1] << 1) | (nibble[2] << 2) | (nibble[3] << 3);
            output.push(char::from_digit(u32::from(value), 16).expect("nibble is hexadecimal"));
        }
        output
    }
}

#[derive(Debug)]
struct BigramStats {
    rows: Vec<[u64; ALPHABET]>,
    row_totals: [u64; ALPHABET],
    unigram: [u64; ALPHABET],
    first: Option<u8>,
}

impl BigramStats {
    fn from_data(data: &[u8]) -> Self {
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
        Self {
            rows,
            row_totals,
            unigram,
            first: data.first().copied(),
        }
    }

    fn reset_counts(&self, assignment: &[u8; ALPHABET]) -> ([[u64; ALPHABET]; 2], [u64; 2]) {
        let mut counts = [[0_u64; ALPHABET]; 2];
        let mut totals = [0_u64; 2];
        if let Some(first) = self.first {
            counts[0][usize::from(first)] += 1;
            totals[0] += 1;
        }
        for (previous, &assigned_state) in assignment.iter().enumerate() {
            let state = usize::from(assigned_state);
            totals[state] += self.row_totals[previous];
            for (next, &amount) in self.rows[previous].iter().enumerate() {
                counts[state][next] += amount;
            }
        }
        (counts, totals)
    }
}

#[derive(Debug, Clone)]
struct Scored {
    table: Dfa2,
    ln_evidence: f64,
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        value ^ (value >> 31)
    }

    fn next_f64(&mut self) -> f64 {
        const SCALE: f64 = 1.0 / ((1_u64 << 53) as f64);
        ((self.next_u64() >> 11) as f64) * SCALE
    }

    fn shuffle(&mut self, values: &mut [usize]) {
        for index in (1..values.len()).rev() {
            let other = (self.next_u64() as usize) % (index + 1);
            values.swap(index, other);
        }
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_usize(name: &str, text: &str, minimum: usize) -> io::Result<usize> {
    text.parse()
        .ok()
        .filter(|value| *value >= minimum)
        .ok_or_else(|| invalid(format!("{name} must be an integer >= {minimum}")))
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut search_bytes = 500_000_usize;
    let mut screen_bytes = 10_000_000_usize;
    let mut population = 64_usize;
    let mut generations = 20_usize;
    let mut elite = 8_usize;
    let mut cem_restarts = 2_usize;
    let mut reset_restarts = 32_usize;
    let mut screen_candidates = 32_usize;
    let mut finalists = 8_usize;
    let mut threads = thread::available_parallelism().map_or(1, usize::from);
    let mut seed = 1_u64;
    let mut positional = false;

    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }

        if !positional && arg.to_string_lossy().starts_with("--") {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid(format!("invalid UTF-8 value for {arg:?}")))?;
            match arg.to_string_lossy().as_ref() {
                "--search-bytes" => search_bytes = parse_usize("--search-bytes", text, 1)?,
                "--screen-bytes" => screen_bytes = parse_usize("--screen-bytes", text, 1)?,
                "--population" => population = parse_usize("--population", text, 2)?,
                "--generations" => generations = parse_usize("--generations", text, 1)?,
                "--elite" => elite = parse_usize("--elite", text, 1)?,
                "--cem-restarts" => cem_restarts = parse_usize("--cem-restarts", text, 1)?,
                "--reset-restarts" => reset_restarts = parse_usize("--reset-restarts", text, 1)?,
                "--screen-candidates" => {
                    screen_candidates = parse_usize("--screen-candidates", text, 1)?
                }
                "--finalists" => finalists = parse_usize("--finalists", text, 1)?,
                "--threads" => threads = parse_usize("--threads", text, 1)?,
                "--seed" => {
                    seed = text
                        .parse()
                        .map_err(|_| invalid("--seed must be an unsigned integer"))?
                }
                _ => return Err(invalid(format!("unknown option {arg:?}"))),
            }
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    if elite > population {
        return Err(invalid("--elite cannot exceed --population"));
    }
    if finalists > screen_candidates {
        return Err(invalid("--finalists cannot exceed --screen-candidates"));
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        search_bytes,
        screen_bytes,
        population,
        generations,
        elite,
        cem_restarts,
        reset_restarts,
        screen_candidates,
        finalists,
        threads,
        seed,
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

fn counts_ln_evidence(counts: &[[u64; ALPHABET]; 2], totals: &[u64; 2]) -> f64 {
    state_ln_evidence(&counts[0], totals[0]) + state_ln_evidence(&counts[1], totals[1])
}

fn table_ln_evidence(data: &[u8], table: &Dfa2) -> f64 {
    let mut counts = [[0_u64; ALPHABET]; 2];
    let mut totals = [0_u64; 2];
    let mut state = 0_usize;
    for &byte in data {
        counts[state][usize::from(byte)] += 1;
        totals[state] += 1;
        state = table.destination(state, byte);
    }
    counts_ln_evidence(&counts, &totals)
}

fn unigram_ln_evidence(stats: &BigramStats) -> f64 {
    state_ln_evidence(&stats.unigram, stats.unigram.iter().sum())
}

fn reset_move_delta(
    stats: &BigramStats,
    counts: &[[u64; ALPHABET]; 2],
    totals: &[u64; 2],
    previous: usize,
    from: usize,
    to: usize,
) -> f64 {
    let row_total = stats.row_totals[previous];
    if row_total == 0 {
        return 0.0;
    }

    let mut delta = ln_gamma(totals[from] as f64 + JEFFREYS_TOTAL)
        - ln_gamma((totals[from] - row_total) as f64 + JEFFREYS_TOTAL)
        + ln_gamma(totals[to] as f64 + JEFFREYS_TOTAL)
        - ln_gamma((totals[to] + row_total) as f64 + JEFFREYS_TOTAL);

    for (next, &amount) in stats.rows[previous].iter().enumerate() {
        if amount == 0 {
            continue;
        }
        delta += ln_gamma((counts[from][next] - amount) as f64 + JEFFREYS_ALPHA)
            - ln_gamma(counts[from][next] as f64 + JEFFREYS_ALPHA)
            + ln_gamma((counts[to][next] + amount) as f64 + JEFFREYS_ALPHA)
            - ln_gamma(counts[to][next] as f64 + JEFFREYS_ALPHA);
    }
    delta
}

fn apply_reset_move(
    stats: &BigramStats,
    counts: &mut [[u64; ALPHABET]; 2],
    totals: &mut [u64; 2],
    previous: usize,
    from: usize,
    to: usize,
) {
    let row_total = stats.row_totals[previous];
    totals[from] -= row_total;
    totals[to] += row_total;
    for (next, &amount) in stats.rows[previous].iter().enumerate() {
        counts[from][next] -= amount;
        counts[to][next] += amount;
    }
}

fn fit_reset_dfa(stats: &BigramStats, restarts: usize, seed: u64) -> Scored {
    let mut rng = SplitMix64::new(seed ^ 0x5245_5345_545f_4446);
    let mut best = Scored {
        table: Dfa2::zero(),
        ln_evidence: unigram_ln_evidence(stats),
    };
    let mut order: Vec<usize> = (0..ALPHABET)
        .filter(|&byte| stats.row_totals[byte] != 0)
        .collect();

    for restart in 0..restarts {
        let mut assignment = [0_u8; ALPHABET];
        if restart != 0 {
            for (byte, assigned_state) in assignment.iter_mut().enumerate() {
                if stats.row_totals[byte] != 0 {
                    *assigned_state = (rng.next_u64() & 1) as u8;
                }
            }
        }
        let (mut counts, mut totals) = stats.reset_counts(&assignment);

        for _ in 0..64 {
            rng.shuffle(&mut order);
            let mut changed = false;
            for &previous in &order {
                let from = usize::from(assignment[previous]);
                let to = 1 - from;
                let delta = reset_move_delta(stats, &counts, &totals, previous, from, to);
                if delta > 1e-10 {
                    apply_reset_move(stats, &mut counts, &mut totals, previous, from, to);
                    assignment[previous] = to as u8;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }

        let ln_evidence = counts_ln_evidence(&counts, &totals);
        if ln_evidence > best.ln_evidence {
            best = Scored {
                table: Dfa2::reset(&assignment),
                ln_evidence,
            };
        }
    }

    best
}

fn score_population(data: &[u8], candidates: &[Dfa2], threads: usize) -> Vec<f64> {
    if candidates.is_empty() {
        return Vec::new();
    }
    let thread_count = threads.min(candidates.len()).max(1);
    let chunk_size = candidates.len().div_ceil(thread_count);

    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in candidates.chunks(chunk_size) {
            handles.push(scope.spawn(move || {
                chunk
                    .iter()
                    .map(|candidate| table_ln_evidence(data, candidate))
                    .collect::<Vec<_>>()
            }));
        }
        let mut scores = Vec::with_capacity(candidates.len());
        for handle in handles {
            scores.extend(handle.join().expect("DFA scorer thread must not panic"));
        }
        scores
    })
}

fn cmp_score_desc(left: &f64, right: &f64) -> Ordering {
    right.total_cmp(left)
}

fn add_hall(hall: &mut Vec<Scored>, candidate: Dfa2, ln_evidence: f64, capacity: usize) {
    if let Some(existing) = hall.iter_mut().find(|entry| entry.table == candidate) {
        existing.ln_evidence = existing.ln_evidence.max(ln_evidence);
    } else {
        hall.push(Scored {
            table: candidate,
            ln_evidence,
        });
    }
    hall.sort_by(|left, right| cmp_score_desc(&left.ln_evidence, &right.ln_evidence));
    hall.truncate(capacity);
}

fn cem_search(data: &[u8], seed_table: &Dfa2, args: &Args, kt_ln_evidence: f64) -> Vec<Scored> {
    let hall_capacity = args.screen_candidates.max(args.finalists).max(8) * 2;
    let mut hall = Vec::new();
    let mut master_rng = SplitMix64::new(args.seed ^ 0x4345_4d5f_4446_4132);

    for restart in 0..args.cem_restarts {
        let mut probabilities = [0.5_f64; TABLE_BITS];
        if restart == 0 {
            for (index, probability) in probabilities.iter_mut().enumerate() {
                *probability = if seed_table.transition[index] == 1 {
                    0.85
                } else {
                    0.15
                };
            }
        }
        let mut champion = seed_table.clone();

        for generation in 0..args.generations {
            let generation_seed = master_rng.next_u64();
            let mut rng = SplitMix64::new(generation_seed);
            let mut candidates = Vec::with_capacity(args.population);
            candidates.push(champion.clone());
            while candidates.len() < args.population {
                let mut transition = [0_u8; TABLE_BITS];
                for index in 0..TABLE_BITS {
                    transition[index] = u8::from(rng.next_f64() < probabilities[index]);
                }
                candidates.push(Dfa2 { transition });
            }

            let scores = score_population(data, &candidates, args.threads);
            let mut indices: Vec<_> = (0..candidates.len()).collect();
            indices.sort_by(|&left, &right| cmp_score_desc(&scores[left], &scores[right]));

            champion = candidates[indices[0]].clone();
            for &index in indices.iter().take(args.elite.min(indices.len())) {
                add_hall(
                    &mut hall,
                    candidates[index].clone(),
                    scores[index],
                    hall_capacity,
                );
            }

            let elite_count = args.elite.min(indices.len());
            for (bit, probability) in probabilities.iter_mut().enumerate() {
                let mean = indices
                    .iter()
                    .take(elite_count)
                    .map(|&index| f64::from(candidates[index].transition[bit]))
                    .sum::<f64>()
                    / elite_count as f64;
                *probability = (0.55 * *probability + 0.45 * mean).clamp(0.02, 0.98);
            }

            let best_cost = -scores[indices[0]];
            let kt_cost = -kt_ln_evidence;
            eprintln!(
                "cem restart={}/{} generation={}/{} search_nats={:.6} coding_ratio_kt={:.9}",
                restart + 1,
                args.cem_restarts,
                generation + 1,
                args.generations,
                best_cost,
                kt_cost / best_cost,
            );
        }
    }

    hall
}

fn rank_candidates(data: &[u8], candidates: Vec<Dfa2>, keep: usize, threads: usize) -> Vec<Scored> {
    let mut unique = HashSet::new();
    let candidates: Vec<_> = candidates
        .into_iter()
        .filter(|candidate| unique.insert(candidate.clone()))
        .collect();
    let scores = score_population(data, &candidates, threads);
    let mut scored: Vec<_> = candidates
        .into_iter()
        .zip(scores)
        .map(|(table, ln_evidence)| Scored { table, ln_evidence })
        .collect();
    scored.sort_by(|left, right| cmp_score_desc(&left.ln_evidence, &right.ln_evidence));
    scored.truncate(keep.min(scored.len()));
    scored
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
    let stats = BigramStats::from_data(&data);
    let kt_ln_evidence = unigram_ln_evidence(&stats);
    let kt_nats = -kt_ln_evidence;
    let uniform_nats = data.len() as f64 * 8.0 * LN_2;

    println!("input: {:?}", args.path);
    println!("bytes: {}", data.len());
    println!("search_bytes: {}", args.search_bytes.min(data.len()));
    println!("screen_bytes: {}", args.screen_bytes.min(data.len()));
    println!("population: {}", args.population);
    println!("generations: {}", args.generations);
    println!("elite: {}", args.elite);
    println!("cem_restarts: {}", args.cem_restarts);
    println!("reset_restarts: {}", args.reset_restarts);
    println!("threads: {}", args.threads);
    println!("seed: {}", args.seed);
    println!("kt_total_nats: {kt_nats:.12}");
    println!(
        "kt_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, kt_nats)
    );
    println!();

    let reset_started = Instant::now();
    let reset = fit_reset_dfa(&stats, args.reset_restarts, args.seed);
    let reset_nats = -reset.ln_evidence;
    println!("reset_full_nats: {reset_nats:.12}");
    println!(
        "reset_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, reset_nats)
    );
    println!(
        "reset_coding_ratio_kt: {:.12}",
        coding_ratio(kt_nats, reset_nats)
    );
    println!(
        "reset_fit_seconds: {:.6}",
        reset_started.elapsed().as_secs_f64()
    );
    println!();

    let search_len = args.search_bytes.min(data.len());
    let search = &data[..search_len];
    let search_stats = BigramStats::from_data(search);
    let search_kt_ln_evidence = unigram_ln_evidence(&search_stats);

    let mut hall = cem_search(search, &reset.table, args, search_kt_ln_evidence);
    add_hall(
        &mut hall,
        reset.table.clone(),
        table_ln_evidence(search, &reset.table),
        args.screen_candidates * 2,
    );
    let zero = Dfa2::zero();
    add_hall(
        &mut hall,
        zero.clone(),
        table_ln_evidence(search, &zero),
        args.screen_candidates * 2,
    );

    let mut search_ranked = hall;
    search_ranked.sort_by(|left, right| cmp_score_desc(&left.ln_evidence, &right.ln_evidence));
    search_ranked.truncate(args.screen_candidates.min(search_ranked.len()));

    let screen_len = args.screen_bytes.min(data.len());
    let screen = &data[..screen_len];
    let screen_ranked = rank_candidates(
        screen,
        search_ranked
            .iter()
            .map(|entry| entry.table.clone())
            .collect(),
        args.finalists,
        args.threads,
    );

    let mut final_tables: Vec<_> = screen_ranked
        .iter()
        .map(|entry| entry.table.clone())
        .collect();
    final_tables.push(reset.table.clone());
    final_tables.push(zero);
    let final_ranked = rank_candidates(&data, final_tables, usize::MAX, args.threads);
    let best = final_ranked
        .first()
        .expect("reset candidate guarantees a final candidate");
    let best_nats = -best.ln_evidence;

    println!();
    println!("full_file_finalists:");
    println!("rank\ttotal_nats\tcoding_ratio_uniform\tcoding_ratio_kt");
    for (rank, entry) in final_ranked.iter().enumerate() {
        let total_nats = -entry.ln_evidence;
        println!(
            "{}\t{:.12}\t{:.12}\t{:.12}",
            rank + 1,
            total_nats,
            coding_ratio(uniform_nats, total_nats),
            coding_ratio(kt_nats, total_nats),
        );
    }

    let transition_prior_nats = TABLE_BITS as f64 * LN_2;
    let mixture_cost_upper_nats = best_nats + transition_prior_nats;
    println!();
    println!("best_candidate_total_nats: {best_nats:.12}");
    println!(
        "best_candidate_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, best_nats)
    );
    println!(
        "best_candidate_coding_ratio_kt: {:.12}",
        coding_ratio(kt_nats, best_nats)
    );
    println!("transition_prior_bits: {TABLE_BITS}");
    println!("transition_prior_bytes: {}", TABLE_BITS / 8);
    println!(
        "certified_mixture_cost_upper_nats: {:.12}",
        mixture_cost_upper_nats
    );
    println!(
        "certified_mixture_coding_ratio_uniform_lower: {:.12}",
        coding_ratio(uniform_nats, mixture_cost_upper_nats)
    );
    println!(
        "certified_mixture_coding_ratio_kt_lower: {:.12}",
        coding_ratio(kt_nats, mixture_cost_upper_nats)
    );
    println!("best_transition_state0_hex: {}", best.table.hex_row(0));
    println!("best_transition_state1_hex: {}", best.table.hex_row(1));
    println!("evaluation_seconds: {:.6}", started.elapsed().as_secs_f64());

    Ok(())
}

pub fn command(args: Vec<OsString>) -> io::Result<()> {
    parse(args).and_then(|args| match args {
        None => {
            println!("{HELP}");
            Ok(())
        }
        Some(args) => run(&args),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn prequential_ln_evidence(data: &[u8], table: &Dfa2) -> f64 {
        let mut counts = [[0_u64; ALPHABET]; 2];
        let mut totals = [0_u64; 2];
        let mut state = 0_usize;
        let mut result = 0.0;
        for &byte in data {
            result += ((counts[state][usize::from(byte)] as f64 + JEFFREYS_ALPHA)
                / (totals[state] as f64 + JEFFREYS_TOTAL))
                .ln();
            counts[state][usize::from(byte)] += 1;
            totals[state] += 1;
            state = table.destination(state, byte);
        }
        result
    }

    #[test]
    fn lanczos_gamma_matches_known_values() {
        assert!((ln_gamma(0.5) - 0.5 * std::f64::consts::PI.ln()).abs() < 1e-13);
        assert!((ln_gamma(1.0) - 0.0).abs() < 1e-13);
        assert!((ln_gamma(5.0) - 24.0_f64.ln()).abs() < 1e-13);
    }

    #[test]
    fn integrated_score_matches_prequential_score() {
        let data = b"mediawiki mediawiki";
        let mut table = Dfa2::zero();
        for byte in 0..ALPHABET {
            table.transition[byte] = (byte & 1) as u8;
            table.transition[ALPHABET + byte] = ((byte >> 1) & 1) as u8;
        }
        let integrated = table_ln_evidence(data, &table);
        let sequential = prequential_ln_evidence(data, &table);
        assert!((integrated - sequential).abs() < 1e-10);
    }

    #[test]
    fn reset_bigram_score_matches_table_scan() {
        let data = b"abracadabra abracadabra";
        let stats = BigramStats::from_data(data);
        let mut assignment = [0_u8; ALPHABET];
        for (byte, assigned_state) in assignment.iter_mut().enumerate() {
            *assigned_state = (byte.count_ones() & 1) as u8;
        }
        let table = Dfa2::reset(&assignment);
        let (counts, totals) = stats.reset_counts(&assignment);
        assert!(
            (counts_ln_evidence(&counts, &totals) - table_ln_evidence(data, &table)).abs() < 1e-10
        );
    }

    #[test]
    fn zero_table_is_byte_kt() {
        let data = b"KRAFT";
        let stats = BigramStats::from_data(data);
        let table = Dfa2::zero();
        assert!((table_ln_evidence(data, &table) - unigram_ln_evidence(&stats)).abs() < 1e-10);
    }
}
