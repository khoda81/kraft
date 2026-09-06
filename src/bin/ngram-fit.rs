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
//! Context length is not packed into a machine word. Instead each observed
//! length-k context receives an exact compact ID, and the next-order context ID
//! is interned from (old_context_id, observed_byte). This removes any context
//! length ceiling while retaining integer-sort performance.

use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    io::{self, BufWriter, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

const ALPHABET: usize = 256;
const JEFFREYS_ALPHA: f64 = 0.5;
const JEFFREYS_TOTAL: f64 = 128.0;
const LN_2: f64 = std::f64::consts::LN_2;
const LOG_TWO_PI_HALF: f64 = 0.918_938_533_204_672_7;
const GAMMA_CACHE_MAX: usize = 65_536;

const HELP: &str = "Usage: ngram-fit <file> [options]

Exactly evaluate fixed-order byte-context predictors on the full corpus.

For order k:
  state_t = previous k bytes
  transition = shift left by one byte and append x_t
  emission = per-context integrated Dirichlet-1/2 categorical model

The first k bytes are encoded uniformly because a full k-byte history does not
yet exist.

Options:
  --orders LIST       comma-separated explicit orders, e.g. 0,1,2,4,8,16
  --max-order N       evaluate every order 0..=N
  --dump-best PATH    after the sweep, dump the best sparse model as TSV

Defaults:
  --orders 0,1,2,3,4

There is no context-length limit for enwik8-sized inputs. Exact compact context
IDs are built recursively. The packed sorting implementation requires a corpus
shorter than 2^28 bytes, which includes enwik8.

If every observed context becomes unique at some order, all larger fixed orders
are provably uniform predictors and are filled analytically without further
sorting.

Sparse model dump format:
  context_hex<TAB>total<TAB>next_counts

where next_counts is comma-separated byte_hex=count. Counts plus the Jeffreys
prior exactly reconstruct the posterior predictive distribution:
  P(byte | context) = (count + 1/2) / (total + 128).

Higher coding ratio is better.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    orders: Vec<usize>,
    dump_best: Option<PathBuf>,
}

#[derive(Debug, Clone)]
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
    let mut dump_best = None;
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
        } else if !positional && arg == "--max-order" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --max-order"))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid("invalid UTF-8 value for --max-order"))?;
            let maximum: usize = text
                .parse()
                .map_err(|_| invalid("--max-order must be a non-negative integer"))?;
            orders = (0..=maximum).collect();
        } else if !positional && arg == "--dump-best" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --dump-best"))?;
            dump_best = Some(PathBuf::from(value));
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        orders,
        dump_best,
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

fn position_bits(data_len: usize) -> io::Result<u32> {
    if data_len == 0 {
        return Ok(0);
    }
    let bits = usize::BITS - (data_len - 1).leading_zeros();
    if 2 * bits + 8 > 64 {
        return Err(invalid(
            "packed exact context-ID scorer currently requires input shorter than 2^28 bytes",
        ));
    }
    Ok(bits)
}

fn position_mask(bits: u32) -> u64 {
    if bits == 0 { 0 } else { (1_u64 << bits) - 1 }
}

fn possible_contexts(order: usize) -> String {
    if order == 0 {
        "1".to_owned()
    } else {
        match order.checked_mul(8) {
            Some(exponent) => format!("2^{exponent}"),
            None => "2^(overflow)".to_owned(),
        }
    }
}

fn uniform_score(order: usize, data_len: usize, seconds: f64) -> Score {
    let contexts = data_len.saturating_sub(order) as u64;
    Score {
        order,
        total_nats: data_len as f64 * 8.0 * LN_2,
        bootstrap_nats: bootstrap_nats(order, data_len),
        observed_contexts: contexts,
        observed_ngrams: contexts,
        seconds,
    }
}

fn build_entries(
    data: &[u8],
    order: usize,
    context_ids: &[u32],
    position_bits: u32,
) -> Vec<u64> {
    let mut entries = Vec::with_capacity(data.len().saturating_sub(order));
    for position in order..data.len() {
        let context = u64::from(context_ids[position]);
        let group = (context << 8) | u64::from(data[position]);
        entries.push((group << position_bits) | position as u64);
    }
    entries
}

fn score_and_advance(
    data: &[u8],
    order: usize,
    context_ids: Vec<u32>,
    context_representatives: &[u32],
    gamma: &GammaCache,
    position_bits: u32,
    mut dump: Option<&mut BufWriter<File>>,
) -> io::Result<(Score, Vec<u32>, Vec<u32>)> {
    let started = Instant::now();
    if data.len() <= order {
        let score = uniform_score(order, data.len(), started.elapsed().as_secs_f64());
        return Ok((score, vec![0; data.len() + 1], Vec::new()));
    }

    let mut entries = build_entries(data, order, &context_ids, position_bits);
    drop(context_ids);
    entries.sort_unstable();

    let mask = position_mask(position_bits);
    let mut next_ids = vec![0_u32; data.len() + 1];
    let mut next_representatives = Vec::new();

    let mut ln_evidence = 0.0;
    let mut observed_contexts = 0_u64;
    let mut observed_ngrams = 0_u64;
    let mut current_context = None;
    let mut context_total = 0_u64;
    let mut context_symbol_terms = 0.0;
    let mut dump_counts: Vec<(u8, u64)> = Vec::new();

    let flush_context = |context: Option<u32>,
                         total: u64,
                         symbol_terms: f64,
                         counts: &mut Vec<(u8, u64)>,
                         ln_evidence: &mut f64,
                         observed_contexts: &mut u64,
                         dump: &mut Option<&mut BufWriter<File>>|
     -> io::Result<()> {
        let Some(context) = context else {
            return Ok(());
        };
        *ln_evidence += gamma.context_evidence(total, symbol_terms);
        *observed_contexts += 1;

        if let Some(writer) = dump.as_deref_mut() {
            let representative = context_representatives[context as usize] as usize;
            let start = representative - order;
            for &byte in &data[start..representative] {
                write!(writer, "{byte:02x}")?;
            }
            write!(writer, "\t{total}\t")?;
            for (index, &(byte, count)) in counts.iter().enumerate() {
                if index != 0 {
                    writer.write_all(b",")?;
                }
                write!(writer, "{byte:02x}={count}")?;
            }
            writer.write_all(b"\n")?;
        }
        counts.clear();
        Ok(())
    };

    let mut index = 0_usize;
    let mut next_context_id = 0_u32;
    while index < entries.len() {
        let entry = entries[index];
        let group = entry >> position_bits;
        let context = (group >> 8) as u32;
        let byte = group as u8;

        if current_context != Some(context) {
            flush_context(
                current_context,
                context_total,
                context_symbol_terms,
                &mut dump_counts,
                &mut ln_evidence,
                &mut observed_contexts,
                &mut dump,
            )?;
            current_context = Some(context);
            context_total = 0;
            context_symbol_terms = 0.0;
        }

        let mut end = index + 1;
        while end < entries.len() && (entries[end] >> position_bits) == group {
            end += 1;
        }
        let count = (end - index) as u64;
        observed_ngrams += 1;
        context_total += count;
        context_symbol_terms += gamma.half(count) - gamma.prior_half;
        dump_counts.push((byte, count));

        let representative_position = (entries[index] & mask) as u32 + 1;
        next_representatives.push(representative_position);
        for &packed in &entries[index..end] {
            let position = (packed & mask) as usize;
            next_ids[position + 1] = next_context_id;
        }
        next_context_id = next_context_id
            .checked_add(1)
            .ok_or_else(|| invalid("more than u32::MAX observed contexts"))?;
        index = end;
    }

    flush_context(
        current_context,
        context_total,
        context_symbol_terms,
        &mut dump_counts,
        &mut ln_evidence,
        &mut observed_contexts,
        &mut dump,
    )?;

    let bootstrap = bootstrap_nats(order, data.len());
    let score = Score {
        order,
        total_nats: -ln_evidence + bootstrap,
        bootstrap_nats: bootstrap,
        observed_contexts,
        observed_ngrams,
        seconds: started.elapsed().as_secs_f64(),
    };
    Ok((score, next_ids, next_representatives))
}

fn coding_ratio(reference_nats: f64, candidate_nats: f64) -> f64 {
    reference_nats / candidate_nats
}

fn evaluate_orders(data: &[u8], orders: &[usize], gamma: &GammaCache) -> io::Result<Vec<Score>> {
    let position_bits = position_bits(data.len())?;
    let maximum = *orders.iter().max().unwrap_or(&0);
    let requested: std::collections::HashSet<_> = orders.iter().copied().collect();

    let mut scores = Vec::new();
    let mut context_ids = vec![0_u32; data.len() + 1];
    let mut representatives = vec![0_u32];
    let mut saturated_at = None;

    for order in 0..=maximum {
        if let Some(saturation_order) = saturated_at {
            if requested.contains(&order) {
                scores.push(uniform_score(order, data.len(), 0.0));
            }
            if order == saturation_order {
                unreachable!("saturation applies only to later orders");
            }
            continue;
        }

        let (score, next_ids, next_representatives) = score_and_advance(
            data,
            order,
            context_ids,
            &representatives,
            gamma,
            position_bits,
            None,
        )?;

        let all_contexts_unique =
            score.observed_contexts as usize == data.len().saturating_sub(order);

        if requested.contains(&order) {
            scores.push(score.clone());
        }

        context_ids = next_ids;
        representatives = next_representatives;

        if all_contexts_unique {
            saturated_at = Some(order);
            eprintln!(
                "all observed contexts are unique at order {order}; larger orders are exactly uniform"
            );
        }
    }

    scores.sort_by_key(|score| score.order);
    Ok(scores)
}

fn dump_best_model(
    data: &[u8],
    best: &Score,
    path: &Path,
    gamma: &GammaCache,
) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let position_bits = position_bits(data.len())?;
    let mut context_ids = vec![0_u32; data.len() + 1];
    let mut representatives = vec![0_u32];

    for order in 0..=best.order {
        let dump_file = if order == best.order {
            let file = File::create(path)?;
            let mut writer = BufWriter::new(file);
            writeln!(writer, "# KRAFT fixed-order byte-context model")?;
            writeln!(writer, "# order={}", best.order)?;
            writeln!(writer, "# total_nats={:.12}", best.total_nats)?;
            writeln!(writer, "# observed_contexts={}", best.observed_contexts)?;
            writeln!(writer, "# observed_ngrams={}", best.observed_ngrams)?;
            writeln!(writer, "# alpha=0.5")?;
            writeln!(writer, "# alphabet=256")?;
            writeln!(writer, "# context_hex\ttotal\tnext_counts")?;
            Some(writer)
        } else {
            None
        };

        let mut dump_file = dump_file;
        let (score, next_ids, next_representatives) = score_and_advance(
            data,
            order,
            context_ids,
            &representatives,
            gamma,
            position_bits,
            dump_file.as_mut(),
        )?;
        if order == best.order {
            debug_assert!((score.total_nats - best.total_nats).abs() < 1e-6);
            if let Some(writer) = dump_file.as_mut() {
                writer.flush()?;
            }
            break;
        }
        context_ids = next_ids;
        representatives = next_representatives;
    }

    Ok(())
}

fn run(args: &Args) -> io::Result<()> {
    let started = Instant::now();
    let data = fs::read(&args.path)?;
    if data.is_empty() {
        return Err(invalid("input file must not be empty"));
    }
    let gamma = GammaCache::new();
    let uniform_nats = data.len() as f64 * 8.0 * LN_2;
    let scores = evaluate_orders(&data, &args.orders, &gamma)?;
    let kt = scores
        .iter()
        .find(|score| score.order == 0)
        .cloned()
        .unwrap_or_else(|| uniform_score(0, data.len(), 0.0));

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

    for score in &scores {
        println!(
            "{}\t{}\t{}\t{}\t{:.6}\t{:.12}\t{:.12}\t{:.12}\t{:.6}",
            score.order,
            possible_contexts(score.order),
            score.observed_contexts,
            score.observed_ngrams,
            score.bootstrap_nats,
            score.total_nats,
            coding_ratio(uniform_nats, score.total_nats),
            coding_ratio(kt.total_nats, score.total_nats),
            score.seconds,
        );
    }

    let best = scores
        .iter()
        .min_by(|left, right| left.total_nats.total_cmp(&right.total_nats))
        .ok_or_else(|| invalid("no requested orders were evaluated"))?;
    println!();
    println!("best_order: {}", best.order);
    println!("best_total_nats: {:.12}", best.total_nats);
    println!(
        "best_coding_ratio_uniform: {:.12}",
        coding_ratio(uniform_nats, best.total_nats)
    );
    println!(
        "best_coding_ratio_kt: {:.12}",
        coding_ratio(kt.total_nats, best.total_nats)
    );

    if let Some(path) = &args.dump_best {
        let dump_started = Instant::now();
        dump_best_model(&data, best, path, &gamma)?;
        println!("model_dump: {:?}", path);
        println!(
            "model_dump_seconds: {:.6}",
            dump_started.elapsed().as_secs_f64()
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
    fn recursive_context_ids_match_prequential_oracle() {
        let data = b"abracadabra abracadabra";
        let gamma = GammaCache::new();
        let orders: Vec<_> = (0..=12).collect();
        let scores = evaluate_orders(data, &orders, &gamma).unwrap();
        for score in scores {
            let sequential = prequential_nats(data, score.order);
            assert!(
                (score.total_nats - sequential).abs() < 1e-9,
                "order {}: integrated={}, sequential={}",
                score.order,
                score.total_nats,
                sequential
            );
        }
    }

    #[test]
    fn ngram_count_becomes_next_order_context_count() {
        let data = b"mediawiki mediawiki mediawiki";
        let gamma = GammaCache::new();
        let orders: Vec<_> = (0..=8).collect();
        let scores = evaluate_orders(data, &orders, &gamma).unwrap();
        for pair in scores.windows(2) {
            assert_eq!(pair[0].observed_ngrams, pair[1].observed_contexts);
        }
    }

    #[test]
    fn model_dump_contains_reconstructable_sparse_counts() {
        let data = b"abababa";
        let gamma = GammaCache::new();
        let scores = evaluate_orders(data, &[0, 1, 2], &gamma).unwrap();
        let best = scores
            .iter()
            .find(|score| score.order == 1)
            .expect("order one score");
        let path = std::env::temp_dir().join(format!(
            "kraft-ngram-dump-{}-{}.tsv",
            std::process::id(),
            data.len()
        ));
        dump_best_model(data, best, &path, &gamma).unwrap();
        let dump = fs::read_to_string(&path).unwrap();
        fs::remove_file(path).unwrap();
        assert!(dump.contains("# order=1"));
        assert!(dump.contains("61\t3\t62=3"));
        assert!(dump.contains("62\t3\t61=3"));
    }

    #[test]
    fn possible_context_count_has_no_integer_width_limit() {
        assert_eq!(possible_contexts(0), "1");
        assert_eq!(possible_contexts(7), "2^56");
        assert_eq!(possible_contexts(9), "2^72");
        assert_eq!(possible_contexts(128), "2^1024");
    }

    #[test]
    fn position_packing_supports_enwik8() {
        assert_eq!(position_bits(100_000_000).unwrap(), 27);
        assert!(position_bits(1_usize << 28).is_err());
    }
}
