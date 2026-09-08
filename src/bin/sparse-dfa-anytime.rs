use std::{fs, num::NonZeroUsize, path::PathBuf, process::ExitCode, time::Instant};

use clap::Parser;
use kraft::{baselines::Kt, evaluate, models::sparse_dfa_anytime::SparseDfaAnytime};

#[derive(Debug, Parser)]
#[command(about = "Anytime evidence bounds for the full sparse-DFA Bayesian prior")]
struct Args {
    path: PathBuf,

    /// Prefix bytes to score.
    #[arg(long, default_value_t = 8)]
    limit: usize,

    /// Structural refinements to perform.
    #[arg(long, default_value_t = 10_000)]
    steps: usize,

    /// Print one progress row every N refinements.
    #[arg(long, default_value = "1000")]
    report_every: NonZeroUsize,
}

fn code_bounds(search: &SparseDfaAnytime) -> (f64, f64) {
    let bounds = search.bounds();
    let lower = -bounds.ln_upper();
    let upper = if bounds.ln_lower() == f64::NEG_INFINITY {
        f64::INFINITY
    } else {
        -bounds.ln_lower()
    };
    (lower, upper)
}

fn range(lower: f64, upper: f64) -> String {
    if upper.is_infinite() {
        return format!("{lower:.3}..inf");
    }
    let width = upper - lower;
    if width == 0.0 {
        return format!("{lower:.6}");
    }

    let places = 1 - width.log10().floor() as i32;
    let scale = 10_f64.powi(places);
    let lower = (lower * scale).floor() / scale;
    let upper = (upper * scale).ceil() / scale;
    let decimals = places.max(0) as usize;
    format!("{lower:.decimals$}..{upper:.decimals$}")
}

fn report(search: &SparseDfaAnytime, uniform_nats: f64, kt_nats: f64, elapsed: f64) {
    let (lower, upper) = code_bounds(search);
    let ratio_uniform_lower = if upper.is_infinite() {
        0.0
    } else {
        uniform_nats / upper
    };
    let ratio_uniform_upper = uniform_nats / lower;
    let ratio_kt_lower = if upper.is_infinite() {
        0.0
    } else {
        kt_nats / upper
    };
    let ratio_kt_upper = kt_nats / lower;

    println!(
        "{}\t{}\t{}\t{}\t{}\t{}\t{:.3}",
        search.steps(),
        search.regions(),
        search.resolved_regions(),
        range(lower, upper),
        range(ratio_uniform_lower, ratio_uniform_upper),
        range(ratio_kt_lower, ratio_kt_upper),
        elapsed,
    );
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let mut data = fs::read(&args.path)?;
    data.truncate(args.limit);

    let uniform_nats = data.len() as f64 * 8.0 * std::f64::consts::LN_2;
    let kt_nats = evaluate(&data[..], &mut Kt::default())?.total_nats;
    let mut search = SparseDfaAnytime::new(&data);
    let started = Instant::now();

    println!("input: {:?}", args.path);
    println!("prefix_bytes: {}", data.len());
    println!("target: exact sparse-DFA Bayesian mixture prequential cost");
    println!(
        "note: this binary certifies joint evidence; it is not yet the finite-compute streaming codec"
    );
    println!();
    println!("steps\tregions\tresolved_regions\tcode_nats\tratio_uniform\tratio_kt\telapsed_s");
    report(
        &search,
        uniform_nats,
        kt_nats,
        started.elapsed().as_secs_f64(),
    );

    let report_every = args.report_every.get();
    while search.steps() < args.steps {
        let remaining = args.steps - search.steps();
        search.run(remaining.min(report_every));
        report(
            &search,
            uniform_nats,
            kt_nats,
            started.elapsed().as_secs_f64(),
        );
        if remaining < report_every {
            break;
        }
    }

    Ok(())
}

fn main() -> ExitCode {
    match run(Args::parse()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sparse-dfa-anytime: {error}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ranges_are_compact_and_outward() {
        assert_eq!(range(1.012144404961, 1.537038328816), "1.01..1.54");
        assert_eq!(range(2382.265, 3617.699), "2300..3700");
        assert_eq!(range(2382.265, 2382.703), "2382.26..2382.71");
        assert_eq!(range(0.0, f64::INFINITY), "0.000..inf");
    }
}
