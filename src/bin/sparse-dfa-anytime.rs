use std::{fs::File, io::Read, num::NonZeroUsize, path::PathBuf, process::ExitCode, time::Instant};

use clap::Parser;
use kraft::{
    baselines::Kt,
    evaluate,
    models::sparse_dfa_anytime::{Schedule, SparseDfaAnytime},
};

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

    /// Print unresolved upper-bound mass and forced-prefix depth by region type to stderr.
    #[arg(long)]
    diagnostics: bool,

    /// Experimental: prioritize tails by the next exact count's upper mass.
    #[arg(long)]
    exposed_tail_priority: bool,
}

#[derive(Clone, Copy)]
struct ReportSnapshot {
    lower: f64,
    upper: f64,
    steps: usize,
}

fn number(value: f64) -> String {
    if value.is_nan() {
        "NA".to_owned()
    } else {
        format!("{value:.9}")
    }
}

fn report(
    search: &SparseDfaAnytime,
    baseline_nats: (f64, f64),
    elapsed: f64,
    work_seconds: f64,
    previous: Option<ReportSnapshot>,
    detailed: bool,
) -> ReportSnapshot {
    let diagnostic = search.diagnostics();
    let lower = -diagnostic.bounds.ln_upper();
    let upper = -diagnostic.bounds.ln_lower();
    let snapshot = ReportSnapshot {
        lower,
        upper,
        steps: search.steps(),
    };
    let improvement = previous.map_or(f64::NAN, |old| {
        if old.upper.is_finite() && upper.is_finite() {
            (lower - old.lower) + (old.upper - upper)
        } else {
            f64::NAN
        }
    });
    let steps = previous.map_or(0, |old| search.steps() - old.steps);
    let per_step = if steps == 0 {
        f64::NAN
    } else {
        improvement / steps as f64
    };
    let per_second = if work_seconds > 0.0 {
        improvement / work_seconds
    } else {
        f64::NAN
    };
    let (uniform, kt) = baseline_nats;
    let values = [
        lower,
        upper,
        upper - lower,
        uniform / upper,
        uniform / lower,
        kt / upper,
        kt / lower,
        previous.map_or(f64::NAN, |old| lower - old.lower),
        previous.map_or(f64::NAN, |old| old.upper - upper),
        per_step,
        per_second,
        elapsed,
        work_seconds,
    ];
    println!(
        "{}\t{}\t{}\t{}\t{}",
        search.steps(),
        search.regions(),
        search.resolved_regions(),
        diagnostic.bound_scan_bytes,
        values
            .into_iter()
            .map(number)
            .collect::<Vec<_>>()
            .join("\t"),
    );
    if detailed {
        eprintln!(
            "diagnostics steps={} ln_unresolved_upper={} (shares are fractions of summed upper bounds, not posterior probabilities)",
            search.steps(),
            number(diagnostic.ln_unresolved_upper)
        );
        eprintln!(
            "kind\tregions\tln_upper\tupper_share\tmean_forced_bytes\tupper_weighted_forced_bytes\trefinements"
        );
        for category in diagnostic.categories {
            eprintln!(
                "{}\t{}\t{}\t{}\t{:.3}\t{:.3}\t{}",
                category.kind,
                category.regions,
                number(category.ln_upper),
                number(category.upper_share),
                category.mean_forced_prefix_bytes,
                category.upper_weighted_forced_prefix_bytes,
                category.refinements,
            );
        }
    }
    snapshot
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    let started = Instant::now();
    let mut data = Vec::new();
    File::open(&args.path)?
        .take(args.limit as u64)
        .read_to_end(&mut data)?;

    let uniform_nats = data.len() as f64 * 8.0 * std::f64::consts::LN_2;
    let kt_nats = evaluate(&data[..], &mut Kt::default())?.total_nats;
    let schedule = if args.exposed_tail_priority {
        Schedule::ExposedTailMass
    } else {
        Schedule::UpperMass
    };
    let mut search = SparseDfaAnytime::with_schedule(&data, schedule);

    println!("input: {:?}", args.path);
    println!("prefix_bytes: {}", data.len());
    println!("schedule: {schedule:?}");
    println!("target: exact sparse-DFA Bayesian mixture prequential cost");
    println!(
        "note: this binary certifies joint evidence; it is not yet the finite-compute streaming codec"
    );
    println!();
    println!(
        "steps\tregions\tresolved_regions\tbound_scan_bytes\tcode_lower_nats\tcode_upper_nats\tgap_nats\tratio_uniform_lower\tratio_uniform_upper\tratio_kt_lower\tratio_kt_upper\tlower_gain_nats\tupper_gain_nats\tgap_gain_nats_per_step\tgap_gain_nats_per_work_s\telapsed_s\twork_s"
    );
    let mut previous = report(
        &search,
        (uniform_nats, kt_nats),
        started.elapsed().as_secs_f64(),
        0.0,
        None,
        args.diagnostics,
    );

    let report_every = args.report_every.get();
    while search.steps() < args.steps && search.has_work() {
        let remaining = args.steps - search.steps();
        let work_started = Instant::now();
        search.run(remaining.min(report_every));
        let work_seconds = work_started.elapsed().as_secs_f64();
        previous = report(
            &search,
            (uniform_nats, kt_nats),
            started.elapsed().as_secs_f64(),
            work_seconds,
            Some(previous),
            args.diagnostics,
        );
    }
    if !search.has_work() {
        eprintln!("stopped: no refinable regions remain (opaque mass, if any, is still included)");
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
    fn diagnostic_numbers_preserve_small_changes_and_undefined_values() {
        assert_eq!(number(2382.265), "2382.265000000");
        assert_eq!(number(f64::NAN), "NA");
        assert_eq!(number(f64::INFINITY), "inf");
    }
}
