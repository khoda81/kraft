use std::{
    fs::File,
    io::{self, BufRead, BufReader, Read},
    path::PathBuf,
    time::Instant,
};

use clap::Parser;
use kraft::{
    Distribution, Model,
    models::{partial_dfa::ExactPartialDfaMixture, predictive_dfa::GroupedDfa},
};

#[derive(Parser)]
#[command(about = "Exact fixed-N DFA posterior with dynamic symbolic prediction groups")]
struct Args {
    /// Raw byte input; omitted or - reads stdin directly.
    input: Option<PathBuf>,
    /// State count defines the fixed-N model family, not a pruning budget.
    #[arg(long, default_value_t = 2)]
    states: u16,
    #[arg(long, default_value_t = 64)]
    limit: u64,
    /// Resource limit: stop on exhaustion without dropping posterior mass.
    #[arg(long, default_value_t = 500_000)]
    max_nodes: usize,
    #[arg(long, default_value_t = 1)]
    report_every: u64,
    /// Also run the original exact leaf oracle and check every observed prediction.
    #[arg(long)]
    compare_oracle: bool,
    #[arg(long, default_value_t = 100_000)]
    max_oracle_components: usize,
}

fn run(args: Args) -> Result<(), Box<dyn std::error::Error>> {
    if args.report_every == 0 {
        return Err("--report-every must be positive".into());
    }
    let mut model = GroupedDfa::new(args.states, args.max_nodes)?;
    let mut oracle = args
        .compare_oracle
        .then(|| ExactPartialDfaMixture::new(args.states))
        .transpose()?;
    let input: Box<dyn Read> = match args.input {
        Some(path) if path.as_os_str() != "-" => Box::new(File::open(path)?),
        _ => Box::new(io::stdin()),
    };
    let mut reader = BufReader::new(input.take(args.limit));
    let mut total_nats = 0.0;
    let mut symbolic_seconds = 0.0;
    let mut oracle_seconds = 0.0;
    let mut max_oracle_error = 0.0_f64;
    let mut oracle_likelihood_evaluations = 0_u64;
    println!(
        "bytes\tstates\tgroups\tnodes\tcount_vectors\tlikelihood_evaluations\tcount_updates\tdiagram_visits\tprojection_visits\ttotal_nats\tcoding_ratio_uniform\toracle_components\toracle_likelihood_evaluations\tmax_oracle_error_nats\tsymbolic_s\toracle_s"
    );
    let report = |model: &GroupedDfa,
                  oracle: &Option<ExactPartialDfaMixture>,
                  cost: f64,
                  error: f64,
                  symbolic_s: f64,
                  oracle_s: f64,
                  oracle_work: u64| {
        let d = model.diagnostics();
        let ratio = if d.observations == 0 {
            "n/a".to_owned()
        } else {
            format!("{:.9}", f64::from(d.observations) * 256_f64.ln() / cost)
        };
        let components = oracle
            .as_ref()
            .map_or_else(|| "n/a".to_owned(), |o| o.component_count().to_string());
        println!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12}\t{}\t{}\t{}\t{:.3e}\t{:.6}\t{:.6}",
            d.observations,
            d.states,
            d.prediction_groups,
            d.allocated_nodes,
            d.count_vectors,
            d.likelihood_evaluations,
            d.count_updates,
            d.diagram_visits,
            d.projection_visits,
            cost,
            ratio,
            components,
            oracle_work,
            error,
            symbolic_s,
            oracle_s
        );
    };
    loop {
        let buffer = reader.fill_buf()?;
        if buffer.is_empty() {
            break;
        }
        let length = buffer.len();
        for &byte in buffer {
            if let Some(oracle) = &oracle
                && oracle.prospective_child_count(byte) > args.max_oracle_components
            {
                return Err(format!(
                    "oracle budget exceeded after {} bytes; no posterior pruning performed",
                    model.diagnostics().observations
                )
                .into());
            }
            let started = Instant::now();
            let ln_prob = model.predict().ln_prob(&byte);
            model.try_observe(byte).map_err(|error| {
                format!("after {} bytes: {error}", model.diagnostics().observations)
            })?;
            symbolic_seconds += started.elapsed().as_secs_f64();
            total_nats -= ln_prob;
            if let Some(oracle) = &mut oracle {
                let started = Instant::now();
                max_oracle_error =
                    max_oracle_error.max((ln_prob - oracle.predict().ln_prob(&byte)).abs());
                oracle_likelihood_evaluations += oracle.component_count() as u64;
                oracle.observe(byte);
                max_oracle_error =
                    max_oracle_error.max((model.ln_evidence() - oracle.ln_evidence()).abs());
                oracle_seconds += started.elapsed().as_secs_f64();
                if max_oracle_error > 1e-8 {
                    return Err(format!("oracle disagreement: {max_oracle_error} nats").into());
                }
            }
            if u64::from(model.diagnostics().observations) % args.report_every == 0 {
                report(
                    &model,
                    &oracle,
                    total_nats,
                    max_oracle_error,
                    symbolic_seconds,
                    oracle_seconds,
                    oracle_likelihood_evaluations,
                );
            }
        }
        reader.consume(length);
    }
    let observations = model.diagnostics().observations;
    if observations == 0 || u64::from(observations) % args.report_every != 0 {
        report(
            &model,
            &oracle,
            total_nats,
            max_oracle_error,
            symbolic_seconds,
            oracle_seconds,
            oracle_likelihood_evaluations,
        );
    }
    Ok(())
}

pub fn command(args: Vec<std::ffi::OsString>) -> io::Result<()> {
    let argv = std::iter::once(std::ffi::OsString::from("kraft infer dfa-grouped")).chain(args);
    let args = match Args::try_parse_from(argv) {
        Ok(args) => args,
        Err(error)
            if matches!(
                error.kind(),
                clap::error::ErrorKind::DisplayHelp | clap::error::ErrorKind::DisplayVersion
            ) =>
        {
            print!("{error}");
            return Ok(());
        }
        Err(error) => {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                error.to_string(),
            ));
        }
    };
    run(args).map_err(|error| io::Error::other(error.to_string()))
}
