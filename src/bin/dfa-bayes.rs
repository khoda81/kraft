use std::{
    env,
    ffi::OsString,
    fs::File,
    io::{self, BufReader, Read},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use kraft::{
    Distribution, Model,
    baselines::Kt,
    models::{dfa_prior::ExactDfaPriorPosterior, partial_dfa::DfaQuotient},
};

const HELP: &str = "Usage: dfa-bayes <file> [options]

Exact Bayesian inference for a finite prefix of a proper prior over all finite
labeled byte-input DFAs.

Prior:
  P(N) = 2^-N, N >= 1
  P(delta | N) = N^(-256N)

Each DFA state uses an integrated Dirichlet-1/2 byte predictor.

The implementation evaluates N=1..=max_states exactly and keeps a certified
upper bound for all omitted N>max_states classes.

Options:
  --max-states N
  --quotient discovery|predictive
  --limit BYTES
  --max-components N
  --report-every N

Defaults:
  --max-states 3
  --quotient discovery
  --limit 32
  --max-components 2000000
  --report-every 1

The command stops before an observation whose total unmerged exact child count
would exceed --max-components.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    max_states: u16,
    quotient: DfaQuotient,
    limit: u64,
    max_components: usize,
    report_every: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut max_states = 3_u16;
    let mut quotient = DfaQuotient::Discovery;
    let mut limit = 32_u64;
    let mut max_components = 2_000_000_usize;
    let mut report_every = 1_u64;
    let mut positional = false;

    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }

        if !positional
            && (arg == "--max-states"
                || arg == "--quotient"
                || arg == "--limit"
                || arg == "--max-components"
                || arg == "--report-every")
        {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid(format!("invalid UTF-8 value for {arg:?}")))?;

            if arg == "--max-states" {
                max_states = text
                    .parse()
                    .ok()
                    .filter(|value| (1..=256).contains(value))
                    .ok_or_else(|| invalid("--max-states must be an integer in 1..=256"))?;
            } else if arg == "--quotient" {
                quotient = match text {
                    "discovery" => DfaQuotient::Discovery,
                    "predictive" => DfaQuotient::Predictive,
                    _ => return Err(invalid("--quotient must be discovery or predictive")),
                };
            } else if arg == "--limit" {
                limit = text
                    .parse()
                    .map_err(|_| invalid("--limit must be a nonnegative integer"))?;
            } else if arg == "--max-components" {
                max_components = text
                    .parse()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| invalid("--max-components must be positive"))?;
            } else {
                report_every = text
                    .parse()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| invalid("--report-every must be positive"))?;
            }
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        max_states,
        quotient,
        limit,
        max_components,
        report_every,
    }))
}

fn coding_ratio_uniform(bytes: u64, total_nats: f64) -> f64 {
    bytes as f64 * 8.0 * std::f64::consts::LN_2 / total_nats
}

fn format_state_posterior(posterior: &ExactDfaPriorPosterior) -> String {
    posterior
        .diagnostics()
        .state_counts
        .into_iter()
        .map(|class| format!("{}:{:.6}", class.states, class.retained_posterior_mass))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_component_counts(posterior: &ExactDfaPriorPosterior) -> String {
    posterior
        .diagnostics()
        .state_counts
        .into_iter()
        .map(|class| format!("{}:{}", class.states, class.components))
        .collect::<Vec<_>>()
        .join(",")
}

fn format_prospective_counts(posterior: &ExactDfaPriorPosterior, byte: u8) -> String {
    posterior
        .prospective_child_counts(byte)
        .into_iter()
        .map(|(states, count)| format!("{states}:{count}"))
        .collect::<Vec<_>>()
        .join(",")
}

fn run(args: &Args) -> io::Result<()> {
    let input = File::open(&args.path)?;
    let mut reader = BufReader::new(input).take(args.limit);
    let mut posterior = ExactDfaPriorPosterior::with_quotient(args.max_states, args.quotient)
        .map_err(|error| invalid(error.to_string()))?;
    let mut kt = Kt::default();

    let mut total_nats = 0.0;
    let mut kt_total_nats = 0.0;
    let mut bytes = 0_u64;
    let started = Instant::now();

    println!("input: {:?}", args.path);
    println!("max_states: {}", args.max_states);
    println!(
        "quotient: {}",
        match args.quotient {
            DfaQuotient::Discovery => "discovery",
            DfaQuotient::Predictive => "predictive",
        }
    );
    println!("limit_bytes: {}", args.limit);
    println!("max_components: {}", args.max_components);
    println!(
        "omitted_prior_mass_initial: {:.12}",
        posterior.omitted_prior_mass_upper()
    );
    println!("state_count_prior: P(N)=2^-N");
    println!("transition_prior: iid uniform destinations within each N");
    println!();
    println!(
        "step\tbyte\tcomponents\tposterior_N\tcomponents_N\tomitted_posterior_upper\tkl_upper_nats\tcoding_ratio_uniform\tcoding_ratio_kt\telapsed_s"
    );

    let mut buffer = [0_u8; 64 * 1024];
    'outer: loop {
        let length = match reader.read(&mut buffer) {
            Ok(0) => break,
            Ok(length) => length,
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error),
        };

        for &byte in &buffer[..length] {
            let prospective = posterior.prospective_child_count(byte);
            if prospective > args.max_components {
                eprintln!(
                    "stopped_before_step: {}\nreason: prospective_unmerged_components {} exceeds max_components {}\nprospective_by_N: {}",
                    bytes + 1,
                    prospective,
                    args.max_components,
                    format_prospective_counts(&posterior, byte),
                );
                break 'outer;
            }

            let ln_probability = posterior.predict().ln_prob(&byte);
            let kt_ln_probability = kt.predict().ln_prob(&byte);
            if !ln_probability.is_finite() || ln_probability > 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid DFA posterior ln_prob {ln_probability} at offset {bytes}"),
                ));
            }

            total_nats -= ln_probability;
            kt_total_nats -= kt_ln_probability;
            posterior.observe(byte);
            kt.observe(byte);
            bytes += 1;

            if bytes.is_multiple_of(args.report_every) || bytes == args.limit {
                let diagnostics = posterior.diagnostics();
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.9}\t{:.9}\t{:.6}",
                    bytes,
                    byte,
                    diagnostics.components,
                    format_state_posterior(&posterior),
                    format_component_counts(&posterior),
                    diagnostics.omitted_posterior_mass_upper,
                    diagnostics.retained_to_full_kl_upper_nats,
                    coding_ratio_uniform(bytes, total_nats),
                    kt_total_nats / total_nats,
                    started.elapsed().as_secs_f64(),
                );
            }
        }
    }

    let diagnostics = posterior.diagnostics();
    println!();
    println!("processed_bytes: {bytes}");
    println!("total_nats: {total_nats:.12}");
    if bytes > 0 {
        println!(
            "coding_ratio_uniform: {:.12}",
            coding_ratio_uniform(bytes, total_nats)
        );
        println!("coding_ratio_kt: {:.12}", kt_total_nats / total_nats);
    }
    println!("components: {}", diagnostics.components);
    println!("posterior_N: {}", format_state_posterior(&posterior));
    println!("components_N: {}", format_component_counts(&posterior));
    println!(
        "omitted_prior_mass_upper: {:.12}",
        diagnostics.omitted_prior_mass_upper
    );
    println!(
        "omitted_posterior_mass_upper: {:.12}",
        diagnostics.omitted_posterior_mass_upper
    );
    println!(
        "retained_to_full_kl_upper_nats: {:.12}",
        diagnostics.retained_to_full_kl_upper_nats
    );
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
            eprintln!("dfa-bayes: {error}");
            ExitCode::FAILURE
        }
    }
}
