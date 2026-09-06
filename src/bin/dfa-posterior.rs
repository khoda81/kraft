use std::{
    env,
    ffi::OsString,
    fs::File,
    io::{self, BufReader, Read},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use kraft::{Distribution, Model, baselines::Kt, models::partial_dfa::ExactPartialDfaMixture};

const HELP: &str = "Usage: dfa-posterior <file> [--states N] [--limit BYTES] [--epsilon NATS] [--max-components N] [--report-every N]

Exact oracle over all labeled byte-input DFAs with a fixed state count.
Transitions are instantiated lazily and unused state labels are canonicalized.
Each state uses an integrated Dirichlet-1/2 byte predictor.

Defaults:
  --states 2
  --limit 64
  --epsilon 0.01
  --max-components 2000000
  --report-every 1

The run stops before an observation whose unmerged child count could exceed
--max-components. The exact posterior itself is never pruned. Instead, each
report gives the minimum number of top-posterior components that would retain
enough mass for D_KL(Q || P) <= epsilon.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    states: u16,
    limit: u64,
    epsilon: f64,
    max_components: usize,
    report_every: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut states = 2_u16;
    let mut limit = 64_u64;
    let mut epsilon = 0.01_f64;
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
            && (arg == "--states"
                || arg == "--limit"
                || arg == "--epsilon"
                || arg == "--max-components"
                || arg == "--report-every")
        {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid(format!("invalid UTF-8 value for {arg:?}")))?;
            if arg == "--states" {
                states = text
                    .parse()
                    .ok()
                    .filter(|value| (1..=256).contains(value))
                    .ok_or_else(|| invalid("--states must be an integer in 1..=256"))?;
            } else if arg == "--limit" {
                limit = text
                    .parse()
                    .map_err(|_| invalid("--limit must be a nonnegative integer"))?;
            } else if arg == "--epsilon" {
                epsilon = text
                    .parse()
                    .ok()
                    .filter(|value: &f64| value.is_finite() && *value >= 0.0)
                    .ok_or_else(|| invalid("--epsilon must be a finite nonnegative number"))?;
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
        states,
        limit,
        epsilon,
        max_components,
        report_every,
    }))
}

fn coding_ratio_uniform(bytes: u64, total_nats: f64) -> f64 {
    bytes as f64 * 8.0 * std::f64::consts::LN_2 / total_nats
}

fn run(args: &Args) -> io::Result<()> {
    let input = File::open(&args.path)?;
    let mut reader = BufReader::new(input).take(args.limit);
    let mut mixture =
        ExactPartialDfaMixture::new(args.states).map_err(|error| invalid(error.to_string()))?;
    let mut kt = Kt::default();

    let mut total_nats = 0.0;
    let mut kt_total_nats = 0.0;
    let mut bytes = 0_u64;
    let started = Instant::now();

    println!("input: {:?}", args.path);
    println!("states: {}", args.states);
    println!("limit_bytes: {}", args.limit);
    println!("epsilon_nats: {:.12}", args.epsilon);
    println!("max_components: {}", args.max_components);
    println!();
    println!(
        "step\tbyte\tcomponents\tretain_eps\tretain_fraction\tretain_mass\tkl_nats\teffective_components\ttop_mass\tassigned_edges\tnonzero_counts\tpayload_MB\tcoding_ratio_uniform\tcoding_ratio_kt\telapsed_s"
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
            let prospective = mixture.prospective_child_count(byte);
            if prospective > args.max_components {
                eprintln!(
                    "stopped_before_step: {}\nreason: prospective_unmerged_components {} exceeds max_components {}",
                    bytes + 1,
                    prospective,
                    args.max_components
                );
                break 'outer;
            }

            let ln_probability = mixture.predict().ln_prob(&byte);
            let kt_ln_probability = kt.predict().ln_prob(&byte);
            if !ln_probability.is_finite() || ln_probability > 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid DFA mixture ln_prob {ln_probability} at offset {bytes}"),
                ));
            }
            if !kt_ln_probability.is_finite() || kt_ln_probability > 0.0 {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("invalid KT ln_prob {kt_ln_probability} at offset {bytes}"),
                ));
            }

            total_nats -= ln_probability;
            kt_total_nats -= kt_ln_probability;
            mixture.observe(byte);
            kt.observe(byte);
            bytes += 1;

            if bytes.is_multiple_of(args.report_every) || bytes == args.limit {
                let diagnostics = mixture.diagnostics(args.epsilon);
                let retain_fraction =
                    diagnostics.retained_components as f64 / diagnostics.components as f64;
                let payload_mb = diagnostics.payload_bytes_estimate as f64 / 1_000_000.0;
                let coding_ratio_uniform = coding_ratio_uniform(bytes, total_nats);
                let coding_ratio_kt = kt_total_nats / total_nats;
                println!(
                    "{}\t{}\t{}\t{}\t{:.9}\t{:.12}\t{:.12}\t{:.3}\t{:.12}\t{}\t{}\t{:.3}\t{:.9}\t{:.9}\t{:.6}",
                    bytes,
                    byte,
                    diagnostics.components,
                    diagnostics.retained_components,
                    retain_fraction,
                    diagnostics.retained_mass,
                    diagnostics.retained_kl_nats,
                    diagnostics.effective_components,
                    diagnostics.top_component_mass,
                    diagnostics.assigned_transitions,
                    diagnostics.nonzero_emission_counts,
                    payload_mb,
                    coding_ratio_uniform,
                    coding_ratio_kt,
                    started.elapsed().as_secs_f64(),
                );
            }
        }
    }

    println!();
    println!("processed_bytes: {bytes}");
    println!("total_nats: {total_nats:.12}");
    if bytes > 0 {
        println!(
            "coding_ratio_uniform: {:.12}",
            coding_ratio_uniform(bytes, total_nats)
        );
        println!("coding_ratio_kt: {:.12}", kt_total_nats / total_nats);
        let diagnostics = mixture.diagnostics(args.epsilon);
        println!("components: {}", diagnostics.components);
        println!(
            "retained_components_epsilon: {}",
            diagnostics.retained_components
        );
        println!("retained_mass: {:.12}", diagnostics.retained_mass);
        println!("retained_kl_nats: {:.12}", diagnostics.retained_kl_nats);
        println!(
            "payload_estimate_MB: {:.3}",
            diagnostics.payload_bytes_estimate as f64 / 1_000_000.0
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
            eprintln!("dfa-posterior: {error}");
            ExitCode::FAILURE
        }
    }
}
