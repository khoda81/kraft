use std::{
    env,
    ffi::OsString,
    fs::{self, File},
    io::{self, BufReader, Read},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use kraft::{
    Distribution, Model,
    baselines::Kt,
    models::{
        dfa_wmc::ExactDfaWmc2,
        partial_dfa::{DfaQuotient, ExactPartialDfaMixture},
    },
};

const HELP: &str = "Usage: dfa-wmc <file> [--limit BYTES] [--max-nodes N] [--report-every N] [--oracle-through BYTES]

Exact N=2 DFA joint evidence using a reduced ordered algebraic decision diagram.
The evaluator averages over uniform Boolean transition-table variables and
integrates the Dirichlet-1/2 emission model prequentially. It does not enumerate
posterior components and does not approximate or prune.

Defaults:
  --limit 128
  --max-nodes 10000000
  --report-every 1
  --oracle-through 0

When --oracle-through is positive, the persistent leaf oracle is evaluated in
parallel through that prefix and every joint log-evidence value must agree within
1e-10 nat. The oracle is then dropped before the symbolic run continues.";

#[derive(Debug)]
struct Args {
    path: PathBuf,
    limit: u64,
    max_nodes: usize,
    report_every: u64,
    oracle_through: u64,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut limit = 128_u64;
    let mut max_nodes = 10_000_000_usize;
    let mut report_every = 1_u64;
    let mut oracle_through = 0_u64;
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
            && (arg == "--limit"
                || arg == "--max-nodes"
                || arg == "--report-every"
                || arg == "--oracle-through")
        {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid(format!("invalid UTF-8 value for {arg:?}")))?;
            if arg == "--limit" {
                limit = text
                    .parse()
                    .map_err(|_| invalid("--limit must be a nonnegative integer"))?;
            } else if arg == "--max-nodes" {
                max_nodes = text
                    .parse()
                    .ok()
                    .filter(|value| *value >= 2)
                    .ok_or_else(|| invalid("--max-nodes must be at least 2"))?;
            } else if arg == "--report-every" {
                report_every = text
                    .parse()
                    .ok()
                    .filter(|value| *value > 0)
                    .ok_or_else(|| invalid("--report-every must be positive"))?;
            } else {
                oracle_through = text
                    .parse()
                    .map_err(|_| invalid("--oracle-through must be a nonnegative integer"))?;
            }
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        limit,
        max_nodes,
        report_every,
        oracle_through,
    }))
}

fn process_rss_mb() -> Option<f64> {
    let status = fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|line| line.starts_with("VmRSS:"))?;
    let kibibytes: u64 = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kibibytes as f64 * 1024.0 / 1_000_000.0)
}

fn coding_ratio_uniform(bytes: u64, total_nats: f64) -> f64 {
    bytes as f64 * 8.0 * std::f64::consts::LN_2 / total_nats
}

fn run(args: &Args) -> io::Result<()> {
    let input = File::open(&args.path)?;
    let mut reader = BufReader::new(input).take(args.limit);
    let mut wmc = ExactDfaWmc2::new(args.max_nodes)
        .map_err(|error| invalid(format!("cannot initialize ADD: {error}")))?;
    let mut oracle = (args.oracle_through > 0)
        .then(|| ExactPartialDfaMixture::with_quotient(2, DfaQuotient::Discovery).unwrap());
    let mut kt = Kt::default();
    let mut kt_total_nats = 0.0;
    let mut bytes = 0_u64;
    let started = Instant::now();

    println!("input: {:?}", args.path);
    println!("states: 2");
    println!("limit_bytes: {}", args.limit);
    println!("max_nodes: {}", args.max_nodes);
    println!("oracle_through: {}", args.oracle_through);
    println!();
    println!(
        "step\tbyte\ttransition_variables\tallocated_nodes\tlive_nodes\tgc_runs\treclaimed_nodes\tpayload_MB\trss_MB\tln_evidence\tcoding_ratio_uniform\tcoding_ratio_kt\toracle_delta_nats\telapsed_s"
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
            if let Err(error) = wmc.observe(byte) {
                eprintln!("stopped_before_step: {}\nreason: {error}", bytes + 1);
                break 'outer;
            }

            let kt_ln_probability = kt.predict().ln_prob(&byte);
            kt_total_nats -= kt_ln_probability;
            kt.observe(byte);
            bytes += 1;

            let mut oracle_delta = None;
            if let Some(leaf_oracle) = oracle.as_mut()
                && bytes <= args.oracle_through
            {
                let prospective = leaf_oracle.prospective_child_count(byte);
                if prospective > 10_000_000 {
                    return Err(io::Error::other(format!(
                        "leaf oracle would generate {prospective} children at step {bytes}; lower --oracle-through"
                    )));
                }
                leaf_oracle.observe(byte);
                let delta = wmc.ln_evidence() - leaf_oracle.ln_evidence();
                if delta.abs() > 1e-10 {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("ADD/leaf log-evidence mismatch {delta} nat at step {bytes}"),
                    ));
                }
                oracle_delta = Some(delta);
            }

            if bytes.is_multiple_of(args.report_every) || bytes == args.limit {
                let diagnostics = wmc.diagnostics();
                let total_nats = -diagnostics.ln_evidence;
                let rss_mb = process_rss_mb()
                    .map(|value| format!("{value:.3}"))
                    .unwrap_or_else(|| "n/a".to_owned());
                let oracle_delta = oracle_delta
                    .map(|value| format!("{value:.3e}"))
                    .unwrap_or_else(|| "n/a".to_owned());
                println!(
                    "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.3}\t{}\t{:.12}\t{:.9}\t{:.9}\t{}\t{:.6}",
                    bytes,
                    byte,
                    diagnostics.transition_variables,
                    diagnostics.allocated_nodes,
                    diagnostics.live_nodes,
                    diagnostics.garbage_collections,
                    diagnostics.reclaimed_nodes,
                    diagnostics.payload_bytes_estimate as f64 / 1_000_000.0,
                    rss_mb,
                    diagnostics.ln_evidence,
                    coding_ratio_uniform(bytes, total_nats),
                    kt_total_nats / total_nats,
                    oracle_delta,
                    started.elapsed().as_secs_f64(),
                );
            }

            if bytes == args.oracle_through {
                oracle = None;
            }
        }
    }

    let diagnostics = wmc.diagnostics();
    println!();
    println!("processed_bytes: {bytes}");
    println!("ln_evidence: {:.12}", diagnostics.ln_evidence);
    if bytes > 0 {
        let total_nats = -diagnostics.ln_evidence;
        println!("total_nats: {total_nats:.12}");
        println!(
            "coding_ratio_uniform: {:.12}",
            coding_ratio_uniform(bytes, total_nats)
        );
        println!("coding_ratio_kt: {:.12}", kt_total_nats / total_nats);
    }
    println!("transition_variables: {}", diagnostics.transition_variables);
    println!("allocated_nodes: {}", diagnostics.allocated_nodes);
    println!("live_nodes: {}", diagnostics.live_nodes);
    println!("garbage_collections: {}", diagnostics.garbage_collections);
    println!("reclaimed_nodes: {}", diagnostics.reclaimed_nodes);
    println!(
        "payload_estimate_MB: {:.3}",
        diagnostics.payload_bytes_estimate as f64 / 1_000_000.0
    );
    match process_rss_mb() {
        Some(rss_mb) => println!("process_rss_MB: {rss_mb:.3}"),
        None => println!("process_rss_MB: n/a"),
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
            eprintln!("dfa-wmc: {error}");
            ExitCode::FAILURE
        }
    }
}
