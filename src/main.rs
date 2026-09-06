use std::{
    env,
    ffi::OsString,
    fs::File,
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
    process::ExitCode,
    time::Instant,
};

use kraft::{
    Model,
    baselines::{Kt, Uniform},
    evaluate_with_costs,
};

const HELP: &str = "Usage: kraft <file> [--model kt|uniform] [--limit BYTES] [--costs PATH]

Scores raw file bytes in order, predicting before each observation.
Default model: kt (online byte unigram, Dirichlet-1/2 prior).
--limit scores only the first BYTES bytes (default: whole file).
--costs streams zero-based byte offsets and costs in nats to a NEW CSV file.
Reports ideal total coding cost in nats/bits and bits per byte.
Use -- before a positional file path starting with a dash.";

struct Args {
    path: PathBuf,
    model: String,
    limit: Option<u64>,
    costs: Option<PathBuf>,
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut model = "kt".to_owned();
    let mut limit = None;
    let mut costs = None;
    let mut positional = false;
    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }
        if !positional && (arg == "--model" || arg == "--limit" || arg == "--costs") {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            if arg == "--model" {
                model = value
                    .into_string()
                    .map_err(|_| invalid("invalid model name"))?;
            } else if arg == "--limit" {
                limit = Some(
                    value
                        .to_str()
                        .and_then(|s| s.parse::<u64>().ok())
                        .ok_or_else(|| invalid("--limit must be a nonnegative integer"))?,
                );
            } else {
                costs = Some(PathBuf::from(value));
            }
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }
    if model != "kt" && model != "uniform" {
        return Err(invalid("--model must be kt or uniform"));
    }
    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        model,
        limit,
        costs,
    }))
}

fn run(args: &Args, mut model: impl Model<u8>) -> io::Result<()> {
    let input = File::open(&args.path)?;
    // create_new prevents accidental truncation, including aliases of the input.
    let mut output = args
        .costs
        .as_ref()
        .map(|path| File::create_new(path).map(BufWriter::new))
        .transpose()?;
    if let Some(output) = &mut output {
        writeln!(output, "byte_offset,cost_nats")?;
    }
    let started = Instant::now();
    let mut offset = 0_u64;
    let report = evaluate_with_costs(
        input.take(args.limit.unwrap_or(u64::MAX)),
        &mut model,
        |cost| {
            if let Some(output) = &mut output {
                writeln!(output, "{offset},{cost:.17}")?;
            }
            offset += 1;
            Ok(())
        },
    )?;
    if let Some(output) = &mut output {
        output.flush()?;
    }
    let elapsed = started.elapsed().as_secs_f64();
    let stdout = io::stdout();
    let mut stdout = stdout.lock();
    writeln!(stdout, "input: {:?}", args.path)?;
    writeln!(stdout, "model: {}", args.model)?;
    writeln!(stdout, "limit_bytes: {:?}", args.limit)?;
    writeln!(stdout, "bytes: {}", report.bytes)?;
    writeln!(stdout, "total_nats: {:.12}", report.total_nats)?;
    writeln!(stdout, "total_bits: {:.12}", report.total_bits())?;
    match report.bits_per_byte() {
        Some(value) => writeln!(stdout, "bits_per_byte: {value:.12}")?,
        None => writeln!(stdout, "bits_per_byte: n/a")?,
    }
    writeln!(stdout, "evaluation_seconds: {elapsed:.6}")?;
    Ok(())
}

fn main() -> ExitCode {
    let result = parse(env::args_os().skip(1)).and_then(|args| match args {
        None => {
            println!("{HELP}");
            Ok(())
        }
        Some(args) if args.model == "uniform" => run(&args, Uniform),
        Some(args) => run(&args, Kt::default()),
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kraft: {error}");
            ExitCode::FAILURE
        }
    }
}
