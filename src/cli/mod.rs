mod context_fit;
mod dfa_bayes;
mod dfa_fit;
mod dfa_posterior;
mod dfa_wmc;
mod ngram_fit;
mod partition_dfa;
mod sparse_dfa_anytime;
mod sparse_dfa_fit;

use std::{
    env,
    ffi::{OsStr, OsString},
    fs::{self, File},
    io::{self, BufWriter, Read, Write},
    path::PathBuf,
    process::ExitCode,
    time::{Instant, SystemTime, UNIX_EPOCH},
};

use kraft::{
    Model,
    baselines::{Kt, Uniform},
    evaluate_with_costs,
};

const HELP: &str = r#"KRAFT — compute-aware Bayesian program mixtures

Usage:
  kraft eval <model> [INPUT] [options]
  kraft infer <family> [INPUT] [options]
  kraft search <family> [INPUT] [options]
  kraft verify <method> [INPUT] [options]

INPUT is optional. If omitted, or if INPUT is -, bytes are read from stdin.
When supplied, INPUT must immediately follow the model/family/method name.

Evaluation:
  kraft eval kt [INPUT] [--limit BYTES] [--costs PATH]
  kraft eval uniform [INPUT] [--limit BYTES] [--costs PATH]
  kraft eval ngram [INPUT] [ngram options]
  kraft eval partition-dfa [INPUT] [partition options]

Bayesian inference:
  kraft infer dfa [INPUT] [--max-states N ...]
  kraft infer dfa-prior [INPUT] [--max-states N ...]
  kraft infer dfa-fixed [INPUT] --states N [...]
  kraft infer sparse-dfa [INPUT] [anytime options]

Hindsight/oracle search:
  kraft search context [INPUT] [context options]
  kraft search dfa [INPUT] [dense-DFA search options]
  kraft search sparse-dfa [INPUT] [sparse-DFA search options]

Verification:
  kraft verify dfa-wmc [INPUT] [WMC options]

Examples:
  head -c 1000000 enwik8 | kraft eval partition-dfa --depth 8
  kraft eval ngram enwik8 --orders 0,1,2,3,4
  kraft infer dfa enwik8 --max-states 3
  kraft infer dfa-fixed enwik8 --states 2 --limit 64
  kraft search sparse-dfa enwik8 --max-exceptions 32
"#;

#[derive(Debug)]
enum Input {
    File(PathBuf),
    Stdin,
}

struct MaterializedInput {
    path: PathBuf,
    temporary: bool,
}

impl Drop for MaterializedInput {
    fn drop(&mut self) {
        if self.temporary {
            let _ = fs::remove_file(&self.path);
        }
    }
}

pub fn main() -> ExitCode {
    match run(env::args_os().skip(1).collect()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("kraft: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run(args: Vec<OsString>) -> io::Result<()> {
    if args.is_empty() || args.first().is_some_and(|arg| is_help(arg)) {
        print!("{HELP}");
        return Ok(());
    }

    let mut args = args.into_iter();
    let group = next_name(&mut args, "command group")?;
    let command = next_name(&mut args, "subcommand")?;
    let rest: Vec<_> = args.collect();

    match (group.as_str(), command.as_str()) {
        ("eval", "kt") => eval_basic("kt", rest),
        ("eval", "uniform") => eval_basic("uniform", rest),
        ("eval", "ngram") => delegated(rest, ngram_fit::command),
        ("eval", "partition-dfa" | "partition") => delegated(rest, partition_dfa::command),

        ("infer", "dfa") => {
            if rest
                .iter()
                .any(|arg| arg == "--states" || arg == "--epsilon")
            {
                delegated(rest, dfa_posterior::command)
            } else {
                delegated(rest, dfa_bayes::command)
            }
        }
        ("infer", "dfa-prior") => delegated(rest, dfa_bayes::command),
        ("infer", "dfa-fixed") => delegated(rest, dfa_posterior::command),
        ("infer", "sparse-dfa" | "sparse") => delegated(rest, sparse_dfa_anytime::command),

        ("search", "context" | "context-partition") => delegated(rest, context_fit::command),
        ("search", "dfa") => delegated(rest, dfa_fit::command),
        ("search", "sparse-dfa" | "sparse") => delegated(rest, sparse_dfa_fit::command),

        ("verify", "dfa-wmc" | "wmc") => delegated(rest, dfa_wmc::command),

        _ => Err(invalid(format!(
            "unknown command {group} {command}; run kraft --help"
        ))),
    }
}

fn next_name(args: &mut impl Iterator<Item = OsString>, what: &str) -> io::Result<String> {
    args.next()
        .ok_or_else(|| invalid(format!("missing {what}; run kraft --help")))?
        .into_string()
        .map_err(|_| invalid(format!("{what} must be valid UTF-8")))
}

fn is_help(arg: &OsStr) -> bool {
    arg == "-h" || arg == "--help"
}

fn split_input(mut args: Vec<OsString>) -> (Input, Vec<OsString>) {
    let Some(first) = args.first() else {
        return (Input::Stdin, args);
    };

    if first == "-" {
        args.remove(0);
        return (Input::Stdin, args);
    }

    if !first.to_string_lossy().starts_with('-') {
        return (Input::File(PathBuf::from(args.remove(0))), args);
    }

    (Input::Stdin, args)
}

fn delegated(args: Vec<OsString>, command: fn(Vec<OsString>) -> io::Result<()>) -> io::Result<()> {
    if args.iter().any(|arg| is_help(arg)) {
        return command(args);
    }

    let (input, rest) = split_input(args);
    let input = materialize(input)?;
    let mut inner = Vec::with_capacity(rest.len() + 1);
    inner.push(input.path.clone().into_os_string());
    inner.extend(rest);
    command(inner)
}

fn materialize(input: Input) -> io::Result<MaterializedInput> {
    match input {
        Input::File(path) => Ok(MaterializedInput {
            path,
            temporary: false,
        }),
        Input::Stdin => {
            let mut data = Vec::new();
            io::stdin().read_to_end(&mut data)?;
            let nonce = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos();
            let path =
                env::temp_dir().join(format!("kraft-stdin-{}-{nonce}.bin", std::process::id()));
            let mut file = File::create_new(&path)?;
            file.write_all(&data)?;
            file.flush()?;
            Ok(MaterializedInput {
                path,
                temporary: true,
            })
        }
    }
}

fn eval_basic(model_name: &str, args: Vec<OsString>) -> io::Result<()> {
    if args.iter().any(|arg| is_help(arg)) {
        println!(
            "Usage: kraft eval {model_name} [INPUT] [--limit BYTES] [--costs PATH]\n\
             INPUT defaults to stdin; use - explicitly for stdin."
        );
        return Ok(());
    }

    let (input, options) = split_input(args);
    let (limit, costs) = parse_basic_options(options)?;
    match model_name {
        "uniform" => eval_model(input, model_name, limit, costs, Uniform),
        "kt" => eval_model(input, model_name, limit, costs, Kt::default()),
        _ => unreachable!("basic evaluator only dispatches known models"),
    }
}

fn parse_basic_options(args: Vec<OsString>) -> io::Result<(Option<u64>, Option<PathBuf>)> {
    let mut args = args.into_iter();
    let mut limit = None;
    let mut costs = None;

    while let Some(arg) = args.next() {
        if arg == "--limit" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --limit"))?;
            limit = Some(
                value
                    .to_str()
                    .and_then(|text| text.parse().ok())
                    .ok_or_else(|| invalid("--limit must be a nonnegative integer"))?,
            );
        } else if arg == "--costs" {
            let value = args
                .next()
                .ok_or_else(|| invalid("missing value for --costs"))?;
            costs = Some(PathBuf::from(value));
        } else {
            return Err(invalid(format!(
                "unknown eval option {arg:?}; run kraft --help"
            )));
        }
    }

    Ok((limit, costs))
}

fn eval_model(
    input: Input,
    model_name: &str,
    limit: Option<u64>,
    costs: Option<PathBuf>,
    mut model: impl Model<u8>,
) -> io::Result<()> {
    let (reader, label): (Box<dyn Read>, String) = match input {
        Input::File(path) => {
            let label = path.display().to_string();
            (Box::new(File::open(path)?), label)
        }
        Input::Stdin => (Box::new(io::stdin()), "<stdin>".to_owned()),
    };

    let mut output = costs
        .as_ref()
        .map(|path| File::create_new(path).map(BufWriter::new))
        .transpose()?;
    if let Some(output) = &mut output {
        writeln!(output, "byte_offset,cost_nats")?;
    }

    let started = Instant::now();
    let mut offset = 0_u64;
    let report = evaluate_with_costs(reader.take(limit.unwrap_or(u64::MAX)), &mut model, |cost| {
        if let Some(output) = &mut output {
            writeln!(output, "{offset},{cost:.17}")?;
        }
        offset += 1;
        Ok(())
    })?;
    if let Some(output) = &mut output {
        output.flush()?;
    }

    println!("input: {label}");
    println!("model: {model_name}");
    println!("limit_bytes: {limit:?}");
    println!("bytes: {}", report.bytes);
    println!("total_nats: {:.12}", report.total_nats);
    println!("total_bits: {:.12}", report.total_bits());
    match report.coding_ratio_uniform() {
        Some(value) => println!("coding_ratio_uniform: {value:.12}"),
        None => println!("coding_ratio_uniform: n/a"),
    }
    println!("evaluation_seconds: {:.6}", started.elapsed().as_secs_f64());
    Ok(())
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn omitted_input_means_stdin() {
        let (input, rest) = split_input(vec![OsString::from("--limit"), OsString::from("10")]);
        assert!(matches!(input, Input::Stdin));
        assert_eq!(rest.len(), 2);
    }

    #[test]
    fn first_positional_is_input() {
        let (input, rest) = split_input(vec![
            OsString::from("sample.bin"),
            OsString::from("--limit"),
            OsString::from("10"),
        ]);
        assert!(matches!(input, Input::File(path) if path.as_os_str() == "sample.bin"));
        assert_eq!(rest.len(), 2);
    }
}
