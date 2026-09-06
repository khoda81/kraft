use std::{
    fs::{self, File},
    io::{self, BufReader, BufWriter, Read, Write},
    path::{Path, PathBuf},
    process::ExitCode,
    time::Instant,
};

use clap::{Parser, ValueEnum};
use kraft::{
    Distribution, Model,
    baselines::Kt,
    models::{dfa_prior::ExactDfaPriorPosterior, partial_dfa::DfaQuotient},
};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum QuotientArg {
    Discovery,
    Predictive,
}

impl From<QuotientArg> for DfaQuotient {
    fn from(value: QuotientArg) -> Self {
        match value {
            QuotientArg::Discovery => Self::Discovery,
            QuotientArg::Predictive => Self::Predictive,
        }
    }
}

#[derive(Debug, Parser)]
#[command(
    about = "Exact Bayesian inference over a finite prefix of the proper recurrent-DFA prior",
    long_about = "Evaluate N=1..=max-states exactly under the proper prior P(N)=2^-N and iid-uniform transition tables. Larger state-count classes remain represented by a certified omitted-posterior bound. Each DFA state integrates a Dirichlet-1/2 byte predictor."
)]
struct Args {
    /// Input byte corpus.
    #[arg(value_name = "FILE")]
    path: PathBuf,

    /// Largest DFA state count evaluated exactly.
    #[arg(long, default_value_t = 3)]
    max_states: u16,

    /// Exact quotient used inside each fixed-N posterior.
    #[arg(long, value_enum, default_value_t = QuotientArg::Discovery)]
    quotient: QuotientArg,

    /// Maximum bytes to process.
    #[arg(long, default_value_t = 32)]
    limit: u64,

    /// Stop before a byte whose exact unmerged child count exceeds this value.
    #[arg(long, default_value_t = 2_000_000)]
    max_components: usize,

    /// Emit a diagnostics row every N processed bytes.
    #[arg(long, default_value_t = 1)]
    report_every: u64,

    /// Number of highest-mass sufficient-state aggregates to dump.
    #[arg(long, default_value_t = 100)]
    top_components: usize,

    /// Optional path for the highest-mass posterior aggregates.
    #[arg(long)]
    dump_top: Option<PathBuf>,
}

impl Args {
    fn validate(&self) -> io::Result<()> {
        if !(1..=256).contains(&self.max_states) {
            return Err(invalid("--max-states must be in 1..=256"));
        }
        if self.max_components == 0 {
            return Err(invalid("--max-components must be positive"));
        }
        if self.report_every == 0 {
            return Err(invalid("--report-every must be positive"));
        }
        if self.top_components == 0 {
            return Err(invalid("--top-components must be positive"));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
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

fn write_top_components(
    posterior: &ExactDfaPriorPosterior,
    limit: usize,
    path: &Path,
) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let diagnostics = posterior.diagnostics();
    let retained_fraction_lower = 1.0 - diagnostics.omitted_posterior_mass_upper;
    let mut writer = BufWriter::new(File::create(path)?);
    writeln!(
        writer,
        "# KRAFT exact DFA posterior sufficient-state aggregates"
    )?;
    writeln!(writer, "# max_states={}", posterior.max_states())?;
    writeln!(
        writer,
        "# omitted_posterior_mass_upper={:.12}",
        diagnostics.omitted_posterior_mass_upper
    )?;
    writeln!(
        writer,
        "# rank\tN\tretained_mass\tfull_mass_lower\twithin_N_mass\tcurrent_state\tdiscovered_states\ttransitions\temissions"
    )?;

    for (rank, hypothesis) in posterior.top_components(limit).into_iter().enumerate() {
        let transitions = hypothesis
            .component
            .assigned_transitions
            .iter()
            .map(|edge| format!("{}:{:02x}>{}", edge.source, edge.byte, edge.destination))
            .collect::<Vec<_>>()
            .join(",");

        let emissions = hypothesis
            .component
            .emissions
            .iter()
            .map(|state| {
                let counts = state
                    .counts
                    .iter()
                    .map(|&(byte, count)| format!("{byte:02x}={count}"))
                    .collect::<Vec<_>>()
                    .join(",");
                format!("{}[total={};{}]", state.state, state.total, counts)
            })
            .collect::<Vec<_>>()
            .join("|");

        writeln!(
            writer,
            "{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{}\t{}\t{}\t{}",
            rank + 1,
            hypothesis.states,
            hypothesis.retained_posterior_mass,
            hypothesis.retained_posterior_mass * retained_fraction_lower,
            hypothesis.within_class_posterior_mass,
            hypothesis.component.current_state,
            hypothesis.component.discovered_states,
            transitions,
            emissions,
        )?;
    }

    writer.flush()
}

fn run(args: &Args) -> io::Result<()> {
    let input = File::open(&args.path)?;
    let mut reader = BufReader::new(input).take(args.limit);
    let mut posterior =
        ExactDfaPriorPosterior::with_quotient(args.max_states, DfaQuotient::from(args.quotient))
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
            QuotientArg::Discovery => "discovery",
            QuotientArg::Predictive => "predictive",
        }
    );
    println!("limit_bytes: {}", args.limit);
    println!("max_components: {}", args.max_components);
    println!("top_components: {}", args.top_components);
    println!(
        "omitted_prior_mass_initial: {:.12}",
        posterior.omitted_prior_mass_upper()
    );
    println!("state_count_prior: P(N)=2^-N");
    println!("transition_prior: iid uniform destinations within each N");
    println!();
    println!(
        "step\tbyte\tcomponents\tposterior_N\tcomponents_N\tln_omitted_joint_upper\tomitted_posterior_upper\tkl_upper_nats\tcoding_ratio_uniform\tcoding_ratio_kt\telapsed_s"
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
                    "{}\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{:.9}\t{:.9}\t{:.6}",
                    bytes,
                    byte,
                    diagnostics.components,
                    format_state_posterior(&posterior),
                    format_component_counts(&posterior),
                    diagnostics.ln_omitted_joint_mass_upper,
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
        "ln_omitted_joint_mass_upper: {:.12}",
        diagnostics.ln_omitted_joint_mass_upper
    );
    println!(
        "omitted_posterior_mass_upper: {:.12}",
        diagnostics.omitted_posterior_mass_upper
    );
    println!(
        "retained_to_full_kl_upper_nats: {:.12}",
        diagnostics.retained_to_full_kl_upper_nats
    );
    if let Some(path) = &args.dump_top {
        write_top_components(&posterior, args.top_components, path)?;
        println!("top_posterior_dump: {:?}", path);
        println!("top_posterior_components: {}", args.top_components);
    }
    println!("evaluation_seconds: {:.6}", started.elapsed().as_secs_f64());

    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = args.validate().and_then(|()| run(&args));

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("dfa-bayes: {error}");
            ExitCode::FAILURE
        }
    }
}
