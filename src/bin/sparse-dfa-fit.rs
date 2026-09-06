//! Heuristic search in the proper sparse-default DFA prior.
//!
//! The prior itself is exact and proper; this binary searches for high-joint-mass
//! descriptions. Every returned candidate h certifies
//!
//! ```text
//! C_sparse_mixture(x) <= C_h(x) - ln P(h).
//! ```
//!
//! so heuristic search quality affects tightness, not validity of the bound.

use std::{collections::HashSet, fs, io, path::PathBuf, process::ExitCode, thread, time::Instant};

use clap::{Parser, ValueEnum};
use kraft::models::sparse_dfa::{DefaultTopology, SparseDfa, SparseOverride};

const LN_2: f64 = std::f64::consts::LN_2;

#[derive(Debug, Clone, Copy, ValueEnum)]
enum TopologyArg {
    Stay,
    Next,
    Cycle,
}

impl From<TopologyArg> for DefaultTopology {
    fn from(value: TopologyArg) -> Self {
        match value {
            TopologyArg::Stay => Self::Stay,
            TopologyArg::Next => Self::Next,
            TopologyArg::Cycle => Self::Cycle,
        }
    }
}

#[derive(Debug, Clone, Parser)]
#[command(
    about = "Search sparse recurrent DFAs under a proper Bayesian description prior",
    long_about = "Heuristic MAP search over sparse recurrent byte-input DFAs. Each state follows a cheap implicit stay/next/cycle topology by default, with a sparse set of byte-specific transition overrides. The model prior is proper; every returned candidate gives a rigorous upper bound on the full Bayesian mixture coding cost."
)]
struct Args {
    /// Input byte corpus.
    #[arg(value_name = "FILE")]
    path: PathBuf,

    /// State counts to search.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "1,2,4,8,16,32,64,128,256"
    )]
    states: Vec<u16>,

    /// Implicit transition skeletons to search.
    #[arg(
        long,
        value_delimiter = ',',
        default_value = "stay,next,cycle",
        value_enum
    )]
    topologies: Vec<TopologyArg>,

    /// Maximum sparse transition overrides per candidate.
    #[arg(long, default_value_t = 4)]
    max_exceptions: usize,

    /// Prefix used for structure search.
    #[arg(long, default_value_t = 100_000)]
    search_bytes: usize,

    /// Prefix used to screen searched candidates.
    #[arg(long, default_value_t = 5_000_000)]
    screen_bytes: usize,

    /// Number of bare (N, topology) skeletons to search.
    #[arg(long, default_value_t = 6)]
    skeletons: usize,

    /// Beam width at each exception depth.
    #[arg(long, default_value_t = 8)]
    beam: usize,

    /// Most-visited transition keys considered per beam parent.
    #[arg(long, default_value_t = 8)]
    keys_per_parent: usize,

    /// Candidate non-default destinations considered per transition key.
    #[arg(long, default_value_t = 6)]
    destinations_per_key: usize,

    /// Candidates retained for the screen prefix.
    #[arg(long, default_value_t = 32)]
    screen_candidates: usize,

    /// Candidates retained for full-corpus scoring.
    #[arg(long, default_value_t = 8)]
    finalists: usize,

    /// Scoring worker threads. Defaults to available parallelism.
    #[arg(long)]
    threads: Option<usize>,

    /// Deterministic search seed.
    #[arg(long, default_value_t = 1)]
    seed: u64,

    /// Write the best sparse DFA description here.
    #[arg(long, default_value = "artifacts/sparse-dfa-best.tsv")]
    dump_best: PathBuf,
}

impl Args {
    fn threads(&self) -> usize {
        self.threads
            .unwrap_or_else(|| thread::available_parallelism().map_or(1, usize::from))
    }

    fn validate(&self) -> io::Result<()> {
        if self.states.is_empty() || self.states.contains(&0) {
            return Err(invalid("--states must contain positive integers"));
        }
        for (name, value) in [
            ("--search-bytes", self.search_bytes),
            ("--screen-bytes", self.screen_bytes),
            ("--skeletons", self.skeletons),
            ("--beam", self.beam),
            ("--keys-per-parent", self.keys_per_parent),
            ("--destinations-per-key", self.destinations_per_key),
            ("--screen-candidates", self.screen_candidates),
            ("--finalists", self.finalists),
        ] {
            if value == 0 {
                return Err(invalid(format!("{name} must be positive")));
            }
        }
        if self.threads == Some(0) {
            return Err(invalid("--threads must be positive"));
        }
        if self.finalists > self.screen_candidates {
            return Err(invalid("--finalists cannot exceed --screen-candidates"));
        }
        Ok(())
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

#[derive(Debug, Clone)]
struct Candidate {
    model: SparseDfa,
    ln_evidence: f64,
    visited_keys: Vec<u64>,
}

impl Candidate {
    fn ln_joint(&self) -> f64 {
        self.ln_evidence + self.model.ln_prior()
    }

    fn exceptions(&self) -> usize {
        self.model.overrides().len()
    }
}

#[derive(Debug, Clone, Copy)]
struct SplitMix64(u64);

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut value = self.0;
        value = (value ^ (value >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        value = (value ^ (value >> 27)).wrapping_mul(0x94d_49bb_1331_11eb);
        value ^ (value >> 31)
    }
}

fn invalid(message: impl Into<String>) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, message.into())
}

fn parse_usize(name: &str, value: &str, minimum: usize) -> io::Result<usize> {
    value
        .parse()
        .ok()
        .filter(|parsed| *parsed >= minimum)
        .ok_or_else(|| invalid(format!("{name} must be an integer >= {minimum}")))
}

fn parse_states(value: &str) -> io::Result<Vec<u16>> {
    let mut states = Vec::new();
    for item in value.split(',') {
        let parsed: u16 = item
            .parse()
            .ok()
            .filter(|state| *state > 0)
            .ok_or_else(|| invalid("--states must contain positive u16 integers"))?;
        states.push(parsed);
    }
    states.sort_unstable();
    states.dedup();
    if states.is_empty() {
        return Err(invalid("--states cannot be empty"));
    }
    Ok(states)
}

fn parse_topologies(value: &str) -> io::Result<Vec<DefaultTopology>> {
    let mut topologies = Vec::new();
    for item in value.split(',') {
        let topology = DefaultTopology::parse(item)
            .ok_or_else(|| invalid("--topologies must use stay,next,cycle"))?;
        if !topologies.contains(&topology) {
            topologies.push(topology);
        }
    }
    if topologies.is_empty() {
        return Err(invalid("--topologies cannot be empty"));
    }
    Ok(topologies)
}

fn parse(args: impl IntoIterator<Item = OsString>) -> io::Result<Option<Args>> {
    let mut args = args.into_iter();
    let mut path = None;
    let mut states = parse_states("1,2,4,8,16,32,64,128,256")?;
    let mut topologies = DefaultTopology::ALL.to_vec();
    let mut max_exceptions = 4_usize;
    let mut search_bytes = 100_000_usize;
    let mut screen_bytes = 5_000_000_usize;
    let mut skeletons = 6_usize;
    let mut beam = 8_usize;
    let mut keys_per_parent = 8_usize;
    let mut destinations_per_key = 6_usize;
    let mut screen_candidates = 32_usize;
    let mut finalists = 8_usize;
    let mut threads = thread::available_parallelism().map_or(1, usize::from);
    let mut seed = 1_u64;
    let mut dump_best = PathBuf::from("artifacts/sparse-dfa-best.tsv");
    let mut positional = false;

    while let Some(arg) = args.next() {
        if !positional && (arg == "--help" || arg == "-h") {
            return Ok(None);
        }
        if !positional && arg == "--" {
            positional = true;
            continue;
        }

        if !positional && arg.to_string_lossy().starts_with("--") {
            let value = args
                .next()
                .ok_or_else(|| invalid(format!("missing value for {arg:?}")))?;
            let text = value
                .to_str()
                .ok_or_else(|| invalid(format!("invalid UTF-8 value for {arg:?}")))?;
            match arg.to_string_lossy().as_ref() {
                "--states" => states = parse_states(text)?,
                "--topologies" => topologies = parse_topologies(text)?,
                "--max-exceptions" => max_exceptions = parse_usize("--max-exceptions", text, 0)?,
                "--search-bytes" => search_bytes = parse_usize("--search-bytes", text, 1)?,
                "--screen-bytes" => screen_bytes = parse_usize("--screen-bytes", text, 1)?,
                "--skeletons" => skeletons = parse_usize("--skeletons", text, 1)?,
                "--beam" => beam = parse_usize("--beam", text, 1)?,
                "--keys-per-parent" => keys_per_parent = parse_usize("--keys-per-parent", text, 1)?,
                "--destinations-per-key" => {
                    destinations_per_key = parse_usize("--destinations-per-key", text, 1)?
                }
                "--screen-candidates" => {
                    screen_candidates = parse_usize("--screen-candidates", text, 1)?
                }
                "--finalists" => finalists = parse_usize("--finalists", text, 1)?,
                "--threads" => threads = parse_usize("--threads", text, 1)?,
                "--seed" => {
                    seed = text
                        .parse()
                        .map_err(|_| invalid("--seed must be an unsigned integer"))?
                }
                "--dump-best" => dump_best = PathBuf::from(value),
                _ => return Err(invalid(format!("unknown option {arg:?}"))),
            }
        } else if !positional && arg.to_string_lossy().starts_with('-') {
            return Err(invalid(format!("unknown option {arg:?}")));
        } else if path.replace(PathBuf::from(arg)).is_some() {
            return Err(invalid("expected exactly one input file"));
        }
    }

    if finalists > screen_candidates {
        return Err(invalid("--finalists cannot exceed --screen-candidates"));
    }

    Ok(Some(Args {
        path: path.ok_or_else(|| invalid("missing input file; use --help"))?,
        states,
        topologies,
        max_exceptions,
        search_bytes,
        screen_bytes,
        skeletons,
        beam,
        keys_per_parent,
        destinations_per_key,
        screen_candidates,
        finalists,
        threads,
        seed,
        dump_best,
    }))
}

fn score_model(model: SparseDfa, data: &[u8]) -> Candidate {
    let score = model.score(data);
    Candidate {
        model,
        ln_evidence: score.ln_evidence,
        visited_keys: score.visited_keys,
    }
}

fn score_models(models: Vec<SparseDfa>, data: &[u8], threads: usize) -> Vec<Candidate> {
    if models.len() <= 1 || threads <= 1 {
        return models
            .into_iter()
            .map(|model| score_model(model, data))
            .collect();
    }

    let worker_count = threads.min(models.len());
    let chunk_size = models.len().div_ceil(worker_count);
    thread::scope(|scope| {
        let mut handles = Vec::new();
        for chunk in models.chunks(chunk_size) {
            let owned = chunk.to_vec();
            handles.push(scope.spawn(move || {
                owned
                    .into_iter()
                    .map(|model| score_model(model, data))
                    .collect::<Vec<_>>()
            }));
        }

        let mut scored = Vec::new();
        for handle in handles {
            scored.extend(handle.join().expect("sparse DFA scoring worker panicked"));
        }
        scored
    })
}

fn sort_best(candidates: &mut [Candidate]) {
    candidates.sort_by(|left, right| right.ln_joint().total_cmp(&left.ln_joint()));
}

fn override_key_exists(model: &SparseDfa, source: u16, byte: u8) -> bool {
    model
        .overrides()
        .iter()
        .any(|edge| edge.source == source && edge.byte == byte)
}

fn top_visited_keys(candidate: &Candidate, limit: usize) -> Vec<(u16, u8, u64)> {
    let mut visited = candidate
        .visited_keys
        .iter()
        .enumerate()
        .filter_map(|(key, &count)| {
            if count == 0 {
                return None;
            }
            let source = u16::try_from(key / 256).ok()?;
            let byte = (key % 256) as u8;
            if override_key_exists(&candidate.model, source, byte) {
                None
            } else {
                Some((source, byte, count))
            }
        })
        .collect::<Vec<_>>();
    visited.sort_by_key(|item| std::cmp::Reverse(item.2));
    visited.truncate(limit);
    visited
}

fn push_destination(output: &mut Vec<u16>, value: i64, states: u16, default: u16, limit: usize) {
    if output.len() >= limit || value < 0 || value >= i64::from(states) {
        return;
    }
    let value = value as u16;
    if value != default && !output.contains(&value) {
        output.push(value);
    }
}

fn destination_candidates(
    states: u16,
    source: u16,
    default: u16,
    limit: usize,
    seed: u64,
) -> Vec<u16> {
    if states <= 1 {
        return Vec::new();
    }

    let mut output = Vec::with_capacity(limit);
    for value in [
        i64::from(source),
        i64::from(source) - 1,
        i64::from(source) + 1,
        i64::from(default) - 1,
        i64::from(default) + 1,
        0,
        i64::from(states) - 1,
    ] {
        push_destination(&mut output, value, states, default, limit);
    }

    let mut rng = SplitMix64::new(seed ^ (u64::from(source) << 32) ^ u64::from(default));
    while output.len() < limit && output.len() < usize::from(states - 1) {
        let value = (rng.next_u64() % u64::from(states)) as u16;
        if value != default && !output.contains(&value) {
            output.push(value);
        }
    }
    output
}

fn generate_children(
    parent: &Candidate,
    keys_per_parent: usize,
    destinations_per_key: usize,
    seed: u64,
) -> Vec<SparseDfa> {
    let states = parent.model.states();
    if states <= 1 {
        return Vec::new();
    }

    let mut children = Vec::new();
    for (key_index, (source, byte, _)) in top_visited_keys(parent, keys_per_parent)
        .into_iter()
        .enumerate()
    {
        let default = parent.model.default_destination(source);
        let destinations = destination_candidates(
            states,
            source,
            default,
            destinations_per_key,
            seed ^ ((key_index as u64) << 48) ^ u64::from(byte),
        );
        for destination in destinations {
            if let Ok(child) = parent.model.with_added_override(SparseOverride {
                source,
                byte,
                destination,
            }) {
                children.push(child);
            }
        }
    }
    children
}

fn search_skeleton(
    initial: Candidate,
    data: &[u8],
    args: &Args,
    skeleton_index: usize,
) -> Vec<Candidate> {
    let mut beam = vec![initial.clone()];
    let mut all = vec![initial];

    for depth in 1..=args.max_exceptions {
        let mut proposals = HashSet::new();
        for (parent_index, parent) in beam.iter().enumerate() {
            let seed = args.seed
                ^ ((skeleton_index as u64) << 40)
                ^ ((depth as u64) << 24)
                ^ parent_index as u64;
            for child in generate_children(
                parent,
                args.keys_per_parent,
                args.destinations_per_key,
                seed,
            ) {
                proposals.insert(child);
            }
        }

        if proposals.is_empty() {
            break;
        }

        eprintln!(
            "[sparse-dfa-fit] skeleton={} N={} topology={} depth={} proposals={}",
            skeleton_index,
            beam[0].model.states(),
            beam[0].model.topology().as_str(),
            depth,
            proposals.len()
        );

        let mut scored = score_models(proposals.into_iter().collect(), data, args.threads());
        sort_best(&mut scored);
        scored.truncate(args.beam);
        if scored.is_empty() {
            break;
        }
        all.extend(scored.iter().cloned());
        beam = scored;
    }

    all
}

fn rescore_candidates(
    candidates: Vec<Candidate>,
    data: &[u8],
    limit: usize,
    threads: usize,
) -> Vec<Candidate> {
    let mut dedup = HashSet::new();
    let models = candidates
        .into_iter()
        .map(|candidate| candidate.model)
        .filter(|model| dedup.insert(model.clone()))
        .collect::<Vec<_>>();
    let mut scored = score_models(models, data, threads);
    sort_best(&mut scored);
    scored.truncate(limit);
    scored
}

fn log_sum_exp(values: impl IntoIterator<Item = f64>) -> f64 {
    values.into_iter().fold(f64::NEG_INFINITY, |left, right| {
        if left == f64::NEG_INFINITY {
            return right;
        }
        if right == f64::NEG_INFINITY {
            return left;
        }
        if left >= right {
            left + (right - left).exp().ln_1p()
        } else {
            right + (left - right).exp().ln_1p()
        }
    })
}

fn write_best(path: &PathBuf, candidate: &Candidate) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::new();
    output.push_str("# KRAFT sparse DFA candidate\n");
    output.push_str(&format!("states\t{}\n", candidate.model.states()));
    output.push_str(&format!(
        "topology\t{}\n",
        candidate.model.topology().as_str()
    ));
    output.push_str(&format!("exceptions\t{}\n", candidate.exceptions()));
    output.push_str(&format!(
        "prior_bits\t{:.12}\n",
        candidate.model.prior_bits()
    ));
    output.push_str(&format!("ln_evidence\t{:.12}\n", candidate.ln_evidence));
    output.push_str("source\tbyte_hex\tdestination\n");
    for edge in candidate.model.overrides() {
        output.push_str(&format!(
            "{}\t{:02x}\t{}\n",
            edge.source, edge.byte, edge.destination
        ));
    }
    fs::write(path, output)
}

fn run(args: &Args) -> io::Result<()> {
    let started = Instant::now();
    let data = fs::read(&args.path)?;
    if data.is_empty() {
        return Err(invalid("input file is empty"));
    }

    let search_len = args.search_bytes.min(data.len());
    let screen_len = args.screen_bytes.min(data.len());
    let search = &data[..search_len];
    let screen = &data[..screen_len];

    eprintln!(
        "[sparse-dfa-fit] loaded {} bytes; search={} screen={} states={:?}",
        data.len(),
        search_len,
        screen_len,
        args.states
    );

    let mut skeleton_models = Vec::new();
    for &states in &args.states {
        for topology in args.topologies.iter().copied().map(DefaultTopology::from) {
            skeleton_models.push(
                SparseDfa::empty(states, topology).map_err(|error| invalid(error.to_string()))?,
            );
        }
    }

    let mut skeleton_scores = score_models(skeleton_models, search, args.threads());
    sort_best(&mut skeleton_scores);

    println!("search_bytes: {search_len}");
    println!("screen_bytes: {screen_len}");
    println!("skeleton_candidates: {}", skeleton_scores.len());
    println!();
    println!("skeleton_rank\tN\ttopology\tdata_nats\tprior_bits\tjoint_nats");
    for (rank, candidate) in skeleton_scores.iter().enumerate() {
        println!(
            "{}\t{}\t{}\t{:.6}\t{:.6}\t{:.6}",
            rank + 1,
            candidate.model.states(),
            candidate.model.topology().as_str(),
            -candidate.ln_evidence,
            candidate.model.prior_bits(),
            -candidate.ln_joint(),
        );
    }

    skeleton_scores.truncate(args.skeletons.min(skeleton_scores.len()));

    let mut explored = Vec::new();
    for (index, skeleton) in skeleton_scores.into_iter().enumerate() {
        eprintln!(
            "[sparse-dfa-fit] searching skeleton {}/{}: N={} topology={}",
            index + 1,
            args.skeletons,
            skeleton.model.states(),
            skeleton.model.topology().as_str()
        );
        explored.extend(search_skeleton(skeleton, search, args, index));
    }

    let mut explored_search = explored;
    sort_best(&mut explored_search);
    explored_search.truncate(args.screen_candidates.min(explored_search.len()));

    eprintln!(
        "[sparse-dfa-fit] screening {} candidates on {} bytes",
        explored_search.len(),
        screen_len
    );
    let screened = rescore_candidates(
        explored_search,
        screen,
        args.finalists.min(args.screen_candidates),
        args.threads(),
    );

    eprintln!(
        "[sparse-dfa-fit] scoring {} finalists on full {} bytes",
        screened.len(),
        data.len()
    );
    let mut finalists = rescore_candidates(screened, &data, args.finalists, args.threads());
    if finalists.is_empty() {
        return Err(invalid("search produced no finalists"));
    }
    sort_best(&mut finalists);

    let kt =
        SparseDfa::empty(1, DefaultTopology::Stay).map_err(|error| invalid(error.to_string()))?;
    let kt_nats = -kt.ln_evidence(&data);
    let uniform_nats = data.len() as f64 * 8.0 * LN_2;
    let finalist_log_mass = log_sum_exp(finalists.iter().map(Candidate::ln_joint));

    println!();
    println!("bytes: {}", data.len());
    println!("kt_total_nats: {kt_nats:.12}");
    println!("kt_coding_ratio_uniform: {:.12}", uniform_nats / kt_nats);
    println!();
    println!(
        "rank\tN\ttopology\tK\tdata_nats\tprior_bits\tjoint_nats\tdata_ratio_uniform\tcertified_ratio_uniform\tdata_ratio_kt\tcertified_ratio_kt\tfinalist_relative_posterior"
    );

    for (rank, candidate) in finalists.iter().enumerate() {
        let data_nats = -candidate.ln_evidence;
        let prior_nats = -candidate.model.ln_prior();
        let joint_nats = data_nats + prior_nats;
        let finalist_relative_posterior = (candidate.ln_joint() - finalist_log_mass).exp();
        println!(
            "{}\t{}\t{}\t{}\t{:.12}\t{:.6}\t{:.12}\t{:.12}\t{:.12}\t{:.12}\t{:.12}\t{:.12}",
            rank + 1,
            candidate.model.states(),
            candidate.model.topology().as_str(),
            candidate.exceptions(),
            data_nats,
            candidate.model.prior_bits(),
            joint_nats,
            uniform_nats / data_nats,
            uniform_nats / joint_nats,
            kt_nats / data_nats,
            kt_nats / joint_nats,
            finalist_relative_posterior,
        );
    }

    let best = &finalists[0];
    write_best(&args.dump_best, best)?;
    println!();
    println!("best_model_dump: {:?}", args.dump_best);
    println!("best_states: {}", best.model.states());
    println!("best_topology: {}", best.model.topology().as_str());
    println!("best_exceptions: {}", best.exceptions());
    println!("best_prior_bits: {:.12}", best.model.prior_bits());
    println!("best_total_nats: {:.12}", -best.ln_evidence);
    println!(
        "best_certified_mixture_upper_nats: {:.12}",
        -best.ln_joint()
    );
    println!(
        "best_certified_coding_ratio_uniform_lower: {:.12}",
        uniform_nats / -best.ln_joint()
    );
    println!(
        "best_certified_coding_ratio_kt_lower: {:.12}",
        kt_nats / -best.ln_joint()
    );
    println!("evaluation_seconds: {:.6}", started.elapsed().as_secs_f64());

    Ok(())
}

fn main() -> ExitCode {
    let args = Args::parse();
    let result = args.validate().and_then(|()| run(&args));

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("sparse-dfa-fit: {error}");
            ExitCode::FAILURE
        }
    }
}
