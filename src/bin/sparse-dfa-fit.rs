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

use std::{
    collections::HashSet,
    fs::{self, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    process::ExitCode,
    thread,
    time::Instant,
};

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

    /// Optional cheap prefix used to prefilter every depth's proposals before
    /// exact scoring on --search-bytes. Disabled when omitted.
    #[arg(long)]
    prefilter_bytes: Option<usize>,

    /// Number of proposal candidates promoted from the cheap prefilter.
    #[arg(long, default_value_t = 96)]
    prefilter_candidates: usize,

    /// Every Nth exception depth, full-score all proposals and report how many
    /// of the true top-beam candidates survived the cheap prefilter. Zero disables.
    #[arg(long, default_value_t = 0)]
    prefilter_audit_every: usize,

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

    /// Write every full-corpus finalist DFA here.
    #[arg(long, default_value = "artifacts/sparse-dfa-finalists.tsv")]
    dump_finalists: PathBuf,

    /// Durably append machine-readable search checkpoints here as work completes.
    #[arg(long, default_value = "artifacts/sparse-dfa-progress.tsv")]
    dump_progress: PathBuf,
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
            ("--prefilter-candidates", self.prefilter_candidates),
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
        if self.prefilter_bytes == Some(0) {
            return Err(invalid("--prefilter-bytes must be positive"));
        }
        if self.prefilter_candidates < self.beam {
            return Err(invalid(
                "--prefilter-candidates cannot be smaller than --beam",
            ));
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
    progress: &ProgressWriter,
) -> io::Result<Vec<Candidate>> {
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

        let mut proposal_models = proposals.into_iter().collect::<Vec<_>>();
        let proposed_total = proposal_models.len();
        let mut audit_promoted: Option<HashSet<SparseDfa>> = None;

        if let Some(prefilter_bytes) = args.prefilter_bytes {
            let prefilter_len = prefilter_bytes.min(data.len());
            if prefilter_len < data.len() && proposal_models.len() > args.prefilter_candidates {
                eprintln!(
                    "[sparse-dfa-fit] skeleton={} depth={} prefiltering {} proposals on {} bytes -> {}",
                    skeleton_index,
                    depth,
                    proposal_models.len(),
                    prefilter_len,
                    args.prefilter_candidates,
                );
                let mut prefiltered =
                    score_models(proposal_models, &data[..prefilter_len], args.threads());
                sort_best(&mut prefiltered);
                let promoted_len = args.prefilter_candidates.min(prefiltered.len());
                progress.append_candidates(
                    "prefilter",
                    Some(skeleton_index + 1),
                    Some(depth),
                    prefilter_len,
                    &prefiltered[..promoted_len],
                )?;

                let audit = args.prefilter_audit_every != 0
                    && depth % args.prefilter_audit_every == 0;
                if audit {
                    audit_promoted = Some(
                        prefiltered[..promoted_len]
                            .iter()
                            .map(|candidate| candidate.model.clone())
                            .collect(),
                    );
                    proposal_models = prefiltered
                        .into_iter()
                        .map(|candidate| candidate.model)
                        .collect();
                    eprintln!(
                        "[sparse-dfa-fit] skeleton={} depth={} prefilter audit: full-scoring all {} proposals",
                        skeleton_index,
                        depth,
                        proposal_models.len(),
                    );
                } else {
                    proposal_models = prefiltered
                        .into_iter()
                        .take(promoted_len)
                        .map(|candidate| candidate.model)
                        .collect();
                }
            }
        }

        let total = proposal_models.len();
        let batch_size = (args.threads() * 4).max(32).min(total);
        let mut scored = Vec::with_capacity(total);
        for (batch_index, batch) in proposal_models.chunks(batch_size).enumerate() {
            scored.extend(score_models(batch.to_vec(), data, args.threads()));
            sort_best(&mut scored);
            eprintln!(
                "[sparse-dfa-fit] skeleton={} depth={} full-scored={}/{} promoted ({} proposed) batches={}/{}",
                skeleton_index,
                depth,
                scored.len(),
                total,
                proposed_total,
                batch_index + 1,
                total.div_ceil(batch_size),
            );
            progress.append_candidates(
                "depth_partial",
                Some(skeleton_index + 1),
                Some(depth),
                data.len(),
                &scored[..1],
            )?;
        }
        if let Some(promoted) = audit_promoted {
            let truth_len = args.beam.min(scored.len());
            let hits = scored[..truth_len]
                .iter()
                .filter(|candidate| promoted.contains(&candidate.model))
                .count();
            eprintln!(
                "[sparse-dfa-fit] skeleton={} depth={} prefilter audit recall={}/{} ({:.1}%)",
                skeleton_index,
                depth,
                hits,
                truth_len,
                100.0 * hits as f64 / truth_len as f64,
            );
            progress.append_candidates(
                "prefilter_audit_truth",
                Some(skeleton_index + 1),
                Some(depth),
                data.len(),
                &scored[..truth_len],
            )?;
        }

        scored.truncate(args.beam);
        if scored.is_empty() {
            break;
        }
        progress.append_candidates(
            "beam",
            Some(skeleton_index + 1),
            Some(depth),
            data.len(),
            &scored,
        )?;
        all.extend(scored.iter().cloned());
        beam = scored;
    }

    Ok(all)
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

struct ProgressWriter {
    path: PathBuf,
}

impl ProgressWriter {
    fn new(path: PathBuf) -> Self {
        Self { path }
    }

    fn reset(
        &self,
        args: &Args,
        data_len: usize,
        search_len: usize,
        screen_len: usize,
    ) -> io::Result<()> {
        if let Some(parent) = self.path.parent()
            && !parent.as_os_str().is_empty()
        {
            fs::create_dir_all(parent)?;
        }

        let mut output = String::new();
        output.push_str("# KRAFT sparse DFA incremental search checkpoints\n");
        output.push_str(&format!("# corpus_bytes={data_len}\n"));
        output.push_str(&format!("# search_bytes={search_len}\n"));
        output.push_str(&format!("# screen_bytes={screen_len}\n"));
        output.push_str(&format!("# max_exceptions={}\n", args.max_exceptions));
        output.push_str(&format!("# beam={}\n", args.beam));
        output.push_str(&format!(
            "# prefilter_bytes={}\n",
            args.prefilter_bytes
                .map_or_else(|| "disabled".to_owned(), |value| value.to_string())
        ));
        output.push_str(&format!(
            "# prefilter_candidates={}\n",
            args.prefilter_candidates
        ));
        output.push_str(&format!(
            "# prefilter_audit_every={}\n",
            args.prefilter_audit_every
        ));
        output.push_str(
            "stage\tskeleton\tdepth\trank\tscore_bytes\tstates\ttopology\texceptions\tprior_bits\tln_evidence\tln_joint\toverrides\n",
        );
        fs::write(&self.path, output)
    }

    fn append_candidates(
        &self,
        stage: &str,
        skeleton: Option<usize>,
        depth: Option<usize>,
        score_bytes: usize,
        candidates: &[Candidate],
    ) -> io::Result<()> {
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)?;

        for (rank, candidate) in candidates.iter().enumerate() {
            writeln!(
                file,
                "{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{}",
                stage,
                skeleton.map_or_else(String::new, |value| value.to_string()),
                depth.map_or_else(String::new, |value| value.to_string()),
                rank + 1,
                score_bytes,
                candidate.model.states(),
                candidate.model.topology().as_str(),
                candidate.exceptions(),
                candidate.model.prior_bits(),
                candidate.ln_evidence,
                candidate.ln_joint(),
                format_overrides(candidate),
            )?;
        }

        file.flush()
    }
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

fn format_overrides(candidate: &Candidate) -> String {
    candidate
        .model
        .overrides()
        .iter()
        .map(|edge| {
            if edge.byte.is_ascii_graphic() || edge.byte == b' ' {
                format!(
                    "{}:{:02x}('{}')->{}",
                    edge.source, edge.byte, edge.byte as char, edge.destination
                )
            } else {
                format!("{}:{:02x}->{}", edge.source, edge.byte, edge.destination)
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn write_finalists(path: &PathBuf, finalists: &[Candidate]) -> io::Result<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }

    let mut output = String::new();
    output.push_str("# KRAFT sparse DFA full-corpus finalists\n");
    output.push_str(
        "rank\tstates\ttopology\texceptions\tprior_bits\tln_evidence\tln_joint\tsource\tbyte_hex\tdestination\n",
    );

    for (rank, candidate) in finalists.iter().enumerate() {
        if candidate.model.overrides().is_empty() {
            output.push_str(&format!(
                "{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t\t\t\n",
                rank + 1,
                candidate.model.states(),
                candidate.model.topology().as_str(),
                candidate.exceptions(),
                candidate.model.prior_bits(),
                candidate.ln_evidence,
                candidate.ln_joint(),
            ));
            continue;
        }

        for edge in candidate.model.overrides() {
            output.push_str(&format!(
                "{}\t{}\t{}\t{}\t{:.12}\t{:.12}\t{:.12}\t{}\t{:02x}\t{}\n",
                rank + 1,
                candidate.model.states(),
                candidate.model.topology().as_str(),
                candidate.exceptions(),
                candidate.model.prior_bits(),
                candidate.ln_evidence,
                candidate.ln_joint(),
                edge.source,
                edge.byte,
                edge.destination,
            ));
        }
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
    let progress = ProgressWriter::new(args.dump_progress.clone());
    progress.reset(args, data.len(), search_len, screen_len)?;

    eprintln!(
        "[sparse-dfa-fit] loaded {} bytes; search={} prefilter={:?}->{} screen={} states={:?}",
        data.len(),
        search_len,
        args.prefilter_bytes,
        args.prefilter_candidates,
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
    println!(
        "prefilter_bytes: {}",
        args.prefilter_bytes
            .map_or_else(|| "disabled".to_owned(), |value| value.to_string())
    );
    println!("prefilter_candidates: {}", args.prefilter_candidates);
    println!("prefilter_audit_every: {}", args.prefilter_audit_every);
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

    progress.append_candidates("skeleton", None, Some(0), search_len, &skeleton_scores)?;
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
        let searched = search_skeleton(skeleton, search, args, index, &progress)?;
        let mut searched_ranked = searched.clone();
        sort_best(&mut searched_ranked);
        searched_ranked.truncate(args.screen_candidates.min(searched_ranked.len()));
        progress.append_candidates(
            "skeleton_complete",
            Some(index + 1),
            Some(args.max_exceptions),
            search_len,
            &searched_ranked,
        )?;
        explored.extend(searched);
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
    progress.append_candidates("screened", None, None, screen_len, &screened)?;

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
    progress.append_candidates("finalist", None, None, data.len(), &finalists)?;

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

    println!();
    println!("finalist_dfas:");
    for (rank, candidate) in finalists.iter().enumerate() {
        println!(
            "{}\tN={}\ttopology={}\tK={}\t{}",
            rank + 1,
            candidate.model.states(),
            candidate.model.topology().as_str(),
            candidate.exceptions(),
            format_overrides(candidate),
        );
    }

    let best = &finalists[0];
    write_best(&args.dump_best, best)?;
    write_finalists(&args.dump_finalists, &finalists)?;
    println!();
    println!("best_model_dump: {:?}", args.dump_best);
    println!("finalists_dump: {:?}", args.dump_finalists);
    println!("progress_dump: {:?}", args.dump_progress);
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
