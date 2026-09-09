use std::{
    io::Write,
    process::{Command, Stdio},
};

fn kraft() -> Command {
    Command::new(env!("CARGO_BIN_EXE_kraft"))
}

#[test]
fn basic_eval_reads_stdin_when_input_is_omitted() {
    let mut child = kraft()
        .args(["eval", "kt", "--limit", "3"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"hello").unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("input: <stdin>"));
    assert!(stdout.contains("bytes: 3"));
}

#[test]
fn delegated_model_reads_stdin_when_input_is_omitted() {
    let mut child = kraft()
        .args(["eval", "partition-dfa", "--limit", "3", "--depth", "2"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"hello").unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.lines().any(|line| line.starts_with("3\t2\t")));
}

#[test]
fn grouped_dfa_reads_stdin_and_matches_the_oracle() {
    let mut child = kraft()
        .args([
            "infer",
            "dfa-grouped",
            "--states",
            "3",
            "--limit",
            "5",
            "--compare-oracle",
            "--report-every",
            "5",
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"ababa trailing data")
        .unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("likelihood_evaluations"));
    assert!(stdout.lines().any(|line| line.starts_with("5\t3\t")));
    assert_eq!(stdout.lines().count(), 2);
}

#[test]
fn grouped_dfa_budget_exhaustion_is_an_error() {
    let mut child = kraft()
        .args(["infer", "dfa-grouped", "-", "--max-nodes", "3"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    child.stdin.take().unwrap().write_all(b"a").unwrap();
    let output = child.wait_with_output().unwrap();
    assert!(!output.status.success());
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert!(stderr.contains("after 0 bytes"));
    assert!(stderr.contains("not pruned"));
}
