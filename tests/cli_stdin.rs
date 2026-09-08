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
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hello")
        .unwrap();

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
    child
        .stdin
        .take()
        .unwrap()
        .write_all(b"hello")
        .unwrap();

    let output = child.wait_with_output().unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.lines().any(|line| line.starts_with("3\t2\t")));
}
