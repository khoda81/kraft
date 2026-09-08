use std::{
    fs,
    process::{Command, Stdio},
    thread,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};

#[test]
fn short_inputs_exit_and_report_exact_bounds() {
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let path = std::env::temp_dir().join(format!("kraft-short-{}-{nonce}", std::process::id()));
    for data in [&b""[..], &b"a"[..]] {
        fs::write(&path, data).unwrap();
        let mut child = Command::new(env!("CARGO_BIN_EXE_kraft"))
            .args(["infer", "sparse-dfa"])
            .arg(&path)
            .args(["--steps", "10000", "--diagnostics"])
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        let started = Instant::now();
        loop {
            if child.try_wait().unwrap().is_some() {
                break;
            }
            if started.elapsed() > Duration::from_secs(10) {
                child.kill().unwrap();
                child.wait().unwrap();
                fs::remove_file(&path).unwrap();
                panic!("short input failed to terminate");
            }
            thread::sleep(Duration::from_millis(10));
        }
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success());
        let text = String::from_utf8(output.stdout).unwrap();
        let rows: Vec<_> = text
            .lines()
            .filter(|line| line.starts_with("0\t"))
            .collect();
        assert_eq!(rows.len(), 1);
        let fields: Vec<_> = rows[0].split('\t').collect();
        assert_eq!(fields.len(), 17);
        assert_eq!(fields[4], fields[5]);
        assert_eq!(fields[6].parse::<f64>().unwrap(), 0.0);
        if data.is_empty() {
            assert_eq!(&fields[7..11], &["NA"; 4]);
        }
    }
    fs::remove_file(path).unwrap();
}
