// SPDX-License-Identifier: AGPL-3.0-or-later
#![cfg(unix)]
mod common;
use anyhow::{bail, Result};
use heretic::core::checkpoint_name;
use pyo3::Python;
use serde_json::json;
use std::{
    io::Write,
    process::{Command, Stdio},
    time::{Duration, Instant},
};

#[test]
#[ignore = "requires framework dependencies; exercises real SIGINT and checkpoint recovery"]
fn interrupted_trial_is_recoverable() -> Result<()> {
    let root = tempfile::tempdir()?;
    let model = root.path().join("model");
    Python::with_gil(|py| common::fixture(py, &model))?;
    let prompts = root.path().join("prompts.txt");
    std::fs::write(&prompts, "Hello world\nGood morning\n")?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_heretic"));
    command.args([
        "--model",
        model.to_str().unwrap(),
        "--dtypes",
        "float32",
        "--batch-size",
        "1",
        "--max-response-length",
        "4",
        "--response-prefix",
        "",
        "--seed",
        "42",
        "--n-trials",
        "10000",
        "--non-interactive",
    ]);
    for key in [
        "--good-prompts",
        "--bad-prompts",
        "--good-evaluation-prompts",
        "--bad-evaluation-prompts",
    ] {
        command.args([
            key,
            &json!({"dataset":prompts.to_string_lossy()}).to_string(),
        ]);
    }
    let log = root.path().join("run.log");
    let file = std::fs::File::create(&log)?;
    let mut child = command
        .current_dir(root.path())
        .env("OMP_NUM_THREADS", "1")
        .stdout(file.try_clone()?)
        .stderr(file)
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(35);
    loop {
        if let Some(status) = child.try_wait()? {
            bail!(
                "Application exited before interruption: {status}; {}",
                std::fs::read_to_string(&log)?
            )
        }
        if std::fs::read_to_string(&log)?.contains("Elapsed:") {
            break;
        }
        if Instant::now() > deadline {
            child.kill()?;
            bail!("Timed out waiting for first trial")
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    let sent = Command::new("kill")
        .args(["-s", "INT", &child.id().to_string()])
        .status()?;
    assert!(sent.success());
    let deadline = Instant::now() + Duration::from_secs(15);
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() > deadline {
            child.kill()?;
            bail!(
                "Application did not handle SIGINT: {}",
                std::fs::read_to_string(&log)?
            )
        };
        std::thread::sleep(Duration::from_millis(20));
    };
    assert!(
        status.success(),
        "SIGINT was not handled gracefully: {}",
        std::fs::read_to_string(&log)?
    );
    let journal = root
        .path()
        .join("checkpoints")
        .join(checkpoint_name(model.to_str().unwrap()));
    Python::with_gil(|py| -> Result<()> {
        let study = common::load_study(py, &journal)?;
        let trials = study.a("trials")?.list()?;
        assert!(!trials.is_empty());
        assert!(trials.len() < 10000);
        for trial in trials {
            assert_ne!(
                trial.a("state.name")?.str()?,
                "RUNNING",
                "interruption left an unclosed trial"
            );
        }
        assert_eq!(study.a("user_attrs")?.json()?["finished"], false);
        Ok(())
    })?;
    // Reopen the interrupted run, inspect its stored Pareto results and exit.
    let mut resume = Command::new(env!("CARGO_BIN_EXE_heretic"));
    // Cancel at the previous-run menu to verify the persisted checkpoint is readable.
    resume
        .args(["--model", model.to_str().unwrap(), "--dtypes", "float32"])
        .current_dir(root.path())
        .env("OMP_NUM_THREADS", "1");
    let mut child = resume
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;
    child.stdin.take().unwrap().write_all(b"3\n")?;
    let result = child.wait_with_output()?;
    assert!(result.status.success());
    assert!(String::from_utf8_lossy(&result.stdout).contains("Previous run found"));
    Ok(())
}
