// SPDX-License-Identifier: AGPL-3.0-or-later
use anyhow::{Context, Result};
use heretic::{
    config::{self, Settings},
    workflow,
};
use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
};

fn run() -> Result<()> {
    let mut args: Vec<String> = std::env::args().skip(1).collect();
    if args.iter().any(|a| a == "--help" || a == "-h") {
        print!("{}", config::help());
        return Ok(());
    }
    if args.iter().any(|a| a == "--version" || a == "-V") {
        println!(
            "heretic-rust {} (upstream v1.4.0)",
            env!("CARGO_PKG_VERSION")
        );
        return Ok(());
    }
    let mut output = None;
    let mut noninteractive = false;
    let mut check = false;
    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                args.remove(i);
                output = Some(PathBuf::from(args.get(i).context("--output needs a path")?));
                args.remove(i);
            }
            "--non-interactive" => {
                noninteractive = true;
                args.remove(i);
            }
            "--check-config" => {
                check = true;
                args.remove(i);
            }
            _ => i += 1,
        }
    }
    let env: BTreeMap<String, String> = std::env::vars().collect();
    let settings = Settings::from_sources(Path::new("config.toml"), env, args)?;
    if check {
        println!("{}", serde_json::to_string_pretty(&settings.0)?);
        return Ok(());
    }
    if std::env::var_os("PYTORCH_ALLOC_CONF").is_none()
        && std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none()
    {
        std::env::set_var("PYTORCH_ALLOC_CONF", "expandable_segments:True");
    }
    pyo3::Python::with_gil(|py| {
        heretic::python::configure_runtime(py)?;
        // Embedded CPython otherwise leaves SIGINT at the process default, which
        // would terminate an active trial before it can be marked as pruned.
        let signal = heretic::python::module(py, "signal")?;
        let handler = heretic::python::O(
            pyo3::wrap_pyfunction!(heretic::python::handle_sigint, py)?.into_any(),
        );
        signal.m("signal", &[signal.a("SIGINT")?, handler], &[])?;
        workflow::run(py, settings, output, noninteractive)
    })
}
fn main() {
    if let Err(e) = run() {
        if heretic::python::interrupted(&e) {
            eprintln!("Shutting down...");
            return;
        }
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
