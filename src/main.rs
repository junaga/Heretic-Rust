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
    // Discover only the project-local environment. Never modify the user's global Python installation.
    if std::env::var_os("VIRTUAL_ENV").is_none() && std::env::var_os("PYTHONPATH").is_none() {
        let site = Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/lib");
        if site.exists() {
            for python in std::fs::read_dir(site)? {
                let candidate = python?.path().join("site-packages");
                if candidate.is_dir() {
                    std::env::set_var("PYTHONPATH", candidate);
                    break;
                }
            }
        }
    }
    if std::env::var_os("PYTORCH_ALLOC_CONF").is_none()
        && std::env::var_os("PYTORCH_CUDA_ALLOC_CONF").is_none()
    {
        std::env::set_var("PYTORCH_ALLOC_CONF", "expandable_segments:True");
    }
    pyo3::Python::with_gil(|py| workflow::run(py, settings, output, noninteractive))
}
fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}
