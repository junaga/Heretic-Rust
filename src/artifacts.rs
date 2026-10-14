// SPDX-License-Identifier: AGPL-3.0-or-later
//! Native reproduction documents and model-card content.
use crate::{config::Settings, model::is_hf_path};
use serde_json::{json, Value};

pub fn version_info() -> Value {
    json!({"version":env!("CARGO_PKG_VERSION"),"is_standard_pypi":false,"metadata":{"type":"git","url":"https://github.com/junaga/Heretic-Rust","commit_hash":env!("HERETIC_SOURCE_REVISION"),"source_sha256":env!("HERETIC_SOURCE_HASH"),"dirty":env!("HERETIC_SOURCE_DIRTY")=="true","implementation":"rust"}})
}
pub fn trial_parameters(attrs: &Value) -> Vec<(String, String)> {
    let mut rows = vec![(
        "direction_index".into(),
        attrs["direction_index"]
            .as_f64()
            .map(|n| format!("{n:.2}"))
            .unwrap_or_else(|| "per layer".into()),
    )];
    if let Some(components) = attrs["parameters"].as_object() {
        for (component, p) in components {
            for key in [
                "max_weight",
                "max_weight_position",
                "min_weight",
                "min_weight_distance",
            ] {
                if let Some(n) = p[key].as_f64() {
                    rows.push((format!("{component}.{key}"), format!("{n:.2}")));
                }
            }
        }
    }
    rows
}
pub fn markdown_table(headers: &[&str], rows: &[Vec<String>]) -> String {
    let widths: Vec<_> = headers
        .iter()
        .enumerate()
        .map(|(i, h)| {
            rows.iter()
                .map(|r| r.get(i).map(|s| s.chars().count()).unwrap_or(0))
                .max()
                .unwrap_or(0)
                .max(h.chars().count())
                .max(3)
        })
        .collect();
    let row = |values: &[String]| {
        format!(
            "| {} |\n",
            values
                .iter()
                .enumerate()
                .map(|(i, s)| format!(
                    "{s}{}",
                    " ".repeat(widths[i].saturating_sub(s.chars().count()))
                ))
                .collect::<Vec<_>>()
                .join(" | ")
        )
    };
    let mut output = row(&headers.iter().map(|s| s.to_string()).collect::<Vec<_>>());
    output += &row(&widths.iter().map(|w| "-".repeat(*w)).collect::<Vec<_>>());
    for r in rows {
        output += &row(r);
    }
    output
}
pub fn hf_link(name: &str, commit: Option<&str>, dataset: bool) -> String {
    if !is_hf_path(name) {
        return if dataset {
            "a local dataset"
        } else {
            "a local model"
        }
        .into();
    }
    let url = format!(
        "https://huggingface.co/{}{name}",
        if dataset { "datasets/" } else { "" }
    );
    match commit {
        Some(c) => format!("[{name} @ {c}]({url}/tree/{c})"),
        None => format!("[{name}]({url})"),
    }
}
pub fn readme_intro(settings: &Settings, attrs: &Value, reproducible: bool) -> String {
    let name = settings.string("model");
    let model = if is_hf_path(&name) {
        hf_link(&name, None, false)
    } else {
        "a model".into()
    };
    let parameters: Vec<_> = trial_parameters(attrs)
        .into_iter()
        .map(|(name, v)| vec![format!("**{name}**"), v])
        .collect();
    let metrics = vec![
        vec![
            "**KL divergence**".into(),
            format!("{:.4}", attrs["kl_divergence"].as_f64().unwrap_or(0.0)),
            "0 (by definition)".into(),
        ],
        vec![
            "**Refusals**".into(),
            format!("{}/{}", attrs["refusals"], attrs["n_bad_prompts"]),
            format!("{}/{}", attrs["base_refusals"], attrs["n_bad_prompts"]),
        ],
    ];
    format!("# Decensored version of {model}\n\nMade using [Heretic](https://heretic-project.org) v1.4.0 algorithms, implemented by [Heretic Rust](https://github.com/junaga/Heretic-Rust).\n{}\n## Abliteration parameters\n\n{}\n## Performance\n\n{}\n-----\n\n",if reproducible{"\n> [!TIP]\n> This model is reproducible. See [the reproduction guide](reproduce/README.md).\n"}else{""},markdown_table(&["Parameter","Value"],&parameters),markdown_table(&["Metric","This model",&format!("Original model ({model})")],&metrics))
}
pub fn reproduction_readme(
    settings: &Settings,
    attrs: &Value,
    checkpoint: &str,
    system: Option<&Value>,
    torch_version: &str,
) -> String {
    let mut out=String::from("# Reproduction guide\n\nThis directory records a Heretic Rust run using the Heretic v1.4.0 algorithms.\n\n## Models\n\n");
    out += &format!(
        "- **Base model:** {}\n\n## Datasets\n\n",
        hf_link(
            &settings.string("model"),
            settings.get("model_commit").as_str(),
            false
        )
    );
    for (key, label) in [
        ("good_prompts", "Good prompts"),
        ("bad_prompts", "Bad prompts"),
        ("good_evaluation_prompts", "Good evaluation prompts"),
        ("bad_evaluation_prompts", "Bad evaluation prompts"),
    ] {
        let spec = settings.get(key);
        let name = spec["dataset"].as_str().unwrap_or("");
        out += &format!(
            "- **{label}:** {}\n",
            hf_link(name, spec["commit"].as_str(), true)
        );
    }
    out+=&format!("\n## Selected trial\n\n- **Trial number:** {}\n- **KL divergence:** {:.6}\n- **Refusals:** {}/{}\n",attrs["index"],attrs["kl_divergence"].as_f64().unwrap_or(0.0),attrs["refusals"],attrs["n_bad_prompts"]);
    if let Some(s) = system {
        out+=&format!("\n## System\n\n- **Python:** {} ({}, {}) [{}]\n- **Operating system:** {} ({})\n- **CPU:** {}\n\n### Accelerators\n\n",show(&s["python"]["version"]),show(&s["python"]["implementation"]),show(&s["python"]["compiler"]),show(&s["python"]["environment"]),show(&s["os"]["platform"]),show(&s["os"]["machine"]),show(&s["cpu"]["brand"]));
        let a = &s["accelerators"];
        if a["type"].is_null() {
            out += "No GPU or other accelerator detected.\n"
        } else {
            out += &format!(
                "- **Type:** {}\n- **API:** {} {}\n- **Driver:** {}\n",
                show(&a["type"]),
                show(&a["api_name"]),
                show(&a["api_version"]),
                show(&a["driver_version"])
            );
            if let Some(devices) = a["devices"].as_array() {
                for (i, d) in devices.iter().enumerate() {
                    out += &format!(
                        "- **Device {i}:** {}{}\n",
                        show(&d["name"]),
                        d["vram_gb"]
                            .as_f64()
                            .map(|v| format!(" ({v:.2} GB)"))
                            .unwrap_or_default()
                    );
                }
                let names: std::collections::BTreeSet<_> =
                    devices.iter().filter_map(|d| d["name"].as_str()).collect();
                if a["type"] == "CUDA" && names.len() > 1 {
                    out+="\n> [!WARNING]\n> Heterogeneous GPUs can introduce nondeterminism. Byte-identical reproduction cannot be guaranteed.\n";
                }
            }
        }
    }
    let version = version_info();
    out+=&format!("\n## Environment\n\n- **Heretic Rust:** {}\n- **Source revision:** {}\n- **Application source SHA-256:** {}\n- **PyTorch:** {torch_version}\n- **Other dependencies:** See `requirements.txt`.\n",version["version"],show(&version["metadata"]["commit_hash"]),show(&version["metadata"]["source_sha256"]));
    if version["metadata"]["dirty"] == true {
        out+="\n> [!WARNING]\n> This executable was built from a modified working tree. The source revision alone is insufficient to reconstruct it; preserve the matching source identified by its SHA-256.\n";
    }
    let install = if let Some((base, suffix)) = torch_version.split_once('+') {
        format!("pip install torch=={base}+{suffix} --index-url https://download.pytorch.org/whl/{suffix}")
    } else {
        format!("pip install torch=={torch_version}")
    };
    out+=&format!("\n## Contents\n\n- `requirements.txt`: Framework package versions.\n- `config.toml`: Configuration, seed and data revisions.\n- `{checkpoint}`: Optuna study journal.\n- `SHA256SUMS`: Exported weight hashes.\n- `reproduce.json`: Machine-readable reproduction record.\n\n## How to reproduce\n\n1. Obtain the matching Heretic Rust source and build with `cargo build --release --locked`.\n2. Prepare the matching Python and accelerator environment.\n3. Install the framework packages: `pip install -r requirements.txt`.\n4. Install the matching PyTorch build: `{install}`.\n5. Run `heretic --reproduce reproduce.json`, review any environment differences, and export the model.\n6. Check exported weight files with `sha256sum -c SHA256SUMS` from their directory.\n\nTo rerun the optimization, place `config.toml` in the working directory and run `heretic`. To reuse the recorded trials, first copy `{checkpoint}` into the configured checkpoint directory, usually `checkpoints/`.\n\nExact reproduction depends on matching source, packages, hardware and deterministic kernels. No local dataset or model paths are published in this guide.\n");
    out
}
fn show(value: &Value) -> String {
    value.as_str().map(str::to_owned).unwrap_or_else(|| {
        if value.is_null() {
            "Unknown".into()
        } else {
            value.to_string()
        }
    })
}
