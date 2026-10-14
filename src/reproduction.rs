// SPDX-License-Identifier: AGPL-3.0-or-later
use crate::{
    config::Settings,
    model::{is_hf_path, Model},
    python::*,
    workflow::select,
};
use anyhow::{bail, Context, Result};
use pyo3::types::PyAnyMethods;
use pyo3::Python;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, io::Read, path::Path};

pub fn load(_py: Python<'_>, path: &str) -> Result<Value> {
    crate::native::load_json(path)
}
pub fn requirements(py: Python<'_>) -> Result<Value> {
    use std::collections::{BTreeSet, VecDeque};
    let mut values = serde_json::Map::new();
    let mut queue: VecDeque<String> = include_str!("../requirements.txt")
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| {
            line.chars()
                .take_while(|c| c.is_ascii_alphanumeric() || *c == '-' || *c == '_')
                .collect::<String>()
        })
        .filter(|s| !s.is_empty())
        .collect();
    queue.extend(["torch", "torchaudio", "torchvision"].map(str::to_owned));
    let metadata = module(py, "importlib.metadata")?;
    let mut visited = BTreeSet::new();
    while let Some(package) = queue.pop_front() {
        let name = package.to_lowercase().replace('_', "-");
        if !visited.insert(name.clone()) {
            continue;
        }
        let distribution = match metadata.m("distribution", &[j(py, &name)?], &[]) {
            Ok(v) => v,
            Err(e) => {
                let missing = metadata.a("PackageNotFoundError")?;
                if e.downcast_ref::<pyo3::PyErr>()
                    .is_some_and(|e| e.matches(py, &missing.0).unwrap())
                {
                    continue;
                }
                return Err(e);
            }
        };
        let version = distribution.a("version")?.str()?;
        values.insert(name, json!(version.split('+').next().unwrap()));
        let requires = distribution.a("requires")?;
        if !requires.is_none() {
            for requirement in requires.strings()? {
                if requirement.contains(';') && requirement.contains("extra ==") {
                    continue;
                }
                let name: String = requirement
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_' || *c == '-')
                    .collect();
                if !name.is_empty() {
                    queue.push_back(name);
                }
            }
        }
    }
    Ok(Value::Object(values))
}
fn driver(command: &str, args: &[&str]) -> Value {
    std::process::Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|x| x.status.success())
        .map(|x| json!(String::from_utf8_lossy(&x.stdout).trim().to_owned()))
        .unwrap_or(Value::Null)
}
pub fn accelerator(py: Python<'_>) -> Result<Value> {
    let torch = module(py, "torch")?;
    let mut result =
        json!({"type":null,"api_name":null,"api_version":null,"driver_version":null,"devices":[]});
    for (backend, label, api) in [
        ("cuda", "CUDA", "CUDA"),
        ("xpu", "XPU", "oneAPI"),
        ("mlu", "MLU", "MLU"),
        ("sdaa", "SDAA", "SDAA"),
        ("musa", "MUSA", "MUSA"),
        ("npu", "NPU", "CANN"),
    ] {
        let Ok(device) = torch.a(backend) else {
            continue;
        };
        if device
            .m("is_available", &[], &[])
            .is_ok_and(|v| v.empty().is_ok_and(|e| !e))
        {
            let hip = torch.a("version.hip").ok().filter(|v| !v.is_none());
            result["type"] = json!(if backend == "cuda" && hip.is_some() {
                "ROCm"
            } else {
                label
            });
            result["api_name"] = json!(if backend == "cuda" && hip.is_some() {
                "ROCm"
            } else {
                api
            });
            result["api_version"] = if let Some(hip) = hip {
                json!(hip.str()?)
            } else {
                torch
                    .a(&format!("version.{backend}"))
                    .and_then(|v| v.str())
                    .map(|s| json!(s))
                    .unwrap_or(Value::Null)
            };
            result["driver_version"] = match backend {
                "cuda" => driver(
                    "nvidia-smi",
                    &["--query-gpu=driver_version", "--format=csv,noheader"],
                ),
                "xpu" => driver("xpu-smi", &["discovery"]),
                _ => Value::Null,
            };
            let count = if backend == "npu" {
                0
            } else {
                device.m("device_count", &[], &[])?.usize()?
            };
            for i in 0..count {
                let name = device.m("get_device_name", &[j(py, i)?], &[])?.str()?;
                let memory = device
                    .m("get_device_properties", &[j(py, i)?], &[])
                    .and_then(|v| v.a("total_memory"))
                    .and_then(|v| v.usize())
                    .ok();
                result["devices"]
                    .as_array_mut()
                    .unwrap()
                    .push(json!({"name":name,"total_memory":memory}));
            }
            break;
        }
    }
    if result["type"].is_null()
        && torch
            .a("backends.mps")
            .and_then(|v| v.m("is_available", &[], &[]))
            .is_ok_and(|v| v.empty().is_ok_and(|e| !e))
    {
        result = json!({"type":"MPS","api_name":"Metal","api_version":null,"driver_version":driver("sw_vers",&["-productVersion"]),"devices":[{"name":"Apple Silicon"}]});
    }
    // Match the upstream reproduction schema, including accelerator-specific API labels.
    let kind = result["type"].as_str().unwrap_or("").to_owned();
    match kind.as_str() {
        "CUDA" | "ROCm" => {
            result["api_name"] = json!(if kind == "ROCm" {
                "HIP Version"
            } else {
                "CUDA Version"
            });
            if kind == "ROCm" {
                let modern = driver("amd-smi", &["version"]);
                result["driver_version"] =
                    if let Some(s) = modern.as_str().filter(|s| !s.is_empty()) {
                        json!(s.replace('\n', " | "))
                    } else {
                        let legacy = driver("rocm-smi", &["--showdriverversion"]);
                        legacy
                            .as_str()
                            .and_then(|s| s.lines().find(|s| s.contains("Driver version")))
                            .map(|s| json!(s.rsplit(':').next().unwrap_or("").trim()))
                            .unwrap_or_else(|| {
                                std::fs::read_to_string("/sys/module/amdgpu/version")
                                    .map(|s| json!(s.trim()))
                                    .unwrap_or(Value::Null)
                            })
                    };
            } else if let Some(s) = result["driver_version"].as_str() {
                result["driver_version"] = json!(s.lines().next().unwrap_or(""));
            }
            for (i, device) in result["devices"]
                .as_array_mut()
                .unwrap()
                .iter_mut()
                .enumerate()
            {
                let total = torch
                    .a("cuda")?
                    .m("mem_get_info", &[j(py, i)?], &[])?
                    .item(1)?
                    .f64()?;
                *device = json!({"name":device["name"],"vram_gb":(total/1073741824.0*100.0).round()/100.0});
            }
        }
        "XPU" => {
            result["api_name"] = Value::Null;
            result["api_version"] = Value::Null;
            result["driver_version"] = result["driver_version"]
                .as_str()
                .and_then(|s| s.lines().find(|s| s.contains("Driver Version")))
                .map(|s| json!(s.rsplit(':').next().unwrap_or("").trim()))
                .unwrap_or(Value::Null);
            for device in result["devices"].as_array_mut().unwrap() {
                device.as_object_mut().unwrap().remove("total_memory");
            }
        }
        "MLU" | "SDAA" | "MUSA" => {
            result["api_name"] = Value::Null;
            result["api_version"] = Value::Null;
            for device in result["devices"].as_array_mut().unwrap() {
                device.as_object_mut().unwrap().remove("total_memory");
            }
        }
        "NPU" => {
            result["api_name"] = json!("CANN Version");
            result["api_version"] = torch
                .a("version.cann")
                .and_then(|v| v.str())
                .map(|s| json!(s))
                .unwrap_or(Value::Null);
            let raw = driver("npu-smi", &["info", "-t", "board", "-i", "0"]);
            result["driver_version"] = raw
                .as_str()
                .and_then(|s| s.lines().find(|s| s.contains("Software Version")))
                .map(|s| json!(s.split_whitespace().last().unwrap_or("")))
                .unwrap_or(Value::Null);
            result["devices"] = json!([]);
        }
        "MPS" => {
            result["api_name"] = Value::Null;
            result["api_version"] = Value::Null;
            result["devices"] = json!([{"name":"Apple Metal"}]);
        }
        _ => result = json!({"type":null}),
    }
    Ok(result)
}
pub fn system(py: Python<'_>) -> Result<Value> {
    let platform = module(py, "platform")?;
    let cpu = crate::native::cpu_info();
    let sys = module(py, "sys")?;
    let env = if std::env::var_os("CONDA_PREFIX").is_some() {
        "Conda"
    } else if sys.a("base_prefix")?.str()? != sys.a("prefix")?.str()? {
        "Virtualenv/Venv"
    } else {
        "System"
    };
    Ok(
        json!({"python":{"version":platform.m("python_version",&[],&[])?.str()?,"implementation":platform.m("python_implementation",&[],&[])?.str()?,"compiler":platform.m("python_compiler",&[],&[])?.str()?,"environment":env},"os":{"platform":platform.m("platform",&[],&[])?.str()?,"machine":platform.m("machine",&[],&[])?.str()?},"cpu":{"brand":cpu["brand"],"vendor":cpu["vendor"],"family":cpu["family"],"model":cpu["model"],"stepping":cpu["stepping"]},"accelerators":accelerator(py)?}),
    )
}
pub fn print_system(py: Python<'_>) -> Result<()> {
    let a = accelerator(py)?;
    println!(
        "Accelerator: {}",
        if a["type"].is_null() {
            "CPU".to_owned()
        } else {
            a.to_string()
        }
    );
    Ok(())
}

#[derive(Default, Debug, PartialEq, Eq)]
pub struct CollectionSummary {
    pub found: usize,
    pub downloaded: usize,
    pub already_stored: usize,
}
pub fn collect(py: Python<'_>, path: &str) -> Result<()> {
    let result = collect_records(py, path)?;
    println!(
        "Found: {} files\nDownloaded: {} files\nAlready stored: {} files",
        result.found, result.downloaded, result.already_stored
    );
    Ok(())
}
pub fn collect_records(py: Python<'_>, path: &str) -> Result<CollectionSummary> {
    let hub = module(py, "huggingface_hub")?;
    let utils = module(py, "huggingface_hub.utils")?;
    let api = hub.a("HfApi")?.c(&[], &[])?;
    let models = api.m(
        "list_models",
        &[],
        &[
            ("filter", j(py, ["heretic", "reproducible"])?),
            ("sort", j(py, "created_at")?),
            ("expand", j(py, ["gated", "tags"])?),
        ],
    )?;
    let disabled = utils
        .m("are_progress_bars_disabled", &[], &[])?
        .0
        .is_truthy()?;
    utils.m("disable_progress_bars", &[], &[])?;
    let result = (|| -> Result<CollectionSummary> {
        let mut result = CollectionSummary::default();
        for model in models.0.try_iter()? {
            let model = O(model?);
            let id = model.a("id")?.str()?;
            if optional(model.a("tags")?.strings())?
                .unwrap_or_default()
                .iter()
                .any(|s| s == "gguf")
            {
                continue;
            }
            if !model.a("gated")?.empty()? {
                if let Err(e) = api.m(
                    "auth_check",
                    &[j(py, &id)?],
                    &[("repo_type", j(py, "model")?)],
                ) {
                    let gated = utils.a("GatedRepoError")?;
                    if e.downcast_ref::<pyo3::PyErr>()
                        .is_some_and(|e| e.matches(py, &gated.0).unwrap())
                    {
                        continue;
                    }
                    return Err(e);
                }
            }
            let paths = api.m(
                "get_paths_info",
                &[j(py, &id)?, j(py, "reproduce/reproduce.json")?],
                &[("expand", j(py, true)?)],
            )?;
            if paths.len()? == 0 {
                println!("{id}: no reproduce.json found");
                continue;
            }
            result.found += 1;
            let commit = paths.item(0)?.a("last_commit.oid")?.str()?;
            let (owner, name) = id.split_once('/').context("Invalid repository ID")?;
            if !crate::native::safe_filename(owner)
                || !crate::native::safe_filename(name)
                || name.contains('/')
            {
                bail!("Invalid repository ID")
            }
            let short: String = commit.chars().take(7).collect();
            if short.len() != 7 || !short.bytes().all(|c| c.is_ascii_hexdigit()) {
                bail!("Invalid repository commit ID")
            }
            let target = Path::new(path)
                .join("huggingface.co")
                .join(owner)
                .join(format!("{name}-{short}.json"));
            if target.exists() {
                result.already_stored += 1;
                continue;
            }
            let cached = hub
                .m(
                    "hf_hub_download",
                    &[j(py, &id)?, j(py, "reproduce/reproduce.json")?],
                    &[],
                )?
                .str()?;
            std::fs::create_dir_all(target.parent().unwrap())?;
            std::fs::copy(cached, &target)?;
            result.downloaded += 1;
            println!("{id}: downloaded");
        }
        Ok(result)
    })();
    if !disabled {
        utils.m("enable_progress_bars", &[], &[])?;
    }
    result
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReproductionLevel {
    None,
    Basic,
    Full,
}
pub struct Publication {
    pub repo_id: String,
    pub token: String,
    pub private: bool,
    pub strategy: String,
    pub reproduction: ReproductionLevel,
    pub trial_count: Option<usize>,
}
pub fn uploaded_hashes(py: Python<'_>, repo_id: &str, token: &str) -> Result<Value> {
    let info = module(py, "huggingface_hub")?.a("HfApi")?.c(&[], &[])?.m(
        "model_info",
        &[j(py, repo_id)?],
        &[("files_metadata", j(py, true)?), ("token", j(py, token)?)],
    )?;
    let files = info.a("siblings")?;
    if files.is_none() || files.len()? == 0 {
        bail!("Could not fetch uploaded model hashes")
    }
    let mut hashes = serde_json::Map::new();
    for file in files.list()? {
        let name = file.a("rfilename")?.str()?;
        if name.ends_with(".safetensors") {
            let lfs = file.a("lfs")?;
            let sha = lfs.key("sha256").or_else(|_| lfs.a("sha256"))?;
            let sha = sha.str()?;
            if sha.is_empty() || sha == "None" {
                bail!("Could not fetch uploaded model hash for {name}")
            }
            hashes.insert(name, json!(sha));
        }
    }
    Ok(Value::Object(hashes))
}
pub fn verify_uploaded(py: Python<'_>, repo: &str, token: &str, original: &Value) -> Result<()> {
    let hashes = uploaded_hashes(py, repo, token)?;
    if let Some(expected) = original["hashes"].as_object() {
        for (name, sha) in expected {
            let outcome = match hashes[name].as_str() {
                Some(actual) if sha.as_str().is_some_and(|s| s.eq_ignore_ascii_case(actual)) => {
                    "Hash matches"
                }
                Some(_) => "Hash does not match",
                None => "File not found",
            };
            println!("{name}: {outcome}");
        }
    }
    Ok(())
}
pub fn publish(
    m: &mut Model<'_>,
    request: &Publication,
    attrs: &Value,
    checkpoint: &Path,
    original: Option<&Value>,
) -> Result<()> {
    if !["merge", "adapter"].contains(&request.strategy.as_str()) {
        bail!("Invalid export strategy")
    }
    let py = m.py;
    let hub = module(py, "huggingface_hub")?;
    let repo = j(py, &request.repo_id)?;
    let token = j(py, &request.token)?;
    let reproducible = original.is_none()
        && is_hf_path(&m.settings.string("model"))
        && [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ]
        .iter()
        .all(|k| is_hf_path(m.settings.get(k)["dataset"].as_str().unwrap_or("")));
    let level = if reproducible {
        request.reproduction
    } else {
        ReproductionLevel::None
    };
    let exported = if request.strategy == "adapter" {
        m.model.clone()
    } else {
        m.merged()?
    };
    exported.m(
        "push_to_hub",
        &[repo.clone()],
        &[
            ("private", j(py, request.private)?),
            ("token", token.clone()),
            ("max_shard_size", j(py, m.settings.get("max_shard_size"))?),
        ],
    )?;
    drop(exported);
    empty_cache(py)?;
    if request.strategy == "merge" {
        let kw = [
            ("private", j(py, request.private)?),
            ("token", token.clone()),
        ];
        m.tokenizer.m("push_to_hub", &[repo.clone()], &kw)?;
        if let Some(processor) = &m.processor {
            processor.m("push_to_hub", &[repo.clone()], &kw)?;
        }
    }
    let source = m.settings.string("model");
    let local_card = Path::new(&source).join("README.md");
    let card = if is_hf_path(&source) {
        Some(
            hub.a("ModelCard")?
                .m("load", &[j(py, &source)?], &[("token", token.clone())])?,
        )
    } else if local_card.is_file() {
        Some(
            hub.a("ModelCard")?
                .m("load", &[j(py, local_card.to_string_lossy())?], &[])?,
        )
    } else {
        None
    };
    if let Some(card) = card {
        if card.a("data")?.is_none() {
            card.set("data", &hub.a("ModelCardData")?.c(&[], &[])?)?;
        }
        let data = card.a("data")?;
        let mut tags = data.a("tags")?.strings().unwrap_or_default();
        tags.extend(["heretic", "uncensored", "decensored", "abliterated"].map(str::to_owned));
        if level != ReproductionLevel::None {
            tags.push("reproducible".into());
        }
        data.set("tags", &j(py, tags)?)?;
        let intro =
            crate::artifacts::readme_intro(&m.settings, attrs, level != ReproductionLevel::None);
        card.set("text", &j(py, intro + &card.a("text")?.str()?)?)?;
        card.m("push_to_hub", &[repo.clone()], &[("token", token.clone())])?;
    }
    if level != ReproductionLevel::None {
        let mut settings = m.settings.clone();
        settings.set("export_strategy", request.strategy.as_str());
        if let Some(n) = request.trial_count {
            settings.set("n_trials", n);
        }
        if settings.get("model_commit").is_null() {
            settings.set(
                "model_commit",
                hub.m(
                    "model_info",
                    &[j(py, &source)?],
                    &[("token", token.clone())],
                )?
                .a("sha")?
                .str()?,
            );
        }
        for key in [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ] {
            if settings.0[key]["commit"].is_null() {
                settings.0[key]["commit"] = json!(hub
                    .m(
                        "dataset_info",
                        &[j(py, &settings.0[key]["dataset"])?],
                        &[("token", token.clone())]
                    )?
                    .a("sha")?
                    .str()?);
            }
        }
        let hashes = uploaded_hashes(py, &request.repo_id, &request.token)?;
        let temporary = tempfile::Builder::new()
            .prefix("heretic-reproduce-")
            .tempdir()?;
        bundle(
            py,
            temporary.path(),
            &settings,
            attrs,
            checkpoint,
            level == ReproductionLevel::Full,
        )?;
        let folder = temporary.path().join("reproduce");
        let path = folder.join("reproduce.json");
        let mut information: Value = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
        information["hashes"] = hashes.clone();
        std::fs::write(&path, serde_json::to_string_pretty(&information)?)?;
        if !hashes.as_object().unwrap().is_empty() {
            std::fs::write(
                folder.join("SHA256SUMS"),
                crate::native::sha256sums(&hashes),
            )?;
        } else {
            std::fs::remove_file(folder.join("SHA256SUMS"))?;
        }
        let mut files = std::fs::read_dir(&folder)?
            .map(|entry| entry.map(|e| e.path()))
            .collect::<std::io::Result<Vec<_>>>()?;
        files.sort();
        for file in files {
            if file.is_file() {
                hub.m(
                    "upload_file",
                    &[],
                    &[
                        ("path_or_fileobj", j(py, file.to_string_lossy())?),
                        (
                            "path_in_repo",
                            j(
                                py,
                                format!(
                                    "reproduce/{}",
                                    file.file_name().unwrap().to_string_lossy()
                                ),
                            )?,
                        ),
                        ("repo_id", repo.clone()),
                        ("token", token.clone()),
                    ],
                )?;
            }
        }
    }
    println!("Model uploaded to {}", request.repo_id);
    if let Some(info) = original {
        if info["version"] == "2" {
            verify_uploaded(py, &request.repo_id, &request.token, info)?;
        }
    }
    Ok(())
}
pub fn upload(m: &mut Model<'_>, attrs: &Value, checkpoint: &Path) -> Result<()> {
    upload_context(m, attrs, checkpoint, None, None)
}
pub fn upload_context(
    m: &mut Model<'_>,
    attrs: &Value,
    checkpoint: &Path,
    original: Option<&Value>,
    trial_count: Option<usize>,
) -> Result<()> {
    let py = m.py;
    let hub = module(py, "huggingface_hub")?;
    let token = hub.m("get_token", &[], &[])?.str()?;
    let token = if token == "None" || token.is_empty() {
        crate::ui::password("Hugging Face access token")?
    } else {
        token
    };
    if token.is_empty() {
        return Ok(());
    }
    let user = hub.m("whoami", &[j(py, &token)?], &[])?.json()?;
    println!(
        "Logged in as {} ({})",
        user["fullname"]
            .as_str()
            .or(user["name"].as_str())
            .unwrap_or("unknown user"),
        user["email"].as_str().unwrap_or("no email found")
    );
    let name = Path::new(&m.settings.string("model"))
        .file_name()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let repo = crate::ui::text(
        "Name of repository",
        &format!(
            "{}/{name}-heretic",
            user["name"].as_str().context("Account name missing")?
        ),
    )?;
    if repo.is_empty() {
        return Ok(());
    }
    let Some(private) = select(
        py,
        "Repository visibility",
        &["Public".into(), "Private".into()],
    )?
    else {
        return Ok(());
    };
    let Some(strategy) = strategy(m)? else {
        return Ok(());
    };
    let eligible = original.is_none()
        && is_hf_path(&m.settings.string("model"))
        && [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ]
        .iter()
        .all(|k| is_hf_path(m.settings.get(k)["dataset"].as_str().unwrap_or("")));
    let level = if eligible {
        match select(py,"Reproduction information to publish (Full includes Python/OS/CPU/accelerator information; neither option publishes local paths)",&["Full: settings, package versions and system information".into(),"Basic: settings and package versions".into(),"None".into()])? {Some(0)=>ReproductionLevel::Full,Some(1)=>ReproductionLevel::Basic,Some(2)=>ReproductionLevel::None,_=>return Ok(())}
    } else {
        ReproductionLevel::None
    };
    if select(
        py,
        &format!(
            "Create or update {repo} ({})?",
            if private == 1 { "private" } else { "public" }
        ),
        &["Upload".into(), "Cancel".into()],
    )? != Some(0)
    {
        return Ok(());
    }
    publish(
        m,
        &Publication {
            repo_id: repo,
            token,
            private: private == 1,
            strategy,
            reproduction: level,
            trial_count,
        },
        attrs,
        checkpoint,
        original,
    )
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum MismatchSeverity {
    Low,
    Medium,
    High,
    Critical,
}
impl std::fmt::Display for MismatchSeverity {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Low => "low",
                Self::Medium => "medium",
                Self::High => "high",
                Self::Critical => "critical",
            }
        )
    }
}
pub fn package_severity(name: &str) -> MismatchSeverity {
    match name {
        "heretic-llm" | "heretic-rust" => MismatchSeverity::Critical,
        "torch" | "transformers" => MismatchSeverity::High,
        "accelerate" | "bitsandbytes" | "kernels" | "optuna" | "peft" | "tokenizers" | "triton" => {
            MismatchSeverity::Medium
        }
        _ => MismatchSeverity::Low,
    }
}
#[derive(Debug)]
pub struct EnvironmentMismatch {
    pub category: &'static str,
    pub component: String,
    pub local: Value,
    pub original: Value,
    pub severity: MismatchSeverity,
}
pub fn format_version(info: &Value) -> String {
    let version = info["version"].as_str().unwrap_or("unknown");
    let m = &info["metadata"];
    let identity = match m["type"].as_str() {
        Some("pypi") => version.to_owned(),
        Some("git") => format!(
            "{version}-git+{}@{}",
            m["url"].as_str().unwrap_or("unknown"),
            m["commit_hash"].as_str().unwrap_or("unknown")
        ),
        _ => format!("{version}-local-{}", rand::random::<u16>() as u32 + 65536),
    };
    if m["implementation"] == "rust" {
        format!(
            "{identity}-rust+{}",
            m["source_sha256"].as_str().unwrap_or("unknown")
        )
    } else {
        identity
    }
}
pub fn environment_mismatches(py: Python<'_>, info: &Value) -> Result<Vec<EnvironmentMismatch>> {
    let mut differences = Vec::new();
    let mut compare = |category, name: &str, local: &Value, original: &Value, severity| {
        if local != original {
            differences.push(EnvironmentMismatch {
                category,
                component: name.into(),
                local: local.clone(),
                original: original.clone(),
                severity,
            });
        }
    };
    if let Some(original) = info.get("system").filter(|s| !s.is_null()) {
        let here = system(py)?;
        for (name, pointer, severity) in [
            ("Python version", "/python/version", MismatchSeverity::Low),
            ("Operating system", "/os/platform", MismatchSeverity::Low),
            ("CPU", "/cpu/brand", MismatchSeverity::Low),
            (
                "Accelerator type",
                "/accelerators/type",
                MismatchSeverity::High,
            ),
        ] {
            compare(
                "System",
                name,
                here.pointer(pointer).unwrap_or(&Value::Null),
                original.pointer(pointer).unwrap_or(&Value::Null),
                severity,
            );
        }
        if !here["accelerators"]["type"].is_null()
            && here["accelerators"]["type"] == original["accelerators"]["type"]
        {
            for (name, pointer) in [
                ("Accelerator API", "/accelerators/api_version"),
                ("Driver version", "/accelerators/driver_version"),
            ] {
                compare(
                    "System",
                    name,
                    here.pointer(pointer).unwrap_or(&Value::Null),
                    original.pointer(pointer).unwrap_or(&Value::Null),
                    MismatchSeverity::Medium,
                );
            }
            let names = |s: &Value| {
                Value::String(
                    s["accelerators"]["devices"]
                        .as_array()
                        .into_iter()
                        .flatten()
                        .filter_map(|d| d["name"].as_str())
                        .collect::<Vec<_>>()
                        .join("\n"),
                )
            };
            compare(
                "System",
                "Devices",
                &names(&here),
                &names(original),
                MismatchSeverity::Medium,
            );
        }
    }
    let mut current = requirements(py)?.as_object().unwrap().clone();
    let mut original = info["environment"]["requirements"]
        .as_object()
        .context("Missing reproduction package versions")?
        .clone();
    current.insert(
        "heretic-llm".into(),
        json!(format_version(&crate::artifacts::version_info())),
    );
    original.insert(
        "heretic-llm".into(),
        json!(format_version(&info["environment"]["heretic"])),
    );
    current.insert(
        "torch".into(),
        json!(module(py, "torch")?.a("__version__")?.str()?),
    );
    original.insert(
        "torch".into(),
        info["environment"]["pytorch_version"].clone(),
    );
    let names: std::collections::BTreeSet<_> = current.keys().chain(original.keys()).collect();
    for name in names {
        compare(
            "Package",
            name,
            current.get(name).unwrap_or(&Value::Null),
            original.get(name).unwrap_or(&Value::Null),
            package_severity(name),
        );
    }
    Ok(differences)
}
pub fn check_environment(py: Python<'_>, info: &Value, noninteractive: bool) -> Result<bool> {
    if info.get("system").is_none_or(Value::is_null) {
        println!("The reproduction record contains no system information; hardware and system compatibility cannot be checked.");
    }
    let differences = environment_mismatches(py, info)?;
    if differences.is_empty() {
        return Ok(true);
    }
    println!("Your environment differs from the original run:");
    let rows: Vec<_> = differences
        .iter()
        .map(|d| {
            vec![
                d.category.into(),
                d.component.clone(),
                d.local.to_string(),
                d.original.to_string(),
                d.severity.to_string(),
            ]
        })
        .collect();
    print!(
        "{}",
        crate::artifacts::markdown_table(
            &[
                "Category",
                "Component",
                "This system",
                "Original system",
                "Severity"
            ],
            &rows
        )
    );
    println!(
        "Highest mismatch severity: {}. Byte-identical reproduction may not be possible.",
        differences.iter().map(|d| d.severity).max().unwrap()
    );
    if noninteractive {
        bail!("Reproduction environment differs; run interactively to review and proceed")
    }
    Ok(select(
        py,
        "How would you like to proceed?",
        &[
            "Attempt to reproduce the model anyway".into(),
            "Exit program".into(),
        ],
    )? == Some(0))
}
pub fn hash(path: &Path) -> Result<String> {
    let mut f = std::fs::File::open(path)?;
    let mut h = Sha256::new();
    let mut buffer = [0u8; 65536];
    loop {
        let n = f.read(&mut buffer)?;
        if n == 0 {
            break;
        }
        h.update(&buffer[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn verify_hashes(path: &Path, info: &Value) -> Result<()> {
    if let Some(hashes) = info["hashes"].as_object() {
        for (filename, expected) in hashes {
            let file = path.join(filename);
            if !file.starts_with(path)
                || Path::new(filename)
                    .components()
                    .any(|x| !matches!(x, std::path::Component::Normal(_)))
            {
                bail!("Invalid reproduction filename")
            }
            if file.is_file() {
                let actual = hash(&file)?;
                println!(
                    "{filename}: {}",
                    if expected
                        .as_str()
                        .is_some_and(|s| s.eq_ignore_ascii_case(&actual))
                    {
                        "Hash matches"
                    } else {
                        "Hash does not match"
                    }
                );
            } else {
                println!("{filename}: file missing");
            }
        }
    }
    Ok(())
}
pub fn strategy(m: &Model<'_>) -> Result<Option<String>> {
    let s = m.settings.string("export_strategy");
    if !s.is_empty() {
        return Ok(Some(s));
    }
    if m.settings.string("quantization") == "bnb_4bit" {
        println!("Merging a quantized model reloads the base model in CPU memory. Ensure sufficient RAM; exporting only the adapter avoids this reload.");
        let py = m.py;
        let estimate = crate::model::model_class(py, &m.settings.string("model"))?
            .m(
                "from_pretrained",
                &[j(py, m.settings.get("model"))?],
                &[
                    ("device_map", j(py, "meta")?),
                    ("torch_dtype", module(py, "torch")?.a("bfloat16")?),
                ],
            )
            .and_then(|meta| meta.m("get_memory_footprint", &[], &[]))
            .and_then(|v| v.f64());
        match estimate {
            Ok(bytes) => println!(
                "Estimated base-model RAM, excluding overhead: {:.2} GB",
                bytes / 1073741824.0
            ),
            Err(e) if interrupted(&e) => return Err(e),
            Err(_) => {
                println!("Allow approximately 3 GB of RAM per billion parameters, plus overhead.")
            }
        }
    }
    let choice = select(
        m.py,
        "Export strategy",
        &[
            "Merge into full model".into(),
            "Save LoRA adapter".into(),
            "Cancel".into(),
        ],
    )?;
    Ok(match choice {
        Some(0) => Some("merge".into()),
        Some(1) => Some("adapter".into()),
        _ => None,
    })
}
pub fn save(m: &mut Model<'_>, path: &Path, attrs: &Value, checkpoint: &Path) -> Result<()> {
    let Some(strategy) = strategy(m)? else {
        return Ok(());
    };
    std::fs::create_dir_all(path)?;
    let py = m.py;
    let kw = vec![("max_shard_size", j(py, m.settings.get("max_shard_size"))?)];
    let exported = if strategy == "adapter" {
        m.model.clone()
    } else {
        m.merged()?
    };
    exported.m("save_pretrained", &[j(py, path.to_string_lossy())?], &kw)?;
    if strategy == "merge" {
        m.tokenizer
            .m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
        if let Some(p) = &m.processor {
            p.m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
        }
    }
    let mut settings = m.settings.clone();
    settings.set("export_strategy", strategy);
    bundle(py, path, &settings, attrs, checkpoint, false)?;
    println!("Model saved to {}", path.display());
    Ok(())
}
pub fn bundle(
    py: Python<'_>,
    path: &Path,
    settings: &Settings,
    attrs: &Value,
    checkpoint: &Path,
    include_system: bool,
) -> Result<()> {
    let mut hashes = BTreeMap::new();
    for entry in std::fs::read_dir(path)? {
        let file = entry?.path();
        if file
            .extension()
            .is_some_and(|e| e == "safetensors" || e == "bin")
        {
            hashes.insert(
                file.file_name().unwrap().to_string_lossy().to_string(),
                hash(&file)?,
            );
        }
    }
    let folder = path.join("reproduce");
    std::fs::create_dir_all(&folder)?;
    let timestamp = crate::native::timestamp();
    let requirements = requirements(py)?;
    let mut info = json!({"version":"2","timestamp":timestamp,"environment":{"heretic":crate::artifacts::version_info(),"pytorch_version":module(py,"torch")?.a("__version__")?.str()?,"requirements":requirements},"settings":settings.exported(),"parameters":{"direction_index":attrs["direction_index"],"abliteration_parameters":attrs["parameters"]},"metrics":{"kl_divergence":attrs["kl_divergence"],"refusals":attrs["refusals"],"base_refusals":attrs["base_refusals"],"n_bad_prompts":attrs["n_bad_prompts"]},"hashes":hashes});
    if include_system {
        info["system"] = system(py)?;
    }
    std::fs::write(
        folder.join("reproduce.json"),
        serde_json::to_string_pretty(&info)?,
    )?;
    let mut sums = String::new();
    for (file, hash) in hashes {
        sums += &format!("{hash} *{file}\n");
    }
    std::fs::write(folder.join("SHA256SUMS"), sums)?;
    let lines = requirements
        .as_object()
        .unwrap()
        .iter()
        .map(|(k, v)| format!("{k}=={}\n", v.as_str().unwrap()))
        .collect::<String>();
    std::fs::write(folder.join("requirements.txt"), lines)?;
    std::fs::write(
        folder.join("config.toml"),
        crate::native::config_toml(settings)?,
    )?;
    let guide = crate::artifacts::reproduction_readme(
        settings,
        attrs,
        checkpoint
            .file_name()
            .unwrap_or_default()
            .to_str()
            .unwrap_or("checkpoint.jsonl"),
        info.get("system"),
        info["environment"]["pytorch_version"]
            .as_str()
            .unwrap_or("unknown"),
    );
    std::fs::write(folder.join("README.md"), guide)?;
    if checkpoint.is_file() {
        std::fs::copy(checkpoint, folder.join(checkpoint.file_name().unwrap()))?;
    }
    Ok(())
}
