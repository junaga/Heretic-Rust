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

pub fn load(py: Python<'_>, path: &str) -> Result<Value> {
    let text = if path.to_lowercase().starts_with("https://")
        || path.to_lowercase().starts_with("http://")
    {
        let url = path
            .replace("/blob/", "/raw/")
            .replace("/src/branch/", "/raw/branch/");
        module(py, "urllib.request")?
            .m("urlopen", &[j(py, url)?], &[("timeout", j(py, 60)?)])?
            .m("read", &[], &[])?
            .m("decode", &[j(py, "utf-8")?], &[])?
            .str()?
    } else {
        std::fs::read_to_string(path)?
    };
    Ok(serde_json::from_str(&text)?)
}
pub fn requirements(py: Python<'_>) -> Result<Value> {
    use std::collections::{BTreeSet, VecDeque};
    let mut values = serde_json::Map::new();
    let mut queue: VecDeque<String> = [
        "accelerate",
        "bitsandbytes",
        "datasets",
        "huggingface-hub",
        "kernels",
        "lm-eval",
        "numpy",
        "optuna",
        "peft",
        "psutil",
        "py-cpuinfo",
        "pydantic-settings",
        "questionary",
        "rich",
        "tomli-w",
        "torch",
        "torchvision",
        "tqdm",
        "transformers",
        "tokenizers",
        "triton",
        "torchaudio",
        "immutabledict",
        "langdetect",
    ]
    .map(str::to_owned)
    .into();
    let metadata = module(py, "importlib.metadata")?;
    let mut visited = BTreeSet::new();
    while let Some(package) = queue.pop_front() {
        let name = package.to_lowercase().replace('_', "-");
        if !visited.insert(name.clone()) {
            continue;
        }
        let Ok(distribution) = metadata.m("distribution", &[j(py, &name)?], &[]) else {
            continue;
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
        ("npu", "NPU", "CANN"),
        ("mlu", "MLU", "MLU"),
        ("sdaa", "SDAA", "SDAA"),
        ("musa", "MUSA", "MUSA"),
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
    let cpu = module(py, "cpuinfo")?.m("get_cpu_info", &[], &[])?.json()?;
    let sys = module(py, "sys")?;
    let env = if std::env::var_os("CONDA_PREFIX").is_some() {
        "Conda"
    } else if sys.a("base_prefix")?.str()? != sys.a("prefix")?.str()? {
        "Virtualenv/Venv"
    } else {
        "System"
    };
    Ok(
        json!({"python":{"version":platform.m("python_version",&[],&[])?.str()?,"implementation":platform.m("python_implementation",&[],&[])?.str()?,"compiler":platform.m("python_compiler",&[],&[])?.str()?,"environment":env},"os":{"platform":platform.m("platform",&[],&[])?.str()?,"machine":platform.m("machine",&[],&[])?.str()?},"cpu":{"brand":cpu["brand_raw"],"vendor":cpu["vendor_id_raw"],"family":cpu["family"],"model":cpu["model"],"stepping":cpu["stepping"]},"accelerators":accelerator(py)?}),
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
pub fn check_environment(py: Python<'_>, info: &Value, noninteractive: bool) -> Result<bool> {
    let ours = requirements(py)?;
    let mut differences = Vec::new();
    if let Some(req) = info["environment"]["requirements"].as_object() {
        for (name, version) in req {
            if ours[name] != *version {
                differences.push(format!("{name}: local {}, original {version}", ours[name]));
            }
        }
    }
    // A Rust port must never claim byte identity with the Python implementation.
    if info["environment"]["heretic"]["metadata"]["implementation"] != "rust" {
        differences.push(
            "Heretic implementation: Rust port; source run used a different implementation".into(),
        );
    }
    if let Some(original) = info.get("system") {
        let here = system(py)?;
        for (label, pointer) in [
            ("Python", "/python/version"),
            ("OS", "/os/platform"),
            ("CPU", "/cpu/brand"),
            ("Accelerator", "/accelerators/type"),
            ("Accelerator API", "/accelerators/api_version"),
            ("Driver", "/accelerators/driver_version"),
            ("Devices", "/accelerators/devices"),
        ] {
            if here.pointer(pointer) != original.pointer(pointer) {
                differences.push(format!("{label} differs"));
            }
        }
    }
    if differences.is_empty() {
        return Ok(true);
    }
    println!(
        "Reproduction environment differences:\n{}",
        differences.join("\n")
    );
    if noninteractive {
        bail!("Reproduction environment differs; run interactively to review and proceed")
    }
    Ok(select(
        py,
        "Attempt reproduction despite environment differences?",
        &["Attempt reproduction".into(), "Exit".into()],
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
    let timestamp = module(py, "datetime")?
        .a("datetime")?
        .m("now", &[module(py, "datetime")?.a("timezone.utc")?], &[])?
        .m("isoformat", &[], &[("timespec", j(py, "seconds")?)])?
        .str()?;
    let requirements = requirements(py)?;
    let mut info = json!({"version":"2","timestamp":timestamp,"environment":{"heretic":{"version":"1.4.0","is_standard_pypi":false,"metadata":{"type":"local","implementation":"rust","upstream":serde_json::from_str::<Value>(include_str!("../resources/upstream.json"))?}},"pytorch_version":module(py,"torch")?.a("__version__")?.str()?,"requirements":requirements},"settings":settings.exported(),"parameters":{"direction_index":attrs["direction_index"],"abliteration_parameters":attrs["parameters"]},"metrics":{"kl_divergence":attrs["kl_divergence"],"refusals":attrs["refusals"],"base_refusals":attrs["base_refusals"],"n_bad_prompts":attrs["n_bad_prompts"]},"hashes":hashes});
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
    // tomli-w handles omission of nulls after recursive cleaning, matching upstream exports.
    let mut config = settings.exported();
    strip_nulls(&mut config);
    let toml = module(py, "tomli_w")?
        .m("dumps", &[j(py, config)?], &[])?
        .str()?;
    std::fs::write(folder.join("config.toml"), toml)?;
    let guide=format!("# Reproducing this model\n\nGenerated by the Rust application port of Heretic v1.4.0. The application uses CPython bindings to PyTorch, Transformers, PEFT and Optuna; it does not call the upstream Heretic Python application.\n\nBuild the Rust project with `cargo build --release`, install the framework versions from `requirements.txt` in a matching Python environment, then run `heretic --reproduce reproduce.json`. The program reports environment differences before attempting reproduction. Original model: `{}`. Selected trial: {}. Refusals: {}/{}. KL divergence: {}.\n\n`SHA256SUMS` records exported weight hashes. Exact reproduction depends on framework versions, hardware and kernel determinism.\n",settings.string("model"),attrs["index"],attrs["refusals"],attrs["n_bad_prompts"],attrs["kl_divergence"]);
    std::fs::write(folder.join("README.md"), guide)?;
    if checkpoint.is_file() {
        std::fs::copy(checkpoint, folder.join(checkpoint.file_name().unwrap()))?;
    }
    Ok(())
}
fn strip_nulls(v: &mut Value) {
    match v {
        Value::Object(map) => {
            map.retain(|_, v| !v.is_null());
            for x in map.values_mut() {
                strip_nulls(x);
            }
        }
        Value::Array(a) => {
            for x in a {
                strip_nulls(x);
            }
        }
        _ => {}
    }
}
pub fn collect(py: Python<'_>, path: &str) -> Result<()> {
    let hub = module(py, "huggingface_hub")?;
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
    for model in models.0.try_iter()? {
        let m = O(model?);
        let id = m.a("id")?.str()?;
        if m.a("tags")
            .and_then(|v| v.strings())
            .is_ok_and(|v| v.iter().any(|s| s == "gguf"))
        {
            continue;
        }
        if !m.a("gated")?.empty()?
            && api
                .m(
                    "auth_check",
                    &[j(py, &id)?],
                    &[("repo_type", j(py, "model")?)],
                )
                .is_err()
        {
            continue;
        }
        let paths = api.m(
            "get_paths_info",
            &[j(py, &id)?, j(py, "reproduce/reproduce.json")?],
            &[("expand", j(py, true)?)],
        )?;
        if paths.len()? == 0 {
            continue;
        }
        let commit = paths.item(0)?.a("last_commit.oid")?.str()?;
        let (user, repo) = id.split_once('/').context("Invalid model ID")?;
        let file = Path::new(path)
            .join("huggingface.co")
            .join(user)
            .join(format!("{repo}-{}.json", &commit[..7.min(commit.len())]));
        if file.exists() {
            continue;
        }
        let cached = hub
            .m(
                "hf_hub_download",
                &[j(py, &id)?, j(py, "reproduce/reproduce.json")?],
                &[],
            )?
            .str()?;
        std::fs::create_dir_all(file.parent().unwrap())?;
        std::fs::copy(cached, &file)?;
        println!("Stored {id}");
    }
    Ok(())
}
pub fn upload(m: &mut Model<'_>, attrs: &Value, checkpoint: &Path) -> Result<()> {
    let py = m.py;
    let hub = module(py, "huggingface_hub")?;
    let mut token = hub.m("get_token", &[], &[])?;
    if token.is_none() {
        token = module(py, "questionary")?
            .m("password", &[j(py, "Hugging Face access token:")?], &[])?
            .m("ask", &[], &[])?;
    }
    if token.is_none() || token.empty()? {
        return Ok(());
    }
    let user = hub.m("whoami", &[token.clone()], &[])?.json()?;
    let default = format!(
        "{}/{}-heretic",
        user["name"].as_str().context("Account name missing")?,
        Path::new(&m.settings.string("model"))
            .file_name()
            .unwrap_or_default()
            .to_string_lossy()
    );
    let repo = module(py, "questionary")?
        .m(
            "text",
            &[j(py, "Repository name:")?],
            &[("default", j(py, default)?)],
        )?
        .m("ask", &[], &[])?;
    if repo.is_none() || repo.empty()? {
        return Ok(());
    }
    let visibility = select(
        py,
        "Repository visibility",
        &["Private".into(), "Public".into(), "Cancel".into()],
    )?;
    let private = match visibility {
        Some(0) => true,
        Some(1) => false,
        _ => return Ok(()),
    };
    let Some(strategy) = strategy(m)? else {
        return Ok(());
    };
    let reproducible = is_hf_path(&m.settings.string("model"))
        && [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ]
        .iter()
        .all(|k| is_hf_path(m.settings.get(k)["dataset"].as_str().unwrap_or("")));
    let repro = if reproducible {
        select(py,"Reproduction information to publish (settings and dependency versions; Full also includes hardware)",&["Full".into(),"Basic".into(),"None".into()])?
    } else {
        Some(2)
    };
    let Some(repro) = repro else { return Ok(()) };
    // The final prompt makes the remote resource name and visibility explicit before creation.
    if select(
        py,
        &format!(
            "Create or update {} ({})?",
            repo.str()?,
            if private { "private" } else { "public" }
        ),
        &["Upload".into(), "Cancel".into()],
    )? != Some(0)
    {
        return Ok(());
    }
    let exported = if strategy == "adapter" {
        m.model.clone()
    } else {
        m.merged()?
    };
    let kw = vec![
        ("private", j(py, private)?),
        ("token", token.clone()),
        ("max_shard_size", j(py, m.settings.get("max_shard_size"))?),
    ];
    exported.m("push_to_hub", &[repo.clone()], &kw)?;
    if strategy == "merge" {
        let kw = [("private", j(py, private)?), ("token", token.clone())];
        m.tokenizer.m("push_to_hub", &[repo.clone()], &kw)?;
        if let Some(p) = &m.processor {
            p.m("push_to_hub", &[repo.clone()], &kw)?;
        }
    }
    let card = hub
        .a("ModelCard")?
        .m("load", &[j(py, m.settings.get("model"))?], &[])
        .ok();
    if let Some(card) = card {
        let data = card.a("data")?;
        let mut tags = data.a("tags").and_then(|v| v.strings()).unwrap_or_default();
        tags.extend(["heretic", "uncensored", "decensored", "abliterated"].map(str::to_owned));
        if repro < 2 {
            tags.push("reproducible".into());
        }
        data.set("tags", &j(py, tags)?)?;
        let intro=format!("Model modified with Heretic Rust (Heretic v1.4.0 algorithms).\n\nRefusals: {}; KL divergence: {}.\n\n",attrs["refusals"],attrs["kl_divergence"]);
        card.set("text", &j(py, intro + &card.a("text")?.str()?)?)?;
        card.m("push_to_hub", &[repo.clone()], &[("token", token.clone())])?;
    }
    if repro < 2 {
        let mut settings = m.settings.clone();
        settings.set("export_strategy", strategy);
        if settings.get("model_commit").is_null() {
            settings.set(
                "model_commit",
                hub.m("model_info", &[j(py, settings.get("model"))?], &[])?
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
                    .m("dataset_info", &[j(py, &settings.0[key]["dataset"])?], &[])?
                    .a("sha")?
                    .str()?);
            }
        }
        let api = hub.a("HfApi")?.c(&[], &[])?;
        let info = api.m(
            "model_info",
            &[repo.clone()],
            &[("files_metadata", j(py, true)?), ("token", token.clone())],
        )?;
        let mut hashes = serde_json::Map::new();
        for file in info.a("siblings")?.list()? {
            let name = file.a("rfilename")?.str()?;
            if name.ends_with(".safetensors") {
                if let Ok(hash) = file.a("lfs.sha256").and_then(|v| v.str()) {
                    hashes.insert(name, json!(hash));
                }
            }
        }
        let suffix = module(py, "secrets")?
            .m("token_hex", &[j(py, 16)?], &[])?
            .str()?;
        let temp = std::env::temp_dir().join(format!("heretic-rust-reproduce-{suffix}"));
        std::fs::create_dir(&temp)?;
        let result = (|| -> Result<()> {
            bundle(py, &temp, &settings, attrs, checkpoint, repro == 0)?;
            let file = temp.join("reproduce/reproduce.json");
            let mut value: Value = serde_json::from_str(&std::fs::read_to_string(&file)?)?;
            let sums = hashes
                .iter()
                .map(|(filename, sha)| format!("{} *{filename}\n", sha.as_str().unwrap_or("")))
                .collect::<String>();
            value["hashes"] = Value::Object(hashes);
            std::fs::write(&file, serde_json::to_string_pretty(&value)?)?;
            std::fs::write(temp.join("reproduce/SHA256SUMS"), sums)?;
            api.m(
                "upload_folder",
                &[],
                &[
                    ("repo_id", repo),
                    (
                        "folder_path",
                        j(py, temp.join("reproduce").to_string_lossy())?,
                    ),
                    ("path_in_repo", j(py, "reproduce")?),
                    ("token", token),
                ],
            )?;
            Ok(())
        })();
        std::fs::remove_dir_all(&temp)?;
        result?;
    }
    Ok(())
}
