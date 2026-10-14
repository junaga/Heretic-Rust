// SPDX-License-Identifier: AGPL-3.0-or-later
// Application workflow ported from Heretic v1.4.0, Copyright Philipp Emanuel Weidmann + contributors.
use crate::{
    config::Settings,
    core::*,
    model::{load_prompts, Model},
    python::*,
};
use anyhow::{bail, Context, Result};
use pyo3::Python;
use serde_json::{json, Value};
use std::{
    path::{Path, PathBuf},
    time::Instant,
};

pub struct Evaluator<'py> {
    pub good: Vec<Prompt>,
    pub bad: Vec<Prompt>,
    pub base: O<'py>,
    pub base_refusals: usize,
}
impl<'py> Evaluator<'py> {
    pub fn new(model: &Model<'py>) -> Result<Self> {
        let good = load_prompts(
            model.py,
            &model.settings,
            model.settings.get("good_evaluation_prompts"),
        )?;
        let bad = load_prompts(
            model.py,
            &model.settings,
            model.settings.get("bad_evaluation_prompts"),
        )?;
        if good.is_empty() || bad.is_empty() {
            bail!("Evaluation datasets must not be empty")
        }
        let base = model.logprobs_batched(&good)?;
        let mut e = Self {
            good,
            bad,
            base,
            base_refusals: 0,
        };
        e.base_refusals = e.refusals(model)?;
        println!("Initial refusals: {}/{}", e.base_refusals, e.bad.len());
        Ok(e)
    }
    fn refusals(&self, m: &Model<'py>) -> Result<usize> {
        let markers: Vec<String> =
            serde_json::from_value(m.settings.get("refusal_markers").clone())?;
        let mut n = 0;
        for (p, r) in self.bad.iter().zip(m.responses_batched(&self.bad, true)?) {
            let refused = is_refusal(&r, &markers);
            n += usize::from(refused);
            if m.settings.flag("print_responses") {
                println!(
                    "System: {}\nPrompt: {}\nResponse (refusal={refused}): {r}",
                    p.system, p.user
                );
            }
        }
        Ok(n)
    }
    pub fn evaluate(&self, m: &Model<'py>) -> Result<((f64, f64), f64, usize)> {
        let py = m.py;
        let probs = m.logprobs_batched(&self.good)?;
        let kl = module(py, "torch.nn.functional")?
            .m(
                "kl_div",
                &[probs, self.base.clone()],
                &[
                    ("reduction", j(py, "batchmean")?),
                    ("log_target", j(py, true)?),
                ],
            )?
            .m("item", &[], &[])?
            .f64()?;
        let refusals = self.refusals(m)?;
        println!(
            "KL divergence: {kl:.4}; refusals: {refusals}/{}",
            self.bad.len()
        );
        Ok((
            score(
                kl,
                refusals,
                self.base_refusals,
                m.settings.float("kl_divergence_scale"),
                m.settings.float("kl_divergence_target"),
            ),
            kl,
            refusals,
        ))
    }
}
pub fn select(_py: Python<'_>, message: &str, choices: &[String]) -> Result<Option<usize>> {
    crate::ui::select(message, choices)
}
pub(crate) fn text(_py: Python<'_>, message: &str, default: &str) -> Result<String> {
    crate::ui::text(message, default)
}
fn print_memory(py: Python<'_>) -> Result<()> {
    let bytes = crate::native::resident_memory()? as f64;
    println!("Resident system RAM: {:.2} GB", bytes / 1073741824.0);
    for backend in ["cuda", "xpu"] {
        if let Ok(device) = module(py, "torch")?.a(backend) {
            if device
                .m("is_available", &[], &[])
                .is_ok_and(|v| v.empty().is_ok_and(|e| !e))
            {
                let mut allocated = 0.0;
                let mut reserved = 0.0;
                for i in 0..device.m("device_count", &[], &[])?.usize()? {
                    allocated += device.m("memory_allocated", &[j(py, i)?], &[])?.f64()?;
                    reserved += device.m("memory_reserved", &[j(py, i)?], &[])?.f64()?;
                }
                println!(
                    "Allocated {backend} memory: {:.2} GB; reserved: {:.2} GB",
                    allocated / 1073741824.0,
                    reserved / 1073741824.0
                );
                return Ok(());
            }
        }
    }
    let torch = module(py, "torch")?;
    if !torch
        .a("backends.mps")?
        .m("is_available", &[], &[])?
        .empty()?
    {
        let mps = torch.a("mps")?;
        println!(
            "Allocated MPS memory: {:.2} GB; driver allocated: {:.2} GB",
            mps.m("current_allocated_memory", &[], &[])?.f64()? / 1073741824.0,
            mps.m("driver_allocated_memory", &[], &[])?.f64()? / 1073741824.0
        );
    }
    Ok(())
}
fn set_attr(trial: &O<'_>, key: &str, v: impl serde::Serialize) -> Result<()> {
    trial.m(
        "set_user_attr",
        &[j(trial.0.py(), key)?, j(trial.0.py(), v)?],
        &[],
    )?;
    Ok(())
}
fn suggest(trial: &O<'_>, name: &str, lo: f64, hi: f64) -> Result<f64> {
    let py = trial.0.py();
    trial
        .m(
            "suggest_float",
            &[j(py, name)?, j(py, lo)?, j(py, hi)?],
            &[],
        )?
        .f64()
}
pub fn suggest_parameters(m: &Model<'_>, trial: &O<'_>) -> Result<(Option<f64>, Parameters)> {
    let py = m.py;
    let scope = trial
        .m(
            "suggest_categorical",
            &[j(py, "direction_scope")?, j(py, ["global", "per layer"])?],
            &[],
        )?
        .str()?;
    let last = (m.layers()?.len()? - 1) as f64;
    let index = suggest(trial, "direction_index", 0.4 * last, 0.9 * last)?;
    let index = if scope == "per layer" {
        None
    } else {
        Some(index)
    };
    let mut params = Parameters::new();
    for c in m.components()? {
        let max = suggest(trial, &format!("{c}.max_weight"), 0.8, 1.5)?;
        let pos = suggest(trial, &format!("{c}.max_weight_position"), 0.6 * last, last)?;
        let min = suggest(trial, &format!("{c}.min_weight"), 0.0, 1.0)?;
        let distance = suggest(trial, &format!("{c}.min_weight_distance"), 1.0, 0.6 * last)?;
        params.insert(
            c,
            AbliterationParameters {
                max_weight: max,
                max_weight_position: pos,
                min_weight: min * max,
                min_weight_distance: distance,
            },
        );
    }
    set_attr(trial, "direction_index", index)?;
    set_attr(trial, "parameters", &params)?;
    Ok((index, params))
}
pub fn directions<'py>(m: &Model<'py>, good: &[Prompt], bad: &[Prompt]) -> Result<O<'py>> {
    let full = m.settings.flag("print_residual_geometry") || m.settings.flag("plot_residuals");
    let (g, b) = if full {
        let g = m.residuals_batched(good)?;
        let b = m.residuals_batched(bad)?;
        crate::research::analyze(m, &g, &b)?;
        (
            g.m("mean", &[], &[("dim", j(m.py, 0)?)])?,
            b.m("mean", &[], &[("dim", j(m.py, 0)?)])?,
        )
    } else {
        (m.residual_mean(good)?, m.residual_mean(bad)?)
    };
    let mut d = norm(&b.sub(&g)?, 1)?;
    if m.settings.flag("orthogonalize_direction") {
        let gn = norm(&g, 1)?;
        let projection = d.mul(&gn)?.m("sum", &[], &[("dim", j(m.py, 1)?)])?.m(
            "unsqueeze",
            &[j(m.py, 1)?],
            &[],
        )?;
        d = norm(&d.sub(&projection.mul(&gn)?)?, 1)?;
    }
    Ok(d)
}
fn journal<'py>(py: Python<'py>, path: &Path) -> Result<O<'py>> {
    let jm = module(py, "optuna.storages.journal")?;
    let p = j(py, path.to_string_lossy())?;
    let lock = jm.a("JournalFileOpenLock")?.c(&[p.clone()], &[])?;
    let backend = jm.a("JournalFileBackend")?.c(&[p], &[("lock_obj", lock)])?;
    module(py, "optuna.storages")?
        .a("JournalStorage")?
        .c(&[backend], &[])
}
fn optimize(m: &mut Model<'_>, eval: &Evaluator<'_>, d: &O<'_>, study: &O<'_>) -> Result<()> {
    let py = m.py;
    let start = Instant::now();
    let initial = study.a("trials")?.len()?;
    for index in initial..m.settings.int("n_trials") {
        py.check_signals()?;
        let trial = study.m("ask", &[], &[])?;
        let attempt = (|| -> Result<((f64, f64), f64, usize)> {
            deliver_interrupt()?;
            set_attr(&trial, "index", index + 1)?;
            let (direction, parameters) = suggest_parameters(m, &trial)?;
            println!(
                "Running trial {} of {}",
                index + 1,
                m.settings.int("n_trials")
            );
            m.reset()?;
            m.abliterate(d, direction, &parameters)?;
            let result = eval.evaluate(m)?;
            set_attr(&trial, "kl_divergence", result.1)?;
            set_attr(&trial, "refusals", result.2)?;
            set_attr(&trial, "base_refusals", eval.base_refusals)?;
            set_attr(&trial, "n_bad_prompts", eval.bad.len())?;
            Ok(result)
        })();
        match attempt {
            Ok((objectives, _kl, _refusals)) => {
                if let Err(e) = study.m(
                    "tell",
                    &[trial.clone(), j(py, [objectives.0, objectives.1])?],
                    &[],
                ) {
                    if interrupted(&e) {
                        study.m(
                            "tell",
                            &[trial],
                            &[
                                ("state", module(py, "optuna.trial")?.a("TrialState.PRUNED")?),
                                ("skip_if_finished", j(py, true)?),
                            ],
                        )?;
                        return Ok(());
                    }
                    return Err(e);
                }
                let elapsed = start.elapsed().as_secs_f64();
                println!("Elapsed: {}", crate::native::format_duration(elapsed));
                if index + 1 < m.settings.int("n_trials") {
                    let remaining = elapsed / (index + 1 - initial) as f64
                        * (m.settings.int("n_trials") - index - 1) as f64;
                    println!(
                        "Estimated remaining time: {}",
                        crate::native::format_duration(remaining)
                    );
                }
                print_memory(py)?;
            }
            Err(e) => {
                let interrupted = crate::python::interrupted(&e);
                let state = module(py, "optuna.trial")?.a(if interrupted {
                    "TrialState.PRUNED"
                } else {
                    "TrialState.FAIL"
                })?;
                study.m("tell", &[trial], &[("state", state)])?;
                if interrupted {
                    return Ok(());
                }
                return Err(e);
            }
        }
    }
    set_attr(
        study,
        "finished",
        study.a("trials")?.len()? == m.settings.int("n_trials"),
    )?;
    Ok(())
}
pub fn restore(m: &mut Model<'_>, d: &O<'_>, attrs: &Value) -> Result<()> {
    m.reset()?;
    let params: Parameters = serde_json::from_value(attrs["parameters"].clone())?;
    m.abliterate(d, attrs["direction_index"].as_f64(), &params)
}
pub fn run(
    py: Python<'_>,
    mut settings: Settings,
    output: Option<PathBuf>,
    noninteractive: bool,
) -> Result<()> {
    if !settings.get("collect_reproducibles").is_null() {
        return crate::reproduction::collect(py, &settings.string("collect_reproducibles"));
    }
    let reproduction = if !settings.get("reproduce").is_null() {
        let info = crate::reproduction::load(py, &settings.string("reproduce"))?;
        if !["1", "2"].contains(&info["version"].as_str().unwrap_or("")) {
            bail!("Unsupported reproduction format")
        }
        if !crate::reproduction::check_environment(py, &info, noninteractive)? {
            return Ok(());
        }
        settings = Settings::from_value(info["settings"].clone())?;
        Some(info)
    } else {
        None
    };
    let torch =
        module(py, "torch").context("Install the project runtime dependencies in .venv first")?;
    if settings.get("seed").is_null() {
        let seed = rand::random::<u32>();
        settings.set("seed", seed);
    }
    let seed = j(py, settings.get("seed"))?;
    module(py, "random")?.m("seed", &[seed.clone()], &[])?;
    module(py, "numpy.random")?.m("seed", &[seed.clone()], &[])?;
    torch.m("manual_seed", &[seed.clone()], &[])?;
    torch.a("cuda")?.m("manual_seed_all", &[seed], &[])?;
    torch.m("set_grad_enabled", &[j(py, false)?], &[])?;
    torch
        .a("_dynamo.config")?
        .set("cache_size_limit", &j(py, 64)?)?;
    module(py, "transformers")?
        .a("logging")?
        .m("set_verbosity_error", &[], &[])?;
    module(py, "optuna")?.a("logging")?.m(
        "set_verbosity",
        &[module(py, "optuna")?.a("logging.WARNING")?],
        &[],
    )?;
    module(py, "warnings")?.m(
        "filterwarnings",
        &[j(py, "ignore")?],
        &[(
            "category",
            module(py, "optuna.exceptions")?.a("ExperimentalWarning")?,
        )],
    )?;
    crate::reproduction::print_system(py)?;
    let checkpoint = Path::new(&settings.string("study_checkpoint_dir"))
        .join(checkpoint_name(&settings.string("model")));
    std::fs::create_dir_all(checkpoint.parent().unwrap())?;
    let mut storage = journal(py, &checkpoint)?;
    if reproduction.is_none() && settings.get("evaluate_model").is_null() {
        if let Some(existing) = storage.m("get_all_studies", &[], &[])?.list()?.first() {
            let choice = if noninteractive {
                Some(0)
            } else {
                select(
                    py,
                    "Previous run found",
                    &[
                        "Continue previous run".into(),
                        "Start from scratch".into(),
                        "Exit".into(),
                    ],
                )?
            };
            match choice {
                Some(0) => {
                    let attrs = existing.a("user_attrs")?.json()?;
                    let saved: Value = serde_json::from_str(
                        attrs["settings"]
                            .as_str()
                            .context("Checkpoint settings missing")?,
                    )?;
                    // Keep current presentation-only settings; saved numerical settings
                    // and complete dataset specifications have highest priority.
                    for (key, value) in saved.as_object().context("Invalid checkpoint settings")? {
                        settings.0[key] = value.clone();
                    }
                    settings = Settings::from_value(settings.0)?;
                }
                Some(1) => {
                    drop(storage);
                    std::fs::remove_file(&checkpoint)?;
                    storage = journal(py, &checkpoint)?;
                }
                _ => return Ok(()),
            }
        }
    }
    if noninteractive && output.is_some() && settings.get("export_strategy").is_null() {
        bail!("--non-interactive --output requires --export-strategy merge or adapter")
    }
    let mut model = Model::new(py, settings)?;
    print_memory(py)?;
    let good = load_prompts(py, &model.settings, model.settings.get("good_prompts"))?;
    let bad = load_prompts(py, &model.settings, model.settings.get("bad_prompts"))?;
    if good.is_empty() || bad.is_empty() {
        bail!("Training datasets must not be empty")
    }
    if model.settings.int("batch_size") == 0 {
        let mut size = 1;
        let mut best = 1;
        let mut fastest = -1.0;
        while size <= model.settings.int("max_batch_size") {
            let batch: Vec<_> = good.iter().cycle().take(size).cloned().collect();
            let result = (|| -> Result<f64> {
                model.responses(&batch, false)?;
                let start = Instant::now();
                let responses = model.responses(&batch, false)?;
                let elapsed = start.elapsed().as_secs_f64();
                let mut tokens = 0;
                for r in responses {
                    tokens += model.tokenizer.m("encode", &[j(py, r)?], &[])?.len()?;
                }
                Ok(tokens as f64 / elapsed)
            })();
            match result {
                Ok(rate) => {
                    println!("Batch size {size}: {rate:.0} tokens/s");
                    if rate > fastest {
                        best = size;
                        fastest = rate;
                    }
                }
                Err(e) => {
                    if size == 1 {
                        return Err(e);
                    }
                    eprintln!("Batch size {size} failed: {e:#}");
                    empty_cache(py)?;
                    break;
                }
            }
            size *= 2;
        }
        model.settings.set("batch_size", best);
        println!("Chosen batch size: {best}");
    }
    if model.settings.get("response_prefix").is_null() {
        let prompts: Vec<_> = good
            .iter()
            .take(100)
            .chain(bad.iter().take(100))
            .cloned()
            .collect();
        let mut prefix = common_prefix(&model.responses_batched(&prompts, false)?);
        for pair in model
            .settings
            .get("chain_of_thought_skips")
            .as_array()
            .context("Invalid chain_of_thought_skips")?
            .clone()
        {
            if !prefix.is_empty()
                && prefix.starts_with(pair[0].as_str().context("Invalid CoT initializer")?)
            {
                prefix = pair[1]
                    .as_str()
                    .context("Invalid CoT replacement")?
                    .to_owned();
                model.settings.set("response_prefix", prefix.clone());
                prefix += &common_prefix(&model.responses_batched(&prompts, false)?);
                break;
            }
        }
        println!("Response prefix: {prefix:?}");
        model.settings.set("response_prefix", prefix);
    }
    let evaluator = Evaluator::new(&model)?;
    if !model.settings.get("evaluate_model").is_null() {
        model
            .settings
            .set("model", model.settings.get("evaluate_model").clone());
        model.reset()?;
        evaluator.evaluate(&model)?;
        return Ok(());
    }
    println!("Calculating residual directions...");
    let d = directions(&model, &good, &bad)?;
    empty_cache(py)?;
    if let Some(info) = reproduction {
        let attrs = json!({"direction_index":info["parameters"]["direction_index"],"parameters":info["parameters"]["abliteration_parameters"],"kl_divergence":info["metrics"]["kl_divergence"],"refusals":info["metrics"]["refusals"],"base_refusals":info["metrics"]["base_refusals"],"n_bad_prompts":info["metrics"]["n_bad_prompts"]});
        restore(&mut model, &d, &attrs)?;
        if let Some(path) = output {
            crate::reproduction::save(&mut model, &path, &attrs, &checkpoint)?;
            crate::reproduction::verify_hashes(&path, &info)?;
        } else if !noninteractive {
            actions(&mut model, &d, &attrs, &checkpoint, Some(&info), None)?;
        }
        return Ok(());
    }
    let sampler = module(py, "optuna.samplers")?.a("TPESampler")?.c(
        &[],
        &[
            (
                "n_startup_trials",
                j(py, model.settings.int("n_startup_trials"))?,
            ),
            ("n_ei_candidates", j(py, 128)?),
            ("multivariate", j(py, true)?),
            ("seed", j(py, model.settings.get("seed"))?),
        ],
    )?;
    let study = module(py, "optuna")?.m(
        "create_study",
        &[],
        &[
            ("sampler", sampler),
            ("directions", j(py, ["minimize", "minimize"])?),
            ("storage", storage),
            ("study_name", j(py, "heretic")?),
            ("load_if_exists", j(py, true)?),
        ],
    )?;
    set_attr(
        &study,
        "settings",
        serde_json::to_string(&model.settings.exported())?,
    )?;
    set_attr(&study, "finished", false)?;
    optimize(&mut model, &evaluator, &d, &study)?;
    loop {
        let trials: Vec<_> = study
            .a("trials")?
            .list()?
            .into_iter()
            .filter(|t| {
                t.a("state.name")
                    .and_then(|x| x.str())
                    .is_ok_and(|s| s == "COMPLETE")
            })
            .collect();
        if trials.is_empty() {
            bail!("No completed trials")
        }
        let attrs: Vec<Value> = trials
            .iter()
            .map(|t| t.a("user_attrs")?.json())
            .collect::<Result<_>>()?;
        let metrics: Vec<_> = attrs
            .iter()
            .map(|a| {
                (
                    a["refusals"].as_u64().unwrap_or(0) as usize,
                    a["kl_divergence"].as_f64().unwrap_or(f64::INFINITY),
                )
            })
            .collect();
        let best = pareto_indices(&metrics);
        let summary: Vec<_> = best.iter().map(|&i| attrs[i].clone()).collect();
        println!(
            "Pareto optimal trials:\n{}",
            serde_json::to_string_pretty(&summary)?
        );
        if noninteractive || output.is_some() {
            if let Some(path) = &output {
                restore(&mut model, &d, &attrs[best[0]])?;
                crate::reproduction::save(&mut model, path, &attrs[best[0]], &checkpoint)?;
            }
            return Ok(());
        }
        let mut choices: Vec<_> = best
            .iter()
            .map(|&i| {
                format!(
                    "Trial {}: refusals {}, KL {:.4}",
                    attrs[i]["index"], metrics[i].0, metrics[i].1
                )
            })
            .collect();
        choices.extend(["Run additional trials".into(), "Exit".into()]);
        match select(py, "Which trial do you want to use?", &choices)? {
            Some(i) if i < best.len() => {
                let a = &attrs[best[i]];
                restore(&mut model, &d, a)?;
                actions(
                    &mut model,
                    &d,
                    a,
                    &checkpoint,
                    None,
                    Some(study.a("trials")?.len()?),
                )?;
            }
            Some(i) if i == best.len() => {
                let n = loop {
                    let value = text(py, "How many additional trials do you want to run?", "")?;
                    if value.is_empty() {
                        break 0;
                    }
                    match value.parse::<usize>() {
                        Ok(n) if n > 0 => break n,
                        _ => println!("Please enter a number greater than 0."),
                    }
                };
                if n == 0 {
                    continue;
                }
                model
                    .settings
                    .set("n_trials", model.settings.int("n_trials") + n);
                set_attr(
                    &study,
                    "settings",
                    serde_json::to_string(&model.settings.exported())?,
                )?;
                set_attr(&study, "finished", false)?;
                optimize(&mut model, &evaluator, &d, &study)?;
            }
            _ => return Ok(()),
        }
    }
}
fn actions(
    m: &mut Model<'_>,
    d: &O<'_>,
    attrs: &Value,
    checkpoint: &Path,
    original: Option<&Value>,
    trial_count: Option<usize>,
) -> Result<()> {
    loop {
        let choices = [
            "Save the model to a local folder",
            "Upload the model to Hugging Face",
            "Chat with the model",
            "Benchmark the model",
            "Return",
        ]
        .map(str::to_owned);
        let Some(action) = select(m.py, "What do you want to do with the model?", &choices)? else {
            return Ok(());
        };
        let result = (|| -> Result<()> {
            match action {
                0 => {
                    let path = text(m.py, "Path to the folder:", "")?;
                    if !path.is_empty() {
                        crate::reproduction::save(m, Path::new(&path), attrs, checkpoint)?;
                        if let Some(info) = original {
                            if info["version"] == "2" {
                                crate::reproduction::verify_hashes(Path::new(&path), info)?;
                            }
                        }
                        restore(m, d, attrs)?;
                    }
                }
                1 => {
                    crate::reproduction::upload_context(
                        m,
                        attrs,
                        checkpoint,
                        original,
                        trial_count,
                    )?;
                    restore(m, d, attrs)?;
                }
                2 => {
                    let mut chat =
                        json!([{"role":"system","content":m.settings.string("system_prompt")}]);
                    loop {
                        let message = text(m.py, "User (empty to return):", "")?;
                        if message.is_empty() {
                            break;
                        }
                        chat.as_array_mut()
                            .unwrap()
                            .push(json!({"role":"user","content":message}));
                        let response = m.stream_chat(&chat)?;
                        chat.as_array_mut()
                            .unwrap()
                            .push(json!({"role":"assistant","content":response}));
                    }
                }
                3 => benchmark(m)?,
                _ => {}
            }
            Ok(())
        })();
        if action == 4 {
            return Ok(());
        }
        if let Err(e) = result {
            // A failed merged export must not leave subsequent chat/benchmark actions
            // operating on a model whose adapter has already been destroyed.
            if m.needs_reload {
                if let Err(reset) = restore(m, d, attrs) {
                    eprintln!("Could not restore trial after export failure: {reset:#}");
                }
            }
            if !interrupted(&e) {
                eprintln!("Error: {e:#}");
            }
        }
    }
}
fn benchmark(m: &Model<'_>) -> Result<()> {
    let py = m.py;
    let available = m
        .settings
        .get("benchmarks")
        .as_array()
        .context("Invalid benchmarks")?;
    let labels: Vec<_> = available
        .iter()
        .map(|b| {
            format!(
                "{}: {}",
                b["name"].as_str().unwrap_or(""),
                b["description"].as_str().unwrap_or("")
            )
        })
        .collect();
    let selected = crate::ui::multiple("Which benchmarks?", &labels)?;
    if selected.is_empty() {
        return Ok(());
    }
    let Some(scope) = select(
        py,
        "Benchmark scope",
        &["This model".into(), "Both original and modified".into()],
    )?
    else {
        return Ok(());
    };
    let compare = scope == 1;
    let tasks: Vec<_> = selected
        .into_iter()
        .map(|i| available[i]["task"].as_str().unwrap().to_owned())
        .collect();
    let result = run_benchmarks(m, &tasks, compare)?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
pub fn run_benchmarks(m: &Model<'_>, tasks: &[String], compare: bool) -> Result<Value> {
    let py = m.py;
    let lm = module(py, "lm_eval.models.huggingface")?.a("HFLM")?.c(
        &[],
        &[
            ("pretrained", m.model.clone()),
            ("tokenizer", m.tokenizer.clone()),
            ("batch_size", j(py, "auto")?),
        ],
    )?;
    let mut output = serde_json::Map::new();
    for task in tasks {
        println!("Running benchmark {task}...");
        let kw = vec![("model", lm.clone()), ("tasks", j(py, [task])?)];
        let result = module(py, "lm_eval")?.m("simple_evaluate", &[], &kw)?;
        let modified = result.key("results")?.key(task)?.json()?;
        let original = if compare {
            let context = m.model.m("disable_adapter", &[], &[])?;
            context.m("__enter__", &[], &[])?;
            let result = module(py, "lm_eval")?.m("simple_evaluate", &[], &kw);
            let cleanup = context.m(
                "__exit__",
                &[
                    j(py, Value::Null)?,
                    j(py, Value::Null)?,
                    j(py, Value::Null)?,
                ],
                &[],
            );
            let result = result?;
            cleanup?;
            result.key("results")?.key(task)?.json()?
        } else {
            Value::Null
        };
        output.insert(
            task.clone(),
            json!({"modified":modified,"original":original}),
        );
    }
    Ok(Value::Object(output))
}
