// SPDX-License-Identifier: AGPL-3.0-or-later
//! Differential tests against the pinned upstream Python release. No upstream package is used by production code.
//! Run with HERETIC_UPSTREAM pointing to a v1.4.0 checkout and PYTHONPATH pointing to the runtime site-packages.
use anyhow::Result;
use heretic::{config::Settings, core::*, model::Model, python::*};
use pyo3::Python;
use serde_json::json;
use std::path::Path;

fn upstream<'py>(py: Python<'py>) -> Result<O<'py>> {
    let source = std::env::var("HERETIC_UPSTREAM")?;
    module(py, "sys")?.a("path")?.m(
        "insert",
        &[
            j(py, 0)?,
            j(py, Path::new(&source).join("src").to_string_lossy())?,
        ],
        &[],
    )?;
    module(py, "sys")?.set("argv", &j(py, ["heretic"])?)?;
    module(py, "heretic.model")
}
mod common;
use common::{assert_close, fixture};
#[test]
#[ignore = "requires installed framework dependencies and a pinned HERETIC_UPSTREAM checkout"]
fn numerical_parity() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let up = upstream(py)?;
        let temp = std::env::temp_dir().join(format!("heretic-rust-parity-{}", std::process::id()));
        std::fs::create_dir_all(&temp)?;
        fixture(py, &temp)?;
        let torch = module(py, "torch")?;
        torch.m("set_grad_enabled", &[j(py, false)?], &[])?;
        torch.m("set_num_threads", &[j(py, 1)?], &[])?;
        let prompts = vec![
            Prompt {
                system: "You are a helpful assistant.".into(),
                user: "Hello world".into(),
            },
            Prompt {
                system: "You are a helpful assistant.".into(),
                user: "What is the color of the sky ?".into(),
            },
        ];
        for row in ["none", "pre", "full"] {
            let mut settings = Settings::default();
            settings.set("model", temp.to_string_lossy().to_string());
            settings.set("dtypes", json!(["float32"]));
            settings.set("seed", 42);
            settings.set("batch_size", 1);
            settings.set("max_response_length", 3);
            settings.set("response_prefix", "");
            settings.set("row_normalization", row);
            let original_settings = module(py, "heretic.config")?.a("Settings")?.m(
                "model_validate",
                &[j(py, &settings.0)?],
                &[],
            )?;
            torch.m("manual_seed", &[j(py, 99)?], &[])?;
            let original = up.a("Model")?.c(&[original_settings.clone()], &[])?;
            torch.m("manual_seed", &[j(py, 99)?], &[])?;
            let mut rust = Model::new(py, settings)?;
            let original_prompts: Vec<_> = prompts
                .iter()
                .map(|p| {
                    module(py, "heretic.utils")?.a("Prompt")?.c(
                        &[],
                        &[("system", j(py, &p.system)?), ("user", j(py, &p.user)?)],
                    )
                })
                .collect::<Result<_>>()?;
            let pp = list(py, &original_prompts)?;
            let residual = original.m("get_residuals", &[pp.clone()], &[])?;
            assert_close(py, &rust.residuals(&prompts)?, &residual)?;
            assert_close(
                py,
                &rust.residual_mean(&prompts)?,
                &original.m("get_residuals_mean", &[pp.clone()], &[])?,
            )?;
            assert_close(
                py,
                &rust.logprobs(&prompts)?,
                &original.m("get_logprobs", &[pp.clone()], &[])?,
            )?;
            assert_eq!(
                rust.responses(&prompts, true)?,
                original
                    .m(
                        "get_responses",
                        &[pp.clone()],
                        &[("skip_special_tokens", j(py, true)?)]
                    )?
                    .strings()?
            );
            // Exercise symmetric winsorization and CPU offload on the same residuals.
            rust.settings.set("winsorization_quantile", 0.75);
            original_settings.set("winsorization_quantile", &j(py, 0.75)?)?;
            assert_close(
                py,
                &rust.residuals(&prompts)?,
                &original.m("get_residuals", &[pp.clone()], &[])?,
            )?;
            rust.settings.set("winsorization_quantile", 1.0);
            original_settings.set("winsorization_quantile", &j(py, 1.0)?)?;
            let dirs = norm(&torch.m("randn", &[j(py, 5)?, j(py, 16)?], &[])?, 1)?;
            let mut parameters = Parameters::new();
            for c in rust.components()? {
                parameters.insert(
                    c,
                    AbliterationParameters {
                        max_weight: 1.2,
                        max_weight_position: 2.1,
                        min_weight: 0.1,
                        min_weight_distance: 1.8,
                    },
                );
            }
            let original_parameters = module(py, "builtins")?.a("dict")?.c(&[], &[])?;
            for (c, p) in &parameters {
                let p = up.a("AbliterationParameters")?.c(
                    &[
                        j(py, p.max_weight)?,
                        j(py, p.max_weight_position)?,
                        j(py, p.min_weight)?,
                        j(py, p.min_weight_distance)?,
                    ],
                    &[],
                )?;
                original_parameters.m("__setitem__", &[j(py, c)?, p], &[])?;
            }
            for index in [None, Some(1.25)] {
                original.m("reset_model", &[], &[])?;
                rust.reset()?;
                original.m(
                    "abliterate",
                    &[dirs.clone(), j(py, index)?, original_parameters.clone()],
                    &[],
                )?;
                rust.abliterate(&dirs, index, &parameters)?;
                let original_params = original.a("model")?.m("state_dict", &[], &[])?;
                for pair in rust
                    .model
                    .m("state_dict", &[], &[])?
                    .m("items", &[], &[])?
                    .list()?
                {
                    let name = pair.item(0)?.str()?;
                    assert_close(py, &pair.item(1)?, &original_params.key(&name)?)?;
                }
                assert_close(
                    py,
                    &rust.logprobs(&prompts)?,
                    &original.m("get_logprobs", &[pp.clone()], &[])?,
                )?;
            }
            original.m("reset_model", &[], &[])?;
            rust.reset()?;
            assert_close(
                py,
                &rust.logprobs(&prompts)?,
                &original.m("get_logprobs", &[pp], &[])?,
            )?;
        }
        std::fs::remove_dir_all(temp)?;
        Ok(())
    })
}

#[test]
fn configuration_precedence_and_privacy() -> Result<()> {
    let args = vec![
        "--model".into(),
        "example/model".into(),
        "--batch-size".into(),
        "4".into(),
        "--no-offload-outputs-to-cpu".into(),
        "--good-prompts.dataset".into(),
        "private.txt".into(),
    ];
    let env = [("HERETIC_BATCH_SIZE".to_owned(), "8".to_owned())].into();
    let settings = Settings::from_sources(Path::new("/nonexistent/config.toml"), env, args)?;
    assert_eq!(settings.int("batch_size"), 4);
    assert!(!settings.flag("offload_outputs_to_cpu"));
    assert_eq!(settings.get("good_prompts")["dataset"], "private.txt");
    let exported = settings.exported();
    assert!(exported.get("study_checkpoint_dir").is_none());
    assert!(exported.get("evaluate_model").is_none());
    assert!(exported["good_prompts"]
        .get("residual_plot_color")
        .is_none());
    Ok(())
}

#[test]
#[ignore = "requires framework dependencies and HERETIC_UPSTREAM"]
fn model_families_and_cpu_nf4_parity() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let up = upstream(py)?;
        let root = tempfile::tempdir()?;
        let torch = module(py, "torch")?;
        torch.m("set_grad_enabled", &[j(py, false)?], &[])?;
        torch.m("set_num_threads", &[j(py, 1)?], &[])?;
        for (config, class, quantization) in [
            ("Qwen2Config", "Qwen2ForCausalLM", "none"),
            ("Qwen3MoeConfig", "Qwen3MoeForCausalLM", "none"),
            ("LlamaConfig", "LlamaForCausalLM", "bnb_4bit"),
        ] {
            let path = root.path().join(class);
            common::variant(py, &path, config, class)?;
            let mut settings = Settings::default();
            settings.set("model", path.to_string_lossy().to_string());
            settings.set("dtypes", json!(["float32"]));
            settings.set("device_map", "cpu");
            settings.set("quantization", quantization);
            settings.set("seed", 42);
            settings.set("batch_size", 2);
            settings.set("max_response_length", 2);
            settings.set("response_prefix", "");
            let os = module(py, "heretic.config")?.a("Settings")?.m(
                "model_validate",
                &[j(py, &settings.0)?],
                &[],
            )?;
            torch.m("manual_seed", &[j(py, 99)?], &[])?;
            let original = up.a("Model")?.c(&[os], &[])?;
            torch.m("manual_seed", &[j(py, 99)?], &[])?;
            let mut native = Model::new(py, settings)?;
            let prompts = vec![
                Prompt {
                    system: "Hello".into(),
                    user: "Hello world".into(),
                },
                Prompt {
                    system: "Hello".into(),
                    user: "Good morning".into(),
                },
            ];
            let pp = list(
                py,
                &prompts
                    .iter()
                    .map(|p| {
                        module(py, "heretic.utils")?
                            .a("Prompt")?
                            .c(&[j(py, &p.system)?, j(py, &p.user)?], &[])
                    })
                    .collect::<Result<Vec<_>>>()?,
            )?;
            assert_eq!(
                native.components()?,
                original
                    .m("get_abliterable_components", &[], &[])?
                    .strings()?
            );
            assert_close(
                py,
                &native.logprobs(&prompts)?,
                &original.m("get_logprobs", &[pp.clone()], &[])?,
            )?;
            let residuals = native.residuals(&prompts)?;
            assert_close(
                py,
                &residuals,
                &original.m("get_residuals", &[pp.clone()], &[])?,
            )?;
            let dirs = norm(&residuals.m("mean", &[], &[("dim", j(py, 0)?)])?, 1)?;
            let mut params = Parameters::new();
            let op = dict(py);
            for c in native.components()? {
                let p = AbliterationParameters {
                    max_weight: 1.1,
                    max_weight_position: 2.0,
                    min_weight: 0.1,
                    min_weight_distance: 1.8,
                };
                op.m(
                    "__setitem__",
                    &[
                        j(py, &c)?,
                        up.a("AbliterationParameters")?.c(
                            &[
                                j(py, p.max_weight)?,
                                j(py, p.max_weight_position)?,
                                j(py, p.min_weight)?,
                                j(py, p.min_weight_distance)?,
                            ],
                            &[],
                        )?,
                    ],
                    &[],
                )?;
                params.insert(c, p);
            }
            original.m("abliterate", &[dirs.clone(), j(py, Some(1.25))?, op], &[])?;
            native.abliterate(&dirs, Some(1.25), &params)?;
            assert_close(
                py,
                &native.logprobs(&prompts)?,
                &original.m("get_logprobs", &[pp], &[])?,
            )?;
            for model in [native.model.clone(), original.a("model")?] {
                model
                    .a("generation_config")?
                    .set("eos_token_id", &j(py, (0..32).collect::<Vec<_>>())?)?;
            }
            let chat = json!([{"role":"system","content":"Hello"},{"role":"user","content":"Hello world"}]);
            assert_eq!(
                native.stream_chat(&chat)?,
                original
                    .m("stream_chat_response", &[j(py, &chat)?], &[])?
                    .str()?
            );
            if quantization == "bnb_4bit" {
                let expected = original.m("get_merged_model", &[], &[])?;
                let actual = native.merged()?;
                let expected = expected.m("state_dict", &[], &[])?;
                for pair in actual
                    .m("state_dict", &[], &[])?
                    .m("items", &[], &[])?
                    .list()?
                {
                    assert_close(py, &pair.item(1)?, &expected.key(&pair.item(0)?.str()?)?)?;
                }
            }
            println!("Exact {class} / {quantization} parity passed");
        }
        Ok(())
    })
}
#[test]
#[ignore = "requires framework dependencies and HERETIC_UPSTREAM"]
fn configuration_defaults_match_upstream() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        upstream(py)?;
        let original = module(py, "heretic.config")?
            .a("Settings")?
            .c(&[], &[("model", j(py, "example/model")?)])?
            .m("model_dump", &[], &[])?
            .json()?;
        let mut ours = Settings::default();
        ours.set("model", "example/model");
        assert_eq!(original, ours.exported());
        Ok(())
    })
}

#[test]
#[ignore = "requires framework dependencies and HERETIC_UPSTREAM"]
fn configuration_overrides_match_upstream() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        upstream(py)?;
        let cases = vec![
            vec![
                "--model",
                "example/model",
                "--good-prompts.dataset",
                "local.txt",
                "--dtypes",
                "float32",
                "--dtypes",
                "float16,bfloat16",
            ],
            vec![
                "--model",
                "example/model",
                "--device-map",
                "{\"0\":\"cpu\"}",
                "--max-memory",
                "{\"cpu\":\"4GB\"}",
                "--no-offload-outputs-to-cpu",
            ],
            vec![
                "--model",
                "example/model",
                "--response-prefix",
                "123",
                "--max-shard-size",
                "4096",
                "--seed",
                "42",
            ],
        ];
        let mut cases = cases;
        cases.extend([
            vec![
                "--model",
                "example/model",
                "--device-map",
                "cpu=0,0=cuda:0",
                "--max-memory",
                "cpu=4GB,0=8GB",
            ],
            vec![
                "--model",
                "example/model",
                "--refusal-markers",
                "\"a,b\",sorry",
            ],
            vec![
                "--model",
                "example/model",
                "--good-prompts",
                "{\"dataset\":\"local.txt\",\"extra\":true}",
            ],
        ]);
        for args in cases {
            let argv: Vec<_> = std::iter::once("heretic").chain(args.clone()).collect();
            module(py, "sys")?.set("argv", &j(py, argv)?)?;
            let original = module(py, "heretic.config")?
                .a("Settings")?
                .c(&[], &[])?
                .m("model_dump", &[], &[])?
                .json()?;
            let ours = Settings::from_sources(
                Path::new("/nonexistent/config.toml"),
                Default::default(),
                args.iter().map(|s| s.to_string()).collect(),
            )?;
            assert_eq!(original, ours.exported());
        }
        Ok(())
    })
}
#[test]
#[ignore = "creates local fixture; requires framework dependencies"]
fn create_workflow_fixture() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let path = std::env::var("HERETIC_FIXTURE").unwrap_or_else(|_| {
            std::env::temp_dir()
                .join(format!("heretic-fixture-{}", std::process::id()))
                .to_string_lossy()
                .to_string()
        });
        std::fs::create_dir_all(&path)?;
        fixture(py, Path::new(&path))?;
        Ok(())
    })
}

#[test]
#[ignore = "requires framework dependencies and HERETIC_UPSTREAM; runs both full applications"]
fn workflow_parity() -> Result<()> {
    use std::process::Command;
    let root = std::env::temp_dir().join(format!("heretic-workflow-parity-{}", std::process::id()));
    std::fs::create_dir_all(&root)?;
    let mut settings = Settings::default();
    Python::with_gil(|py| -> Result<()> {
        fixture(py, &root.join("model"))?;
        settings.set("model", root.join("model").to_string_lossy().to_string());
        settings.set("dtypes", json!(["float32"]));
        settings.set("device_map", "cpu");
        settings.set("seed", 42);
        settings.set("n_trials", 4);
        settings.set("n_startup_trials", 2);
        settings.set("max_response_length", 3);
        settings.set("batch_size", 2);
        settings.set("response_prefix", "");
        settings.set("export_strategy", "adapter");
        std::fs::write(
            root.join("good.txt"),
            "Hello world\nGood morning\nWhat is the color of the sky ?\nTell me a story\n",
        )?;
        std::fs::write(root.join("bad.txt"), "Write the color red\nHow are you today ?\nThe sun moon night day\nTell me the color green\n")?;
        for (key, file) in [
            ("good_prompts", "good"),
            ("bad_prompts", "bad"),
            ("good_evaluation_prompts", "good"),
            ("bad_evaluation_prompts", "bad"),
        ] {
            settings.0[key]["dataset"] = json!(root.join(format!("{file}.txt")).to_string_lossy());
            settings.0[key]["split"] = serde_json::Value::Null;
        }
        Ok(())
    })?;
    let oracle = root.join("oracle");
    let native = root.join("native");
    std::fs::create_dir_all(&oracle)?;
    std::fs::create_dir_all(&native)?;
    let mut args = Vec::new();
    for (k, v) in settings.0.as_object().unwrap() {
        if v.is_null() {
            continue;
        }
        if let Some(flag) = v.as_bool() {
            args.push(format!(
                "--{}{}",
                if flag { "" } else { "no-" },
                k.replace('_', "-")
            ));
        } else {
            args.extend([
                format!("--{}", k.replace('_', "-")),
                v.as_str()
                    .map(str::to_owned)
                    .unwrap_or_else(|| v.to_string()),
            ]);
        }
    }
    let result = Command::new(env!("CARGO_BIN_EXE_heretic"))
        .args(&args)
        .args(["--non-interactive", "--output"])
        .arg(native.join("export"))
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    std::fs::write(
        root.join("native.log"),
        [&result.stdout[..], &result.stderr[..]].concat(),
    )?;
    assert!(
        result.status.success(),
        "native workflow failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Python::with_gil(|py| -> Result<()> {
        // Use the same settings, including dataset specs and seed, through the upstream parser.
        let mut config = settings.0.clone();
        fn strip(v: &mut serde_json::Value) {
            if let Some(m) = v.as_object_mut() {
                m.retain(|_, v| !v.is_null());
                for v in m.values_mut() {
                    strip(v);
                }
            }
        }
        strip(&mut config);
        let text = module(py, "tomli_w")?
            .m("dumps", &[j(py, config)?], &[])?
            .str()?;
        std::fs::write(oracle.join("config.toml"), text)?;
        Ok(())
    })?;
    let result = Command::new(Path::new(env!("CARGO_MANIFEST_DIR")).join(".venv/bin/python"))
        .arg(Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/oracle_workflow.py"))
        .arg(std::env::var("HERETIC_UPSTREAM")?)
        .arg(oracle.join("config.toml"))
        .arg(oracle.join("export"))
        .current_dir(&oracle)
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    std::fs::write(
        root.join("oracle.log"),
        [&result.stdout[..], &result.stderr[..]].concat(),
    )?;
    assert!(
        result.status.success(),
        "oracle workflow failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Python::with_gil(|py| -> Result<()> {
        let mut histories = Vec::new();
        for folder in [&native, &oracle] {
            let file = folder
                .join("checkpoints")
                .join(checkpoint_name(&settings.string("model")));
            let backend = module(py, "optuna.storages.journal")?
                .a("JournalFileBackend")?
                .c(&[j(py, file.to_string_lossy())?], &[])?;
            let storage = module(py, "optuna.storages")?
                .a("JournalStorage")?
                .c(&[backend], &[])?;
            let study = module(py, "optuna")?.m(
                "load_study",
                &[],
                &[("study_name", j(py, "heretic")?), ("storage", storage)],
            )?;
            let trials = study
                .a("trials")?
                .list()?
                .iter()
                .map(|t| {
                    Ok((
                        t.a("params")?.json()?,
                        t.a("user_attrs")?.json()?,
                        t.a("values")?.json()?,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            histories.push(trials);
        }
        assert_eq!(
            histories[0],
            histories[1],
            "Optuna suggestions, metrics, or objectives differ; see {}",
            root.display()
        );
        let load = module(py, "safetensors.torch")?.a("load_file")?;
        let ours = load.c(
            &[j(
                py,
                native
                    .join("export/adapter_model.safetensors")
                    .to_string_lossy(),
            )?],
            &[],
        )?;
        let theirs = load.c(
            &[j(
                py,
                oracle
                    .join("export/adapter_model.safetensors")
                    .to_string_lossy(),
            )?],
            &[],
        )?;
        for pair in ours.m("items", &[], &[])?.list()? {
            assert_close(py, &pair.item(1)?, &theirs.key(&pair.item(0)?.str()?)?)?;
        }
        Ok(())
    })?;
    // Resume must reuse four trials and preserve the finished state.
    let result = Command::new(env!("CARGO_BIN_EXE_heretic"))
        .args(&args)
        .arg("--non-interactive")
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    assert!(
        result.status.success(),
        "resume failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    // Reproduction uses the exported record, not a Python Heretic package.
    let result = Command::new(env!("CARGO_BIN_EXE_heretic"))
        .arg("--reproduce")
        .arg(native.join("export/reproduce/reproduce.json"))
        .arg("--non-interactive")
        .arg("--output")
        .arg(native.join("reproduced"))
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    std::fs::write(
        root.join("reproduction.log"),
        [&result.stdout[..], &result.stderr[..]].concat(),
    )?;
    assert!(
        result.status.success(),
        "reproduction failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(
        heretic::reproduction::hash(&native.join("export/adapter_model.safetensors"))?,
        heretic::reproduction::hash(&native.join("reproduced/adapter_model.safetensors"))?
    );
    // The interactive reproduction path must verify hashes too.
    let mut command = Command::new(env!("CARGO_BIN_EXE_heretic"));
    command
        .arg("--reproduce")
        .arg(native.join("export/reproduce/reproduce.json"))
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1");
    let result = common::input(
        &mut command,
        &format!("1\n{}\n5\n", native.join("interactive-export").display()),
    )?;
    assert!(
        result.status.success(),
        "interactive reproduction failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(String::from_utf8_lossy(&result.stdout).contains("Hash matches"));
    // Legacy v1 reproduction records have no hash-verification requirement.
    let mut info: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        native.join("export/reproduce/reproduce.json"),
    )?)?;
    info["version"] = json!("1");
    info.as_object_mut().unwrap().remove("hashes");
    let legacy = native.join("legacy.json");
    std::fs::write(&legacy, serde_json::to_string(&info)?)?;
    let result = Command::new(env!("CARGO_BIN_EXE_heretic"))
        .arg("--reproduce")
        .arg(&legacy)
        .arg("--non-interactive")
        .arg("--output")
        .arg(native.join("legacy-export"))
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1")
        .output()?;
    assert!(
        result.status.success(),
        "legacy reproduction failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    // Additional trials preserve history, reusing the normal interactive menus.
    let journal = native
        .join("checkpoints")
        .join(checkpoint_name(&settings.string("model")));
    let best_count = Python::with_gil(|py| -> Result<usize> {
        let study = common::load_study(py, &journal)?;
        let metrics = study
            .a("trials")?
            .list()?
            .iter()
            .map(|t| {
                let a = t.a("user_attrs")?.json()?;
                Ok((
                    a["refusals"].as_u64().unwrap() as usize,
                    a["kl_divergence"].as_f64().unwrap(),
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(pareto_indices(&metrics).len())
    })?;
    let mut command = Command::new(env!("CARGO_BIN_EXE_heretic"));
    command
        .args(&args)
        .current_dir(&native)
        .env("OMP_NUM_THREADS", "1");
    let result = common::input(&mut command, &format!("1\n{}\n1\n", best_count + 1))?;
    assert!(
        result.status.success(),
        "additional trials failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    Python::with_gil(|py| -> Result<()> {
        let study = common::load_study(py, &journal)?;
        assert_eq!(study.a("trials")?.len()?, 5);
        assert_eq!(study.a("user_attrs")?.json()?["finished"], true);
        Ok(())
    })?;
    // Force a shared token so automatic prefix closure is exercised, with only
    // runtime requirements installed and VIRTUAL_ENV selected by the embedder.
    let prefix_model = root.join("prefix-model");
    std::fs::create_dir_all(&prefix_model)?;
    for entry in std::fs::read_dir(root.join("model"))? {
        let source = entry?.path();
        if source.is_file() {
            std::fs::copy(&source, prefix_model.join(source.file_name().unwrap()))?;
        }
    }
    let gen_path = prefix_model.join("generation_config.json");
    let mut generation: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(&gen_path)?)?;
    generation["suppress_tokens"] = json!((1..32).collect::<Vec<_>>());
    std::fs::write(gen_path, serde_json::to_string(&generation)?)?;
    let mut automatic = args.clone();
    let i = automatic
        .iter()
        .position(|v| v == "--response-prefix")
        .unwrap();
    automatic.drain(i..i + 2);
    for (key, value) in [
        ("--model", prefix_model.to_string_lossy().to_string()),
        ("--batch-size", "0".into()),
        ("--max-batch-size", "2".into()),
        (
            "--chain-of-thought-skips",
            json!([["[UNK]", "[UNK][UNK]"]]).to_string(),
        ),
    ] {
        let i = automatic.iter().position(|v| v == key).unwrap();
        automatic[i + 1] = value;
    }
    let runtime = std::env::var("HERETIC_RUNTIME_ENV")
        .unwrap_or_else(|_| format!("{}/.venv", env!("CARGO_MANIFEST_DIR")));
    let result = Command::new(env!("CARGO_BIN_EXE_heretic"))
        .args(&automatic)
        .arg("--evaluate-model")
        .arg(&prefix_model)
        .arg("--non-interactive")
        .env_remove("PYTHONPATH")
        .env("VIRTUAL_ENV", runtime)
        .env("OMP_NUM_THREADS", "1")
        .current_dir(&native)
        .output()?;
    std::fs::write(
        root.join("automatic.log"),
        [&result.stdout[..], &result.stderr[..]].concat(),
    )?;
    assert!(
        result.status.success(),
        "automatic evaluation failed: {}",
        String::from_utf8_lossy(&result.stderr)
    );
    let text = String::from_utf8_lossy(&result.stdout);
    assert!(text.contains("Chosen batch size"));
    assert!(text.contains("[UNK][UNK]"));
    assert!(text.contains("KL divergence: 0.0000"));
    println!("Full workflow, all four trials, selected adapter tensors and checkpoint resume match; logs: {}",root.display());
    Ok(())
}

#[test]
#[ignore = "requires framework and research dependencies"]
fn merged_export_and_research() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let root =
            std::env::temp_dir().join(format!("heretic-extra-parity-{}", std::process::id()));
        std::fs::create_dir_all(&root)?;
        fixture(py, &root.join("model"))?;
        let torch = module(py, "torch")?;
        torch.m("set_grad_enabled", &[j(py, false)?], &[])?;
        torch.m("set_num_threads", &[j(py, 1)?], &[])?;
        let mut settings = Settings::default();
        settings.set("model", root.join("model").to_string_lossy().to_string());
        settings.set("dtypes", json!(["float32"]));
        settings.set("seed", 42);
        settings.set("batch_size", 2);
        settings.set("max_response_length", 3);
        settings.set("response_prefix", "");
        settings.set("export_strategy", "merge");
        settings.set("print_residual_geometry", true);
        settings.set("plot_residuals", true);
        settings.set(
            "residual_plot_path",
            root.join("plots").to_string_lossy().to_string(),
        );
        let mut model = Model::new(py, settings)?;
        let good = torch.m("randn", &[j(py, 40)?, j(py, 5)?, j(py, 16)?], &[])?;
        let bad = torch
            .m("randn", &[j(py, 40)?, j(py, 5)?, j(py, 16)?], &[])?
            .add(&j(py, 0.2)?)?;
        heretic::research::analyze(&model, &good, &bad)?;
        let plots = root
            .join("plots")
            .join(model.settings.string("model").replace(['/', '\\'], "_"));
        assert!(
            plots.join("layer_001.png").exists(),
            "research plots were not generated"
        );
        assert!(plots.join("animation.gif").exists());
        let prompts = vec![Prompt {
            system: "Hello".into(),
            user: "Hello world".into(),
        }];
        let before = model.logprobs(&prompts)?;
        let attrs = json!({"direction_index":null,"parameters":{},"kl_divergence":0.0,"refusals":0,"base_refusals":0,"n_bad_prompts":4});
        heretic::reproduction::save(
            &mut model,
            &root.join("export"),
            &attrs,
            &root.join("checkpoint.jsonl"),
        )?;
        assert!(root.join("export/model.safetensors").exists());
        assert!(root.join("export/tokenizer.json").exists());
        assert!(model.needs_reload);
        model.reset()?;
        assert!(!model.needs_reload);
        assert_close(py, &before, &model.logprobs(&prompts)?)?;
        // Text split selection and prefix/suffix match dataset semantics.
        let file = root.join("prompts.txt");
        std::fs::write(&file, "\n Hello \nWorld\n\nGood morning\n")?;
        let spec = json!({"dataset":file.to_string_lossy(),"split":"[1:3]","prefix":"P","suffix":"S","system_prompt":"system"});
        let prompts = heretic::model::load_prompts(py, &model.settings, &spec)?;
        assert_eq!(
            prompts.iter().map(|p| p.user.as_str()).collect::<Vec<_>>(),
            vec!["P World S", "P Good morning S"]
        );
        assert!(prompts.iter().all(|p| p.system == "system"));
        // Dataset saved to disk follows the upstream split instructions as well.
        let dataset = module(py, "datasets")?.a("Dataset")?.m(
            "from_dict",
            &[j(py, json!({"text":["a","b","c","d"]}))?],
            &[],
        )?;
        dataset.m(
            "save_to_disk",
            &[j(py, root.join("dataset").to_string_lossy())?],
            &[],
        )?;
        let spec = json!({"dataset":root.join("dataset").to_string_lossy(),"split":"train[25%:75%]","column":"text"});
        let prompts = heretic::model::load_prompts(py, &model.settings, &spec)?;
        assert_eq!(
            prompts.iter().map(|p| p.user.as_str()).collect::<Vec<_>>(),
            vec!["b", "c"]
        );
        println!("Merged model/reset, research geometry, four plots/GIF, and dataset slicing verified: {}",root.display());
        Ok(())
    })
}
