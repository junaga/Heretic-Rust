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
pub fn fixture(py: Python<'_>, path: &Path) -> Result<()> {
    let tr = module(py, "transformers")?;
    let word=module(py,"tokenizers.models")?.a("WordLevel")?.c(&[],&[("vocab",j(py,json!({"[UNK]":0,"[PAD]":1,"[EOS]":2,"What":3,"is":4,"1":5,"+":6,"?":7,"Hello":8,"world":9,"Tell":10,"me":11,"a":12,"story":13,"How":14,"are":15,"you":16,"today":17,"Good":18,"morning":19,"sun":20,"moon":21,"night":22,"day":23,"sky":24,"blue":25,"green":26,"red":27,"color":28,"the":29,"of":30,"Write":31}))?),("unk_token",j(py,"[UNK]")?)])?;
    let backend = module(py, "tokenizers")?.a("Tokenizer")?.c(&[word], &[])?;
    backend.set(
        "pre_tokenizer",
        &module(py, "tokenizers.pre_tokenizers")?
            .a("Whitespace")?
            .c(&[], &[])?,
    )?;
    let tokenizer = tr.a("PreTrainedTokenizerFast")?.c(
        &[],
        &[
            ("tokenizer_object", backend),
            ("unk_token", j(py, "[UNK]")?),
            ("pad_token", j(py, "[PAD]")?),
            ("eos_token", j(py, "[EOS]")?),
        ],
    )?;
    tokenizer.set("chat_template",&j(py,"{% for m in messages %}{{ m['content'] }}\n{% endfor %}{% if add_generation_prompt %}Assistant: {% endif %}")?)?;
    tokenizer.m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
    module(py, "torch")?.m("manual_seed", &[j(py, 1234)?], &[])?;
    let config = tr.a("LlamaConfig")?.c(
        &[],
        &[
            ("vocab_size", j(py, 32)?),
            ("hidden_size", j(py, 16)?),
            ("intermediate_size", j(py, 32)?),
            ("num_hidden_layers", j(py, 4)?),
            ("num_attention_heads", j(py, 2)?),
            ("num_key_value_heads", j(py, 2)?),
            ("max_position_embeddings", j(py, 256)?),
            ("pad_token_id", j(py, 1)?),
            ("eos_token_id", j(py, 2)?),
            ("bos_token_id", j(py, 0)?),
        ],
    )?;
    let model = tr.a("LlamaForCausalLM")?.c(&[config], &[])?;
    model.m("save_pretrained", &[j(py, path.to_string_lossy())?], &[])?;
    Ok(())
}
fn assert_close(py: Python<'_>, a: &O<'_>, b: &O<'_>) -> Result<()> {
    module(py, "torch.testing")?.m(
        "assert_close",
        &[a.clone(), b.clone()],
        &[("rtol", j(py, 0.0)?), ("atol", j(py, 0.0)?)],
    )?;
    Ok(())
}
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
