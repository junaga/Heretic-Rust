// SPDX-License-Identifier: AGPL-3.0-or-later
//! Integration coverage for service protocols without creating remote resources.
mod common;
use anyhow::Result;
use heretic::{config::Settings, model::Model, python::*, reproduction::*};
use pyo3::{prelude::*, types::PyModule};
use serde_json::{json, Value};
use std::ffi::CString;

fn fixtures(py: Python<'_>) -> Result<O<'_>> {
    Ok(O(PyModule::from_code(
        py,
        &CString::new(include_str!("fake_services.py"))?,
        c"fake_services.py",
        c"heretic_test_services",
    )?
    .into_any()))
}
struct Context<'py>(O<'py>);
impl Drop for Context<'_> {
    fn drop(&mut self) {
        let py = self.0 .0.py();
        let _ = self.0.m(
            "__exit__",
            &[
                j(py, Value::Null).unwrap(),
                j(py, Value::Null).unwrap(),
                j(py, Value::Null).unwrap(),
            ],
            &[],
        );
    }
}
fn enter(context: O<'_>) -> Result<Context<'_>> {
    context.m("__enter__", &[], &[])?;
    Ok(Context(context))
}
fn model<'py>(py: Python<'py>, path: &std::path::Path) -> Result<Model<'py>> {
    common::fixture(py, path)?;
    module(py, "torch")?.m("set_grad_enabled", &[j(py, false)?], &[])?;
    module(py, "torch")?.m("set_num_threads", &[j(py, 1)?], &[])?;
    let mut settings = Settings::default();
    settings.set("model", path.to_string_lossy().to_string());
    settings.set("dtypes", json!(["float32"]));
    settings.set("batch_size", 2);
    settings.set("seed", 42);
    settings.set("response_prefix", "");
    settings.set("max_response_length", 2);
    Model::new(py, settings)
}
fn attrs() -> Value {
    json!({"index":3,"direction_index":1.25,"parameters":{"attn.o_proj":{"max_weight":1.2,"max_weight_position":2.0,"min_weight":0.2,"min_weight_distance":1.5}},"kl_divergence":0.123,"refusals":2,"base_refusals":4,"n_bad_prompts":4})
}

#[test]
#[ignore = "requires framework dependencies"]
fn publishing_collection_and_metadata() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let root = tempfile::tempdir()?;
        let cache = root.path().join("cached.json");
        std::fs::write(&cache, "{}")?;
        let fixture = fixtures(py)?;
        let mocks = enter(
            fixture
                .a("Services")?
                .c(&[j(py, cache.to_string_lossy())?], &[])?,
        )?;
        let mut m = model(py, &root.path().join("model"))?;
        m.settings.set("model", "owner/base");
        for key in [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ] {
            m.settings.0[key]["dataset"] = json!("owner/dataset");
        }
        let journal = root.path().join("history.jsonl");
        std::fs::write(&journal, "JOURNAL\n")?;
        let request = Publication {
            repo_id: "owner/output".into(),
            token: "test-token".into(),
            private: true,
            strategy: "adapter".into(),
            reproduction: ReproductionLevel::Full,
            trial_count: Some(7),
        };
        publish(&mut m, &request, &attrs(), &journal, None)?;
        let uploads = mocks.0.a("uploads")?.json()?;
        let record: Value =
            serde_json::from_str(uploads["reproduce/reproduce.json"].as_str().unwrap())?;
        assert_eq!(record["settings"]["n_trials"], 7);
        assert_eq!(record["settings"]["model_commit"], "base-commit");
        assert_eq!(
            record["settings"]["good_prompts"]["commit"],
            "dataset-commit"
        );
        assert_eq!(record["hashes"]["model.safetensors"], "abc123");
        assert!(record.get("system").is_some());
        assert!(record["environment"]["heretic"]["metadata"]["source_sha256"].is_string());
        assert_eq!(
            uploads["reproduce/SHA256SUMS"],
            "abc123 *model.safetensors\n"
        );
        assert_eq!(uploads["reproduce/history.jsonl"], "JOURNAL\n");
        let guide = uploads["reproduce/README.md"].as_str().unwrap();
        for section in [
            "## Models",
            "## Datasets",
            "## Selected trial",
            "## System",
            "## Environment",
            "## Contents",
            "## How to reproduce",
        ] {
            assert!(guide.contains(section));
        }
        let config: toml::Value =
            toml::from_str(uploads["reproduce/config.toml"].as_str().unwrap())?;
        assert_eq!(config["n_trials"].as_integer(), Some(7));
        let card = mocks.0.a("card")?;
        let text = card.a("text")?.str()?;
        assert!(text.contains("Abliteration parameters"));
        assert!(text.contains("0.1230"));
        assert!(text.contains("ORIGINAL MODEL CARD"));
        assert!(card
            .a("data.tags")?
            .strings()?
            .contains(&"reproducible".into()));
        assert!(!uploads.to_string().contains(root.path().to_str().unwrap()));
        assert!(!uploads.to_string().contains("test-token"));
        let archive = root.path().join("archive");
        let first = collect_records(py, archive.to_str().unwrap())?;
        assert_eq!(
            first,
            CollectionSummary {
                found: 1,
                downloaded: 1,
                already_stored: 0
            }
        );
        assert!(archive
            .join("huggingface.co/owner/model-abcdef0.json")
            .is_file());
        let second = collect_records(py, archive.to_str().unwrap())?;
        assert_eq!(
            second,
            CollectionSummary {
                found: 1,
                downloaded: 0,
                already_stored: 1
            }
        );
        mocks.0.set("network_error", &j(py, true)?)?;
        assert!(collect_records(py, archive.to_str().unwrap())
            .unwrap_err()
            .to_string()
            .contains("network unavailable"));
        mocks.0.set("missing_hash", &j(py, true)?)?;
        assert!(uploaded_hashes(py, "owner/output", "test-token").is_err());
        mocks.0.set("missing_hash", &j(py, false)?)?;
        // Reproduction publishing verifies hashes and does not create a new reproduction record.
        mocks.0.a("uploads")?.m("clear", &[], &[])?;
        publish(
            &mut m,
            &request,
            &attrs(),
            &journal,
            Some(&json!({"version":"2","hashes":{"model.safetensors":"ABC123"}})),
        )?;
        assert!(mocks.0.a("uploads")?.empty()?);
        // Remote card failures must be visible rather than silently omitting metadata.
        mocks.0.set("card_error", &j(py, true)?)?;
        assert!(publish(&mut m, &request, &attrs(), &journal, None)
            .unwrap_err()
            .to_string()
            .contains("card endpoint"));
        Ok(())
    })
}

#[test]
#[ignore = "requires framework dependencies and HERETIC_UPSTREAM"]
fn accelerator_and_cpu_schema_parity() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let upstream = std::env::var("HERETIC_UPSTREAM")?;
        module(py, "sys")?.a("path")?.m(
            "insert",
            &[j(py, 0)?, j(py, format!("{upstream}/src"))?],
            &[],
        )?;
        let original = module(py, "heretic.system")?;
        let fixture = fixtures(py)?;
        for backend in [
            "cpu", "cuda", "rocm", "xpu", "mlu", "sdaa", "musa", "npu", "mps",
        ] {
            let _context = enter(
                fixture
                    .a("Accelerators")?
                    .c(&[j(py, backend)?, original.clone()], &[])?,
            )?;
            let expected = original.m("get_accelerator_info_dict", &[], &[])?.json()?;
            assert_eq!(
                accelerator(py)?,
                expected,
                "accelerator schema differs for {backend}"
            );
            empty_cache(py)?;
        }
        assert_eq!(
            heretic::native::cpu_info(),
            original.m("get_cpu_info_dict", &[], &[])?.json()?
        );
        Ok(())
    })
}

#[test]
#[ignore = "requires framework dependencies"]
fn environment_versions_and_native_bridge() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let value = json!({"null":null,"bool":true,"int":i64::MIN,"uint":u64::MAX,"real":0.12345678901234568,"unicode":"é🦀","list":[1,"a",false]});
        assert_eq!(j(py, &value)?.json()?, value);
        let settings = Settings::default();
        let mut settings = settings;
        settings.set("model", "owner/model");
        let exported = heretic::native::config_toml(&settings)?;
        let parsed: toml::Value = toml::from_str(&exported)?;
        assert!(parsed.get("study_checkpoint_dir").is_none());
        assert_eq!(parsed["row_normalization"].as_str(), Some("full"));
        let req = requirements(py)?;
        let torch = module(py, "torch")?.a("__version__")?.str()?;
        let info = json!({"version":"2","environment":{"requirements":req,"heretic":heretic::artifacts::version_info(),"pytorch_version":torch},"system":system(py)?});
        assert!(environment_mismatches(py, &info)?.is_empty());
        let mut altered = info.clone();
        altered["environment"]["pytorch_version"] = json!("different+cuda");
        let differences = environment_mismatches(py, &altered)?;
        assert!(differences
            .iter()
            .any(|d| d.component == "torch" && d.severity == MismatchSeverity::High));
        let mut altered = info.clone();
        altered["environment"]["requirements"]["missing-package"] = json!("1.0");
        assert!(environment_mismatches(py, &altered)?
            .iter()
            .any(|d| d.component == "missing-package"));
        assert_eq!(package_severity("heretic-llm"), MismatchSeverity::Critical);
        assert_eq!(package_severity("peft"), MismatchSeverity::Medium);
        let original = json!({"version":"1.4.0","metadata":{"type":"pypi"}});
        assert_eq!(format_version(&original), "1.4.0");
        let inner = pyo3::exceptions::PyKeyboardInterrupt::new_err(());
        let outer = pyo3::exceptions::PyRuntimeError::new_err("wrapped");
        outer.value(py).setattr("__context__", inner.value(py))?;
        assert!(interrupted(&anyhow::Error::from(outer)));
        Ok(())
    })
}

#[test]
fn native_http_and_path_rewriting() -> Result<()> {
    use std::{
        io::{Read, Write},
        net::TcpListener,
    };
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let address = listener.local_addr()?;
    let server = std::thread::spawn(move || {
        let (mut stream, _) = listener.accept().unwrap();
        let mut request = [0; 2048];
        let n = stream.read(&mut request).unwrap();
        let request = String::from_utf8_lossy(&request[..n]);
        assert!(request.contains("GET /owner/repo/raw/main/reproduce.json "));
        let body = "{\"version\":\"2\"}";
        write!(
            stream,
            "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
            body.len()
        )
        .unwrap();
    });
    assert_eq!(
        heretic::native::load_json(&format!(
            "http://{address}/owner/repo/blob/main/reproduce.json"
        ))?["version"],
        "2"
    );
    server.join().unwrap();
    Ok(())
}

#[test]
#[ignore = "requires framework and lm-eval dependencies"]
fn real_benchmark_and_quantized_merge_branch() -> Result<()> {
    Python::with_gil(|py| -> Result<()> {
        let root = tempfile::tempdir()?;
        let fixture = fixtures(py)?;
        let mut m = model(py, &root.path().join("model"))?;
        let _bench = enter(
            fixture
                .a("Benchmarks")?
                .c(&[j(py, root.path().to_string_lossy())?], &[])?,
        )?;
        let result = heretic::workflow::run_benchmarks(&m, &["heretic_fixture".into()], true)?;
        assert!(result["heretic_fixture"]["modified"]["acc,none"].is_number());
        assert_eq!(
            result["heretic_fixture"]["modified"],
            result["heretic_fixture"]["original"]
        );
        // The quantized merge branch must reload on CPU and copy adapter state. This
        // exercises that control path with a CPU fixture, without claiming GPU kernel coverage.
        let prompts = vec![heretic::core::Prompt {
            system: "Hello".into(),
            user: "Hello world".into(),
        }];
        let before = m.logprobs(&prompts)?;
        m.settings.set("quantization", "bnb_4bit");
        let merged = m.merged()?;
        assert!(!m.needs_reload);
        assert!(merged.a("config")?.key("unused").is_err());
        assert_eq!(merged.a("device")?.str()?, "cpu");
        common::assert_close(py, &before, &m.logprobs(&prompts)?)?;
        Ok(())
    })
}
