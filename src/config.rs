// SPDX-License-Identifier: AGPL-3.0-or-later
use anyhow::{bail, Context, Result};
use serde_json::{Map, Value};
use std::{collections::BTreeMap, path::Path};

#[derive(Clone, Debug)]
pub struct Settings(pub Value);

pub fn schema() -> Value {
    serde_json::from_str(include_str!("../resources/settings.json")).unwrap()
}
pub fn merge(dst: &mut Value, src: Value) {
    if let (Some(a), Some(b)) = (dst.as_object_mut(), src.as_object()) {
        for (k, v) in b {
            merge(a.entry(k).or_insert(Value::Null), v.clone());
        }
    } else {
        *dst = src;
    }
}
impl Default for Settings {
    fn default() -> Self {
        let fields = schema();
        let mut v = Map::new();
        for (k, f) in fields["Settings"].as_object().unwrap() {
            v.insert(k.clone(), f["default"].clone());
        }
        for key in [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ] {
            let mut d = Map::new();
            for (k, f) in fields["DatasetSpecification"].as_object().unwrap() {
                d.insert(k.clone(), f["default"].clone());
            }
            let mut d = Value::Object(d);
            merge(&mut d, v[key].clone());
            v.insert(key.into(), d);
        }
        Self(Value::Object(v))
    }
}
impl Settings {
    pub fn get(&self, k: &str) -> &Value {
        &self.0[k]
    }
    pub fn string(&self, k: &str) -> String {
        self.0[k].as_str().unwrap_or("").to_owned()
    }
    pub fn int(&self, k: &str) -> usize {
        self.0[k].as_u64().unwrap_or(0) as usize
    }
    pub fn float(&self, k: &str) -> f64 {
        self.0[k].as_f64().unwrap_or(0.0)
    }
    pub fn flag(&self, k: &str) -> bool {
        self.0[k].as_bool().unwrap_or(false)
    }
    pub fn set(&mut self, k: &str, v: impl Into<Value>) {
        self.0[k] = v.into();
    }
    pub fn from_sources(
        path: &Path,
        env: BTreeMap<String, String>,
        args: Vec<String>,
    ) -> Result<Self> {
        // Combine explicit sources before adding defaults: supplying a dataset
        // replaces the default dataset object, as in Pydantic's nested models.
        let mut settings = Self(Value::Object(Map::new()));
        if path.exists() {
            let parsed: toml::Value = toml::from_str(&std::fs::read_to_string(path)?)?;
            merge(&mut settings.0, serde_json::to_value(parsed)?);
        }
        let fields = schema();
        for (key, f) in fields["Settings"].as_object().unwrap() {
            if let Some(raw) = env.get(&format!("HERETIC_{}", key.to_uppercase())) {
                merge(
                    settings
                        .0
                        .as_object_mut()
                        .unwrap()
                        .entry(key.clone())
                        .or_insert(Value::Null),
                    parse_field(raw, f)?,
                );
            }
        }
        let mut i = 0;
        let mut positional_model = false;
        let mut list_options = std::collections::BTreeSet::new();
        let explicit_model = args
            .iter()
            .any(|a| a == "--model" || a.starts_with("--model="));
        while i < args.len() {
            let arg = &args[i];
            if !arg.starts_with("--") {
                if positional_model || explicit_model {
                    bail!("Unexpected positional argument: {arg}")
                }
                settings.set("model", arg.clone());
                positional_model = true;
                i += 1;
                continue;
            }
            let (name, inline) = arg[2..]
                .split_once('=')
                .map_or((&arg[2..], None), |(a, b)| (a, Some(b)));
            let negative = name.starts_with("no-");
            let name = if negative { &name[3..] } else { name };
            let key = name.replace('-', "_");
            let top = key.split('.').next().unwrap();
            let f = fields["Settings"]
                .get(top)
                .with_context(|| format!("Unknown option --{name}"))?;
            if f["default"].is_boolean() && !key.contains('.') {
                if inline.is_some() {
                    bail!("Boolean options use --{name} or --no-{name}");
                }
                settings.0[&key] = Value::Bool(!negative);
                i += 1;
                continue;
            }
            if negative {
                bail!("--no-{name} is only valid for boolean options")
            }
            let raw = if let Some(v) = inline {
                v
            } else {
                i += 1;
                args.get(i)
                    .with_context(|| format!("Missing value for --{name}"))?
            };
            if let Some((parent, child)) = key.split_once('.') {
                let df = &fields["DatasetSpecification"][child];
                if df.is_null() {
                    bail!("Unknown dataset option: {child}")
                }
                if settings.0.get(parent).is_none() {
                    settings.0[parent] = Value::Object(Map::new());
                }
                settings.0[parent][child] = parse_field(raw, df)?;
            } else {
                let value = if f["annotation"] == "list[str]" {
                    if let Ok(Value::Array(a)) = serde_json::from_str::<Value>(raw) {
                        Value::Array(a)
                    } else {
                        Value::Array(
                            raw.split(',')
                                .map(|s| Value::String(s.trim().trim_matches('"').to_owned()))
                                .collect(),
                        )
                    }
                } else {
                    parse_field(raw, f)?
                };
                if f["default"].is_array() && !list_options.insert(key.clone()) {
                    settings.0[&key]
                        .as_array_mut()
                        .context("Invalid repeated list option")?
                        .extend(value.as_array().context("Expected a list")?.clone());
                } else if value.is_object() {
                    merge(
                        settings
                            .0
                            .as_object_mut()
                            .unwrap()
                            .entry(key)
                            .or_insert(Value::Null),
                        value,
                    );
                } else {
                    settings.0[&key] = value;
                }
            }
            i += 1;
        }
        Self::from_value(settings.0)
    }
    pub fn from_value(input: Value) -> Result<Self> {
        let mut result = Self::default();
        let fields = schema();
        for (key, value) in input.as_object().context("Settings must be an object")? {
            if [
                "good_prompts",
                "bad_prompts",
                "good_evaluation_prompts",
                "bad_evaluation_prompts",
            ]
            .contains(&key.as_str())
            {
                let mut base = Value::Object(
                    fields["DatasetSpecification"]
                        .as_object()
                        .unwrap()
                        .iter()
                        .map(|(k, f)| (k.clone(), f["default"].clone()))
                        .collect(),
                );
                merge(&mut base, value.clone());
                result.0[key] = base;
            } else {
                result.0[key] = value.clone();
            }
        }
        result.validate()?;
        Ok(result)
    }
    pub fn validate(&self) -> Result<()> {
        let fields = schema();
        for (key, v) in self.0.as_object().context("Settings must be an object")? {
            let f = fields["Settings"]
                .get(key)
                .with_context(|| format!("Unknown setting: {key}"))?;
            let d = &f["default"];
            if v.is_null() {
                if !f["annotation"].as_str().unwrap_or("").contains("None") && key != "model" {
                    bail!("{key} cannot be null")
                }
                continue;
            }
            let annotation = f["annotation"].as_str().unwrap_or("");
            if (annotation == "str" || annotation == "str | None") && !v.is_string() {
                bail!("{key} must be a string")
            }
            if annotation == "int | None" && v.as_i64().is_none() && v.as_u64().is_none() {
                bail!("{key} must be an integer")
            }
            if (d.is_boolean() && !v.is_boolean())
                || (d.is_array() && !v.is_array())
                || (d.is_number() && !v.is_number())
            {
                bail!("Invalid type for {key}")
            }
        }
        if self.string("model").is_empty()
            && self.0["collect_reproducibles"].is_null()
            && self.0["reproduce"].is_null()
        {
            bail!("A model ID or local model path is required")
        }
        for (key, values) in [
            ("quantization", vec!["none", "bnb_4bit"]),
            ("row_normalization", vec!["none", "pre", "full"]),
            ("export_strategy", vec!["", "merge", "adapter"]),
        ] {
            if !values.contains(&self.string(key).as_str()) {
                bail!("Invalid {key}: {}", self.0[key])
            }
        }
        for k in [
            "max_batch_size",
            "max_response_length",
            "full_normalization_lora_rank",
        ] {
            if self.int(k) == 0 {
                bail!("{k} must be positive")
            }
        }
        for k in ["batch_size", "n_trials", "n_startup_trials"] {
            if self.0[k].as_u64().is_none() {
                bail!("{k} must be a nonnegative integer")
            }
        }
        if self.float("kl_divergence_scale") <= 0.0 {
            bail!("kl_divergence_scale must be positive")
        }
        for key in [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ] {
            let spec = self.0[key]
                .as_object()
                .with_context(|| format!("{key} must be a dataset specification"))?;
            if spec
                .get("dataset")
                .and_then(Value::as_str)
                .is_none_or(str::is_empty)
            {
                bail!("{key}.dataset is required")
            }
            for (field, value) in spec {
                let df = &fields["DatasetSpecification"][field];
                if df.is_null() {
                    bail!("Unknown dataset setting: {key}.{field}")
                }
                if value.is_null() {
                    if !df["annotation"].as_str().unwrap_or("").contains("None") {
                        bail!("{key}.{field} cannot be null")
                    }
                } else if !value.is_string() {
                    bail!("{key}.{field} must be a string")
                }
            }
        }
        for key in ["dtypes", "refusal_markers"] {
            if !self.0[key]
                .as_array()
                .is_some_and(|a| a.iter().all(Value::is_string))
            {
                bail!("{key} must be a list of strings")
            }
        }
        if self.0["dtypes"].as_array().is_none_or(Vec::is_empty) {
            bail!("dtypes must not be empty")
        }
        if !self.0["chain_of_thought_skips"]
            .as_array()
            .is_some_and(|a| {
                a.iter().all(|p| {
                    p.as_array()
                        .is_some_and(|p| p.len() == 2 && p.iter().all(Value::is_string))
                })
            })
        {
            bail!("chain_of_thought_skips must contain pairs of strings")
        }
        if let Some(memory) = self.0["max_memory"].as_object() {
            if !memory.values().all(Value::is_string) {
                bail!("max_memory values must be strings")
            }
        } else if !self.0["max_memory"].is_null() {
            bail!("max_memory must be an object")
        }
        if !self.0["device_map"].is_string()
            && !self.0["device_map"].as_object().is_some_and(|m| {
                m.values()
                    .all(|v| v.is_string() || v.is_i64() || v.is_u64())
            })
        {
            bail!("device_map must be a string or a device mapping")
        }
        if !self.0["benchmarks"].as_array().is_some_and(|a| {
            a.iter().all(|b| {
                ["task", "name", "description"]
                    .iter()
                    .all(|k| b[*k].is_string())
            })
        }) {
            bail!("benchmarks must contain task, name, and description strings")
        }
        Ok(())
    }
    pub fn exported(&self) -> Value {
        let mut v = self.0.clone();
        let s = schema();
        for (k, f) in s["Settings"].as_object().unwrap() {
            if f["exclude"] == true {
                v.as_object_mut().unwrap().remove(k);
            }
        }
        for k in [
            "good_prompts",
            "bad_prompts",
            "good_evaluation_prompts",
            "bad_evaluation_prompts",
        ] {
            for (n, f) in s["DatasetSpecification"].as_object().unwrap() {
                if f["exclude"] == true {
                    v[k].as_object_mut().unwrap().remove(n);
                }
            }
        }
        v
    }
}
fn parse_value(raw: &str, default: &Value) -> Result<Value> {
    if default.is_string() {
        return Ok(Value::String(raw.into()));
    }
    if default.is_null() {
        return Ok(serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into())));
    }
    serde_json::from_str(raw).with_context(|| format!("Expected JSON value, got {raw:?}"))
}
fn parse_field(raw: &str, field: &Value) -> Result<Value> {
    let annotation = field["annotation"].as_str().unwrap_or("");
    if annotation == "bool" {
        return match raw.to_ascii_lowercase().as_str() {
            "true" | "1" | "yes" | "y" | "on" | "t" => Ok(Value::Bool(true)),
            "false" | "0" | "no" | "n" | "off" | "f" => Ok(Value::Bool(false)),
            _ => bail!("Invalid boolean value: {raw}"),
        };
    }
    if annotation == "int" || annotation == "int | None" {
        if raw == "null" && annotation.contains("None") {
            return Ok(Value::Null);
        }
        return Ok(Value::from(
            raw.trim().parse::<i64>().context("Expected an integer")?,
        ));
    }
    if annotation == "float" {
        return Ok(serde_json::to_value(
            raw.trim().parse::<f64>().context("Expected a number")?,
        )?);
    }
    if annotation == "str" || annotation == "str | None" {
        return Ok(if raw == "null" && annotation.ends_with("None") {
            Value::Null
        } else {
            Value::String(raw.into())
        });
    }
    if annotation.starts_with("str | Dict") {
        return Ok(serde_json::from_str(raw).unwrap_or_else(|_| Value::String(raw.into())));
    }
    if annotation == "int | str" {
        return Ok(Value::String(raw.into()));
    }
    parse_value(raw, &field["default"])
}
pub fn help() -> String {
    let mut out=String::from("Heretic Rust 1.4.0\nUsage: heretic [OPTIONS] MODEL\n\nSettings: config.toml < HERETIC_* environment < command line\nComplex values use JSON. Boolean options support --no-NAME.\n\n");
    for (k, f) in schema()["Settings"].as_object().unwrap() {
        out += &format!(
            "  --{}\n      {} [default: {}]\n",
            k.replace('_', "-"),
            f["description"].as_str().unwrap_or(""),
            f["default"]
        );
        if f["annotation"] == "DatasetSpecification" {
            for (child, field) in schema()["DatasetSpecification"].as_object().unwrap() {
                out += &format!(
                    "      --{}.{}  {}\n",
                    k.replace('_', "-"),
                    child.replace('_', "-"),
                    field["description"].as_str().unwrap_or("")
                );
            }
        }
    }
    out+"\nAdditional options: --help, --version, --check-config, --output PATH, --non-interactive\n"
}
