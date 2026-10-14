// SPDX-License-Identifier: AGPL-3.0-or-later
// Port of Heretic v1.4.0 model operations, Copyright Philipp Emanuel Weidmann + contributors.
use crate::{
    config::Settings,
    core::{Parameters, Prompt},
    python::*,
};
use anyhow::{bail, Context, Result};
use pyo3::Python;
use serde_json::{json, Value};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};

pub struct Model<'py> {
    pub py: Python<'py>,
    pub settings: Settings,
    pub model: O<'py>,
    pub tokenizer: O<'py>,
    pub processor: Option<O<'py>>,
    pub peft_config: O<'py>,
    pub dtype: O<'py>,
    pub needs_reload: bool,
    trusted: BTreeSet<String>,
}
pub fn model_class<'py>(py: Python<'py>, name: &str) -> Result<O<'py>> {
    let tr = module(py, "transformers")?;
    let configs = tr
        .a("PretrainedConfig")?
        .m("get_config_dict", &[j(py, name)?], &[])?;
    let vision = configs
        .list()?
        .iter()
        .any(|c| c.key("vision_config").is_ok());
    tr.a(if vision {
        "AutoModelForImageTextToText"
    } else {
        "AutoModelForCausalLM"
    })
}
impl<'py> Model<'py> {
    pub fn new(py: Python<'py>, settings: Settings) -> Result<Self> {
        let tr = module(py, "transformers")?;
        let name = settings.string("model");
        println!("Loading model {name}...");
        let rev = if settings.get("model_commit").is_null() {
            vec![]
        } else {
            vec![("revision", j(py, settings.get("model_commit"))?)]
        };
        let tokenizer = tr
            .a("AutoTokenizer")?
            .m("from_pretrained", &[j(py, &name)?], &rev)?;
        if tokenizer.a("pad_token")?.is_none() {
            tokenizer.set("pad_token", &tokenizer.a("eos_token")?)?;
        }
        tokenizer.set("padding_side", &j(py, "left")?)?;
        let processor = if model_class(py, &name)?
            .str()?
            .contains("AutoModelForImageTextToText")
        {
            Some(
                tr.a("AutoProcessor")?
                    .m("from_pretrained", &[j(py, &name)?], &rev)?,
            )
        } else {
            None
        };
        let none = j(py, Value::Null)?;
        let mut m = Self {
            py,
            settings,
            model: none.clone(),
            tokenizer,
            processor,
            peft_config: none.clone(),
            dtype: none,
            needs_reload: false,
            trusted: BTreeSet::new(),
        };
        let mut last_error = None;
        for dtype in m
            .settings
            .get("dtypes")
            .as_array()
            .context("dtypes must be an array")?
            .clone()
        {
            println!("  Trying dtype {dtype}");
            let attempt = (|| -> Result<()> {
                let kw = m.load_kwargs(j(py, &dtype)?, false)?;
                m.model = model_class(py, &name)?.m("from_pretrained", &[j(py, &name)?], &kw)?;
                m.dtype = m.model.a("dtype")?;
                m.trusted.insert(name.clone());
                m.generate(
                    &[Prompt {
                        system: m.settings.string("system_prompt"),
                        user: "What is 1+1?".into(),
                    }],
                    &[("max_new_tokens", j(py, 1)?)],
                )?;
                Ok(())
            })();
            match attempt {
                Ok(()) => {
                    last_error = None;
                    break;
                }
                Err(e) => {
                    if control_flow(&e) {
                        return Err(e);
                    }
                    eprintln!("  Failed: {e:#}");
                    last_error = Some(e);
                    m.model = j(py, Value::Null)?;
                    empty_cache(py)?;
                }
            }
        }
        if m.model.is_none() {
            bail!(
                "Failed to load model with all configured dtypes: {:?}",
                last_error
            )
        }
        m.apply_lora()?;
        println!("Transformer model with {} layers", m.layers()?.len()?);
        Ok(m)
    }
    fn load_kwargs(&self, dtype: O<'py>, cpu: bool) -> Result<Vec<(&'static str, O<'py>)>> {
        let py = self.py;
        let mut kw = vec![
            ("dtype", dtype.clone()),
            (
                "device_map",
                j(
                    py,
                    if cpu {
                        json!("cpu")
                    } else {
                        self.settings.get("device_map").clone()
                    },
                )?,
            ),
        ];
        if !self.settings.get("model_commit").is_null() {
            kw.push(("revision", j(py, self.settings.get("model_commit"))?));
        }
        if self.trusted.contains(&self.settings.string("model")) {
            kw.push(("trust_remote_code", j(py, true)?));
        }
        if !cpu {
            let memory = dict(py);
            if let Some(map) = self.settings.get("max_memory").as_object() {
                for (k, v) in map {
                    let key = if k.chars().all(|c| c.is_ascii_digit()) {
                        json!(k.parse::<usize>()?)
                    } else {
                        json!(k)
                    };
                    memory.m("__setitem__", &[j(py, key)?, j(py, v)?], &[])?;
                }
                kw.push(("max_memory", memory));
            }
            if self.settings.string("quantization") == "bnb_4bit" {
                let d = dtype.str()?;
                let d = d.trim_matches('"').trim_start_matches("torch.");
                let compute = module(py, "torch")?.a(if d == "auto" { "bfloat16" } else { d })?;
                let config = module(py, "transformers")?.a("BitsAndBytesConfig")?.c(
                    &[],
                    &[
                        ("load_in_4bit", j(py, true)?),
                        ("bnb_4bit_compute_dtype", compute),
                        ("bnb_4bit_quant_type", j(py, "nf4")?),
                        ("bnb_4bit_use_double_quant", j(py, true)?),
                    ],
                )?;
                kw.push(("quantization_config", config));
            }
        }
        Ok(kw)
    }
    pub fn layers(&self) -> Result<O<'py>> {
        let base = if self
            .model
            .instance_of(&module(self.py, "peft")?.a("PeftModel")?)?
        {
            self.model.a("base_model.model")?
        } else {
            self.model.clone()
        };
        match optional(base.a("model.language_model.layers"))? {
            Some(layers) => Ok(layers),
            None => base.a("model.layers"),
        }
    }
    pub fn layer_modules(&self, index: usize) -> Result<BTreeMap<String, Vec<O<'py>>>> {
        let layer = self.layers()?.item(index as isize)?;
        let mut modules: BTreeMap<String, Vec<O<'py>>> = BTreeMap::new();
        let nn = module(self.py, "torch.nn")?.a("Module")?;
        let mut add = |component: &str, m: O<'py>| -> Result<()> {
            if m.instance_of(&nn)? {
                modules.entry(component.into()).or_default().push(m);
            }
            Ok(())
        };
        for (c, p) in [
            ("attn.o_proj", "self_attn.o_proj"),
            ("attn.o_proj", "linear_attn.out_proj"),
            ("mlp.down_proj", "mlp.down_proj"),
            ("attn.o_proj", "conv.out_proj"),
            ("mlp.down_proj", "feed_forward.w2"),
            ("attn.o_proj", "self_attn.out_proj"),
            ("mlp.down_proj", "shared_mlp.output_linear"),
        ] {
            if let Some(m) = optional(layer.a(p))? {
                add(c, m)?;
            }
        }
        for (path, projection) in [
            ("mlp.experts", "down_proj"),
            ("block_sparse_moe.experts", "w2"),
            ("feed_forward.experts", "w2"),
            ("moe.experts", "output_linear"),
        ] {
            if let Some(experts) = optional(layer.a(path).and_then(|x| x.list()))? {
                for expert in experts {
                    if let Some(m) = optional(expert.a(projection))? {
                        add("mlp.down_proj", m)?;
                    }
                }
            }
        }
        if modules.is_empty() {
            bail!("No abliterable modules found in layer {index}")
        }
        Ok(modules)
    }
    pub fn components(&self) -> Result<Vec<String>> {
        let mut set = BTreeSet::new();
        for i in 0..self.layers()?.len()? {
            set.extend(self.layer_modules(i)?.into_keys());
        }
        Ok(set.into_iter().collect())
    }
    pub fn apply_lora(&mut self) -> Result<()> {
        let py = self.py;
        let mut names = BTreeMap::new();
        for pair in self.model.m("named_modules", &[], &[])?.list()? {
            names.insert(pair.item(1)?.0.as_ptr() as usize, pair.item(0)?.str()?);
        }
        let mut targets = BTreeSet::new();
        for i in 0..self.layers()?.len()? {
            for m in self.layer_modules(i)?.into_values().flatten() {
                if let Some(name) = names.get(&(m.0.as_ptr() as usize)) {
                    targets.insert(name.clone());
                }
            }
        }
        let r = if self.settings.string("row_normalization") == "full" {
            self.settings.int("full_normalization_lora_rank")
        } else {
            1
        };
        self.peft_config = module(py, "peft")?.a("LoraConfig")?.c(
            &[],
            &[
                ("r", j(py, r)?),
                ("target_modules", j(py, targets)?),
                ("lora_alpha", j(py, r)?),
                ("lora_dropout", j(py, 0)?),
                ("bias", j(py, "none")?),
                ("task_type", j(py, "CAUSAL_LM")?),
            ],
        )?;
        self.model = module(py, "peft")?.m(
            "get_peft_model",
            &[self.model.clone(), self.peft_config.clone()],
            &[],
        )?;
        Ok(())
    }
    pub fn reset(&mut self) -> Result<()> {
        let name = optional(self.model.a("config.name_or_path").and_then(|x| x.str()))?
            .unwrap_or_default();
        if name == self.settings.string("model") && !self.needs_reload {
            for pair in self.model.m("named_modules", &[], &[])?.list()? {
                if pair.item(0)?.str()?.contains("lora_B") {
                    if let Some(w) = optional(pair.item(1)?.a("weight"))? {
                        module(self.py, "torch.nn.init")?.m("zeros_", &[w], &[])?;
                    }
                }
            }
        } else {
            self.model = j(self.py, Value::Null)?;
            empty_cache(self.py)?;
            self.model = model_class(self.py, &self.settings.string("model"))?.m(
                "from_pretrained",
                &[j(self.py, self.settings.get("model"))?],
                &self.load_kwargs(self.dtype.clone(), false)?,
            )?;
            self.apply_lora()?;
            self.needs_reload = false;
        }
        Ok(())
    }
    pub fn merged(&mut self) -> Result<O<'py>> {
        if self.settings.string("quantization") == "bnb_4bit" {
            let mut state = BTreeMap::new();
            for p in self.model.m("named_parameters", &[], &[])?.list()? {
                let name = p.item(0)?.str()?;
                if name.contains("lora_") {
                    state.insert(
                        name,
                        p.item(1)?
                            .a("data")?
                            .m("clone", &[], &[])?
                            .m("cpu", &[], &[])?,
                    );
                }
            }
            let base = model_class(self.py, &self.settings.string("model"))?.m(
                "from_pretrained",
                &[j(self.py, self.settings.get("model"))?],
                &self.load_kwargs(self.dtype.clone(), true)?,
            )?;
            let peft = module(self.py, "peft")?.m(
                "get_peft_model",
                &[base, self.peft_config.clone()],
                &[],
            )?;
            for p in peft.m("named_parameters", &[], &[])?.list()? {
                if let Some(w) = state.get(&p.item(0)?.str()?) {
                    let param = p.item(1)?;
                    param.set("data", &w.m("to", &[param.a("device")?], &[])?)?;
                }
            }
            peft.m("merge_and_unload", &[], &[])
        } else {
            let merged = self.model.m("merge_and_unload", &[], &[])?;
            self.needs_reload = true;
            Ok(merged)
        }
    }
    pub fn generate(
        &self,
        prompts: &[Prompt],
        extra: &[(&str, O<'py>)],
    ) -> Result<(O<'py>, O<'py>)> {
        let py = self.py;
        let chats: Vec<Value> = prompts
            .iter()
            .map(|p| json!([{"role":"system","content":p.system},{"role":"user","content":p.user}]))
            .collect();
        let strings = self
            .tokenizer
            .m(
                "apply_chat_template",
                &[j(py, chats)?],
                &[
                    ("add_generation_prompt", j(py, true)?),
                    ("tokenize", j(py, false)?),
                ],
            )?
            .strings()?;
        let prefix = self.settings.string("response_prefix");
        let strings: Vec<_> = strings.into_iter().map(|s| s + &prefix).collect();
        let inputs = self
            .tokenizer
            .c(
                &[j(py, strings)?],
                &[
                    ("return_tensors", j(py, "pt")?),
                    ("padding", j(py, true)?),
                    ("return_token_type_ids", j(py, false)?),
                ],
            )?
            .m("to", &[self.model.a("device")?], &[])?;
        let mut kv = Vec::new();
        for pair in inputs.m("items", &[], &[])?.list()? {
            kv.push((pair.item(0)?.str()?, pair.item(1)?));
        }
        let mut kw: Vec<_> = kv.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        kw.extend_from_slice(extra);
        kw.push(("pad_token_id", self.tokenizer.a("pad_token_id")?));
        kw.push(("do_sample", j(py, false)?));
        Ok((inputs, self.model.m("generate", &[], &kw)?))
    }
    pub fn responses(&self, prompts: &[Prompt], skip: bool) -> Result<Vec<String>> {
        let (inputs, outputs) = self.generate(
            prompts,
            &[(
                "max_new_tokens",
                j(self.py, self.settings.int("max_response_length"))?,
            )],
        )?;
        let start = inputs.key("input_ids")?.dim(1)? as isize;
        let generated =
            outputs.tensor_slice(&[all(self.py), slice(self.py, Some(start), None)?])?;
        self.tokenizer
            .m(
                "batch_decode",
                &[generated],
                &[("skip_special_tokens", j(self.py, skip)?)],
            )?
            .strings()
    }
    pub fn responses_batched(&self, prompts: &[Prompt], skip: bool) -> Result<Vec<String>> {
        let mut result = Vec::new();
        for batch in prompts.chunks(self.settings.int("batch_size").max(1)) {
            result.extend(self.responses(batch, skip)?);
        }
        Ok(result)
    }
    pub fn residuals(&self, prompts: &[Prompt]) -> Result<O<'py>> {
        let py = self.py;
        let torch = module(py, "torch")?;
        let (_, out) = self.generate(
            prompts,
            &[
                ("max_new_tokens", j(py, 1)?),
                ("output_hidden_states", j(py, true)?),
                ("return_dict_in_generate", j(py, true)?),
                ("use_cache", j(py, false)?),
            ],
        )?;
        let hidden = out.a("hidden_states")?.item(0)?;
        let layers: Vec<_> = hidden
            .list()?
            .iter()
            .map(|h| h.tensor_slice(&[all(py), j(py, -1)?, all(py)]))
            .collect::<Result<_>>()?;
        let mut residuals = torch
            .m("stack", &[list(py, &layers)?], &[("dim", j(py, 1)?)])?
            .m("to", &[torch.a("float32")?], &[])?;
        let q = self.settings.float("winsorization_quantile");
        if (0.0..1.0).contains(&q) {
            let threshold = torch.m(
                "quantile",
                &[residuals.m("abs", &[], &[])?, j(py, q)?],
                &[("dim", j(py, 2)?), ("keepdim", j(py, true)?)],
            )?;
            residuals = torch.m("clamp", &[residuals, threshold.neg()?, threshold], &[])?;
        }
        if self.settings.flag("offload_outputs_to_cpu") {
            residuals = residuals.m("cpu", &[], &[])?;
        }
        Ok(residuals)
    }
    pub fn residuals_batched(&self, prompts: &[Prompt]) -> Result<O<'py>> {
        let data = prompts
            .chunks(self.settings.int("batch_size").max(1))
            .map(|batch| self.residuals(batch))
            .collect::<Result<Vec<_>>>()?;
        module(self.py, "torch")?.m("cat", &[list(self.py, &data)?], &[("dim", j(self.py, 0)?)])
    }
    pub fn residual_mean(&self, prompts: &[Prompt]) -> Result<O<'py>> {
        if prompts.is_empty() {
            bail!("prompts must not be empty")
        }
        let mut running: Option<O<'py>> = None;
        for batch in prompts.chunks(self.settings.int("batch_size").max(1)) {
            let sum = self
                .residuals(batch)?
                .m(
                    "sum",
                    &[],
                    &[
                        ("dim", j(self.py, 0)?),
                        ("dtype", module(self.py, "torch")?.a("float64")?),
                    ],
                )?
                .m("cpu", &[], &[])?;
            running = Some(if let Some(s) = running {
                s.add(&sum)?
            } else {
                sum
            });
        }
        running.unwrap().div(&j(self.py, prompts.len())?)?.m(
            "to",
            &[module(self.py, "torch")?.a("float32")?],
            &[],
        )
    }
    pub fn logprobs(&self, prompts: &[Prompt]) -> Result<O<'py>> {
        let py = self.py;
        let (_, out) = self.generate(
            prompts,
            &[
                ("max_new_tokens", j(py, 1)?),
                ("output_logits", j(py, true)?),
                ("return_dict_in_generate", j(py, true)?),
                ("use_cache", j(py, false)?),
            ],
        )?;
        let mut probs = module(py, "torch.nn.functional")?.m(
            "log_softmax",
            &[out.a("logits")?.item(0)?],
            &[("dim", j(py, -1)?)],
        )?;
        if self.settings.flag("offload_outputs_to_cpu") {
            probs = probs.m("cpu", &[], &[])?;
        }
        Ok(probs)
    }
    pub fn logprobs_batched(&self, prompts: &[Prompt]) -> Result<O<'py>> {
        let data = prompts
            .chunks(self.settings.int("batch_size").max(1))
            .map(|b| self.logprobs(b))
            .collect::<Result<Vec<_>>>()?;
        module(self.py, "torch")?.m("cat", &[list(self.py, &data)?], &[("dim", j(self.py, 0)?)])
    }
    pub fn abliterate(
        &self,
        directions: &O<'py>,
        index: Option<f64>,
        parameters: &Parameters,
    ) -> Result<()> {
        let py = self.py;
        let torch = module(py, "torch")?;
        let direction = if let Some(index) = index {
            let shifted = index + 1.0;
            let i = shifted.trunc() as isize;
            Some(norm(
                &directions.item(i)?.m(
                    "lerp",
                    &[directions.item(i + 1)?, j(py, shifted.fract())?],
                    &[],
                )?,
                0,
            )?)
        } else {
            None
        };
        for i in 0..self.layers()?.len()? {
            for (component, modules) in self.layer_modules(i)? {
                let params = parameters
                    .get(&component)
                    .with_context(|| format!("Missing parameters for {component}"))?;
                let Some(weight) = params.weight(i) else {
                    continue;
                };
                let v = if let Some(v) = &direction {
                    v.clone()
                } else {
                    directions.item((i + 1) as isize)?
                };
                for m in modules {
                    let v = v.m("to", &[m.a("weight.device")?], &[])?;
                    let base = m.a("base_layer.weight")?;
                    let quant = base.a("quant_state").ok().filter(|x| !x.is_none());
                    let w = if let Some(q) = quant {
                        module(py, "bitsandbytes.functional")?.m(
                            "dequantize_4bit",
                            &[base.a("data")?, q],
                            &[],
                        )?
                    } else {
                        base
                    };
                    let w = w.m("to", &[torch.a("float32")?], &[])?;
                    let mut w = w.m("view", &[j(py, w.dim(0)?)?, j(py, -1)?], &[])?;
                    let original = w.clone();
                    let row = self.settings.string("row_normalization");
                    let norms = if row != "none" {
                        Some(module(py, "torch.linalg")?.m(
                            "vector_norm",
                            &[w.clone()],
                            &[("dim", j(py, 1)?), ("keepdim", j(py, true)?)],
                        )?)
                    } else {
                        None
                    };
                    if row != "none" {
                        w = norm(&w, 1)?;
                    }
                    let mut a = v.mat(&w)?.m("view", &[j(py, 1)?, j(py, -1)?], &[])?;
                    let mut b =
                        v.mul(&j(py, -weight)?)?
                            .m("view", &[j(py, -1)?, j(py, 1)?], &[])?;
                    if row == "pre" {
                        b = norms.as_ref().unwrap().mul(&b)?;
                    } else if row == "full" {
                        w = norm(&w.add(&b.mat(&a)?)?, 1)?
                            .mul(norms.as_ref().unwrap())?
                            .sub(&original)?;
                        let r = self.peft_config.a("r")?.usize()?;
                        torch.m("manual_seed", &[j(py, self.settings.get("seed"))?], &[])?;
                        let svd = torch.m(
                            "svd_lowrank",
                            &[w],
                            &[("q", j(py, 2 * r + 4)?), ("niter", j(py, 6)?)],
                        )?;
                        let u = svd
                            .item(0)?
                            .tensor_slice(&[all(py), slice(py, None, Some(r as isize))?])?;
                        let s = svd.item(1)?.idx(&slice(py, None, Some(r as isize))?)?.m(
                            "sqrt",
                            &[],
                            &[],
                        )?;
                        let vh = svd
                            .item(2)?
                            .tensor_slice(&[all(py), slice(py, None, Some(r as isize))?])?
                            .a("T")?;
                        let diag = torch.m("diag", &[s], &[])?;
                        b = u.mat(&diag)?;
                        a = diag.mat(&vh)?;
                    }
                    let wa = m.a("lora_A")?.key("default")?.a("weight")?;
                    let wb = m.a("lora_B")?.key("default")?.a("weight")?;
                    wa.set("data", &a.m("to", &[wa.a("dtype")?], &[])?)?;
                    wb.set("data", &b.m("to", &[wb.a("dtype")?], &[])?)?;
                }
            }
        }
        Ok(())
    }
    pub fn stream_chat(&self, chat: &Value) -> Result<String> {
        let py = self.py;
        let prompt = self.tokenizer.m(
            "apply_chat_template",
            &[j(py, chat)?],
            &[
                ("add_generation_prompt", j(py, true)?),
                ("tokenize", j(py, false)?),
            ],
        )?;
        let inputs = self
            .tokenizer
            .c(
                &[prompt],
                &[
                    ("return_tensors", j(py, "pt")?),
                    ("return_token_type_ids", j(py, false)?),
                ],
            )?
            .m("to", &[self.model.a("device")?], &[])?;
        let streamer = module(py, "transformers")?.a("TextStreamer")?.c(
            &[self.tokenizer.clone()],
            &[
                ("skip_prompt", j(py, true)?),
                ("skip_special_tokens", j(py, true)?),
            ],
        )?;
        let mut owned = Vec::new();
        for pair in inputs.m("items", &[], &[])?.list()? {
            owned.push((pair.item(0)?.str()?, pair.item(1)?));
        }
        let mut kw: Vec<_> = owned.iter().map(|(k, v)| (k.as_str(), v.clone())).collect();
        kw.extend([("streamer", streamer), ("max_new_tokens", j(py, 4096)?)]);
        let outputs = self.model.m("generate", &[], &kw)?;
        let new = outputs.tensor_slice(&[
            j(py, 0)?,
            slice(py, Some(inputs.key("input_ids")?.dim(1)? as isize), None)?,
        ])?;
        self.tokenizer
            .m("decode", &[new], &[("skip_special_tokens", j(py, true)?)])?
            .str()
    }
}

pub fn load_prompts(py: Python<'_>, settings: &Settings, spec: &Value) -> Result<Vec<Prompt>> {
    let path = spec["dataset"]
        .as_str()
        .context("Dataset needs a path or ID")?;
    let ds = module(py, "datasets")?;
    let rows = if Path::new(path).is_file() {
        let mut lines: Vec<String> = std::fs::read_to_string(path)?
            .lines()
            .map(str::trim)
            .filter(|s| !s.is_empty())
            .map(str::to_owned)
            .collect();
        if let Some(split) = spec["split"].as_str() {
            let (a, b) = split_slice(py, &format!("_{split}"), lines.len())?;
            lines = lines[a..b].to_vec();
        }
        lines
    } else {
        let split = spec["split"]
            .as_str()
            .context("split is required for datasets")?;
        let col = spec["column"]
            .as_str()
            .context("column is required for datasets")?;
        let data = if Path::new(path).join("state.json").exists() {
            let data = ds.m("load_from_disk", &[j(py, path)?], &[])?;
            let (a, b) = split_slice(py, split, data.len()?)?;
            data.idx(&slice(py, Some(a as isize), Some(b as isize))?)?
        } else {
            let mut kw = vec![("split", j(py, split)?)];
            if is_hf_path(path) {
                kw.push(("revision", j(py, &spec["commit"])?));
            } else {
                kw.extend([
                    ("verification_mode", j(py, "no_checks")?),
                    ("download_mode", j(py, "force_redownload")?),
                ]);
            }
            ds.m("load_dataset", &[j(py, path)?], &kw)?
        };
        data.key(col)?.strings()?
    };
    let system = spec["system_prompt"]
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| settings.string("system_prompt"));
    let pre = spec["prefix"].as_str().unwrap_or("");
    let post = spec["suffix"].as_str().unwrap_or("");
    Ok(rows
        .into_iter()
        .map(|mut user| {
            if !pre.is_empty() {
                user = format!("{pre} {user}");
            }
            if !post.is_empty() {
                user = format!("{user} {post}");
            }
            Prompt {
                system: system.clone(),
                user,
            }
        })
        .collect())
}
pub fn is_hf_path(path: &str) -> bool {
    if Path::new(path).exists() {
        return false;
    }
    let parts: Vec<_> = path.split('/').collect();
    parts.len() <= 2
        && !path.contains("--")
        && !path.contains("..")
        && !path.ends_with(".git")
        && parts.last().is_some_and(|s| s.chars().count() <= 96)
        && parts.iter().all(|s| {
            !s.is_empty()
                && !s.starts_with(['.', '-'])
                && !s.ends_with(['.', '-'])
                && s.chars()
                    .all(|c| c.is_alphanumeric() || matches!(c, '_' | '-' | '.'))
        })
}
fn split_slice(py: Python<'_>, split: &str, length: usize) -> Result<(usize, usize)> {
    let name = split.split('[').next().unwrap();
    let read = module(py, "datasets")?
        .a("ReadInstruction")?
        .m("from_spec", &[j(py, split)?], &[])?
        .m("to_absolute", &[j(py, json!({name:length}))?], &[])?
        .item(0)?;
    Ok((read.a("from_")?.usize()?, read.a("to")?.usize()?))
}
